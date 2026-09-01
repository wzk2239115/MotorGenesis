"""JAX-differentiable electromechanical and thermal transient primitives.

The functions in this module are intentionally configuration-independent so a
future 3-D forward model can compose them inside ``jax.lax.scan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass
class RotorState:
    """Mechanical rotor state in SI units."""

    angle: Array
    angular_velocity: Array

    def tree_flatten(self):
        return (self.angle, self.angular_velocity), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class ThreePhaseState:
    """Three phase currents ordered ``(a, b, c)``."""

    currents: Array

    def tree_flatten(self):
        return (self.currents,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class DQState:
    """Direct/quadrature axis currents."""

    current_d: Array
    current_q: Array

    def tree_flatten(self):
        return (self.current_d, self.current_q), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class Transient3DState:
    """Composable state for a three-phase, lumped-temperature forward model."""

    rotor: RotorState
    circuit: ThreePhaseState
    temperature: Array

    def tree_flatten(self):
        return (self.rotor, self.circuit, self.temperature), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def load_torque(
    angular_velocity: Array,
    constant: Array | float = 0.0,
    viscous: Array | float = 0.0,
    quadratic: Array | float = 0.0,
) -> Array:
    """Constant, viscous, and signed-quadratic opposing load torque."""
    omega = jnp.asarray(angular_velocity)
    direction = jnp.where(omega == 0.0, 0.0, jnp.sign(omega))
    return constant * direction + viscous * omega + quadratic * omega * jnp.abs(
        omega
    )


def advance_rotor(
    state: RotorState,
    electromagnetic_torque: Array,
    load: Array,
    inertia: Array | float,
    dt: Array | float,
    viscous_friction: Array | float = 0.0,
    wrap_angle: bool = False,
) -> RotorState:
    """Semi-implicit integration of rotor angular speed and angle."""
    acceleration = (
        electromagnetic_torque
        - load
        - viscous_friction * state.angular_velocity
    ) / inertia
    angular_velocity = state.angular_velocity + dt * acceleration
    angle = state.angle + dt * angular_velocity
    if wrap_angle:
        angle = jnp.mod(angle + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    return RotorState(angle=angle, angular_velocity=angular_velocity)


def sinusoidal_back_emf(
    mechanical_angle: Array,
    mechanical_angular_velocity: Array,
    pole_pairs: int | Array,
    flux_linkage: Array | float,
    phase_offset: Array | float = 0.0,
) -> Array:
    """Return sinusoidal phase back-EMF ``(a,b,c)``."""
    electrical_angle = pole_pairs * mechanical_angle + phase_offset
    electrical_speed = pole_pairs * mechanical_angular_velocity
    shifts = jnp.asarray([0.0, -2.0 * jnp.pi / 3.0, 2.0 * jnp.pi / 3.0])
    return electrical_speed * flux_linkage * jnp.sin(
        electrical_angle[..., None] + shifts
    )


def advance_three_phase_rl(
    state: ThreePhaseState,
    phase_voltage: Array,
    back_emf: Array,
    resistance: Array | float,
    inductance: Array | float,
    dt: Array | float,
    enforce_zero_sequence: bool = True,
) -> ThreePhaseState:
    """Advance independent phase RL equations with an exact constant-input step."""
    currents = state.currents
    resistance = jnp.asarray(resistance, dtype=currents.dtype)
    inductance = jnp.asarray(inductance, dtype=currents.dtype)
    decay = jnp.exp(-resistance * dt / inductance)
    safe_r = jnp.where(jnp.abs(resistance) > 1e-12, resistance, 1.0)
    forced = (phase_voltage - back_emf) / safe_r
    rl_current = decay * currents + (1.0 - decay) * forced
    lossless_current = currents + dt * (phase_voltage - back_emf) / inductance
    next_currents = jnp.where(
        jnp.abs(resistance) > 1e-12, rl_current, lossless_current
    )
    if enforce_zero_sequence:
        next_currents = next_currents - jnp.mean(
            next_currents, axis=-1, keepdims=True
        )
    return ThreePhaseState(currents=next_currents)


def advance_dq(
    state: DQState,
    voltage_d: Array,
    voltage_q: Array,
    electrical_angular_velocity: Array,
    resistance: Array | float,
    inductance_d: Array | float,
    inductance_q: Array | float,
    pm_flux_linkage: Array | float,
    dt: Array | float,
) -> DQState:
    """Explicit dq-axis PMSM circuit step, including cross-coupling and EMF."""
    derivative_d = (
        voltage_d
        - resistance * state.current_d
        + electrical_angular_velocity * inductance_q * state.current_q
    ) / inductance_d
    derivative_q = (
        voltage_q
        - resistance * state.current_q
        - electrical_angular_velocity
        * (inductance_d * state.current_d + pm_flux_linkage)
    ) / inductance_q
    return DQState(
        current_d=state.current_d + dt * derivative_d,
        current_q=state.current_q + dt * derivative_q,
    )


def dq_electromagnetic_torque(
    state: DQState,
    pole_pairs: int | Array,
    pm_flux_linkage: Array | float,
    inductance_d: Array | float,
    inductance_q: Array | float,
) -> Array:
    """PMSM dq torque, including reluctance torque."""
    return (
        1.5
        * pole_pairs
        * (
            pm_flux_linkage * state.current_q
            + (inductance_d - inductance_q)
            * state.current_d
            * state.current_q
        )
    )


def transient_joule_loss(
    current_density: Array,
    conductivity: Array,
    active_mask: Array | float = 1.0,
) -> Array:
    """Native-3D Joule loss density ``|J|^2/sigma`` [W/m^3]."""
    if current_density.ndim != 4 or current_density.shape[-1] != 3:
        raise ValueError("current_density must have shape (nx, ny, nz, 3)")
    return active_mask * jnp.sum(current_density * current_density, axis=-1) / (
        conductivity + 1e-30
    )


def transient_iron_loss(
    magnetic_flux_density: Array,
    flux_density_rate: Array,
    hysteresis_coefficient: Array | float,
    eddy_coefficient: Array | float,
    electrical_frequency: Array | float,
    iron_mask: Array | float = 1.0,
) -> Array:
    """Differentiable transient hysteresis-plus-eddy loss density."""
    if (
        magnetic_flux_density.ndim != 4
        or magnetic_flux_density.shape[-1] != 3
        or flux_density_rate.shape != magnetic_flux_density.shape
    ):
        raise ValueError("B and dB/dt must have shape (nx, ny, nz, 3)")
    b_magnitude = jnp.sqrt(
        jnp.sum(magnetic_flux_density * magnetic_flux_density, axis=-1) + 1e-30
    )
    db2 = jnp.sum(flux_density_rate * flux_density_rate, axis=-1)
    return iron_mask * (
        hysteresis_coefficient * jnp.abs(electrical_frequency) * b_magnitude**2
        + eddy_coefficient * db2
    )


def advance_lumped_temperature(
    temperature: Array,
    heat_power: Array,
    ambient_temperature: Array | float,
    heat_capacity: Array | float,
    thermal_conductance: Array | float,
    dt: Array | float,
) -> Array:
    """Advance a lumped thermal state by one explicit time step."""
    temperature_rate = (
        heat_power
        - thermal_conductance * (temperature - ambient_temperature)
    ) / heat_capacity
    return temperature + dt * temperature_rate


def _axis_flux_divergence(
    temperature: Array,
    conductivity: Array,
    spacing: Array,
    axis: int,
) -> Array:
    """Finite-volume ``div(k grad(T))`` contribution with insulated faces."""
    count = temperature.shape[axis]
    if count < 2:
        return jnp.zeros_like(temperature)
    low = jnp.arange(count - 1)
    high = jnp.arange(1, count)
    t_low = jnp.take(temperature, low, axis=axis)
    t_high = jnp.take(temperature, high, axis=axis)
    k_low = jnp.take(conductivity, low, axis=axis)
    k_high = jnp.take(conductivity, high, axis=axis)
    face_k = 2.0 * k_low * k_high / (k_low + k_high + 1e-30)
    face_flux = face_k * (t_high - t_low) / spacing
    pad_shape = list(temperature.shape)
    pad_shape[axis] = 1
    zero_face = jnp.zeros(pad_shape, dtype=temperature.dtype)
    incoming = jnp.concatenate([zero_face, face_flux], axis=axis)
    outgoing = jnp.concatenate([face_flux, zero_face], axis=axis)
    return (outgoing - incoming) / spacing


def advance_voxel_temperature(
    temperature: Array,
    heat_density: Array,
    conductivity: Array,
    volumetric_heat_capacity: Array,
    spacing: float | tuple[float, float, float],
    dt: Array | float,
    ambient_temperature: Array | float | None = None,
    cooling_coefficient: Array | float = 0.0,
    cooling_mask: Array | float = 0.0,
) -> Array:
    """Advance a true 3-D variable-property heat equation explicitly.

    Grid exterior faces are adiabatic.  ``cooling_coefficient * cooling_mask``
    supplies an optional volumetric ambient sink for boundary/interface models.
    """
    if temperature.ndim != 3:
        raise ValueError("temperature must have shape (nx, ny, nz)")
    shape = temperature.shape
    heat_density = jnp.broadcast_to(heat_density, shape)
    conductivity = jnp.broadcast_to(conductivity, shape)
    heat_capacity = jnp.broadcast_to(volumetric_heat_capacity, shape)
    h = jnp.asarray(spacing, dtype=temperature.dtype)
    if h.ndim == 0:
        h = jnp.repeat(h[None], 3)
    if h.shape != (3,):
        raise ValueError("spacing must be scalar or length three")
    conduction = sum(
        _axis_flux_divergence(temperature, conductivity, h[axis], axis)
        for axis in range(3)
    )
    cooling = 0.0
    if ambient_temperature is not None:
        cooling = (
            cooling_coefficient
            * cooling_mask
            * (temperature - ambient_temperature)
        )
    return temperature + dt * (conduction + heat_density - cooling) / (
        heat_capacity + 1e-30
    )


def advance_transient_step(
    state: Transient3DState,
    phase_voltage: Array,
    electromagnetic_torque: Array,
    heat_power: Array,
    dt: Array | float,
    *,
    pole_pairs: int | Array,
    flux_linkage: Array | float,
    phase_resistance: Array | float,
    phase_inductance: Array | float,
    rotor_inertia: Array | float,
    ambient_temperature: Array | float,
    thermal_capacity: Array | float,
    thermal_conductance: Array | float,
    load_model: Callable[[Array], Array] | None = None,
) -> Transient3DState:
    """One composable electromechanical/thermal step for a future 3-D forward."""
    back_emf = sinusoidal_back_emf(
        state.rotor.angle,
        state.rotor.angular_velocity,
        pole_pairs,
        flux_linkage,
    )
    circuit = advance_three_phase_rl(
        state.circuit,
        phase_voltage,
        back_emf,
        phase_resistance,
        phase_inductance,
        dt,
    )
    load = (
        jnp.asarray(0.0)
        if load_model is None
        else load_model(state.rotor.angular_velocity)
    )
    rotor = advance_rotor(
        state.rotor, electromagnetic_torque, load, rotor_inertia, dt
    )
    temperature = advance_lumped_temperature(
        state.temperature,
        heat_power,
        ambient_temperature,
        thermal_capacity,
        thermal_conductance,
        dt,
    )
    return Transient3DState(
        rotor=rotor, circuit=circuit, temperature=temperature
    )
