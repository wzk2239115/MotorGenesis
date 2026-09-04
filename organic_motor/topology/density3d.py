"""Four-phase 3-D topology fields with continuous rotor ownership."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.domain3d import domain_masks3d
from organic_motor.topology.filters3d import helmholtz_filter3d
from organic_motor.topology.projection import heaviside_projection


@dataclass
class TopologyFields3D:
    """Physical fractions and moving-domain ownership, all ``(Nx, Ny, Nz)``."""

    rho_air: jnp.ndarray
    rho_iron: jnp.ndarray
    rho_copper: jnp.ndarray
    rho_pm: jnp.ndarray
    rotor_ownership: jnp.ndarray
    # Printed dielectric: electromagnetically it IS air (it rides inside
    # rho_air for the Maxwell/electric solves via the realize() complement);
    # carried separately so the thermal blend can give it a real (low, but
    # non-zero) conductivity instead of floating every coil thermally.
    rho_insulator: jnp.ndarray | None = None
    # Printed coolant: a pure void thermally dead-ends the heat path; the
    # thermal solve turns its density into the internal convection sink
    # (h * S_v * rho_coolant), the reduced-order conjugate heat transfer.
    rho_coolant: jnp.ndarray | None = None

    @property
    def phases(self) -> jnp.ndarray:
        """Stacked ``(air, iron, copper, pm)`` fractions."""
        return jnp.stack(
            (self.rho_air, self.rho_iron, self.rho_copper, self.rho_pm), axis=0
        )


def assemble3d(
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    cfg: MotorConfig3D,
    temperature: float | None = None,
) -> TopologyFields3D:
    """Map unconstrained fields to a conservative native 3-D topology.

    Material fractions form a simplex at every voxel.  Iron may occupy both
    design regions, copper is stationary, PM is rotor-owned, and the shaft,
    closed air gap and exterior remain free of optimized material.
    """
    expected = (4,) + cfg.shape
    if logits.shape != expected:
        raise ValueError(f"logits shape must be {expected}, got {logits.shape}")
    if rotor_logits.shape != cfg.shape:
        raise ValueError(
            f"rotor_logits shape must be {cfg.shape}, got {rotor_logits.shape}"
        )
    temp = cfg.sm_temp_init if temperature is None else temperature
    if temp <= 0.0:
        raise ValueError("temperature must be positive")

    masks = domain_masks3d(cfg)
    design = masks["design"].astype(logits.dtype)
    rotor_design = masks["rotor_design"].astype(logits.dtype)
    stator_design = masks["stator_design"].astype(logits.dtype)

    phases = jax.nn.softmax(logits / temp, axis=0)
    if cfg.filt_radius > 0.0:
        phases = jnp.stack(
            [helmholtz_filter3d(phases[i], cfg) for i in range(4)], axis=0
        )
    phases = jnp.clip(phases, 0.0, None)

    ownership = jax.nn.sigmoid(rotor_logits / temp)
    if cfg.filt_radius > 0.0:
        ownership = helmholtz_filter3d(ownership, cfg)
    if cfg.projection_beta > 0.0:
        ownership = heaviside_projection(ownership, cfg.projection_beta)
    ownership = jnp.clip(ownership, 0.0, 1.0) * rotor_design

    # Disallowed solid fractions are returned to air before renormalization.
    iron = phases[1] * design
    copper = phases[2] * stator_design
    pm = phases[3] * rotor_design
    if cfg.projection_beta > 0.0:
        iron = heaviside_projection(iron, cfg.projection_beta) * design
        copper = heaviside_projection(copper, cfg.projection_beta) * stator_design
        pm = heaviside_projection(pm, cfg.projection_beta) * rotor_design

    solids = iron + copper + pm
    scale = jnp.maximum(1.0, solids)
    iron, copper, pm = iron / scale, copper / scale, pm / scale
    air = 1.0 - iron - copper - pm
    return TopologyFields3D(air, iron, copper, pm, ownership)


def random_init3d(
    cfg: MotorConfig3D, key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize four phase logits and rotor-ownership logits."""
    phase_key, ownership_key = jax.random.split(key)
    logits = 0.2 * jax.random.normal(phase_key, (4,) + cfg.shape)
    rotor_logits = 0.2 * jax.random.normal(ownership_key, cfg.shape)
    return logits, rotor_logits


def material_label3d(fields: TopologyFields3D) -> jnp.ndarray:
    """Integer labels: 0 air, 1 iron, 2 copper, 3 permanent magnet."""
    return jnp.argmax(fields.phases, axis=0).astype(jnp.int32)


MaterialFields3D = TopologyFields3D
DensityFields3D = TopologyFields3D
assemble = assemble3d
random_init = random_init3d
material_label = material_label3d
