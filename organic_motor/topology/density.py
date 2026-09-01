"""Material density field: design variables -> physical material fields.

The design is described by two continuous fields defined on the node grid:

  z     : (3, N, N) logits for the three material phases (air, iron, PM),
          mapped through a softmax so ``rho_air + rho_iron + rho_pm == 1``.
  theta : (N, N) magnetisation direction (angle) of the permanent magnet.

From these we build the physical fields consumed by the Maxwell solver:

  nu           : magnetic reluctivity (1/mu)     -- iron via SIMP, air elsewhere
  Mx, My       : remanent magnetisation [A/m]    -- ``rho_pm * M_sat * mhat``

The whole chain (softmax -> mask -> Helmholtz filter -> projection -> SIMP) is
pure JAX and therefore fully differentiable w.r.t. ``z`` and ``theta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks
from organic_motor.physics.material import interpolate_nu, magnetization
from organic_motor.topology.filters import (
    filtered_density_pair,
    helmholtz_filter,
)
from organic_motor.topology.projection import heaviside_projection


@dataclass
class MaterialFields:
    rho_air: jnp.ndarray
    rho_iron: jnp.ndarray
    rho_pm: jnp.ndarray
    nu: jnp.ndarray
    Mx: jnp.ndarray
    My: jnp.ndarray


def assemble(z: jnp.ndarray, theta: jnp.ndarray, cfg: MotorConfig,
             temperature: float | None = None) -> MaterialFields:
    """Map design variables (z, theta) to physical material fields."""
    T = cfg.sm_temp_init if temperature is None else temperature
    design = domain_masks(cfg)["design"]

    # --- softmax -> volume fractions (sum to 1) ---
    rho = jax.nn.softmax(z / T, axis=0)
    rho_air, rho_iron, rho_pm = rho[0], rho[1], rho[2]

    # --- confine materials to the design annulus ---
    rho_iron = rho_iron * design
    rho_pm = rho_pm * design

    # --- density filter (min feature size) ---
    rho_iron, rho_pm = filtered_density_pair(rho_iron, rho_pm, cfg)

    # --- optional Heaviside projection (crisp boundaries) ---
    if cfg.projection_beta > 0.0:
        rho_iron = heaviside_projection(rho_iron, cfg.projection_beta)
        rho_pm = heaviside_projection(rho_pm, cfg.projection_beta)

    rho_iron = rho_iron * design
    rho_pm = rho_pm * design
    rho_air = jnp.clip(1.0 - rho_iron - rho_pm, 0.0, 1.0)

    # --- SIMP-interpolated reluctance (PM ~ air permeability) ---
    nu = interpolate_nu(rho_iron, cfg)

    # --- remanent magnetisation (filtered for smooth direction) ---
    Mx, My = magnetization(rho_pm, theta, cfg)
    if cfg.filt_radius > 0.0:
        Mx = helmholtz_filter(Mx, cfg) * design
        My = helmholtz_filter(My, cfg) * design

    return MaterialFields(
        rho_air=rho_air, rho_iron=rho_iron, rho_pm=rho_pm,
        nu=nu, Mx=Mx, My=My,
    )


def random_init(cfg: MotorConfig, key: jax.Array):
    """Random initial design: near-uniform material + random magnetisation."""
    key_z, key_t = jax.random.split(key)
    z = 0.2 * jax.random.normal(key_z, (3, cfg.N, cfg.N))
    theta = jax.random.uniform(key_t, (cfg.N, cfg.N), minval=0.0,
                               maxval=2.0 * jnp.pi)
    return z, theta


def material_label(rho_iron: jnp.ndarray, rho_pm: jnp.ndarray) -> jnp.ndarray:
    """Argmax material index: 0 air, 1 iron, 2 PM (for export/plotting)."""
    rho_air = jnp.clip(1.0 - rho_iron - rho_pm, 0.0, 1.0)
    stack = jnp.stack([rho_air, rho_iron, rho_pm], axis=0)
    return jnp.argmax(stack, axis=0).astype(jnp.int32)
