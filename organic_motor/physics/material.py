"""Material constitutive laws (physical layer).

Soft-magnetic iron, permanent magnet and air are captured by two scalar laws:

  * reluctance ``nu(rho_iron)`` via SIMP interpolation between air and iron,
  * remanent magnetisation ``M(rho_pm, theta)`` proportional to PM volume
    fraction, oriented along ``theta``.

This module contains only the *physics* of the materials; the mapping from the
design variables to the volume fractions lives in
:mod:`organic_motor.topology.density`.
"""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config import MotorConfig


def interpolate_nu(rho_iron: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """SIMP reluctance: nu = nu_air + (nu_iron - nu_air) * rho_iron^p."""
    return cfg.nu_air + (cfg.nu_iron - cfg.nu_air) * (rho_iron ** cfg.simp_p)


def magnetization(rho_pm: jnp.ndarray, theta: jnp.ndarray,
                  cfg: MotorConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Remanent magnetisation M = M_sat * rho_pm * [cos, sin](theta)  [A/m]."""
    return cfg.M_sat * rho_pm * jnp.cos(theta), cfg.M_sat * rho_pm * jnp.sin(theta)


def mass_density(rho_iron: jnp.ndarray, rho_pm: jnp.ndarray,
                 cfg: MotorConfig, rho_copper: jnp.ndarray | None = None) -> jnp.ndarray:
    """Volumetric mass density field [kg/m^3] (air contributes zero)."""
    copper = 0.0 if rho_copper is None else cfg.rho_copper_kg * rho_copper
    return cfg.rho_iron_kg * rho_iron + cfg.rho_pm_kg * rho_pm + copper
