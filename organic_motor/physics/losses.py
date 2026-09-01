"""Differentiable reduced-order electromagnetic loss models."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from organic_motor.config import MotorConfig


@dataclass
class LossFields:
    copper: jnp.ndarray
    iron: jnp.ndarray
    total: jnp.ndarray


def electromagnetic_losses(Bx: jnp.ndarray, By: jnp.ndarray, Jz: jnp.ndarray,
                           rho_iron: jnp.ndarray, rho_copper: jnp.ndarray,
                           cfg: MotorConfig) -> LossFields:
    """Return volumetric copper and first-order iron losses [W/m^3]."""
    # Homogenized conductor: J scales with rho_copper, so division by rho gives
    # the correct linear volume scaling without rewarding diffuse copper.
    copper = (Jz * Jz / (cfg.sigma_copper * (rho_copper + 1e-6)))
    copper = jnp.where(rho_copper > 1e-6, copper, 0.0)
    b2 = Bx * Bx + By * By
    iron = (cfg.iron_loss_coeff * cfg.electrical_frequency
            * b2 / (cfg.iron_loss_B_ref ** 2) * rho_iron)
    return LossFields(copper=copper, iron=iron, total=copper + iron)


def saturation_penalty(Bx: jnp.ndarray, By: jnp.ndarray,
                       rho_iron: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Mean squared flux-density excess in iron, normalized by B_sat."""
    bmag = jnp.sqrt(Bx * Bx + By * By + 1e-12)
    excess = jnp.maximum(bmag / cfg.B_sat_iron - 1.0, 0.0)
    return jnp.mean(rho_iron * excess * excess)
