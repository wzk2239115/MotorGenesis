"""Electrical excitation models for rotating-field motor simulations."""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks, meshgrid


def three_phase_current_density(electrical_angle: float,
                                cfg: MotorConfig,
                                rho_copper: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return a balanced distributed three-phase axial current density.

    The equivalent sinusoidal current sheet is the fundamental of a balanced
    three-phase winding.  Its spatial field rotates with ``electrical_angle``.
    """
    X, Y, _ = meshgrid(cfg)
    alpha = jnp.arctan2(Y, X)
    phase = electrical_angle + cfg.electrical_phase_offset
    winding = domain_masks(cfg)["winding"].astype(X.dtype)
    conductor = winding if rho_copper is None else winding * rho_copper
    return (cfg.current_density_peak * conductor
            * jnp.cos(cfg.pole_pairs * alpha - phase))


def synchronous_electrical_angle(mechanical_angle: float,
                                 cfg: MotorConfig) -> float:
    """Electrical phase tracking a rotor at the configured pole-pair count."""
    return cfg.pole_pairs * mechanical_angle
