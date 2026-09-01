"""Differentiable steady-state 2-D heat-conduction model."""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks
from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner
from organic_motor.physics.maxwell import diffusion, diffusion_diagonal


def thermal_conductivity(rho_air, rho_iron, rho_copper, rho_pm,
                         cfg: MotorConfig) -> jnp.ndarray:
    return (cfg.thermal_k_air * rho_air
            + cfg.thermal_k_iron * rho_iron
            + cfg.thermal_k_copper * rho_copper
            + cfg.thermal_k_pm * rho_pm)


def steady_temperature(loss_density: jnp.ndarray, rho_air: jnp.ndarray,
                       rho_iron: jnp.ndarray, rho_copper: jnp.ndarray,
                       rho_pm: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Solve temperature with the outer square held at ambient temperature."""
    conductivity = thermal_conductivity(rho_air, rho_iron, rho_copper,
                                        rho_pm, cfg)
    boundary = domain_masks(cfg)["boundary"]

    def operator(rise):
        return jnp.where(boundary, rise, diffusion(conductivity, rise, cfg))

    rhs = jnp.where(boundary, 0.0, loss_density)
    diag = jnp.where(boundary, 1.0,
                     diffusion_diagonal(conductivity, cfg))
    rise = cg_fixed(operator, rhs, jnp.zeros_like(rhs),
                    jacobi_preconditioner(diag), cfg.thermal_maxiter,
                    cfg.thermal_tol)
    return cfg.ambient_temperature + rise


def thermal_relative_residual(temperature, loss_density, rho_air, rho_iron,
                              rho_copper, rho_pm, cfg: MotorConfig):
    from organic_motor.physics.linear import relative_residual
    conductivity = thermal_conductivity(rho_air, rho_iron, rho_copper,
                                        rho_pm, cfg)
    boundary = domain_masks(cfg)["boundary"]
    rhs = jnp.where(boundary, 0.0, loss_density)

    def operator(rise):
        return jnp.where(boundary, rise, diffusion(conductivity, rise, cfg))

    return relative_residual(operator,
                             temperature - cfg.ambient_temperature, rhs)
