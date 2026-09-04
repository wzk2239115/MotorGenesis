"""Differentiable native 3-D steady heat-conduction model."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from organic_motor.physics.electric3d import (
    _diffusion_diagonal,
    _outer_boundary,
    _shape,
    _spacing,
    _variable_diffusion,
)
from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner, relative_residual


def thermal_conductivity(rho_air: jnp.ndarray, rho_iron: jnp.ndarray,
                         rho_copper: jnp.ndarray, rho_pm: jnp.ndarray,
                         cfg: Any,
                         rho_insulator: jnp.ndarray | None = None) -> jnp.ndarray:
    """Mixture conductivity in W/(m K), preserving full 3-D phase fields.

    ``rho_insulator`` (optional, printed dielectric) carries its own low
    conductivity (~1.5 W/mK, a filled ceramic) instead of being lumped
    into air -- for the magnetic/electric solves the insulator IS air;
    thermally it must conduct, or every coil would be thermally floated.
    """
    k = (getattr(cfg, "thermal_k_air", 0.026) * rho_air
         + getattr(cfg, "thermal_k_iron", 25.0) * rho_iron
         + getattr(cfg, "thermal_k_copper", 385.0) * rho_copper
         + getattr(cfg, "thermal_k_pm", 8.0) * rho_pm)
    if rho_insulator is not None:
        k = k + getattr(cfg, "thermal_k_insulator", 1.5) * jnp.asarray(rho_insulator)
    return k


def total_loss_density(losses, shape: tuple[int, int, int] | None = None):
    """Combine one array, a sequence, or a named mapping of volumetric losses."""
    if isinstance(losses, Mapping):
        values = tuple(losses.values())
    elif isinstance(losses, (tuple, list)):
        values = tuple(losses)
    else:
        return jnp.asarray(losses)
    if not values:
        if shape is None:
            raise ValueError("shape is required when combining an empty loss collection")
        return jnp.zeros(shape)
    return sum((jnp.asarray(value) for value in values), jnp.zeros_like(values[0]))


def _surface_volume_ratio(cfg: Any) -> jnp.ndarray:
    """Exposed box area per boundary-node control volume [1/m]."""
    nx, ny, nz = _shape(cfg)
    hx, hy, hz = _spacing(cfg)
    ratio = jnp.zeros((nx, ny, nz))
    ratio = ratio.at[0, :, :].add(2.0 / hx).at[-1, :, :].add(2.0 / hx)
    ratio = ratio.at[:, 0, :].add(2.0 / hy).at[:, -1, :].add(2.0 / hy)
    return ratio.at[:, :, 0].add(2.0 / hz).at[:, :, -1].add(2.0 / hz)


def boundary_heat_source(surface_heat_flux, cfg: Any) -> jnp.ndarray:
    """Convert outward-positive surface heat flux [W/m²] to volume source."""
    return -jnp.asarray(surface_heat_flux) * _surface_volume_ratio(cfg)


def solve_temperature(loss_density, conductivity: jnp.ndarray, cfg: Any,
                      dirichlet_mask: jnp.ndarray | None = None,
                      boundary_temperature: jnp.ndarray | float | None = None,
                      convection_coefficient: jnp.ndarray | float | None = None,
                      ambient_temperature: float | None = None,
                      surface_heat_flux: jnp.ndarray | float | None = None,
                      internal_sink_beta: jnp.ndarray | None = None,
                      internal_sink_temperature: float = 40.0):
    """Solve ``-div(k grad(T))=q`` with Dirichlet and/or Robin box boundaries.

    If no boundary arguments are supplied the outer box is held at ambient,
    matching the existing 2-D thermal model.  Supplying a convection
    coefficient without a Dirichlet mask selects Robin cooling on the box.

    ``internal_sink_beta`` (W/(m^3 K), same shape as the grid) adds a
    volumetric Robin sink ``beta*(T - internal_sink_temperature)``: the
    reduced-order conjugate heat transfer of coolant flowing through
    printed channels -- ``beta = h * S_v`` with surface-to-volume ratio
    ``S_v ~ 4/D`` for a channel of hydraulic diameter D.
    """
    conductivity = jnp.asarray(conductivity)
    shape = conductivity.shape
    q = total_loss_density(loss_density, shape).astype(conductivity.dtype)
    ambient = (getattr(cfg, "ambient_temperature", 25.0)
               if ambient_temperature is None else ambient_temperature)
    outer = _outer_boundary(shape)

    if dirichlet_mask is None:
        dirichlet_mask = outer if convection_coefficient is None else jnp.zeros(shape, bool)
    mask = jnp.asarray(dirichlet_mask, dtype=bool)
    prescribed = ambient if boundary_temperature is None else boundary_temperature
    fixed_rise = jnp.where(
        mask,
        jnp.broadcast_to(jnp.asarray(prescribed, conductivity.dtype), shape) - ambient,
        0.0,
    )
    free = (~mask).astype(conductivity.dtype)

    beta = jnp.zeros(shape, dtype=conductivity.dtype)
    if convection_coefficient is not None:
        htc = jnp.broadcast_to(jnp.asarray(convection_coefficient,
                                          conductivity.dtype), shape)
        beta = jnp.where(outer, htc * _surface_volume_ratio(cfg), 0.0)
    if surface_heat_flux is not None:
        q = q + boundary_heat_source(surface_heat_flux, cfg)
    if internal_sink_beta is not None:
        beta = beta + jnp.broadcast_to(
            jnp.asarray(internal_sink_beta, conductivity.dtype), shape
        )
        q = q + jnp.broadcast_to(
            jnp.asarray(internal_sink_temperature - ambient, conductivity.dtype), shape
        ) * jnp.asarray(internal_sink_beta, conductivity.dtype)

    def operator(rise):
        free_rise = free * rise
        weak = _variable_diffusion(conductivity, free_rise, cfg) + beta * free_rise
        return jnp.where(mask, rise, weak)

    fixed_action = (_variable_diffusion(conductivity, fixed_rise, cfg)
                    + beta * fixed_rise)
    rhs = jnp.where(mask, fixed_rise, q - fixed_action)
    diagonal = jnp.where(mask, 1.0,
                         _diffusion_diagonal(conductivity, cfg) + beta)
    n_iter = int(getattr(cfg, "thermal_maxiter", getattr(cfg, "maxiter", 400)))
    tol = getattr(cfg, "thermal_tol", getattr(cfg, "tol", 1e-8))
    rise = cg_fixed(operator, rhs, fixed_rise,
                    jacobi_preconditioner(diagonal), n_iter, tol)
    return ambient + rise


def steady_temperature(loss_density: jnp.ndarray, rho_air: jnp.ndarray,
                       rho_iron: jnp.ndarray, rho_copper: jnp.ndarray,
                       rho_pm: jnp.ndarray, cfg: Any,
                       rho_insulator: jnp.ndarray | None = None,
                       rho_coolant: jnp.ndarray | None = None,
                       **boundary_kwargs):
    """Phase-field compatibility interface for the steady temperature solve.

    ``rho_coolant`` enables the internal convection sink of the printed
    cooling channels: ``beta = h_coolant * S_v * rho_coolant`` everywhere
    the channel fluid is present -- the reduced-order conjugate heat
    transfer (real channel walls convect; a void with k_air cannot).
    """
    conductivity = thermal_conductivity(
        rho_air, rho_iron, rho_copper, rho_pm, cfg, rho_insulator=rho_insulator
    )
    if rho_coolant is not None:
        h = getattr(cfg, "thermal_h_coolant", 3000.0)
        s_v = getattr(cfg, "thermal_channel_s_v", 2000.0)
        t_cool = getattr(cfg, "thermal_coolant_temperature", 40.0)
        beta = h * s_v * jnp.asarray(rho_coolant)
        return solve_temperature(
            loss_density, conductivity, cfg,
            internal_sink_beta=beta, internal_sink_temperature=t_cool,
        )
    return solve_temperature(loss_density, conductivity, cfg, **boundary_kwargs)


def thermal_relative_residual(temperature: jnp.ndarray, loss_density,
                              conductivity: jnp.ndarray, cfg: Any,
                              dirichlet_mask: jnp.ndarray | None = None,
                              boundary_temperature: jnp.ndarray | float | None = None,
                              convection_coefficient=None,
                              ambient_temperature: float | None = None,
                              internal_sink_beta: jnp.ndarray | None = None,
                              internal_sink_temperature: float = 40.0):
    """Relative residual for :func:`solve_temperature` (without surface flux)."""
    conductivity = jnp.asarray(conductivity)
    shape = conductivity.shape
    ambient = (getattr(cfg, "ambient_temperature", 25.0)
               if ambient_temperature is None else ambient_temperature)
    outer = _outer_boundary(shape)
    if dirichlet_mask is None:
        dirichlet_mask = outer if convection_coefficient is None else jnp.zeros(shape, bool)
    mask = jnp.asarray(dirichlet_mask, bool)
    prescribed = ambient if boundary_temperature is None else boundary_temperature
    fixed = jnp.where(mask, jnp.broadcast_to(jnp.asarray(
        prescribed, conductivity.dtype), shape) - ambient, 0.0)
    free = (~mask).astype(conductivity.dtype)
    beta = jnp.zeros(shape, dtype=conductivity.dtype)
    if convection_coefficient is not None:
        beta = jnp.where(
            outer,
            jnp.broadcast_to(jnp.asarray(convection_coefficient,
                                        conductivity.dtype), shape)
            * _surface_volume_ratio(cfg),
            0.0,
        )
    q = total_loss_density(loss_density, shape)
    if internal_sink_beta is not None:
        beta = beta + jnp.asarray(internal_sink_beta, conductivity.dtype)
        q = q + (internal_sink_temperature - ambient) * jnp.asarray(
            internal_sink_beta, conductivity.dtype
        )

    def operator(rise):
        y = free * rise
        return jnp.where(mask, rise,
                         _variable_diffusion(conductivity, y, cfg) + beta * y)

    rhs = jnp.where(
        mask, fixed,
        q - _variable_diffusion(conductivity, fixed, cfg) - beta * fixed,
    )
    return relative_residual(operator, temperature - ambient, rhs)


steady_heat_solve = solve_temperature
