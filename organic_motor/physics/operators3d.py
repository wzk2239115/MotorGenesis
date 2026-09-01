"""Differentiable finite-difference vector calculus on anisotropic 3-D grids."""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


class _HasSpacing(Protocol):
    spacing: tuple[float, float, float]


def _spacing(value: _HasSpacing | tuple[float, float, float]) -> tuple[float, float, float]:
    spacing = value.spacing if hasattr(value, "spacing") else value
    if len(spacing) != 3 or any(h <= 0.0 for h in spacing):
        raise ValueError("spacing must contain three positive values")
    return spacing


def gradient3d(
    scalar: jnp.ndarray, grid_or_spacing: _HasSpacing | tuple[float, float, float]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Node-centred gradient in x/y/z axis order."""
    dx, dy, dz = _spacing(grid_or_spacing)
    return (
        jnp.gradient(scalar, dx, axis=0),
        jnp.gradient(scalar, dy, axis=1),
        jnp.gradient(scalar, dz, axis=2),
    )


def divergence3d(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vz: jnp.ndarray,
    grid_or_spacing: _HasSpacing | tuple[float, float, float],
) -> jnp.ndarray:
    """Node-centred divergence of a vector field."""
    dx, dy, dz = _spacing(grid_or_spacing)
    return (
        jnp.gradient(vx, dx, axis=0)
        + jnp.gradient(vy, dy, axis=1)
        + jnp.gradient(vz, dz, axis=2)
    )


def curl3d(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vz: jnp.ndarray,
    grid_or_spacing: _HasSpacing | tuple[float, float, float],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Node-centred curl ``nabla x v``."""
    dx, dy, dz = _spacing(grid_or_spacing)
    return (
        jnp.gradient(vz, dy, axis=1) - jnp.gradient(vy, dz, axis=2),
        jnp.gradient(vx, dz, axis=2) - jnp.gradient(vz, dx, axis=0),
        jnp.gradient(vy, dx, axis=0) - jnp.gradient(vx, dy, axis=1),
    )


def variable_diffusion3d(
    coefficient: jnp.ndarray,
    scalar: jnp.ndarray,
    grid_or_spacing: _HasSpacing | tuple[float, float, float],
) -> jnp.ndarray:
    """Apply positive-semidefinite ``-div(coefficient * grad(scalar))``.

    Arithmetic face averaging and omitted exterior face fluxes impose a
    homogeneous Neumann condition on all six box faces.
    """
    dx, dy, dz = _spacing(grid_or_spacing)
    if coefficient.shape != scalar.shape or scalar.ndim != 3:
        raise ValueError("coefficient and scalar must be equal-shape 3-D arrays")
    out = jnp.zeros_like(scalar)

    cx = 0.5 * (coefficient[1:, :, :] + coefficient[:-1, :, :])
    fx = cx * (scalar[1:, :, :] - scalar[:-1, :, :]) / dx
    out = out.at[1:, :, :].add(fx / dx)
    out = out.at[:-1, :, :].add(-fx / dx)

    cy = 0.5 * (coefficient[:, 1:, :] + coefficient[:, :-1, :])
    fy = cy * (scalar[:, 1:, :] - scalar[:, :-1, :]) / dy
    out = out.at[:, 1:, :].add(fy / dy)
    out = out.at[:, :-1, :].add(-fy / dy)

    cz = 0.5 * (coefficient[:, :, 1:] + coefficient[:, :, :-1])
    fz = cz * (scalar[:, :, 1:] - scalar[:, :, :-1]) / dz
    out = out.at[:, :, 1:].add(fz / dz)
    return out.at[:, :, :-1].add(-fz / dz)


def diffusion_diagonal3d(
    coefficient: jnp.ndarray,
    grid_or_spacing: _HasSpacing | tuple[float, float, float],
) -> jnp.ndarray:
    """Exact diagonal of :func:`variable_diffusion3d`."""
    if coefficient.ndim != 3:
        raise ValueError("coefficient must be a 3-D array")
    dx, dy, dz = _spacing(grid_or_spacing)
    diagonal = jnp.zeros_like(coefficient)

    cx = 0.5 * (coefficient[1:, :, :] + coefficient[:-1, :, :]) / dx**2
    diagonal = diagonal.at[1:, :, :].add(cx)
    diagonal = diagonal.at[:-1, :, :].add(cx)
    cy = 0.5 * (coefficient[:, 1:, :] + coefficient[:, :-1, :]) / dy**2
    diagonal = diagonal.at[:, 1:, :].add(cy)
    diagonal = diagonal.at[:, :-1, :].add(cy)
    cz = 0.5 * (coefficient[:, :, 1:] + coefficient[:, :, :-1]) / dz**2
    diagonal = diagonal.at[:, :, 1:].add(cz)
    return diagonal.at[:, :, :-1].add(cz)


# Familiar short names for users of the 2-D Maxwell module.
gradient = gradient3d
divergence = divergence3d
curl = curl3d
diffusion = variable_diffusion3d
diffusion_diagonal = diffusion_diagonal3d
variable_coefficient_diffusion3d = variable_diffusion3d
