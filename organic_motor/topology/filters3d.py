"""Differentiable 3-D filters and morphology regularisers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner
from organic_motor.physics.operators3d import (
    diffusion_diagonal3d,
    divergence3d,
    gradient3d,
    variable_diffusion3d,
)


def helmholtz_filter3d(
    field: jnp.ndarray,
    cfg: MotorConfig3D,
    radius: float | None = None,
    n_iter: int = 80,
    tol: float | None = None,
) -> jnp.ndarray:
    """Solve ``(I - r^2 Laplace) filtered = field`` with Neumann faces."""
    r = cfg.filt_radius if radius is None else radius
    if r <= 0.0:
        return field
    if field.shape != cfg.shape:
        raise ValueError(f"field shape {field.shape} does not match {cfg.shape}")
    coefficient = jnp.full_like(field, r**2)

    def operator(value: jnp.ndarray) -> jnp.ndarray:
        return value + variable_diffusion3d(coefficient, value, cfg)

    diagonal = 1.0 + diffusion_diagonal3d(coefficient, cfg)
    tolerance = cfg.filter_tol if tol is None else tol
    return cg_fixed(
        operator,
        field,
        jnp.zeros_like(field),
        jacobi_preconditioner(diagonal),
        n_iter,
        tolerance,
    )


def total_variation3d(
    field: jnp.ndarray, cfg: MotorConfig3D, epsilon: float = 1e-12
) -> jnp.ndarray:
    """Mean isotropic 3-D total variation density."""
    gx, gy, gz = gradient3d(field, cfg)
    return jnp.mean(jnp.sqrt(gx * gx + gy * gy + gz * gz + epsilon))


def mean_curvature3d(
    field: jnp.ndarray, cfg: MotorConfig3D, epsilon: float = 1e-12
) -> jnp.ndarray:
    """Level-set mean-curvature field ``div(grad(f)/|grad(f)|)``."""
    gx, gy, gz = gradient3d(field, cfg)
    norm = jnp.sqrt(gx * gx + gy * gy + gz * gz + epsilon)
    return divergence3d(gx / norm, gy / norm, gz / norm, cfg)


def curvature_penalty3d(
    field: jnp.ndarray, cfg: MotorConfig3D, epsilon: float = 1e-12
) -> jnp.ndarray:
    """Interface-weighted squared curvature proxy."""
    gx, gy, gz = gradient3d(field, cfg)
    interface = jnp.sqrt(gx * gx + gy * gy + gz * gz + epsilon)
    curvature = mean_curvature3d(field, cfg, epsilon)
    return jnp.mean(interface * curvature * curvature)


def _neighbor_mean(field: jnp.ndarray) -> jnp.ndarray:
    padded = jnp.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")
    return (
        padded[2:, 1:-1, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 1:-1, 2:]
        + padded[1:-1, 1:-1, :-2]
    ) / 6.0


def soft_connected_field3d(
    density: jnp.ndarray,
    seed: jnp.ndarray,
    steps: int = 32,
    sharpness: float = 8.0,
) -> jnp.ndarray:
    """Propagate differentiable ownership from ``seed`` through ``density``.

    The result lies in ``[0, 1]`` and approximates the portion of material
    reachable on the six-neighbour voxel graph.  A probabilistic union makes
    propagation monotone while retaining useful gradients.
    """
    material = jax.nn.sigmoid(sharpness * (density - 0.5))
    reached0 = jnp.clip(seed, 0.0, 1.0) * material

    def body(reached: jnp.ndarray, _unused: None):
        candidate = material * _neighbor_mean(reached)
        reached = 1.0 - (1.0 - reached) * (1.0 - candidate)
        return jnp.clip(reached, 0.0, 1.0), None

    reached, _ = jax.lax.scan(body, reached0, xs=None, length=steps)
    return reached


def island_penalty3d(
    density: jnp.ndarray,
    seed: jnp.ndarray,
    steps: int = 32,
    sharpness: float = 8.0,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
    """Fraction of soft material not connected to the supplied anchor seed."""
    material = jax.nn.sigmoid(sharpness * (density - 0.5))
    connected = soft_connected_field3d(density, seed, steps, sharpness)
    return jnp.sum(material * (1.0 - connected)) / (
        jnp.sum(material) + epsilon
    )


def connectivity_proxy3d(
    density: jnp.ndarray,
    seed: jnp.ndarray,
    steps: int = 32,
    sharpness: float = 8.0,
) -> jnp.ndarray:
    """Return one minus the differentiable island fraction."""
    return 1.0 - island_penalty3d(density, seed, steps, sharpness)


helmholtz_filter = helmholtz_filter3d
total_variation = total_variation3d
mean_curvature = mean_curvature3d
curvature_penalty = curvature_penalty3d
island_penalty = island_penalty3d
connectivity_proxy = connectivity_proxy3d
