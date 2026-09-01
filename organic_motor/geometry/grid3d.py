"""Cartesian 3-D grids, trilinear sampling and rigid z-axis rotations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D


@dataclass(frozen=True)
class Grid3D:
    """A node-centred Cartesian grid using ``(x, y, z)`` array axes."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(n < 2 for n in self.shape):
            raise ValueError("shape must be (Nx, Ny, Nz), with every size >= 2")
        if len(self.spacing) != 3 or any(h <= 0.0 for h in self.spacing):
            raise ValueError("spacing must contain three positive values")

    @classmethod
    def from_config(cls, cfg: MotorConfig3D) -> "Grid3D":
        return cls(cfg.shape, cfg.spacing, cfg.origin)

    def coordinates(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return tuple(
            o + h * jnp.arange(n) for n, h, o in zip(self.shape, self.spacing, self.origin)
        )

    def meshgrid(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return jnp.meshgrid(*self.coordinates(), indexing="ij")


def as_grid3d(grid_or_cfg: Grid3D | MotorConfig3D) -> Grid3D:
    return (
        grid_or_cfg
        if isinstance(grid_or_cfg, Grid3D)
        else Grid3D.from_config(grid_or_cfg)
    )


def coords_3d(
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the three one-dimensional node coordinate vectors."""
    return as_grid3d(grid_or_cfg).coordinates()


def meshgrid3d(
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``X, Y, Z``, each with shape ``(Nx, Ny, Nz)``."""
    return as_grid3d(grid_or_cfg).meshgrid()


def trilinear_sample(
    field: jnp.ndarray,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    zs: jnp.ndarray,
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> jnp.ndarray:
    """Sample a scalar volume at arbitrary coordinates.

    Coordinates outside the box are clamped to its nearest boundary node.
    The operation is differentiable with respect to the field and coordinates
    away from voxel boundaries.
    """
    grid = as_grid3d(grid_or_cfg)
    if field.shape != grid.shape:
        raise ValueError(f"field shape {field.shape} does not match grid {grid.shape}")

    indices = []
    fractions = []
    for q, origin, spacing, n in zip(
        (xs, ys, zs), grid.origin, grid.spacing, grid.shape
    ):
        u = jnp.clip((q - origin) / spacing, 0.0, n - 1.0)
        i0 = jnp.floor(u).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, n - 1)
        indices.append((i0, i1))
        fractions.append(u - i0)

    (i0, i1), (j0, j1), (k0, k1) = indices
    fx, fy, fz = fractions
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def rotate_volume_z(
    field: jnp.ndarray,
    theta: float,
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> jnp.ndarray:
    """Rigidly rotate a scalar volume about the grid centre's z axis."""
    grid = as_grid3d(grid_or_cfg)
    X, Y, Z = grid.meshgrid()
    cx, cy, _ = (
        grid.origin[i] + 0.5 * grid.spacing[i] * (grid.shape[i] - 1)
        for i in range(3)
    )
    c, s = jnp.cos(theta), jnp.sin(theta)
    x = X - cx
    y = Y - cy
    # Inverse warp: value at a destination node comes from R(-theta) x.
    xs = cx + c * x + s * y
    ys = cy - s * x + c * y
    return trilinear_sample(field, xs, ys, Z, grid)


def rotate_vector_z(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vz: jnp.ndarray,
    theta: float,
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rotate vector locations and components rigidly about the z axis."""
    ax = rotate_volume_z(vx, theta, grid_or_cfg)
    ay = rotate_volume_z(vy, theta, grid_or_cfg)
    az = rotate_volume_z(vz, theta, grid_or_cfg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return c * ax - s * ay, s * ax + c * ay, az


def rotate_owned_volume_z(
    field: jnp.ndarray,
    ownership: jnp.ndarray,
    theta: float,
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> jnp.ndarray:
    """Rotate only the continuously rotor-owned fraction of a volume."""
    moved = rotate_volume_z(field * ownership, theta, grid_or_cfg)
    stationary = field * (1.0 - ownership)
    return stationary + moved


def rotate_owned_vector_z(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vz: jnp.ndarray,
    ownership: jnp.ndarray,
    theta: float,
    grid_or_cfg: Grid3D | MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rotate only a continuously rotor-owned vector-field fraction."""
    mx, my, mz = rotate_vector_z(
        vx * ownership,
        vy * ownership,
        vz * ownership,
        theta,
        grid_or_cfg,
    )
    stationary = 1.0 - ownership
    return (
        vx * stationary + mx,
        vy * stationary + my,
        vz * stationary + mz,
    )


# Concise compatibility names.
rotate_field_z = rotate_volume_z
rotate_vector_field_z = rotate_vector_z
trilinear_interpolate = trilinear_sample
