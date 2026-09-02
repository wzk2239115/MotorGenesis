"""SDF voxel field kernel: a reduced instruction set for constructive geometry.

Inspired by LEAP 71 / PicoGK.  The kernel exposes exactly five capabilities,
which compose to build any motor geometry and, by design, never fail:

  1. a voxel field carrying a narrow-band signed distance value per cell
  2. ``from_implicit``  -- render an implicit ``f(x,y,z) -> d`` into a field
  3. ``from_mesh``      -- rasterise a triangle mesh into a field
  4. ``boolean_add / subtract / intersect`` -- exact SDF Booleans
  5. ``offset``         -- grow or shrink a solid by a signed distance

Every operation is total and deterministic: Booleans on signed distance
fields are closed-form (``min``/``max``) and cannot produce degenerate
topology, which is the property a code-generating agent depends on.

Densities are intentionally ``numpy`` (not ``jax``): construction is a
discrete, robust backend and must not live on the gradient tape.  The
differentiable solver consumes the realised densities later, as a critic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class SDFVoxelField:
    """A dense signed-distance voxel field on a regular Cartesian grid.

    ``sdf`` is negative inside matter, zero on the surface, positive outside.
    ``spacing`` and ``origin`` give the physical node-to-node step and the
    coordinate of node ``(0, 0, 0)``.  The field is dense rather than
    narrow-band for simplicity; at the grid sizes a motor uses this is
    negligible memory and keeps every Boolean a single array op.
    """

    sdf: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        self.sdf = np.asarray(self.sdf, dtype=np.float32)
        if self.sdf.ndim != 3:
            raise ValueError(f"sdf must be 3-D, got shape {self.sdf.shape}")
        if len(self.spacing) != 3 or any(h <= 0.0 for h in self.spacing):
            raise ValueError("spacing must be three positive values")
        if len(self.origin) != 3:
            raise ValueError("origin must be three values")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.sdf.shape)

    @property
    def cell_size(self) -> float:
        return float(min(self.spacing))

    def coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Physical coordinate arrays for every voxel centre."""
        nx, ny, nz = self.shape
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        x = ox + dx * np.arange(nx, dtype=np.float32)
        y = oy + dy * np.arange(ny, dtype=np.float32)
        z = oz + dz * np.arange(nz, dtype=np.float32)
        return np.meshgrid(x, y, z, indexing="ij")

    def inside(self) -> np.ndarray:
        """Boolean mask of voxels strictly inside the solid."""
        return self.sdf < 0.0

    def to_density(self, bandwidth: float | None = None) -> np.ndarray:
        """Smoothstep the SDF into a ``[0, 1]`` occupancy density.

        ``bandwidth`` defaults to one cell; it controls the thickness of the
        diffuse interface that the differentiable critic consumes.  A cubic
        smoothstep is used so the density is C1-continuous across the surface,
        which keeps the solver well-conditioned.
        """
        h = self.cell_size if bandwidth is None else float(bandwidth)
        if h <= 0.0:
            raise ValueError("bandwidth must be positive")
        t = np.clip(0.5 - 0.5 * self.sdf / h, 0.0, 1.0)
        return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def empty_field(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SDFVoxelField:
    """An empty field (everything outside, SDF = +1 by convention)."""
    return SDFVoxelField(
        sdf=np.full(shape, 1.0, dtype=np.float32),
        spacing=spacing,
        origin=origin,
    )


def from_implicit(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> SDFVoxelField:
    """Build a field by evaluating an implicit ``fn(x, y, z) -> d`` per voxel."""
    field = SDFVoxelField(
        sdf=np.zeros(shape, dtype=np.float32),
        spacing=spacing,
        origin=origin,
    )
    x, y, z = field.coords()
    sdf = np.asarray(fn(x, y, z), dtype=np.float32)
    if sdf.shape != shape:
        raise ValueError(
            f"implicit returned shape {sdf.shape}, expected {shape}"
        )
    field.sdf = sdf
    return field


def from_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> SDFVoxelField:
    """Rasterise a triangle mesh into an unsigned-then-signed distance field.

    The unsigned distance is computed via a k-d tree over the mesh surface
    (``trimesh.proximity.ProximityQuery``).  The sign is recovered with a
    fast winding-number test, so the result is a true SDF and composes
    correctly under Booleans.  Use this to import existing CAD meshes.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces))
    if len(mesh.faces) == 0:
        return empty_field(shape, spacing, origin)
    field = SDFVoxelField(
        sdf=np.zeros(shape, dtype=np.float32),
        spacing=spacing,
        origin=origin,
    )
    x, y, z = field.coords()
    query = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    pq = trimesh.proximity.ProximityQuery(mesh)
    _closest, distance, _tri = pq.on_surface(query)
    distance = distance.astype(np.float32).reshape(shape)
    contained = mesh.contains(query).reshape(shape)
    field.sdf = np.where(contained, -distance, distance)
    return field


def _same_grid(a: SDFVoxelField, b: SDFVoxelField) -> None:
    if a.shape != b.shape or a.spacing != b.spacing or a.origin != b.origin:
        raise ValueError(
            f"fields must share a grid; got {a.shape}@{a.spacing} vs "
            f"{b.shape}@{b.spacing}"
        )


def boolean_add(a: SDFVoxelField, b: SDFVoxelField) -> SDFVoxelField:
    """Union of two solids: ``min(a.sdf, b.sdf)``."""
    _same_grid(a, b)
    return SDFVoxelField(
        sdf=np.minimum(a.sdf, b.sdf), spacing=a.spacing, origin=a.origin
    )


def boolean_subtract(a: SDFVoxelField, b: SDFVoxelField) -> SDFVoxelField:
    """Solid ``a`` minus solid ``b``: ``max(a.sdf, -b.sdf)``.

    Subtraction is the intersection of ``a`` with the complement of ``b``;
    the complement flips the sign of the SDF, and intersection is the maximum,
    so ``a \\ b = max(a.sdf, -b.sdf)``.
    """
    _same_grid(a, b)
    return SDFVoxelField(
        sdf=np.maximum(a.sdf, -b.sdf), spacing=a.spacing, origin=a.origin
    )


def boolean_intersect(a: SDFVoxelField, b: SDFVoxelField) -> SDFVoxelField:
    """Intersection of two solids: ``max(a.sdf, b.sdf)``."""
    _same_grid(a, b)
    return SDFVoxelField(
        sdf=np.maximum(a.sdf, b.sdf), spacing=a.spacing, origin=a.origin
    )


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum (Ricci blend) of two SDF arrays.

    When ``|a - b| < k`` the two surfaces blend with a smooth fillet of
    radius ~k/2; outside that zone it falls back to the hard ``min``.
    This is the single operation that turns hard Boolean assemblies into
    the organic, bone-like fused structures LEAP 71 is known for.
    """
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smooth_boolean_add(a: SDFVoxelField, b: SDFVoxelField, blend: float = 0.001) -> SDFVoxelField:
    """Smooth union: like :func:`boolean_add` but with a fillet at the seam.

    ``blend`` is the blend radius in metres.  At 0 it degenerates to the
    hard ``min``; at 1–2 mm it produces visibly rounded junctions.
    """
    _same_grid(a, b)
    if blend <= 0.0:
        return boolean_add(a, b)
    return SDFVoxelField(
        sdf=_smooth_min(a.sdf, b.sdf, float(blend)), spacing=a.spacing, origin=a.origin
    )


def smooth_boolean_subtract(a: SDFVoxelField, b: SDFVoxelField, blend: float = 0.001) -> SDFVoxelField:
    """Smooth subtraction: like :func:`boolean_subtract` with a fillet."""
    _same_grid(a, b)
    if blend <= 0.0:
        return boolean_subtract(a, b)
    return SDFVoxelField(
        sdf=-_smooth_min(-a.sdf, b.sdf, float(blend)), spacing=a.spacing, origin=a.origin
    )


def offset(field: SDFVoxelField, distance: float) -> SDFVoxelField:
    """Grow (``distance > 0``) or shrink (``distance < 0``) a solid.

    Adding a constant to an SDF is an exact offset operation; it is the
    primitive that makes shells, clearance gaps and fillets one-liners.
    """
    return SDFVoxelField(
        sdf=field.sdf + float(distance), spacing=field.spacing, origin=field.origin
    )


def shell(field: SDFVoxelField, thickness: float) -> SDFVoxelField:
    """A hollow shell of ``thickness`` centred on the surface of ``field``."""
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    outer = offset(field, thickness * 0.5)
    inner = offset(field, -thickness * 0.5)
    return boolean_subtract(outer, inner)


def resample(
    field: SDFVoxelField,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] | None = None,
) -> SDFVoxelField:
    """Trilinearly resample an SDF onto a new grid.

    Used to continue a coarse-stage construction onto a finer grid.  The SDF
    remains a valid distance field up to trilinear interpolation error, which
    is bounded by the source cell size.
    """
    ox, oy, oz = field.origin if origin is None else origin
    sx, sy, sz = field.spacing if spacing is None else spacing
    nx, ny, nz = shape
    x = ox + sx * np.arange(nx, dtype=np.float32)
    y = oy + sy * np.arange(ny, dtype=np.float32)
    z = oz + sz * np.arange(nz, dtype=np.float32)
    src_x = field.origin[0] + field.spacing[0] * np.arange(field.shape[0], dtype=np.float32)
    src_y = field.origin[1] + field.spacing[1] * np.arange(field.shape[1], dtype=np.float32)
    src_z = field.origin[2] + field.spacing[2] * np.arange(field.shape[2], dtype=np.float32)

    def _coord(v, src):
        t = (v - src[0]) / (src[1] - src[0]) if len(src) > 1 and src[1] != src[0] else np.zeros_like(v)
        return np.clip(t, 0.0, len(src) - 1.0)

    xi = _coord(x, src_x)
    yi = _coord(y, src_y)
    zi = _coord(z, src_z)
    from scipy.ndimage import map_coordinates

    sdf = map_coordinates(field.sdf, np.meshgrid(xi, yi, zi, indexing="ij"), order=1, mode="nearest")
    return SDFVoxelField(
        sdf=sdf.astype(np.float32), spacing=(sx, sy, sz), origin=(ox, oy, oz)
    )


# ---------------------------------------------------------------------------
# Sampleable scalar/vector fields (the PicoGK ScalarField/VectorField contract)
# ---------------------------------------------------------------------------

@dataclass
class ScalarField:
    """A scalar field over a voxel grid, sampleable at any world point.

    Port of PicoGK ``ScalarField``: trilinear-sample at a world point; points
    outside the grid return ``background``.  This is the contract a reduced
    physics field speaks so it can drive geometry pointwise.
    """

    data: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    background: float = 0.0

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)

    @classmethod
    def from_sdf(cls, field: SDFVoxelField) -> "ScalarField":
        """A scalar field whose values are a voxel SDF (sample-as-distance)."""
        return cls(data=field.sdf, spacing=field.spacing, origin=field.origin)

    def sample(self, p) -> float:
        from scipy.ndimage import map_coordinates

        idx = (np.asarray(p, dtype=np.float32) - np.asarray(self.origin)) / np.asarray(self.spacing)
        if np.any(idx < 0) or np.any(idx > np.asarray(self.data.shape) - 1):
            return float(self.background)
        return float(map_coordinates(self.data, idx.reshape(-1, 1), order=1, mode="nearest")[0])

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.data.shape
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        x = ox + dx * np.arange(nx, dtype=np.float32)
        y = oy + dy * np.arange(ny, dtype=np.float32)
        z = oz + dz * np.arange(nz, dtype=np.float32)
        return np.meshgrid(x, y, z, indexing="ij")


@dataclass
class VectorField:
    """A 3-component vector field over a voxel grid, sampleable at a point."""

    data: np.ndarray  # shape (..., 3) or (3, ...)
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.shape[-1] != 3 and arr.shape[0] == 3 and arr.ndim == 4:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] != 3:
            raise ValueError("VectorField data must have a trailing size-3 axis")
        self.data = arr

    def sample(self, p) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        idx = (np.asarray(p, dtype=np.float32) - np.asarray(self.origin)) / np.asarray(self.spacing)
        shape = self.data.shape[:-1]
        if np.any(idx < 0) or np.any(idx > np.asarray(shape) - 1):
            return np.zeros(3, dtype=np.float32)
        out = np.empty(3, dtype=np.float32)
        for c in range(3):
            out[c] = float(map_coordinates(self.data[..., c], idx.reshape(-1, 1), order=1, mode="nearest")[0])
        return out
