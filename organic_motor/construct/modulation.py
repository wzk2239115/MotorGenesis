"""Field-driven geometry parameters, ported from LEAP 71 ShapeKernel.

The core idea (see ``vendor/leap71/LEAP71_ShapeKernel``): a geometry
parameter is never a ``float`` constant -- it is a callable sampled
pointwise during construction.  Two families:

  * :class:`LineMod` / :class:`SurfaceMod` -- 1D/2D *parameter fields*
    sampled by a normalised local coordinate (length ratio, polar angle)
    along a spine.  These drive pipe radii, pole widths, arc extents.

  * :class:`BeamField` -- the ``IBeamThickness`` equivalent: a true world-
    position spatial field ``sample(p) -> thickness``, backed here by either
    a constant, an analytic function of position, or a distance-to-surface
    SDF (the boundary-ramp implementation).  This is what lets a lattice
    strut or a gyroid wall thicken where a physics field says so.

Algebra (``+``, ``*``) composes fields, mirroring the ShapeKernel operator
overloads, so ``0.5*iron_field + copper_field`` is itself a field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# 1D / 2D parameter fields (local coordinates)
# ---------------------------------------------------------------------------

class LineMod:
    """A 1D parameter field ``f(ratio) -> float`` over a normalised spine coordinate."""

    def __init__(self, value: float | Callable[[np.ndarray | float], np.ndarray] | "LineMod"):
        if isinstance(value, LineMod):
            self._fn = value._fn
        elif callable(value):
            self._fn = value
        else:
            const = float(value)
            self._fn = lambda r: np.full_like(np.asarray(r, dtype=np.float32), const)

    def __call__(self, ratio) -> np.ndarray:
        return np.asarray(self._fn(ratio), dtype=np.float32)

    @staticmethod
    def constant(value: float) -> "LineMod":
        return LineMod(value)

    def __add__(self, other) -> "LineMod":
        o = other if isinstance(other, LineMod) else LineMod(other)
        return LineMod(lambda r: self(r) + o(r))

    def __mul__(self, other) -> "LineMod":
        if isinstance(other, LineMod):
            return LineMod(lambda r: self(r) * other(r))
        s = float(other)
        return LineMod(lambda r: self(r) * s)

    __rmul__ = __mul__
    __radd__ = __add__


class SurfaceMod:
    """A 2D parameter field ``f(phi, length_ratio) -> float`` over a swept surface."""

    def __init__(self, value: float | Callable | LineMod | "SurfaceMod"):
        if isinstance(value, SurfaceMod):
            self._fn = value._fn
        elif isinstance(value, LineMod):
            line = value
            self._fn = lambda phi, s: line(s) * np.ones_like(np.asarray(phi, dtype=np.float32))
        elif callable(value):
            self._fn = value
        else:
            const = float(value)
            self._fn = lambda phi, s: np.full_like(
                np.asarray(phi, dtype=np.float32), const
            )

    def __call__(self, phi, length_ratio) -> np.ndarray:
        return np.asarray(self._fn(phi, length_ratio), dtype=np.float32)

    @staticmethod
    def constant(value: float) -> "SurfaceMod":
        return SurfaceMod(value)

    @staticmethod
    def from_line(line: LineMod) -> "SurfaceMod":
        return SurfaceMod(line)

    def __add__(self, other) -> "SurfaceMod":
        o = other if isinstance(other, SurfaceMod) else SurfaceMod(other)
        return SurfaceMod(lambda phi, s: self(phi, s) + o(phi, s))

    def __mul__(self, other) -> "SurfaceMod":
        if isinstance(other, SurfaceMod):
            return SurfaceMod(lambda phi, s: self(phi, s) * other(phi, s))
        s_ = float(other)
        return SurfaceMod(lambda phi, s: self(phi, s) * s_)

    __rmul__ = __mul__
    __radd__ = __add__


# ---------------------------------------------------------------------------
# World-position spatial field (IBeamThickness equivalent)
# ---------------------------------------------------------------------------

@runtime_checkable
class SpatialField(Protocol):
    """A scalar field sampled at a world point: ``sample(p) -> float``."""

    def sample(self, p: np.ndarray) -> float: ...


@dataclass
class ConstField:
    """A spatially-constant field."""

    value: float

    def sample(self, p: np.ndarray) -> float:
        return float(self.value)


@dataclass
class FuncField:
    """A spatial field from a function ``f(x, y, z) -> float``."""

    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    vectorized: bool = True

    def sample(self, p: np.ndarray) -> float:
        return float(self.fn(p[..., 0], p[..., 1], p[..., 2]))


@dataclass
class BoundaryRampField:
    """Thickness that ramps with distance to a surface (BoundaryBeamThickness).

    ``fMin`` at the surface, ``fMax`` deep inside, with a smoothstep over
    ``band``.  The boundary is the SDF of a :class:`SDFVoxelField`, so this is
    how a physics-derived geometry field modulates strut/wall thickness.
    """

    boundary: object  # SDFVoxelField
    f_min: float
    f_max: float
    band: float = 0.01

    def sample(self, p: np.ndarray) -> float:
        from scipy.ndimage import map_coordinates

        sdf = self.boundary.sdf
        spacing = np.asarray(self.boundary.spacing)
        origin = np.asarray(self.boundary.origin)
        idx = (np.asarray(p, dtype=np.float32) - origin) / spacing
        d = float(map_coordinates(sdf, idx.reshape(-1, 1), order=1, mode="nearest")[0])
        t = np.clip(-d / max(self.band, 1e-12), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        return float(self.f_min + (self.f_max - self.f_min) * t)


# Alias mirroring the ShapeKernel interface name.
BeamField = SpatialField


def field_sample_grid(field: SpatialField, shape, spacing, origin) -> np.ndarray:
    """Evaluate a :class:`SpatialField` on a full voxel grid (vectorised).

    Used to rasterise a field-driven parameter once, when a part needs the
    whole array rather than pointwise sampling.
    """
    nx, ny, nz = shape
    ox, oy, oz = origin
    dx, dy, dz = spacing
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1)
    out = np.empty(nx * ny * nz, dtype=np.float32)
    flat = pts.reshape(-1, 3)
    if isinstance(field, ConstField):
        out[:] = field.value
    elif isinstance(field, FuncField) and field.vectorized:
        out[:] = np.asarray(field.fn(flat[:, 0], flat[:, 1], flat[:, 2]), dtype=np.float32)
    elif isinstance(field, BoundaryRampField):
        from scipy.ndimage import map_coordinates

        idx = (flat - np.asarray(field.boundary.origin)) / np.asarray(field.boundary.spacing)
        d = map_coordinates(field.boundary.sdf, idx.T, order=1, mode="nearest")
        t = np.clip(-d / max(field.band, 1e-12), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        out[:] = field.f_min + (field.f_max - field.f_min) * t
    else:
        for i, p in enumerate(flat):
            out[i] = field.sample(p)
    return out.reshape(shape)
