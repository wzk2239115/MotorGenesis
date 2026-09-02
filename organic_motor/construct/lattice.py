"""Lattice primitives for cooling jackets, winding supports and infill.

Lattices are periodic Boolean compositions of implicit primitives: they
exploit the fact that a signed distance to a repeating structure is the
minimum over one cell, which the numpy minimum over a phase-shifted stack
computes exactly.  These give a motor its ``flesh'' -- the gyroid-sheet
cooling wall between stator and housing, the strut network that stiffens a
winding overhang, the infill that lightens an iron yoke.
"""

from __future__ import annotations

import numpy as np

from organic_motor.construct.field import SDFVoxelField
from organic_motor.construct.implicit import Implicit, sphere


def _grid(shape, spacing, origin):
    nx, ny, nz = shape
    ox, oy, oz = origin
    dx, dy, dz = spacing
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    return np.meshgrid(x, y, z, indexing="ij")


def _field(sdf, spacing, origin):
    return SDFVoxelField(sdf=sdf, spacing=spacing, origin=origin)


def strut_lattice(
    period: float,
    radius: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> SDFVoxelField:
    """A 3-axis strut lattice: the periodic min of spheres on a cubic grid.

    Each node hosts a sphere of ``radius``; the periodic repetition yields an
    interconnected strut network through the touching points.
    """
    p = float(period)
    if p <= 0.0:
        raise ValueError("period must be positive")
    X, Y, Z = _grid(shape, spacing, origin)

    fx = (X - np.round(X / p) * p) / radius
    fy = (Y - np.round(Y / p) * p) / radius
    fz = (Z - np.round(Z / p) * p) / radius
    sx = np.sqrt(np.maximum(fx ** 2 + fy ** 2, 0.0)) - 1.0
    sy = np.sqrt(np.maximum(fy ** 2 + fz ** 2, 0.0)) - 1.0
    sz = np.sqrt(np.maximum(fz ** 2 + fx ** 2, 0.0)) - 1.0
    sdf = (np.minimum(np.minimum(sx, sy), sz) * radius).astype(np.float32)
    return _field(sdf, spacing, origin)


def gyroid_sheet(
    scale: float,
    thickness: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> SDFVoxelField:
    """A thickened gyroid minimal-surface sheet.

    The wall is the region where ``|gyroid| < thickness/2``; the returned
    SDF is ``|gyroid| - thickness/2`` so it is negative inside the wall.
    This is the canonical LEAP 71 / nTopology heat-exchanger wall.
    """
    s, t = float(scale), float(thickness)
    X, Y, Z = _grid(shape, spacing, origin)
    g = (
        np.sin(s * X) * np.cos(s * Y)
        + np.sin(s * Y) * np.cos(s * Z)
        + np.sin(s * Z) * np.cos(s * X)
    )
    sdf = (np.abs(g / s) - t * 0.5).astype(np.float32)
    return _field(sdf, spacing, origin)


def sheet_lattice(
    period: float,
    half_thickness: float,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> SDFVoxelField:
    """A three-perpendicular thin-sheet lattice (a discrete gyroid analogue).

    Plates at spacing ``period`` with half-thickness ``half_thickness``
    along each axis; the union is an orthogonal grid of walls.
    """
    p, t = float(period), float(half_thickness)
    X, Y, Z = _grid(shape, spacing, origin)
    ox, oy, oz = origin
    px = np.abs(((X - ox + 0.5 * p) % p) - 0.5 * p) - t
    py = np.abs(((Y - oy + 0.5 * p) % p) - 0.5 * p) - t
    pz = np.abs(((Z - oz + 0.5 * p) % p) - 0.5 * p) - t
    sdf = np.minimum(np.minimum(px, py), pz).astype(np.float32)
    return _field(sdf, spacing, origin)


def lattice_field(
    sdf_array: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
) -> SDFVoxelField:
    """Wrap a raw lattice SDF array into an :class:`SDFVoxelField`."""
    return SDFVoxelField(sdf=sdf_array, spacing=spacing, origin=origin)
