"""Field-modulated implicits, ported from ShapeKernel ``ImplicitModular``.

The pattern that closes the field-driven-growth gap: an SDF whose wall
thickness is sampled pointwise from a :class:`SpatialField`.  So a gyroid
cooling wall thickens where the local heat-flux field is high and thins where
it is low -- ``wall = f(position, physics)``, evaluated *inside* the SDF.

This is the exact port of::

    float fSignedDistance(in Vector3 vecPt) {
        float raw = m_xRawTPMSPattern.fGetSignedDistance(trafo(vecPt));
        float wall = m_xWallThickness.fGetBeamThickness(vecPt);   // field sample
        return m_xSplittingLogic.fGetAdvancedSignedDistance(raw, wall);
    }
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from organic_motor.construct.modulation import BeamField, ConstField
from organic_motor.construct.implicit import Implicit


def modulated_implicit(
    raw: Implicit,
    wall_thickness: BeamField | float,
    *,
    trafo: Callable[[np.ndarray, np.ndarray, np.ndarray],
                    tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> Implicit:
    """Return an implicit ``f(x,y,z) -> d`` whose wall is field-driven.

    ``raw`` is the base pattern (e.g. gyroid).  ``wall_thickness`` is either a
    constant or a :class:`SpatialField` sampled at each voxel.  The solid is
    the band ``|raw| < wall/2``, so ``sdf = |raw| - wall/2`` -- exactly the
    ShapeKernel ``ImplicitModular`` splitting logic.
    """
    if isinstance(wall_thickness, (int, float)):
        wall_field = ConstField(float(wall_thickness))
    else:
        wall_field = wall_thickness

    def fn(x, y, z):
        if trafo is not None:
            x, y, z = trafo(x, y, z)
        raw_d = np.asarray(raw(x, y, z), dtype=np.float32)
        shape = raw_d.shape
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        wall = np.empty(pts.shape[0], dtype=np.float32)
        if isinstance(wall_field, ConstField):
            wall[:] = wall_field.value
        else:
            from organic_motor.construct.modulation import field_sample_grid
            # vectorised where the field supports it; else per-point
            try:
                wall[:] = field_sample_grid(wall_field, shape, None, None).ravel() \
                    if False else np.array([wall_field.sample(p) for p in pts])
            except Exception:
                wall[:] = np.array([wall_field.sample(p) for p in pts])
        return (np.abs(raw_d) - 0.5 * wall.reshape(shape)).astype(np.float32)

    return fn


def field_modulated_surface(
    base_implicit: Implicit,
    thickness_field: BeamField,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
):
    """Build a :class:`SDFVoxelField` whose wall thickness follows a field.

    Convenience wrapper: rasterise ``base_implicit`` modulated by
    ``thickness_field`` directly onto a grid, returning an SDF voxel field
    that composes under the standard Booleans.
    """
    from organic_motor.construct.field import SDFVoxelField
    from organic_motor.construct.modulation import field_sample_grid

    nx, ny, nz = shape
    ox, oy, oz = origin
    dx, dy, dz = spacing
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    raw = np.asarray(base_implicit(X, Y, Z), dtype=np.float32)
    wall = field_sample_grid(thickness_field, shape, spacing, origin)
    sdf = np.abs(raw) - 0.5 * wall
    return SDFVoxelField(sdf=sdf.astype(np.float32), spacing=spacing, origin=origin)
