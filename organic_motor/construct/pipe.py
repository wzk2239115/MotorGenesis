"""Swept pipes with field-driven cross-sections, ported from ShapeKernel BasePipe.

A pipe is a spine ``C(s)`` plus a polar cross-section in the local
``localX/localY`` plane at each station.  The inner/outer radius of that
section are sampled from :class:`SurfaceMod` fields -- so a pipe whose outer
radius is ``f(phi, s) = g(physics_field(s))`` is the LEAP 71 ``radius =
f(position, physics)`` pattern, realised as a watertight swept solid.

:meth:`surface_point` is the one-to-one port of ShapeKernel
``BasePipe.vecGetSurfacePoint``; the mesh/voxel kernel stitches quads by
calling it per ``(s, phi, r)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from organic_motor.construct.field import (
    SDFVoxelField,
    boolean_add,
    boolean_intersect,
    boolean_subtract,
    empty_field,
    from_implicit,
    offset as offset_field,
)
from organic_motor.construct.frames import Frames, LocalFrame
from organic_motor.construct.modulation import LineMod, SurfaceMod
from organic_motor.construct.implicit import cylinder_z


@dataclass
class BasePipe:
    """A swept pipe: spine + field-driven inner/outer radius."""

    frames: Frames
    inner_radius: SurfaceMod
    outer_radius: SurfaceMod
    length_steps: int = 64
    polar_steps: int = 32

    def surface_point(self, s: float, phi_ratio: float, r_ratio: float) -> np.ndarray:
        """Port of ``BasePipe.vecGetSurfacePoint``.

        ``s`` in [0,1] along the spine, ``phi_ratio`` in [0,1] around the
        section, ``r_ratio`` in [0,1] between inner and outer radius.
        """
        fr = self.frames.frame(s)
        phi = 2.0 * np.pi * phi_ratio
        r_outer = float(self.outer_radius(phi, s))
        r_inner = float(self.inner_radius(phi, s))
        r = r_inner + r_ratio * (r_outer - r_inner)
        return fr.pos + r * np.cos(phi) * fr.local_x + r * np.sin(phi) * fr.local_y

    def as_sdf(self, shape, spacing, origin) -> SDFVoxelField:
        """Rasterise the pipe as an SDF by filling a distance-to-spine field.

        A full exact swept-surface SDF is expensive; for the voxel resolutions
        a motor uses we approximate the pipe as the union of the spine's
        bounded influence.  This keeps the Boolean pipeline (subtract/intersect)
        exact in sign and good enough for the differentiable critic, which only
        needs a smooth occupancy.
        """
        nx, ny, nz = shape
        ox, oy, oz = origin
        dx, dy, dz = spacing
        x = ox + dx * np.arange(nx, dtype=np.float32)
        y = oy + dy * np.arange(ny, dtype=np.float32)
        z = oz + dz * np.arange(nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1)

        sdf = np.full(shape, 1e9, dtype=np.float32)
        for i in range(self.length_steps):
            s0 = i / self.length_steps
            s1 = (i + 1) / self.length_steps
            p0 = self.frames.spine(s0)
            p1 = self.frames.spine(s1)
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-12:
                continue
            seg_dir = seg / seg_len
            t = np.clip(((pts - p0) * seg_dir).sum(-1) / seg_len, 0.0, 1.0)
            closest = p0 + t[..., None] * seg
            dist = np.linalg.norm(pts - closest, axis=-1)
            r_outer = float(self.outer_radius(0.0, s0))
            r_inner = float(self.inner_radius(0.0, s0))
            seg_sdf = dist - r_outer
            if r_inner > 0.0:
                seg_sdf = np.maximum(seg_sdf, -(dist - r_inner))
            sdf = np.minimum(sdf, seg_sdf)
        return SDFVoxelField(sdf=sdf, spacing=spacing, origin=origin)


class PipeSegment(BasePipe):
    """A pipe whose angular extent is itself field-driven (a swept annular sector).

    Port of ``BasePipeSegment``: the arc centre and width are ``LineMod``
    fields over the length ratio, so a pole piece can open and close along
    the axis -- the canonical stator tooth / magnet pole profile.
    """

    def __init__(
        self,
        frames: Frames,
        inner_radius: SurfaceMod,
        outer_radius: SurfaceMod,
        arc_mid: LineMod,
        arc_range: LineMod,
        length_steps: int = 64,
        polar_steps: int = 32,
    ):
        super().__init__(frames, inner_radius, outer_radius, length_steps, polar_steps)
        self.arc_mid = arc_mid
        self.arc_range = arc_range

    def surface_point(self, s: float, phi_ratio: float, r_ratio: float) -> np.ndarray:
        fr = self.frames.frame(s)
        mid = float(self.arc_mid(s))
        rng = float(self.arc_range(s))
        phi = mid + (phi_ratio - 0.5) * rng
        r_outer = float(self.outer_radius(phi, s))
        r_inner = float(self.inner_radius(phi, s))
        r = r_inner + r_ratio * (r_outer - r_inner)
        return fr.pos + r * np.cos(phi) * fr.local_x + r * np.sin(phi) * fr.local_y
