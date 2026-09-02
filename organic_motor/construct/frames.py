"""Local frames and along-spine frame sequences, ported from ShapeKernel.

A :class:`LocalFrame` is an oriented coordinate system (position + localX/Y/Z);
``localZ`` is the tooling/spine tangent and ``localX`` the in-plane reference.
:class:`Frames` carries a sequence of frames along a spline and resolves the
in-plane rotation by a targeting strategy.  ``CYLINDRICAL`` targeting makes
every station's ``localX`` point radially outward in the XY plane -- the
natural frame for motor pole pieces and annular cross-sections.

This is the orientation engine that :mod:`organic_motor.construct.pipe` uses to
sweep a cross-section along a spine with a field-driven radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


@dataclass
class LocalFrame:
    """An oriented coordinate system: position + orthonormal localX/Y/Z."""

    pos: np.ndarray
    local_x: np.ndarray
    local_y: np.ndarray
    local_z: np.ndarray

    @classmethod
    def from_z(cls, pos: np.ndarray, local_z: np.ndarray, local_x: np.ndarray | None = None) -> "LocalFrame":
        z = _unit(local_z)
        if local_x is None:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(z @ ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x = _unit(ref - (ref @ z) * z)
        else:
            x = _unit(local_x - (local_x @ z) * z)
        y = np.cross(z, x)
        return cls(pos=np.asarray(pos, dtype=np.float32), local_x=x, local_y=y, local_z=z)

    @classmethod
    def identity(cls) -> "LocalFrame":
        return cls(
            pos=np.zeros(3, dtype=np.float32),
            local_x=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            local_y=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            local_z=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )


class FrameType(Enum):
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    Z = "z"
    MIN_ROTATION = "min_rotation"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


def _target_x(p: np.ndarray, frame_type: FrameType) -> np.ndarray:
    if frame_type is FrameType.CYLINDRICAL:
        r = np.array([p[0], p[1], 0.0], dtype=np.float32)
        return _unit(r) if np.linalg.norm(r) > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if frame_type is FrameType.SPHERICAL:
        return _unit(p) if np.linalg.norm(p) > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _align_local_x(local_z: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """Rotate an in-plane localX around localZ to best align with target_x."""
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(local_z @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x0 = _unit(ref - (ref @ local_z) * local_z)
    best = x0
    best_dot = float(x0 @ target_x)
    for deg in np.linspace(0.0, 180.0, 181):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        x = c * x0 + s * np.cross(local_z, x0)
        d = float(x @ target_x)
        if d > best_dot:
            best_dot, best = d, x
    return best


@dataclass
class Frames:
    """A sequence of :class:`LocalFrame` along a polyline spine.

    ``points`` is the spine ``C(s)``; tangents come from finite differences.
    The in-plane ``localX`` is resolved per station by ``frame_type`` (or a
    constant ``target_x``).  Query with :meth:`frame` at a length ratio.
    """

    points: np.ndarray
    frame_type: FrameType | None
    target_x: np.ndarray | None
    frames: list[LocalFrame]

    @classmethod
    def from_points(
        cls,
        points: Sequence[np.ndarray],
        frame_type: FrameType | None = None,
        target_x: np.ndarray | None = None,
    ) -> "Frames":
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 2:
            raise ValueError("spine needs at least two points")
        tangents = np.zeros_like(pts)
        tangents[1:-1] = pts[2:] - pts[:-2]
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
        tangents = np.array([_unit(t) for t in tangents])
        frames = []
        for p, z in zip(pts, tangents):
            if frame_type is not None:
                tgt = _target_x(p, frame_type)
            elif target_x is not None:
                tgt = np.asarray(target_x, dtype=np.float32)
            else:
                tgt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x = _align_local_x(z, tgt)
            y = np.cross(z, x)
            frames.append(LocalFrame(pos=p, local_x=x, local_y=y, local_z=z))
        return cls(points=pts, frame_type=frame_type, target_x=target_x, frames=frames)

    @classmethod
    def straight(cls, length: float, frame: LocalFrame, n: int = 2) -> "Frames":
        p0 = frame.pos
        p1 = p0 + length * frame.local_z
        pts = np.linspace(p0, p1, n, dtype=np.float32)
        return cls.from_points(pts, target_x=frame.local_x)

    def frame(self, length_ratio: float) -> LocalFrame:
        r = float(np.clip(length_ratio, 0.0, 1.0)) * (len(self.frames) - 1)
        i = int(np.floor(r))
        if i >= len(self.frames) - 1:
            return self.frames[-1]
        f = r - i
        a, b = self.frames[i], self.frames[i + 1]
        return LocalFrame(
            pos=(1 - f) * a.pos + f * b.pos,
            local_x=_unit((1 - f) * a.local_x + f * b.local_x),
            local_y=_unit((1 - f) * a.local_y + f * b.local_y),
            local_z=_unit((1 - f) * a.local_z + f * b.local_z),
        )

    def spine(self, length_ratio: float) -> np.ndarray:
        return self.frame(length_ratio).pos
