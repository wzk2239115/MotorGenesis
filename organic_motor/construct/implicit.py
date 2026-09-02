"""Implicit signed-distance functions ``f(x, y, z) -> d``.

Each function returns a closure that is vectorised over numpy arrays and
evaluates to a (possibly approximate) signed distance: negative inside the
solid, zero on the surface, positive outside.  Approximate SDFs (gyroid,
finite cylinders) are still valid Booleans because the kernel only requires
the sign and a monotone distance ordering near the surface.

These are the constructive primitives a motor is built from: a rotor bore is
a cylinder, a magnet pole is a box, a cooling jacket wall is a gyroid sheet,
a winding slot is a rounded box subtracted from the stator annulus.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


Implicit = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def sphere(center: tuple[float, float, float], radius: float) -> Implicit:
    """An exact SDF for a sphere."""
    cx, cy, cz = center
    r = float(radius)

    def fn(x, y, z):
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - r

    return fn


def cylinder_z(center: tuple[float, float], radius: float, half_length: float) -> Implicit:
    """A finite cylinder about the z axis (exact SDF)."""
    cx, cy = center
    r, h = float(radius), float(half_length)

    def fn(x, y, z):
        radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
        axial = np.abs(z) - h
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return outside + inside

    return fn


def cylinder(
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
    radius: float,
    half_length: float,
) -> Implicit:
    """A finite cylinder about an arbitrary (unit) axis."""
    c = np.asarray(center, dtype=np.float32)
    a = np.asarray(axis, dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-12)
    r, h = float(radius), float(half_length)

    def fn(x, y, z):
        p = np.stack([x - c[0], y - c[1], z - c[2]], axis=-1)
        ax = np.sum(p * a, axis=-1)[..., None] * a
        orth = p - ax
        radial = np.linalg.norm(orth, axis=-1) - r
        axial = np.abs(np.sum(p * a, axis=-1)) - h
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return outside + inside

    return fn


def box(center: tuple[float, float, float], half_extents: tuple[float, float, float]) -> Implicit:
    """An exact SDF for an axis-aligned box."""
    c = np.asarray(center, dtype=np.float32)
    he = np.asarray(half_extents, dtype=np.float32)

    def fn(x, y, z):
        p = np.stack([np.abs(x - c[0]), np.abs(y - c[1]), np.abs(z - c[2])], axis=-1)
        q = p - he
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0)
        return outside + inside

    return fn


def rounded_box(
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    radius: float,
) -> Implicit:
    """A box with rounded edges and corners (exact SDF)."""
    c = np.asarray(center, dtype=np.float32)
    he = np.asarray(half_extents, dtype=np.float32) - float(radius)

    def fn(x, y, z):
        p = np.stack([np.abs(x - c[0]), np.abs(y - c[1]), np.abs(z - c[2])], axis=-1)
        q = p - he
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0)
        return outside + inside - radius

    return fn


def torus(
    center: tuple[float, float, float],
    major_radius: float,
    minor_radius: float,
) -> Implicit:
    """A torus in the z = cz plane (exact SDF)."""
    cx, cy, cz = center
    R, r = float(major_radius), float(minor_radius)

    def fn(x, y, z):
        radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - R
        return np.sqrt(radial ** 2 + (z - cz) ** 2) - r

    return fn


def plane(normal: tuple[float, float, float], offset: float) -> Implicit:
    """A half-space: negative where ``normal . p + offset < 0``."""
    n = np.asarray(normal, dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(offset)

    def fn(x, y, z):
        return n[0] * x + n[1] * y + n[2] * z + d

    return fn


def capsule(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    radius: float,
) -> Implicit:
    """A capsule (cylinder with hemispherical caps) between points ``a`` and ``b``."""
    pa = np.asarray(a, dtype=np.float32)
    pb = np.asarray(b, dtype=np.float32)
    r = float(radius)

    def fn(x, y, z):
        p = np.stack([x, y, z], axis=-1)
        ab = pb - pa
        t = np.clip(
            np.sum((p - pa) * ab, axis=-1) / (float(np.dot(ab, ab)) + 1e-12),
            0.0,
            1.0,
        )[..., None]
        closest = pa + t * ab
        return np.linalg.norm(p - closest, axis=-1) - r

    return fn


def annular_sector(
    center: tuple[float, float],
    r_inner: float,
    r_outer: float,
    a0: float,
    a1: float,
    half_z: float,
) -> Implicit:
    """A finite annular sector about the z axis: the universal motor primitive.

    Radial span ``[r_inner, r_outer]``, angular span ``[a0, a1]`` (radians,
    counterclockwise) and axial half-length ``half_z``.  This is the building
    block for magnet poles, stator teeth, slot openings and rotor poles --
    every part of a motor that is naturally specified in (r, theta, z).
    """
    cx, cy = center
    rm = 0.5 * (r_inner + r_outer)
    dr = 0.5 * (r_outer - r_inner)
    hz = float(half_z)
    a0, a1 = float(a0), float(a1)
    c0, s0 = np.cos(a0), np.sin(a0)
    c1, s1 = np.cos(a1), np.sin(a1)

    def fn(x, y, z):
        px = x - cx
        py = y - cy
        r = np.sqrt(px * px + py * py)
        radial = np.abs(r - rm) - dr
        axial = np.abs(z) - hz
        cyl = np.maximum(radial, axial)
        # Signed distance to the two radial boundary lines; positive inside
        # the angular span, scaled by radius to be in metres.
        h0 = c0 * py - s0 * px
        h1 = s1 * px - c1 * py
        return np.maximum(cyl, np.maximum(-h0, -h1))

    return fn


def gyroid(scale: float, thickness: float = 0.0) -> Implicit:
    """The triply-periodic minimal surface ``sin x cos y + sin y cos z + sin z cos x``.

    The level set at ``thickness`` (positive => solid struts, negative =>
    solid walls) is returned as an approximate SDF scaled by ``1/scale`` so it
    composes sensibly with other primitives at the same cell size.
    """
    s = float(scale)
    iso = float(thickness)

    def fn(x, y, z):
        f = (
            np.sin(s * x) * np.cos(s * y)
            + np.sin(s * y) * np.cos(s * z)
            + np.sin(s * z) * np.cos(s * x)
            - iso
        )
        return f / s

    return fn
