"""Three-axis force and torque from a closed cylindrical Maxwell-stress surface."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from organic_motor.physics.electric3d import _shape, _spacing


def _coordinates(cfg: Any):
    nx, ny, nz = _shape(cfg)
    hx, hy, hz = _spacing(cfg)
    origin = getattr(cfg, "origin", None)

    def axis(n, h, name, axis_index):
        lower = getattr(cfg, f"{name}_min", None)
        if lower is None and origin is not None:
            lower = origin[axis_index]
        if lower is None:
            half = getattr(cfg, f"L{name}", getattr(cfg, "L", 0.5 * (n - 1) * h))
            lower = -half
        return lower + h * jnp.arange(n)

    return (axis(nx, hx, "x", 0), axis(ny, hy, "y", 1),
            axis(nz, hz, "z", 2))


def _trilinear(field: jnp.ndarray, x, y, z, cfg: Any) -> jnp.ndarray:
    """Vectorised differentiable trilinear sampling on the node grid."""
    xs, ys, zs = _coordinates(cfg)
    hx, hy, hz = _spacing(cfg)
    nx, ny, nz = field.shape
    ux = jnp.clip((x - xs[0]) / hx, 0.0, nx - 1.0)
    uy = jnp.clip((y - ys[0]) / hy, 0.0, ny - 1.0)
    uz = jnp.clip((z - zs[0]) / hz, 0.0, nz - 1.0)
    i0 = jnp.minimum(jnp.floor(ux).astype(jnp.int32), nx - 2)
    j0 = jnp.minimum(jnp.floor(uy).astype(jnp.int32), ny - 2)
    k0 = jnp.minimum(jnp.floor(uz).astype(jnp.int32), nz - 2)
    tx, ty, tz = ux - i0, uy - j0, uz - k0

    value = jnp.zeros_like(tx, dtype=field.dtype)
    for di in (0, 1):
        wx = (1.0 - tx) if di == 0 else tx
        for dj in (0, 1):
            wy = (1.0 - ty) if dj == 0 else ty
            for dk in (0, 1):
                wz = (1.0 - tz) if dk == 0 else tz
                value = value + wx * wy * wz * field[i0 + di, j0 + dj, k0 + dk]
    return value


def _traction(B: jnp.ndarray, normal: jnp.ndarray, mu0: float) -> jnp.ndarray:
    bdotn = jnp.sum(B * normal, axis=-1, keepdims=True)
    b2 = jnp.sum(B * B, axis=-1, keepdims=True)
    return (bdotn * B - 0.5 * b2 * normal) / mu0


def maxwell_force_torque(Bx: jnp.ndarray, By: jnp.ndarray, Bz: jnp.ndarray,
                         cfg: Any, radius: float | None = None,
                         z_min: float | None = None,
                         z_max: float | None = None,
                         center: tuple[float, float, float] | None = None,
                         n_theta: int | None = None,
                         n_z: int | None = None,
                         n_r: int | None = None):
    """Integrate Maxwell stress over cylinder side and both end caps.

    Returns ``(force_xyz, torque_xyz)`` in N and N.m.  Midpoint quadrature is
    used on every surface, avoiding duplicated seams and the polar singularity.
    """
    nx, ny, nz = _shape(cfg)
    xs, ys, zs = _coordinates(cfg)
    if center is None:
        center = getattr(cfg, "center", (0.0, 0.0, 0.0))
    cx, cy, cz = center
    if radius is None:
        radius = getattr(cfg, "R_torque", getattr(cfg, "torque_radius",
                         0.4 * min(xs[-1] - xs[0], ys[-1] - ys[0])))
    if z_min is None:
        z_min = getattr(cfg, "torque_z_min", None)
    if z_max is None:
        z_max = getattr(cfg, "torque_z_max", None)
    if z_min is None or z_max is None:
        rotor_half = getattr(cfg, "rotor_half_length", None)
        if rotor_half is None:
            default_min, default_max = zs[0], zs[-1]
        else:
            half = getattr(
                cfg, "torque_half_length",
                rotor_half + 0.5 * getattr(cfg, "axial_airgap", 0.0),
            )
            default_min = jnp.maximum(zs[0], cz - half)
            default_max = jnp.minimum(zs[-1], cz + half)
        z_min = default_min if z_min is None else z_min
        z_max = default_max if z_max is None else z_max
    nt = int(getattr(cfg, "n_theta", 128) if n_theta is None else n_theta)
    nzq = int(getattr(cfg, "torque_n_z", max(nz - 1, 1)) if n_z is None else n_z)
    nrq = int(getattr(cfg, "torque_n_r", max(min(nx, ny) // 2, 1))
              if n_r is None else n_r)
    mu0 = getattr(cfg, "mu0", 1.25663706127e-6)

    theta = (jnp.arange(nt) + 0.5) * (2.0 * jnp.pi / nt)
    zq = z_min + (jnp.arange(nzq) + 0.5) * ((z_max - z_min) / nzq)
    th, zz = jnp.meshgrid(theta, zq, indexing="ij")
    side_x = cx + radius * jnp.cos(th)
    side_y = cy + radius * jnp.sin(th)
    side_z = jnp.broadcast_to(zz, th.shape)
    side_n = jnp.stack((jnp.cos(th), jnp.sin(th), jnp.zeros_like(th)), axis=-1)
    side_B = jnp.stack(tuple(_trilinear(f, side_x, side_y, side_z, cfg)
                             for f in (Bx, By, Bz)), axis=-1)
    side_t = _traction(side_B, side_n, mu0)
    side_area = radius * (2.0 * jnp.pi / nt) * ((z_max - z_min) / nzq)
    side_r = jnp.stack((side_x - cx, side_y - cy, side_z - cz), axis=-1)

    rq = (jnp.arange(nrq) + 0.5) * (radius / nrq)
    th_cap, rr = jnp.meshgrid(theta, rq, indexing="ij")
    cap_x = cx + rr * jnp.cos(th_cap)
    cap_y = cy + rr * jnp.sin(th_cap)
    cap_area = rr * (radius / nrq) * (2.0 * jnp.pi / nt)

    force = jnp.sum(side_t * side_area, axis=(0, 1))
    torque = jnp.sum(jnp.cross(side_r, side_t) * side_area, axis=(0, 1))
    for cap_z, sign in ((z_min, -1.0), (z_max, 1.0)):
        z_arr = jnp.full_like(cap_x, cap_z)
        normal = jnp.broadcast_to(jnp.asarray((0.0, 0.0, sign)), cap_x.shape + (3,))
        cap_B = jnp.stack(tuple(_trilinear(f, cap_x, cap_y, z_arr, cfg)
                                for f in (Bx, By, Bz)), axis=-1)
        cap_t = _traction(cap_B, normal, mu0)
        cap_r = jnp.stack((cap_x - cx, cap_y - cy, z_arr - cz), axis=-1)
        force = force + jnp.sum(cap_t * cap_area[..., None], axis=(0, 1))
        torque = torque + jnp.sum(
            jnp.cross(cap_r, cap_t) * cap_area[..., None], axis=(0, 1))
    return force, torque


def maxwell_torque(Bx: jnp.ndarray, By: jnp.ndarray, Bz: jnp.ndarray,
                   cfg: Any, **kwargs) -> jnp.ndarray:
    """Return the three torque components from the closed cylinder."""
    return maxwell_force_torque(Bx, By, Bz, cfg, **kwargs)[1]


def torque_from_solution(vector_potential_field, cfg: Any, **kwargs):
    from organic_motor.physics.maxwell3d import flux_density
    return maxwell_torque(*flux_density(vector_potential_field, cfg), cfg, **kwargs)


maxwell_stress_torque = maxwell_torque
