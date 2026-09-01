"""Geometry primitives for the circular free-topology motor domain.

The design envelope is fixed (a set of concentric circular partitions inside a
square Dirichlet box); what is *free* is the material distribution inside the
design annulus.  This module only builds the unchanging geometric scaffolding:
coordinates, SDF-style radial masks, torque-circle sampling points and a
differentiable in-plane rotation warp used by the ripple experiment.

Everything returns JAX arrays so the whole pipeline stays traceable.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp

from organic_motor.config import MotorConfig


def coords_1d(cfg: MotorConfig) -> jnp.ndarray:
    """1-D node coordinate vector x in [-L, L], shape (N,)."""
    return jnp.linspace(-cfg.L, cfg.L, cfg.N)


@lru_cache(maxsize=16)
def _meshgrid(N: int, L: float):
    """Cached (X, Y, R) node grids, each shape (N, N)."""
    c = jnp.linspace(-L, L, N)
    X, Y = jnp.meshgrid(c, c, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2)
    return X, Y, R


def meshgrid(cfg: MotorConfig):
    """(X, Y, R) node grids; X/Y cartesian [m], R radius from origin."""
    return _meshgrid(cfg.N, cfg.L)


def radial_mask_piecewise(R: jnp.ndarray, r0: float, r1: float) -> jnp.ndarray:
    """Smooth indicator of r0 <= r < r1 (hard step, but with soft edges).

    Uses a hard comparison; the density filtering downstream provides the true
    length-scale control.
    """
    return (R >= r0) & (R < r1)


def domain_masks(cfg: MotorConfig):
    """Return the boolean masks partitioning the box.

    Returns a dict with keys:
      shaft    : r < R_shaft           (fixed air)
      rotor_design: R_shaft <= r < R_rotor_outer
      airgap   : R_rotor_outer <= r < R_stator_inner
      stator_design: R_stator_inner <= r < R_design
      outer    : r >= R_design         (fixed air)
      rotor    : r < R_rotor_outer     (rotates in ripple experiment)
      stator   : R_stator_inner <= r   (fixed in ripple experiment)
      boundary : outer square edge (Dirichlet nodes)
    """
    X, Y, R = meshgrid(cfg)
    shaft = R < cfg.R_shaft
    rotor_design = (R >= cfg.R_shaft) & (R < cfg.R_rotor_outer)
    airgap = (R >= cfg.R_rotor_outer) & (R < cfg.R_stator_inner)
    stator_design = (R >= cfg.R_stator_inner) & (R < cfg.R_design)
    design = rotor_design | stator_design
    outer = R >= cfg.R_design
    rotor = R < cfg.R_rotor_outer
    stator = R >= cfg.R_stator_inner
    winding = ((R >= cfg.R_winding_inner) & (R < cfg.R_winding_outer))

    # Dirichlet boundary: the outer square edges of the node grid
    boundary = jnp.zeros_like(R, dtype=bool)
    boundary = boundary.at[0, :].set(True)
    boundary = boundary.at[-1, :].set(True)
    boundary = boundary.at[:, 0].set(True)
    boundary = boundary.at[:, -1].set(True)

    return {
        "shaft": shaft,
        "airgap": airgap,
        "rotor_design": rotor_design,
        "stator_design": stator_design,
        "design": design,
        "outer": outer,
        "rotor": rotor,
        "stator": stator,
        "winding": winding,
        "boundary": boundary,
    }


def torque_circle(cfg: MotorConfig, n_theta: int | None = None):
    """Sample points (x, y, nx, ny, dl) on the torque-evaluation circle.

    Returns:
      x, y   : coordinates of the n_theta sample points [m]
      nx, ny : outward unit normal (radial)
      dl     : arc length weight = R * dtheta   [m]
      theta  : angles [rad]
    """
    n = cfg.n_theta if n_theta is None else n_theta
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    dtheta = 2.0 * jnp.pi / n
    x = cfg.R_torque * jnp.cos(theta)
    y = cfg.R_torque * jnp.sin(theta)
    nx = jnp.cos(theta)
    ny = jnp.sin(theta)
    dl = cfg.R_torque * dtheta
    return x, y, nx, ny, dl, theta


def bilinear_sample(field: jnp.ndarray, xs: jnp.ndarray, ys: jnp.ndarray,
                    cfg: MotorConfig) -> jnp.ndarray:
    """Sample ``field`` (shape (N, N)) at arbitrary coordinates via bilinear interp.

    Coordinates outside the box are clamped to the edge (matches the Dirichlet
    wall where the field is held to zero).
    """
    c = coords_1d(cfg)
    h = cfg.h
    lo = -cfg.L
    # normalised coordinate in node index space
    u = (xs - lo) / h
    v = (ys - lo) / h
    u = jnp.clip(u, 0.0, cfg.N - 1.0)
    v = jnp.clip(v, 0.0, cfg.N - 1.0)
    i0 = jnp.floor(u).astype(jnp.int32)
    j0 = jnp.floor(v).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, cfg.N - 1)
    j1 = jnp.minimum(j0 + 1, cfg.N - 1)
    fu = u - i0
    fv = v - j0
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]
    return (f00 * (1 - fu) * (1 - fv) + f10 * fu * (1 - fv)
            + f01 * (1 - fu) * fv + f11 * fu * fv)


def rotate_field(field: jnp.ndarray, theta: float, cfg: MotorConfig) -> jnp.ndarray:
    """Return ``field`` rigidly rotated by ``theta`` (about the origin).

    Samples ``field`` at R(-theta) @ [x, y]; i.e. the material that was at
    angle phi is now at angle phi + theta.  Fully differentiable.
    """
    X, Y, _ = meshgrid(cfg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    xs_rot = c * X + s * Y
    ys_rot = -s * X + c * Y
    return bilinear_sample(field, xs_rot, ys_rot, cfg)


def rotate_rotor(field: jnp.ndarray, theta: float, cfg: MotorConfig) -> jnp.ndarray:
    """Rotate the *rotor* part of ``field`` by ``theta``; stator stays fixed.

    The shaft is fixed air and the interpolation is blended only inside the
    moving rotor boundary.  The mechanical air gap remains source-free.
    """
    rotor = domain_masks(cfg)["rotor"].astype(field.dtype)
    rotated = rotate_field(field, theta, cfg)
    return rotor * rotated + (1.0 - rotor) * field


def rotate_rotor_vector(fx: jnp.ndarray, fy: jnp.ndarray, theta: float,
                        cfg: MotorConfig):
    """Rigidly rotate a rotor-bound in-plane vector field.

    Rotation requires both spatial advection and rotation of vector components;
    treating the components as independent scalar images is physically wrong.
    """
    rotor = domain_masks(cfg)["rotor"].astype(fx.dtype)
    c, s = jnp.cos(theta), jnp.sin(theta)
    fx_adv = rotate_field(fx, theta, cfg)
    fy_adv = rotate_field(fy, theta, cfg)
    fx_rot = c * fx_adv - s * fy_adv
    fy_rot = s * fx_adv + c * fy_adv
    return (rotor * fx_rot + (1.0 - rotor) * fx,
            rotor * fy_rot + (1.0 - rotor) * fy)
