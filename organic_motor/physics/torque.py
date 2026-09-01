"""Torque evaluation for the 2-D permanent-magnet motor.

Two equivalent definitions are provided, both returning torque per unit axial
depth [N.m / m] (multiply by stack length for the physical torque):

* :func:`lorentz_torque` -- integrate the magnetisation-current Lorentz force
  ``f = curl(M)_z x B`` over the rotor (``r < R_split``).  For a rigid rotor the
  self-torque cancels, so this gives the rotor-stator interaction directly.

* :func:`maxwell_torque` -- integrate the vacuum Maxwell stress tensor over a
  closed circle in the air gap, ``tau = int_C r x (T.n) dl`` with
  ``T_ij = (1/mu0)(B_i B_j - (1/2) delta_ij B^2)``.  This is exact only when the
  circle encloses all sources; here the design annulus lies outside the air gap,
  so this is kept as a reference/visualisation quantity.
"""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import bilinear_sample, domain_masks, meshgrid
from organic_motor.physics.maxwell import curl_magnetization


def lorentz_torque(Bx: jnp.ndarray, By: jnp.ndarray,
                   Mx: jnp.ndarray, My: jnp.ndarray,
                   cfg: MotorConfig) -> jnp.ndarray:
    """Torque [N.m/m] on the rotor via the magnetisation-current Lorentz force.

    The force density on the permanent magnet is ``f = Jm x B`` with the
    equivalent magnetisation current ``Jm = (0, 0, curl(M)_z)``.  The torque
    about the axis on the rotor (``r < R_split``) is therefore

        tau = int_{rotor} curl(M)_z * (x Bx + y By) dA .

    For a rigid rotor the self-torque (rotor PM in its own field) cancels, so
    this yields the rotor-stator interaction torque without needing an explicit
    air-gap integration circle.
    """
    Jz = curl_magnetization(Mx, My, cfg)
    X, Y, _ = meshgrid(cfg)
    rotor = domain_masks(cfg)["rotor"]
    integrand = Jz * (X * Bx + Y * By)
    return jnp.sum(integrand * rotor) * (cfg.h ** 2)


def maxwell_torque(Bx: jnp.ndarray, By: jnp.ndarray, cfg: MotorConfig,
                   radius: float | None = None) -> jnp.ndarray:
    """Torque [N.m/m] from the flux density field, integrated on a circle."""
    Rt = cfg.R_torque if radius is None else radius
    n = cfg.n_theta
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    dl = Rt * 2.0 * jnp.pi / n

    x = Rt * jnp.cos(theta)
    y = Rt * jnp.sin(theta)
    nx, ny = jnp.cos(theta), jnp.sin(theta)

    bx = bilinear_sample(Bx, x, y, cfg)
    by = bilinear_sample(By, x, y, cfg)

    bdotn = bx * nx + by * ny
    b2 = bx * bx + by * by

    # (T . n) = (1/mu0) [ (B.n) B - (B^2 / 2) n ]
    tn_x = (bdotn * bx - 0.5 * b2 * nx) / cfg.mu0
    tn_y = (bdotn * by - 0.5 * b2 * ny) / cfg.mu0

    # r x (T.n)  ->  z-component = x * (T.n)_y - y * (T.n)_x
    tau_density = x * tn_y - y * tn_x
    return jnp.sum(tau_density) * dl


def torque_from_solution(az: jnp.ndarray, cfg: MotorConfig,
                         radius: float | None = None) -> jnp.ndarray:
    """Torque from the vector potential ``az`` (computes B internally)."""
    from organic_motor.physics.maxwell import flux_density
    Bx, By = flux_density(az, cfg)
    return maxwell_torque(Bx, By, cfg, radius)
