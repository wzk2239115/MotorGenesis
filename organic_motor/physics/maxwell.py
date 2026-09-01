"""Differentiable 2-D magnetostatic Maxwell solver.

Solves the scalar magnetic vector potential ``Az`` (out-of-plane component) from

    - div( nu grad Az ) = Jz + curl(M)_z ,      curl(M)_z = dMy/dx - dMx/dy

on a uniform square node grid with a Dirichlet wall ``Az = 0`` on the outer
boundary (models a flux-returning yoke).  ``nu`` is the field of magnetic
reluctivity (1/mu) and ``M = (Mx, My)`` the remanent magnetisation.

The elliptic operator is applied matrix-free and solved with preconditioned
conjugate gradients; JAX differentiates *through* the solve implicitly (one
extra CG solve for the adjoint), so gradients flow back to ``nu`` and ``M``.

Grid convention: ``axis 0 == x``, ``axis 1 == y`` (meshgrid ``indexing="ij"``).
"""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner


def diffusion(nu: jnp.ndarray, u: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Return ``-div(nu grad u)`` with zero-flux (Neumann) at the box edges.

    ``nu``, ``u`` shape (N, N); the result is a pure finite-difference action
    with no boundary masking (callers apply Dirichlet masking as needed).
    """
    h = cfg.h
    nux = (nu[:-1, :] + nu[1:, :]) / 2.0      # x-faces, shape (N-1, N)
    nuy = (nu[:, :-1] + nu[:, 1:]) / 2.0      # y-faces, shape (N, N-1)

    fx = nux * (u[1:, :] - u[:-1, :]) / h     # x flux at (i+1/2, j)
    fy = nuy * (u[:, 1:] - u[:, :-1]) / h     # y flux at (i, j+1/2)

    out = jnp.zeros_like(u)
    out = out.at[1:, :].add(fx / h)
    out = out.at[:-1, :].add(-fx / h)
    out = out.at[:, 1:].add(fy / h)
    out = out.at[:, :-1].add(-fy / h)
    return out


def diffusion_diagonal(nu: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Diagonal of the diffusion operator (sum of face reluctivities / h^2)."""
    nux = (nu[:-1, :] + nu[1:, :]) / 2.0
    nuy = (nu[:, :-1] + nu[:, 1:]) / 2.0
    diag = jnp.zeros_like(nu)
    diag = diag.at[1:, :].add(nux)
    diag = diag.at[:-1, :].add(nux)
    diag = diag.at[:, 1:].add(nuy)
    diag = diag.at[:, :-1].add(nuy)
    return diag / (cfg.h ** 2)


def curl_magnetization(Mx: jnp.ndarray, My: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Return ``curl(M)_z = dMy/dx - dMx/dy`` (central differences)."""
    return jnp.gradient(My, cfg.h, axis=0) - jnp.gradient(Mx, cfg.h, axis=1)


def magnetostatic_solve(nu: jnp.ndarray, Mx: jnp.ndarray, My: jnp.ndarray,
                        Jz: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Solve for the magnetic vector potential ``Az`` (N, N).

    Uses Jacobi-preconditioned conjugate gradients with a fixed iteration count
    (differentiated through implicitly for exact gradients).
    """
    src = curl_magnetization(Mx, My, cfg) + Jz
    boundary = _boundary_mask(cfg)

    def A(u):
        return jnp.where(boundary, u, diffusion(nu, u, cfg))

    b = jnp.where(boundary, 0.0, src)
    diag = jnp.where(boundary, 1.0, diffusion_diagonal(nu, cfg))
    x0 = jnp.zeros_like(b)
    az = cg_fixed(A, b, x0, jacobi_preconditioner(diag), cfg.maxwell_maxiter)
    return az


def flux_density(az: jnp.ndarray, cfg: MotorConfig):
    """Magnetic flux density ``B = curl(A) = (dAz/dy, -dAz/dx)``."""
    Bx = jnp.gradient(az, cfg.h, axis=1)
    By = -jnp.gradient(az, cfg.h, axis=0)
    return Bx, By


def _boundary_mask(cfg: MotorConfig) -> jnp.ndarray:
    from organic_motor.geometry.sdf import domain_masks
    return domain_masks(cfg)["boundary"]
