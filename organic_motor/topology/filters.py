"""Density filtering: Helmholtz PDE filter (and a Gaussian fallback).

The Helmholtz filter solves (I - r^2 Laplace) rho_tilde = rho with homogeneous
Neumann boundary conditions, which enforces a minimum length scale on the
design and is the standard method used in modern density-based topology
optimisation (cf. ARL_Topologies' density filtering).

Because the operator is linear and preserves the constant function, applying the
same filter to ``rho_iron`` and ``rho_pm`` keeps ``rho_air = 1 - rho_iron -
rho_pm`` non-negative automatically.
"""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner
from organic_motor.physics.maxwell import diffusion, diffusion_diagonal


def helmholtz_filter(field: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Return the Helmholtz-PDE-filtered field (range-preserving, Neumann BC).

    Solves ``(I - r^2 Laplace) rho_tilde = rho`` with a fixed number of
    preconditioned CG iterations.  The operator is well conditioned (spectral
    radius ``~ 1 + (r/h)^2``), so few iterations suffice.
    """
    r = cfg.filt_radius
    if r <= 0.0:
        return field
    ones = jnp.ones_like(field)

    def L(u):
        return u + (r ** 2) * diffusion(ones, u, cfg)

    diag = 1.0 + (r ** 2) * diffusion_diagonal(ones, cfg)
    return cg_fixed(L, field, jnp.zeros_like(field),
                    jacobi_preconditioner(diag), 80, cfg.filter_tol)


def gaussian_filter(field: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Separable Gaussian convolution filter (simple, differentiable fallback)."""
    from jax.scipy.signal import convolve2d

    sigma = cfg.filt_radius / 2.0
    n_kernel = max(3, int(round(4.0 * sigma / cfg.h)) | 1)
    x = jnp.arange(-(n_kernel // 2), n_kernel // 2 + 1)
    k1 = jnp.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k1 = k1 / k1.sum()
    k = k1[:, None] * k1[None, :]
    return convolve2d(field, k, mode="same", boundary="fill")


def filtered_density_pair(rho_iron: jnp.ndarray, rho_pm: jnp.ndarray,
                          cfg: MotorConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Filter iron & PM densities independently (sum-preserving)."""
    if cfg.filt_radius <= 0.0:
        return rho_iron, rho_pm
    return helmholtz_filter(rho_iron, cfg), helmholtz_filter(rho_pm, cfg)


def total_variation(field: jnp.ndarray, cfg: MotorConfig) -> jnp.ndarray:
    """Isotropic total-variation measure (perimeter proxy) of a field."""
    dx = jnp.gradient(field, cfg.h, axis=0)
    dy = jnp.gradient(field, cfg.h, axis=1)
    return jnp.mean(jnp.sqrt(dx ** 2 + dy ** 2 + 1e-12))
