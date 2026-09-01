"""Differentiable three-dimensional steady electric-conduction model.

Arrays use the node convention ``(Nx, Ny, Nz)``.  Terminal voltages are
imposed strongly while all other box faces are electrically insulating.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner, relative_residual


def _spacing(cfg: Any) -> tuple[float, float, float]:
    """Read anisotropic spacing from both current and planned 3-D configs."""
    if all(hasattr(cfg, name) for name in ("hx", "hy", "hz")):
        return cfg.hx, cfg.hy, cfg.hz
    spacing = getattr(cfg, "spacing", getattr(cfg, "h", 1.0))
    if hasattr(spacing, "__len__"):
        return spacing[0], spacing[1], spacing[2]
    return spacing, spacing, spacing


def _shape(cfg: Any) -> tuple[int, int, int]:
    return int(cfg.Nx), int(cfg.Ny), int(cfg.Nz)


def _outer_boundary(shape: tuple[int, int, int]) -> jnp.ndarray:
    mask = jnp.zeros(shape, dtype=bool)
    mask = mask.at[0, :, :].set(True).at[-1, :, :].set(True)
    mask = mask.at[:, 0, :].set(True).at[:, -1, :].set(True)
    return mask.at[:, :, 0].set(True).at[:, :, -1].set(True)


def _gradient(u: jnp.ndarray, cfg: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    try:
        from organic_motor.physics import operators3d
    except ImportError:
        operators3d = None
    gradient_scalar = (None if operators3d is None else
                       getattr(operators3d, "gradient_scalar",
                               getattr(operators3d, "gradient3d", None)))
    if gradient_scalar is not None:
        try:
            value = gradient_scalar(u, cfg)
        except TypeError:
            value = gradient_scalar(u, *_spacing(cfg))
        if isinstance(value, (tuple, list)):
            return value[0], value[1], value[2]
        if value.shape[0] == 3:
            return value[0], value[1], value[2]
        return value[..., 0], value[..., 1], value[..., 2]
    hx, hy, hz = _spacing(cfg)
    return (jnp.gradient(u, hx, axis=0),
            jnp.gradient(u, hy, axis=1),
            jnp.gradient(u, hz, axis=2))


def _variable_diffusion(k: jnp.ndarray, u: jnp.ndarray, cfg: Any) -> jnp.ndarray:
    """Return the positive operator ``-div(k grad(u))``."""
    try:
        from organic_motor.physics import operators3d
    except ImportError:
        operators3d = None
    variable_diffusion = (None if operators3d is None else
                          getattr(operators3d, "variable_diffusion",
                                  getattr(operators3d, "variable_diffusion3d", None)))
    if variable_diffusion is not None:
        try:
            return variable_diffusion(k, u, cfg)
        except TypeError:
            return variable_diffusion(k, u, *_spacing(cfg))

    hx, hy, hz = _spacing(cfg)
    out = jnp.zeros_like(u)
    for axis, h in enumerate((hx, hy, hz)):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        face_k = 0.5 * (k[lo] + k[hi])
        flux = face_k * (u[hi] - u[lo]) / h
        out = out.at[hi].add(flux / h)
        out = out.at[lo].add(-flux / h)
    return out


def _diffusion_diagonal(k: jnp.ndarray, cfg: Any) -> jnp.ndarray:
    try:
        from organic_motor.physics import operators3d
    except ImportError:
        operators3d = None
    variable_diffusion_diagonal = (
        None if operators3d is None else
        getattr(operators3d, "variable_diffusion_diagonal",
                getattr(operators3d, "diffusion_diagonal3d", None))
    )
    if variable_diffusion_diagonal is not None:
        try:
            return variable_diffusion_diagonal(k, cfg)
        except TypeError:
            return variable_diffusion_diagonal(k, *_spacing(cfg))

    hx, hy, hz = _spacing(cfg)
    diag = jnp.zeros_like(k)
    for axis, h in enumerate((hx, hy, hz)):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        face_k = 0.5 * (k[lo] + k[hi]) / (h * h)
        diag = diag.at[lo].add(face_k)
        diag = diag.at[hi].add(face_k)
    return diag


def ersatz_conductivity(rho_copper: jnp.ndarray, cfg: Any,
                        sigma_solid: float | None = None,
                        sigma_void: float | None = None,
                        penalization: float | None = None) -> jnp.ndarray:
    """SIMP/ersatz conductivity, bounded away from zero in void."""
    solid = getattr(cfg, "sigma_copper", 5.8e7) if sigma_solid is None else sigma_solid
    void = getattr(cfg, "sigma_void", getattr(cfg, "electric_sigma_void", 1e-6))
    if sigma_void is not None:
        void = sigma_void
    power = getattr(cfg, "electric_simp_p", getattr(cfg, "simp_p", 1.0))
    if penalization is not None:
        power = penalization
    rho = jnp.clip(jnp.asarray(rho_copper), 0.0, 1.0)
    return void + (solid - void) * rho ** power


def solve_potential(conductivity: jnp.ndarray, terminal_mask: jnp.ndarray,
                    terminal_potential: jnp.ndarray | float, cfg: Any) -> jnp.ndarray:
    """Solve ``div(sigma grad(phi)) = 0`` with arbitrary Dirichlet terminals."""
    conductivity = jnp.asarray(conductivity)
    mask = jnp.asarray(terminal_mask, dtype=bool)
    values = jnp.broadcast_to(jnp.asarray(terminal_potential, conductivity.dtype),
                              conductivity.shape)
    free = (~mask).astype(conductivity.dtype)
    fixed = jnp.where(mask, values, 0.0)

    def operator(x):
        free_x = free * x
        return jnp.where(mask, x, _variable_diffusion(conductivity, free_x, cfg))

    # L(fixed + free*x)=0 on free nodes; this correction also keeps the
    # projected free-node operator symmetric positive definite.
    rhs = jnp.where(mask, values, -_variable_diffusion(conductivity, fixed, cfg))
    diag = jnp.where(mask, 1.0, _diffusion_diagonal(conductivity, cfg))
    n_iter = int(getattr(cfg, "electric_maxiter", getattr(cfg, "maxiter", 300)))
    tol = getattr(cfg, "electric_tol", getattr(cfg, "tol", 1e-8))
    return cg_fixed(operator, rhs, fixed, jacobi_preconditioner(diag), n_iter, tol)


def electric_relative_residual(conductivity: jnp.ndarray,
                               terminal_mask: jnp.ndarray,
                               terminal_potential: jnp.ndarray | float,
                               potential: jnp.ndarray, cfg: Any) -> jnp.ndarray:
    mask = jnp.asarray(terminal_mask, dtype=bool)
    values = jnp.broadcast_to(jnp.asarray(terminal_potential, potential.dtype),
                              potential.shape)
    free = (~mask).astype(potential.dtype)
    fixed = jnp.where(mask, values, 0.0)

    def operator(x):
        return jnp.where(mask, x,
                         _variable_diffusion(conductivity, free * x, cfg))

    rhs = jnp.where(mask, values, -_variable_diffusion(conductivity, fixed, cfg))
    return relative_residual(operator, potential, rhs)


def current_density(potential: jnp.ndarray, conductivity: jnp.ndarray,
                    cfg: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``J = -sigma grad(phi)`` in A/m²."""
    gx, gy, gz = _gradient(potential, cfg)
    return -conductivity * gx, -conductivity * gy, -conductivity * gz


def joule_loss(potential: jnp.ndarray, conductivity: jnp.ndarray,
               cfg: Any) -> jnp.ndarray:
    """Volumetric Joule heat ``sigma |grad(phi)|²`` in W/m³."""
    gx, gy, gz = _gradient(potential, cfg)
    return conductivity * (gx * gx + gy * gy + gz * gz)


def solve_electric(conductivity: jnp.ndarray, terminal_mask: jnp.ndarray,
                   terminal_potential: jnp.ndarray | float, cfg: Any):
    """Convenience solve returning ``(phi, (Jx,Jy,Jz), joule_density)``."""
    phi = solve_potential(conductivity, terminal_mask, terminal_potential, cfg)
    return phi, current_density(phi, conductivity, cfg), joule_loss(phi, conductivity, cfg)


electric_potential = solve_potential
joule_heating = joule_loss
