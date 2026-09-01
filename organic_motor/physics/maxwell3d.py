"""Native differentiable 3-D magnetostatic vector-potential solver.

The discretisation solves the Coulomb-stabilised weak form

``C.T nu C A + alpha D.T D A = J + curl(M)``,

where ``C`` and ``D`` are the discrete curl and divergence.  Constructing the
adjoints explicitly with JAX makes the free-node matrix symmetric positive
definite even when the underlying finite-difference boundary stencils are not
self-adjoint.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from organic_motor.physics.electric3d import _outer_boundary, _shape, _spacing
from organic_motor.physics.linear import cg_fixed, jacobi_preconditioner, relative_residual


def _stack_vector(v) -> jnp.ndarray:
    if isinstance(v, (tuple, list)):
        return jnp.stack((v[0], v[1], v[2]), axis=-1)
    value = jnp.asarray(v)
    if value.shape[-1] == 3:
        return value
    if value.shape[0] == 3:
        return jnp.moveaxis(value, 0, -1)
    raise ValueError("3-D vector fields must be a 3-tuple or have a component axis of length 3")


def _split_vector(v: jnp.ndarray):
    return v[..., 0], v[..., 1], v[..., 2]


def _curl(v: jnp.ndarray, cfg: Any) -> jnp.ndarray:
    try:
        from organic_motor.physics import operators3d
    except ImportError:
        operators3d = None
    curl_vector = (None if operators3d is None else
                   getattr(operators3d, "curl_vector",
                           getattr(operators3d, "curl3d", None)))
    if curl_vector is not None:
        try:
            result = curl_vector(v, cfg)
        except (TypeError, ValueError):
            try:
                result = curl_vector(*_split_vector(v), cfg)
            except TypeError:
                result = curl_vector(v, _spacing(cfg))
        return _stack_vector(result)

    hx, hy, hz = _spacing(cfg)
    vx, vy, vz = _split_vector(v)
    return jnp.stack((
        jnp.gradient(vz, hy, axis=1) - jnp.gradient(vy, hz, axis=2),
        jnp.gradient(vx, hz, axis=2) - jnp.gradient(vz, hx, axis=0),
        jnp.gradient(vy, hx, axis=0) - jnp.gradient(vx, hy, axis=1),
    ), axis=-1)


def _divergence(v: jnp.ndarray, cfg: Any) -> jnp.ndarray:
    try:
        from organic_motor.physics import operators3d
    except ImportError:
        operators3d = None
    divergence_vector = (None if operators3d is None else
                         getattr(operators3d, "divergence_vector",
                                 getattr(operators3d, "divergence3d", None)))
    if divergence_vector is not None:
        try:
            return divergence_vector(v, cfg)
        except (TypeError, ValueError):
            try:
                return divergence_vector(*_split_vector(v), cfg)
            except TypeError:
                return divergence_vector(v, _spacing(cfg))
    hx, hy, hz = _spacing(cfg)
    vx, vy, vz = _split_vector(v)
    return (jnp.gradient(vx, hx, axis=0)
            + jnp.gradient(vy, hy, axis=1)
            + jnp.gradient(vz, hz, axis=2))


def _adjoint(linear_fun, x_template: jnp.ndarray, cotangent: jnp.ndarray) -> jnp.ndarray:
    """Apply the exact transpose of a JAX-linear finite-difference map."""
    return jax.linear_transpose(linear_fun, x_template)(cotangent)[0]


def curl_magnetization(magnetization, cfg: Any) -> jnp.ndarray:
    """Equivalent bound-current density ``curl(M)``."""
    return _curl(_stack_vector(magnetization), cfg)


def _system(nu: jnp.ndarray, cfg: Any, gauge_penalty: float | None = None):
    shape = _shape(cfg)
    boundary = _outer_boundary(shape)
    interior = (~boundary).astype(nu.dtype)[..., None]
    boundary3 = boundary[..., None]
    zero_vector = jnp.zeros(shape + (3,), dtype=nu.dtype)
    zero_scalar = jnp.zeros(shape, dtype=nu.dtype)
    alpha = (getattr(cfg, "coulomb_gauge_penalty", None)
             if gauge_penalty is None else gauge_penalty)
    if alpha is None:
        alpha = jnp.mean(nu)
    alpha = jnp.asarray(alpha, dtype=nu.dtype)

    curl_fun = lambda x: _curl(x, cfg)
    div_fun = lambda x: _divergence(x, cfg)

    def energy_operator(x):
        y = interior * x
        cy = curl_fun(y)
        curl_term = _adjoint(curl_fun, zero_vector, nu[..., None] * cy)
        dy = div_fun(y)
        gauge_term = _adjoint(div_fun, zero_vector, alpha * dy)
        weak = interior * (curl_term + gauge_term)
        return jnp.where(boundary3, x, weak)

    hx, hy, hz = _spacing(cfg)
    inv_h2 = 1.0 / hx**2 + 1.0 / hy**2 + 1.0 / hz**2
    # A positive scale preconditioner; exact diagonal assembly is deliberately
    # avoided because operator3d may use either node- or face-centred stencils.
    scale = jnp.maximum(2.0 * (nu + alpha) * inv_h2,
                        jnp.finfo(nu.dtype).tiny)
    diagonal = jnp.where(boundary3, 1.0, scale[..., None])
    return energy_operator, diagonal, boundary, alpha


def magnetostatic_solve(nu: jnp.ndarray, magnetization, current_density,
                        cfg: Any, gauge_penalty: float | None = None) -> jnp.ndarray:
    """Solve for ``A=(Ax,Ay,Az)`` and return an array ``(Nx,Ny,Nz,3)``."""
    nu = jnp.asarray(nu)
    M = _stack_vector(magnetization)
    J = _stack_vector(current_density)
    source = J + curl_magnetization(M, cfg)
    operator, diagonal, boundary, _ = _system(nu, cfg, gauge_penalty)
    rhs = jnp.where(boundary[..., None], 0.0, source)
    n_iter = int(getattr(cfg, "maxwell_maxiter", getattr(cfg, "maxiter", 400)))
    tol = getattr(cfg, "maxwell_tol", getattr(cfg, "tol", 1e-8))
    return cg_fixed(operator, rhs, jnp.zeros_like(source),
                    jacobi_preconditioner(diagonal), n_iter, tol)


def vector_potential(nu: jnp.ndarray, Mx: jnp.ndarray, My: jnp.ndarray,
                     Mz: jnp.ndarray, Jx: jnp.ndarray, Jy: jnp.ndarray,
                     Jz: jnp.ndarray, cfg: Any,
                     gauge_penalty: float | None = None) -> jnp.ndarray:
    """Component-wise compatibility wrapper around :func:`magnetostatic_solve`."""
    return magnetostatic_solve(nu, (Mx, My, Mz), (Jx, Jy, Jz), cfg,
                               gauge_penalty)


def flux_density(vector_potential_field, cfg: Any):
    """Return the three components of ``B = curl(A)``."""
    return _split_vector(_curl(_stack_vector(vector_potential_field), cfg))


def maxwell_relative_residual(nu: jnp.ndarray, magnetization, current_density,
                              vector_potential_field, cfg: Any,
                              gauge_penalty: float | None = None) -> jnp.ndarray:
    source = (_stack_vector(current_density)
              + curl_magnetization(magnetization, cfg))
    operator, _, boundary, _ = _system(jnp.asarray(nu), cfg, gauge_penalty)
    rhs = jnp.where(boundary[..., None], 0.0, source)
    return relative_residual(operator, _stack_vector(vector_potential_field), rhs)


def coulomb_gauge_residual(vector_potential_field, cfg: Any) -> jnp.ndarray:
    """Dimensionless ``h ||div(A)|| / ||A||`` Coulomb-gauge diagnostic."""
    A = _stack_vector(vector_potential_field)
    h = min(_spacing(cfg))
    tiny = jnp.finfo(A.dtype).tiny
    return h * jnp.linalg.norm(_divergence(A, cfg)) / jnp.maximum(
        jnp.linalg.norm(A), tiny)


def residuals(nu: jnp.ndarray, magnetization, current_density,
              vector_potential_field, cfg: Any,
              gauge_penalty: float | None = None):
    """Return ``(equation_residual, gauge_residual)``."""
    return (maxwell_relative_residual(nu, magnetization, current_density,
                                      vector_potential_field, cfg, gauge_penalty),
            coulomb_gauge_residual(vector_potential_field, cfg))
