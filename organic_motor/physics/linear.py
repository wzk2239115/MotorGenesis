"""Preconditioned conjugate-gradient solver (fixed iteration count).

A fixed-iteration, Jacobi-preconditioned CG implemented with ``lax.scan`` so it
unrolls/vectorises cleanly on GPU.  Differentiating *through* the iterations
gives exact gradients of the (approximately) solved linear system without an
extra adjoint solve; the memory cost is just the per-iteration carry (negligible
for our grid sizes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cg_fixed(A, b, x0, M_inv, n_iter: int, tol: float | None = None):
    """Solve A x = b with preconditioned CG for a fixed number of iterations.

    ``A`` is a callable implementing the (symmetric positive-definite) matvec;
    ``M_inv`` is a callable applying the inverse preconditioner.
    """
    bnorm2 = jnp.sum(b * b)
    tiny = jnp.finfo(b.dtype).tiny
    # Asking float32 for a residual far below its roundoff floor causes CG to
    # over-iterate, lose conjugacy and sometimes degrade an already good solve.
    # Float64 keeps the configured engineering tolerance unchanged.
    effective_tol = (0.0 if tol is None else
                     jnp.maximum(tol, 100.0 * jnp.finfo(b.dtype).eps))
    threshold2 = (effective_tol ** 2
                  * jnp.maximum(bnorm2, tiny))

    def body(carry, _):
        x, r, p, rho, active = carry
        Ap = A(p)
        pAp = jnp.sum(p * Ap)
        safe_pAp = jnp.where(jnp.abs(pAp) > tiny, pAp, 1.0)
        alpha = jnp.where(active, rho / safe_pAp, 0.0)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv(r)
        rho_new = jnp.sum(r * z)
        safe_rho = jnp.where(jnp.abs(rho) > tiny, rho, 1.0)
        beta = jnp.where(active, rho_new / safe_rho, 0.0)
        p_new = z + beta * p
        active_new = active & (jnp.sum(r * r) > threshold2)
        p = jnp.where(active_new, p_new, jnp.zeros_like(p_new))
        return (x, r, p, rho_new, active_new), None

    r0 = b - A(x0)
    z0 = M_inv(r0)
    p0 = z0
    rho0 = jnp.sum(r0 * z0)
    active0 = jnp.asarray(True) if tol is None else jnp.sum(r0 * r0) > threshold2
    (x, _r, _p, _rho, _active), _ = jax.lax.scan(
        body, (x0, r0, p0, rho0, active0), xs=None, length=n_iter)
    return x


def relative_residual(A, x: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Dimensionless ||b-Ax||2 / ||b||2 diagnostic."""
    tiny = jnp.finfo(jnp.result_type(x, b)).tiny
    return jnp.linalg.norm(b - A(x)) / jnp.maximum(jnp.linalg.norm(b), tiny)


def jacobi_preconditioner(diag: jnp.ndarray):
    """Return M_inv(x) = x / diag (diag clamped away from zero)."""
    inv = 1.0 / jnp.clip(diag, 1e-30, None)
    return lambda x: x * inv
