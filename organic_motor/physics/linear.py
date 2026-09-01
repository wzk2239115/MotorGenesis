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


def cg_fixed(A, b, x0, M_inv, n_iter: int):
    """Solve A x = b with preconditioned CG for a fixed number of iterations.

    ``A`` is a callable implementing the (symmetric positive-definite) matvec;
    ``M_inv`` is a callable applying the inverse preconditioner.
    """
    def body(carry, _):
        x, r, p, rho = carry
        Ap = A(p)
        pAp = jnp.sum(p * Ap)
        alpha = rho / (pAp + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv(r)
        rho_new = jnp.sum(r * z)
        beta = rho_new / (rho + 1e-12)
        p = z + beta * p
        return (x, r, p, rho_new), None

    r0 = b - A(x0)
    z0 = M_inv(r0)
    p0 = z0
    rho0 = jnp.sum(r0 * z0)
    (x, _r, _p, _rho), _ = jax.lax.scan(body, (x0, r0, p0, rho0),
                                         xs=None, length=n_iter)
    return x


def jacobi_preconditioner(diag: jnp.ndarray):
    """Return M_inv(x) = x / diag (diag clamped away from zero)."""
    inv = 1.0 / jnp.clip(diag, 1e-30, None)
    return lambda x: x * inv
