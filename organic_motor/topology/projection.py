"""Heaviside projection (optional sharpening of the density field).

A smooth Heaviside with projection threshold ``eta`` and sharpness ``beta``:

    rho_proj = tanh(beta*eta) + tanh(beta*(rho - eta))
             / (tanh(beta*eta) + tanh(beta*(1 - eta)))

For ``beta -> inf`` this pushes densities that are above ``eta`` toward 1 and
the rest toward 0, producing crisp material boundaries.  ``beta`` is typically
annealed upward over the course of optimisation (continuation scheme).
"""

from __future__ import annotations

import jax.numpy as jnp


def heaviside_projection(rho: jnp.ndarray, beta: float, eta: float = 0.5) -> jnp.ndarray:
    if beta <= 0.0:
        return rho
    a = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return (jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))) / a