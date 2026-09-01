"""Magnetostatic solver tests (correctness against a dense reference)."""

from __future__ import annotations

import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks
from organic_motor.physics.maxwell import (
    diffusion,
    flux_density,
    magnetostatic_solve,
)


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig(N=32, maxwell_maxiter=600)


def _scipy_reference(nu, src, cfg):
    """Assemble -div(nu grad) with Dirichlet BC and solve with scipy."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    N = cfg.N
    h = cfg.h
    nu = np.asarray(nu)
    src = np.asarray(src)
    boundary = np.asarray(domain_masks(cfg)["boundary"]).astype(bool).ravel()
    n = N * N
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            if boundary[idx]:
                rows.append(idx); cols.append(idx); vals.append(1.0)
                b[idx] = 0.0
                continue
            nl = (nu[i, j] + nu[i - 1, j]) / 2 if i > 0 else 0.0
            nr = (nu[i, j] + nu[i + 1, j]) / 2 if i < N - 1 else 0.0
            nd = (nu[i, j] + nu[i, j - 1]) / 2 if j > 0 else 0.0
            nu_ = (nu[i, j] + nu[i, j + 1]) / 2 if j < N - 1 else 0.0
            rows.append(idx); cols.append(idx)
            vals.append((nl + nr + nd + nu_) / h ** 2)
            if i > 0:
                rows.append(idx); cols.append(idx - N); vals.append(-nl / h ** 2)
            if i < N - 1:
                rows.append(idx); cols.append(idx + N); vals.append(-nr / h ** 2)
            if j > 0:
                rows.append(idx); cols.append(idx - 1); vals.append(-nd / h ** 2)
            if j < N - 1:
                rows.append(idx); cols.append(idx + 1); vals.append(-nu_ / h ** 2)
            b[idx] = src[i, j]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return spsolve(A, b).reshape(N, N)


def test_matches_dense_reference(cfg):
    N = cfg.N
    nu = np.full((N, N), cfg.nu_air)
    nu[N // 3:2 * N // 3, N // 3:2 * N // 3] = cfg.nu_iron
    X = np.linspace(-cfg.L, cfg.L, N)
    Xv, Yv = np.meshgrid(X, X, indexing="ij")
    src = np.exp(-(Xv ** 2 + Yv ** 2) / (2 * 0.02 ** 2))
    Mx = np.zeros((N, N)); My = np.zeros((N, N))

    az = magnetostatic_solve(nu, Mx, My, src, cfg)
    ref = _scipy_reference(nu, src, cfg)

    assert np.allclose(np.asarray(az), ref, atol=1e-3 * np.abs(ref).max())


def test_dirichlet_boundary(cfg):
    N = cfg.N
    nu = np.full((N, N), cfg.nu_air)
    src = np.random.RandomState(0).rand(N, N)
    az = magnetostatic_solve(nu, np.zeros((N, N)), np.zeros((N, N)), src, cfg)
    a = np.asarray(az)
    assert np.allclose(a[0, :], 0.0, atol=1e-8)
    assert np.allclose(a[-1, :], 0.0, atol=1e-8)
    assert np.allclose(a[:, 0], 0.0, atol=1e-8)
    assert np.allclose(a[:, -1], 0.0, atol=1e-8)


def test_symmetric_source_gives_symmetric_solution(cfg):
    N = cfg.N
    nu = np.full((N, N), cfg.nu_air)
    X = np.linspace(-cfg.L, cfg.L, N)
    Xv, Yv = np.meshgrid(X, X, indexing="ij")
    src = np.exp(-(Xv ** 2 + Yv ** 2) / (2 * 0.02 ** 2))  # radially symmetric
    az = magnetostatic_solve(nu, np.zeros((N, N)), np.zeros((N, N)), src, cfg)
    a = np.asarray(az)
    # even in x and y
    assert np.allclose(a, a[::-1, :], atol=1e-4 * np.abs(a).max())
    assert np.allclose(a, a[:, ::-1], atol=1e-4 * np.abs(a).max())


def test_diffusion_linearity(cfg):
    nu = np.ones((cfg.N, cfg.N))
    u = np.random.RandomState(1).rand(cfg.N, cfg.N)
    v = np.random.RandomState(2).rand(cfg.N, cfg.N)
    lhs = np.asarray(diffusion(nu, 2 * u + 3 * v, cfg))
    rhs = 2 * np.asarray(diffusion(nu, u, cfg)) + 3 * np.asarray(diffusion(nu, v, cfg))
    assert np.allclose(lhs, rhs, rtol=1e-3, atol=1e-2)


def test_flux_density_curl_zero_for_constant(cfg):
    az = np.full((cfg.N, cfg.N), 1.23)
    Bx, By = flux_density(az, cfg)
    assert np.allclose(np.asarray(Bx), 0.0, atol=1e-6)
    assert np.allclose(np.asarray(By), 0.0, atol=1e-6)