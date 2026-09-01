"""Density-filter tests (Helmholtz filter)."""

from __future__ import annotations

import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.topology.filters import (
    filtered_density_pair,
    gaussian_filter,
    helmholtz_filter,
    total_variation,
)


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig(N=48)


def test_filter_preserves_constant(cfg):
    c = np.full((cfg.N, cfg.N), 0.7)
    out = helmholtz_filter(c, cfg)
    assert np.allclose(np.asarray(out), 0.7, atol=1e-2)


def test_filter_smooths_delta(cfg):
    f = np.zeros((cfg.N, cfg.N))
    f[cfg.N // 2, cfg.N // 2] = 1.0
    out = np.asarray(helmholtz_filter(f, cfg))
    assert out.max() < 1.0            # peak reduced
    assert (out > 1e-3).sum() > 1     # support widened beyond the single cell


def test_filter_positive_and_bounded(cfg):
    f = np.random.RandomState(0).rand(cfg.N, cfg.N)
    out = np.asarray(helmholtz_filter(f, cfg))
    assert out.min() >= -1e-3
    assert out.max() <= 1.0 + 1e-2


def test_filtered_density_pair_nonnegative(cfg):
    ri = np.random.RandomState(1).rand(cfg.N, cfg.N) * 0.5
    rp = np.random.RandomState(2).rand(cfg.N, cfg.N) * 0.3
    ri_f, rp_f = filtered_density_pair(ri, rp, cfg)
    air = 1.0 - np.asarray(ri_f) - np.asarray(rp_f)
    assert air.min() > -1e-3


def test_gaussian_filter_runs(cfg):
    f = np.random.RandomState(3).rand(cfg.N, cfg.N)
    g = np.asarray(gaussian_filter(f, cfg))
    assert g.shape == f.shape
    assert np.all(np.isfinite(g))


def test_total_variation_constant_zero(cfg):
    c = np.full((cfg.N, cfg.N), 0.5)
    tv = total_variation(c, cfg)
    assert float(tv) < 1e-4