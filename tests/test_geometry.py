"""Geometry scaffolding tests: masks, sampling, rotation."""

from __future__ import annotations

import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import (
    bilinear_sample,
    domain_masks,
    meshgrid,
    rotate_rotor,
    torque_circle,
)


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig(N=64)


def test_grid_bounds(cfg):
    X, Y, R = meshgrid(cfg)
    assert X.shape == (cfg.N, cfg.N)
    assert float(X.min()) == pytest.approx(-cfg.L)
    assert float(X.max()) == pytest.approx(cfg.L)
    assert float(R.max()) == pytest.approx(cfg.L * np.sqrt(2), rel=1e-3)


def test_domain_masks_partition(cfg):
    m = domain_masks(cfg)
    for name in ("shaft", "airgap", "design", "outer"):
        assert m[name].dtype == np.bool_
    # shaft, airgap, design, outer are disjoint and cover the box
    exact = (m["shaft"].astype(int) + m["airgap"].astype(int)
             + m["design"].astype(int) + m["outer"].astype(int))
    assert np.allclose(np.asarray(exact), 1.0)


def test_boundary_is_square_edges(cfg):
    b = np.asarray(domain_masks(cfg)["boundary"])
    assert np.all(b[0, :]) and np.all(b[-1, :])
    assert np.all(b[:, 0]) and np.all(b[:, -1])
    assert b[1:-1, 1:-1].sum() == 0


def test_bilinear_sample_exact_at_nodes(cfg):
    X, Y, _ = meshgrid(cfg)
    field = X * Y * 0.0 + X  # linear field
    s = bilinear_sample(field, X, Y, cfg)
    assert np.allclose(np.asarray(s), np.asarray(X), atol=1e-5)


def test_bilinear_sample_clamps_outside(cfg):
    X, Y, _ = meshgrid(cfg)
    field = X * 0.0 + 1.0
    s = bilinear_sample(field, np.array([1e6]), np.array([0.0]), cfg)
    assert float(s[0]) == pytest.approx(1.0)


def test_torque_circle_on_airgap(cfg):
    x, y, nx, ny, dl, th = torque_circle(cfg)
    r = np.hypot(np.asarray(x), np.asarray(y))
    assert np.allclose(r, cfg.R_torque)
    assert cfg.R_shaft < cfg.R_torque < cfg.R_gap


def test_rotate_rotor_leaves_outer_stator(cfg):
    X, Y, _ = meshgrid(cfg)
    field = X + Y
    out = rotate_rotor(field, 0.5, cfg)
    stator = domain_masks(cfg)["stator"]
    # outside the rotating rotor region the field is untouched
    assert np.allclose(np.asarray(out)[np.asarray(stator)],
                       np.asarray(field)[np.asarray(stator)], atol=1e-4)