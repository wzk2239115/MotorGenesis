"""Torque tests: Maxwell stress tensor and Lorentz (magnetisation-current)."""

from __future__ import annotations

import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import meshgrid
from organic_motor.physics.maxwell import flux_density, magnetostatic_solve
from organic_motor.physics.torque import lorentz_torque, maxwell_torque


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig(N=96, maxwell_maxiter=600)


def _blob(X, Y, cx, cy, s=0.008):
    return np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2)) / (2 * s ** 2))


def _solve(cfg, Mx, My):
    nu = np.full((cfg.N, cfg.N), cfg.nu_air)
    az = magnetostatic_solve(nu, Mx, My, np.zeros((cfg.N, cfg.N)), cfg)
    return flux_density(az, cfg)


# --- Maxwell stress tensor (used for air-gap / reference eval) ---

def test_uniform_field_zero_torque(cfg):
    Bx = np.full((cfg.N, cfg.N), 1.0)
    By = np.zeros((cfg.N, cfg.N))
    assert abs(float(maxwell_torque(Bx, By, cfg))) < 1e-3


def test_pure_radial_field_zero_torque(cfg):
    X, Y, _ = meshgrid(cfg)
    R = np.hypot(X, Y) + 1e-12
    tau = maxwell_torque(X / R, Y / R, cfg)
    assert abs(float(tau)) < 1e-3


def test_stress_torque_zero_for_external_source(cfg):
    # With all magnetisation outside the evaluation circle, the source-free
    # interior integrates to (numerically) zero torque.
    X, Y, _ = meshgrid(cfg)
    Mx = _blob(X, Y, 0.03, 0.03) * cfg.M_sat
    Bx, By = _solve(cfg, Mx, np.zeros_like(Mx))
    # evaluation circle R_torque=0.012 encloses no source
    tau = float(maxwell_torque(Bx, By, cfg, radius=0.012))
    assert abs(tau) < 1e-2


def test_torque_scales_quadratically_with_B(cfg):
    Bx = np.full((cfg.N, cfg.N), 0.5)
    By = np.zeros((cfg.N, cfg.N))
    t1 = float(maxwell_torque(Bx, By, cfg))
    t2 = float(maxwell_torque(2 * Bx, 2 * By, cfg))
    assert t2 == pytest.approx(4 * t1, rel=0.05)


# --- Lorentz (magnetisation-current) torque ---

def test_self_torque_vanishes(cfg):
    # PM confined to the rotor; a rigid rotor cannot torque itself.
    X, Y, _ = meshgrid(cfg)
    Mx = _blob(X, Y, 0.02, 0.0) * cfg.M_sat
    Bx, By = _solve(cfg, Mx, np.zeros_like(Mx))
    tau = float(lorentz_torque(Bx, By, Mx, np.zeros_like(Mx), cfg))
    assert abs(tau) < 1e-2


def test_rotor_stator_interaction_nonzero(cfg):
    # Rotor PM (magnetised +y) and stator PM (magnetised +x) interact.
    X, Y, _ = meshgrid(cfg)
    Mx = _blob(X, Y, 0.04, 0.0) * cfg.M_sat       # stator, +x magnetised
    My = _blob(X, Y, 0.02, 0.0) * cfg.M_sat       # rotor, +y magnetised
    Bx, By = _solve(cfg, Mx, My)
    tau = float(lorentz_torque(Bx, By, Mx, My, cfg))
    assert abs(tau) > 0.5


def test_lorentz_torque_finite_and_differentiable(cfg):
    import jax
    import jax.numpy as jnp
    X, Y, _ = meshgrid(cfg)
    Mx = jnp.asarray(_blob(X, Y, 0.04, 0.0) * cfg.M_sat)
    My = jnp.asarray(_blob(X, Y, 0.02, 0.0) * cfg.M_sat)

    def tau_of(Mxarr):
        Bx, By = _solve(cfg, Mxarr, My)
        return lorentz_torque(Bx, By, Mxarr, My, cfg)

    g = jax.grad(tau_of)(Mx)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 1e-6
