"""Steady native-3D electric-conduction tests."""

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.grid3d import meshgrid3d
from organic_motor.geometry.domain3d import domain_masks3d
from organic_motor.optimization.objective3d import (
    three_phase_terminal_conduction3d,
)
from organic_motor.physics.electric3d import (
    current_density,
    electric_relative_residual,
    joule_loss,
    solve_potential,
)


def _terminal_problem():
    cfg = MotorConfig3D(shape=(6, 4, 3), box_size=(1.0, 0.6, 0.4))
    conductivity = np.full(cfg.shape, 2.0)
    terminal_mask = np.zeros(cfg.shape, dtype=bool)
    terminal_mask[0, :, :] = True
    terminal_mask[-1, :, :] = True
    terminal_values = np.zeros(cfg.shape)
    terminal_values[-1, :, :] = 3.0
    return cfg, conductivity, terminal_mask, terminal_values


def test_terminal_solve_produces_linear_voltage_and_small_residual():
    cfg, conductivity, mask, values = _terminal_problem()
    potential = solve_potential(conductivity, mask, values, cfg)
    x, _, _ = meshgrid3d(cfg)
    expected = 3.0 * (x - x.min()) / (x.max() - x.min())

    assert potential.shape == cfg.shape
    assert np.allclose(np.asarray(potential)[mask], values[mask], atol=2e-5)
    assert np.allclose(np.asarray(potential), np.asarray(expected), atol=2e-3)
    residual = electric_relative_residual(
        conductivity, mask, values, potential, cfg
    )
    assert np.isfinite(float(residual))
    assert float(residual) < 2e-3


def test_current_and_joule_loss_match_linear_field():
    cfg, conductivity, _, _ = _terminal_problem()
    x, _, _ = meshgrid3d(cfg)
    potential = 3.0 * (x - x.min()) / (x.max() - x.min())
    jx, jy, jz = current_density(potential, conductivity, cfg)
    heat = joule_loss(potential, conductivity, cfg)

    assert np.allclose(np.asarray(jx), -6.0, atol=2e-4)
    assert np.allclose(np.asarray(jy), 0.0, atol=2e-4)
    assert np.allclose(np.asarray(jz), 0.0, atol=2e-4)
    assert np.allclose(np.asarray(heat), 18.0, atol=2e-3)


def test_three_phase_terminal_drive_uses_native_z_conduction():
    cfg = MotorConfig3D(
        shape=(14, 14, 7),
        filt_radius=0.0,
        electric_maxiter=100,
        electric_tol=1e-7,
    )
    copper = domain_masks3d(cfg)["winding"].astype(float)
    total, phases, heat, residual, balance = three_phase_terminal_conduction3d(
        copper, 0.2, cfg
    )

    assert total.shape == cfg.shape + (3,)
    assert phases.shape == (3,) + cfg.shape + (3,)
    assert heat.shape == cfg.shape
    assert np.ptp(np.asarray(phases[..., 2]), axis=3).max() > 0.0
    assert float(residual) < 1e-6
    assert float(balance) < 1e-5
