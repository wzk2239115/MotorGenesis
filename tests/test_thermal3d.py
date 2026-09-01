"""Native-3D steady thermal-model tests."""

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.physics.thermal3d import (
    solve_temperature,
    thermal_relative_residual,
)


def _cfg():
    cfg = MotorConfig3D(shape=(5, 5, 5), box_size=(0.04, 0.04, 0.04))
    cfg.thermal_maxiter = 160
    cfg.thermal_tol = 2e-5
    return cfg


def test_zero_heat_stays_at_ambient_in_full_3d_volume():
    cfg = _cfg()
    conductivity = np.ones(cfg.shape)
    temperature = solve_temperature(np.zeros(cfg.shape), conductivity, cfg)

    assert temperature.shape == cfg.shape
    assert np.allclose(
        np.asarray(temperature), cfg.ambient_temperature, atol=2e-4
    )


def test_interior_heat_creates_finite_hotspot_with_small_residual():
    cfg = _cfg()
    conductivity = np.full(cfg.shape, 2.0)
    heat = np.zeros(cfg.shape)
    heat[2, 2, 2] = 1.0e5
    temperature = solve_temperature(heat, conductivity, cfg)
    residual = thermal_relative_residual(
        temperature, heat, conductivity, cfg
    )

    values = np.asarray(temperature)
    assert np.all(np.isfinite(values))
    assert values[2, 2, 2] > cfg.ambient_temperature
    assert np.allclose(values[0, :, :], cfg.ambient_temperature, atol=2e-3)
    assert np.isfinite(float(residual))
    assert float(residual) < 5e-3
