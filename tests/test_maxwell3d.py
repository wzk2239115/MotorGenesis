"""Native-3D magnetostatic solver smoke and residual tests."""

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.physics.maxwell3d import (
    flux_density,
    magnetostatic_solve,
    maxwell_relative_residual,
)


def _cfg():
    cfg = MotorConfig3D(shape=(5, 5, 5), box_size=(0.04, 0.04, 0.04))
    cfg.maxwell_maxiter = 160
    cfg.maxwell_tol = 2e-5
    return cfg


def test_zero_source_has_zero_solution_and_field():
    cfg = _cfg()
    scalar = np.ones(cfg.shape)
    zero_vector = np.zeros(cfg.shape + (3,))
    potential = magnetostatic_solve(scalar, zero_vector, zero_vector, cfg)
    field = flux_density(potential, cfg)

    assert potential.shape == cfg.shape + (3,)
    assert np.allclose(np.asarray(potential), 0.0, atol=1e-6)
    for component in field:
        assert component.shape == cfg.shape
        assert np.allclose(np.asarray(component), 0.0, atol=1e-6)
    residual = maxwell_relative_residual(
        scalar, zero_vector, zero_vector, potential, cfg
    )
    assert float(residual) == 0.0


def test_simple_current_source_returns_finite_solution_and_residual():
    cfg = _cfg()
    reluctivity = np.ones(cfg.shape)
    magnetization = np.zeros(cfg.shape + (3,))
    current = np.zeros(cfg.shape + (3,))
    current[2, 2, 2, 2] = 1.0
    potential = magnetostatic_solve(
        reluctivity, magnetization, current, cfg
    )
    residual = maxwell_relative_residual(
        reluctivity, magnetization, current, potential, cfg
    )

    assert np.all(np.isfinite(np.asarray(potential)))
    assert np.linalg.norm(np.asarray(potential)) > 0.0
    assert np.isfinite(float(residual))
    assert float(residual) < 5e-3
