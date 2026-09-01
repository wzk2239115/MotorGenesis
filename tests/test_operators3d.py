"""Analytic checks for native three-dimensional differential operators."""

import numpy as np

from organic_motor.geometry.grid3d import Grid3D
from organic_motor.physics.operators3d import curl3d, divergence3d, gradient3d


def test_gradient_of_affine_scalar_field():
    grid = Grid3D((5, 4, 6), (0.2, 0.3, 0.4), (-0.4, -0.3, -1.0))
    x, y, z = grid.meshgrid()
    gradient = gradient3d(2.0 * x - 3.0 * y + 4.0 * z, grid)

    for actual, expected in zip(gradient, (2.0, -3.0, 4.0)):
        assert actual.shape == grid.shape
        assert np.allclose(np.asarray(actual), expected, atol=2e-5)


def test_divergence_of_diagonal_linear_field():
    grid = Grid3D((4, 5, 6), (0.3, 0.2, 0.1), (0.0, 0.0, 0.0))
    x, y, z = grid.meshgrid()
    divergence = divergence3d(2.0 * x, -3.0 * y, 4.0 * z, grid)

    assert divergence.shape == grid.shape
    assert np.allclose(np.asarray(divergence), 3.0, atol=2e-5)


def test_curl_of_solid_body_rotation_field():
    grid = Grid3D((5, 5, 4), (0.25, 0.25, 0.4), (-0.5, -0.5, -0.6))
    x, y, z = grid.meshgrid()
    zeros = 0.0 * z
    curl = curl3d(-y, x, zeros, grid)

    assert np.allclose(np.asarray(curl[0]), 0.0, atol=2e-5)
    assert np.allclose(np.asarray(curl[1]), 0.0, atol=2e-5)
    assert np.allclose(np.asarray(curl[2]), 2.0, atol=2e-5)
