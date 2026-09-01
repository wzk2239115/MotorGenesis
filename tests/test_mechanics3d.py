"""Native three-dimensional solid-mechanics tests."""

import numpy as np

from organic_motor.physics.mechanics3d import (
    centrifugal_body_force,
    small_strain,
    solve_linear_elasticity,
)


def test_affine_displacement_has_expected_three_dimensional_strain():
    shape = (4, 3, 5)
    x, y, z = np.meshgrid(
        np.arange(shape[0]) * 0.2,
        np.arange(shape[1]) * 0.3,
        np.arange(shape[2]) * 0.4,
        indexing="ij",
    )
    displacement = np.stack((2.0 * x, -y, 0.5 * z), axis=-1)
    strain = np.asarray(small_strain(displacement, (0.2, 0.3, 0.4)))

    assert strain.shape == shape + (3, 3)
    assert np.allclose(strain[..., 0, 0], 2.0, atol=2e-5)
    assert np.allclose(strain[..., 1, 1], -1.0, atol=2e-5)
    assert np.allclose(strain[..., 2, 2], 0.5, atol=2e-5)
    off_diagonal = strain.copy()
    for axis in range(3):
        off_diagonal[..., axis, axis] = 0.0
    assert np.allclose(off_diagonal, 0.0, atol=2e-5)


def test_centrifugal_force_and_small_elastic_solve_are_finite():
    shape = (3, 3, 3)
    x, y, z = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]),
        np.linspace(-0.5, 0.5, shape[1]),
        np.linspace(-0.5, 0.5, shape[2]),
        indexing="ij",
    )
    coordinates = np.stack((x, y, z), axis=-1)
    centrifugal = centrifugal_body_force(
        np.ones(shape), coordinates, angular_velocity=2.0
    )
    assert np.all(np.isfinite(np.asarray(centrifugal)))
    assert np.all(np.asarray(centrifugal)[..., 2] == 0.0)

    fixed = np.zeros(shape, dtype=bool)
    fixed[0, :, :] = True
    force = np.zeros(shape + (3,))
    force[-1, :, :, 0] = 1e-2
    result = solve_linear_elasticity(
        np.full(shape, 10.0),
        np.full(shape, 0.25),
        fixed,
        spacing=0.5,
        body_force=force,
        maxiter=100,
        tol=2e-5,
    )

    assert result.displacement.shape == shape + (3,)
    assert result.strain.shape == result.stress.shape == shape + (3, 3)
    assert result.von_mises.shape == shape
    assert np.all(np.isfinite(np.asarray(result.displacement)))
    assert np.allclose(np.asarray(result.displacement)[0], 0.0, atol=2e-5)
    assert float(result.relative_residual) < 2e-2
