"""Grid, trilinear interpolation and rigid-rotation tests."""

import numpy as np

from organic_motor.geometry.grid3d import (
    Grid3D,
    rotate_vector_z,
    rotate_volume_z,
    trilinear_sample,
)


def test_meshgrid_uses_xyz_axis_order_and_varies_in_z():
    grid = Grid3D((4, 5, 6), (0.2, 0.3, 0.4), (-0.3, -0.6, -1.0))
    x, y, z = grid.meshgrid()

    assert x.shape == y.shape == z.shape == (4, 5, 6)
    assert np.all(np.diff(np.asarray(z)[0, 0, :]) > 0.0)
    assert not np.allclose(np.asarray(z)[:, :, 0], np.asarray(z)[:, :, -1])
    assert np.allclose(np.diff(np.asarray(x)[:, 0, 0]), 0.2)


def test_trilinear_sampling_is_exact_for_affine_field():
    grid = Grid3D((5, 5, 5), (0.5, 0.5, 0.5), (-1.0, -1.0, -1.0))
    x, y, z = grid.meshgrid()
    field = 1.0 + 2.0 * x - 3.0 * y + 0.5 * z
    xs = np.array([-0.75, 0.1, 0.8])
    ys = np.array([0.25, -0.4, 0.6])
    zs = np.array([-0.2, 0.35, 0.7])

    sampled = trilinear_sample(field, xs, ys, zs, grid)
    expected = 1.0 + 2.0 * xs - 3.0 * ys + 0.5 * zs
    assert np.allclose(np.asarray(sampled), expected, atol=2e-5)


def test_quarter_turn_rotates_locations_and_vector_components():
    grid = Grid3D((5, 5, 3), (0.5, 0.5, 0.5), (-1.0, -1.0, -0.5))
    x, y, z = grid.meshgrid()
    rotated_x = rotate_volume_z(x, np.pi / 2.0, grid)
    vx, vy, vz = rotate_vector_z(
        np.ones(grid.shape), np.zeros(grid.shape), np.zeros(grid.shape),
        np.pi / 2.0, grid,
    )

    assert np.allclose(np.asarray(rotated_x), np.asarray(y), atol=3e-5)
    assert np.allclose(np.asarray(vx), 0.0, atol=3e-5)
    assert np.allclose(np.asarray(vy), 1.0, atol=3e-5)
    assert np.allclose(np.asarray(vz), 0.0, atol=3e-5)
