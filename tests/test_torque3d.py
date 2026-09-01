"""Closed-surface Maxwell stress tests."""

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.physics.torque3d import maxwell_force_torque


def test_uniform_field_has_negligible_closed_surface_force_and_torque():
    cfg = MotorConfig3D(shape=(7, 7, 5), box_size=(0.06, 0.06, 0.04))
    bx = np.full(cfg.shape, 0.3)
    by = np.full(cfg.shape, -0.2)
    bz = np.full(cfg.shape, 0.1)
    force, torque = maxwell_force_torque(
        bx,
        by,
        bz,
        cfg,
        radius=0.015,
        z_min=-0.01,
        z_max=0.01,
        n_theta=32,
        n_z=4,
        n_r=5,
    )

    stress_area_scale = (
        (0.3**2 + 0.2**2 + 0.1**2)
        / cfg.mu0
        * 2.0
        * np.pi
        * 0.015
        * (0.02 + 0.015)
    )
    assert np.linalg.norm(np.asarray(force)) < 2e-4 * stress_area_scale
    assert np.linalg.norm(np.asarray(torque)) < (
        2e-4 * stress_area_scale * 0.02
    )
