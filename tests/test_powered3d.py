"""Fast integration tests for the quasi-static powered 3-D workflow."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.experiments.motor3d_powered import (
    Powered3DSettings,
    load_design3d,
    periodic_interpolate,
    run_powered3d,
)
from organic_motor.optimization.objective3d import ForwardResult3D


def test_periodic_interpolation_wraps_scalar_and_volume_maps():
    angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    values = np.array([0.0, 1.0, 0.0, -1.0])
    assert np.isclose(periodic_interpolate(values, angles, 2.0 * np.pi, 2.0 * np.pi), 0.0)
    assert np.isclose(periodic_interpolate(values, angles, 1.75 * np.pi, 2.0 * np.pi), -0.5)

    volume = values[:, None, None, None] * np.ones((4, 2, 3, 4))
    interpolated = periodic_interpolate(
        volume, angles, 0.25 * np.pi, 2.0 * np.pi
    )
    assert interpolated.shape == (2, 3, 4)
    assert np.allclose(interpolated, 0.5)


def test_design_loader_supports_reference_and_final_design_npz(tmp_path: Path):
    cfg = MotorConfig3D(shape=(5, 5, 5), filt_radius=0.0)
    logits, rotor, magnetization, source = load_design3d(cfg, None)
    assert source == "reference_design3d"
    path = tmp_path / "final_design3d.npz"
    np.savez(
        path,
        logits=np.asarray(logits),
        rotor_logits=np.asarray(rotor),
        magnetization_raw=np.asarray(magnetization),
    )
    loaded = load_design3d(cfg, path)
    assert loaded[0].shape == (4,) + cfg.shape
    assert loaded[1].shape == cfg.shape
    assert loaded[2].shape == (3,) + cfg.shape


def test_powered_workflow_preserves_native_xyz_fields():
    cfg = MotorConfig3D(shape=(5, 5, 5), filt_radius=0.0, projection_beta=0.0)
    logits, rotor_logits, magnetization, _ = load_design3d(cfg, None)

    def fake_forward(
        _cfg, _logits, _rotor_logits, _magnetization, angles, _temperature
    ):
        shape = _cfg.shape
        angle = float(angles[0])
        scalar = jnp.ones(shape)
        vector = jnp.zeros(shape + (3,))
        vector = vector.at[..., 0].set(0.4 + 0.1 * np.cos(angle))
        current = jnp.zeros(shape + (3,))
        current = current.at[..., 2].set(2.0e4)
        phase_current = jnp.stack((current, -0.5 * current, -0.5 * current))
        rho_air = 0.6 * scalar
        rho_iron = 0.2 * scalar
        rho_copper = 0.1 * scalar
        rho_pm = 0.1 * scalar
        return ForwardResult3D(
            rho_air=rho_air,
            rho_iron=rho_iron,
            rho_copper=rho_copper,
            rho_pm=rho_pm,
            rotor_ownership=0.5 * scalar,
            nu=scalar,
            magnetization=vector,
            vector_potential=vector,
            flux_density=vector,
            current_density=current,
            phase_current_density=phase_current,
            torques=jnp.asarray([1.0 + 0.2 * np.cos(angle)]),
            joule_loss=2.0 * scalar,
            iron_loss=3.0 * scalar,
            loss_total=5.0 * scalar,
            temperature=30.0 * scalar,
            maxwell_residual=jnp.asarray(0.0),
            thermal_residual=jnp.asarray(0.0),
            electric_residual=jnp.asarray(0.0),
            source_divergence_residual=jnp.asarray(0.0),
            phase_balance_residual=jnp.asarray(0.0),
        )

    settings = Powered3DSettings(
        steps=3,
        dt=1.0e-6,
        mechanical_maxiter=8,
        mechanical_tol=None,
        eddy_loss_coefficient=0.0,
    )
    data, summary = run_powered3d(
        cfg,
        logits,
        rotor_logits,
        magnetization,
        [0.0, 0.5 * np.pi],
        settings,
        forward_solver=fake_forward,
    )

    assert summary["model"] == "quasi-static field-map transient"
    assert summary["full_time_domain_eddy_current"] is False
    assert data["flux_density_map_T"].shape == (2,) + cfg.shape + (3,)
    assert data["current_density_map_A_m2"].shape == (2,) + cfg.shape + (3,)
    assert data["temperature_map_C"].shape == (2,) + cfg.shape
    assert data["displacement_m"].shape == cfg.shape + (3,)
    assert data["von_mises_Pa"].shape == cfg.shape
    assert data["temperature_final"].shape == cfg.shape
    assert np.all(np.isfinite(data["angular_velocity_rad_s"]))
    assert np.all(np.isfinite(data["temperature_final"]))
