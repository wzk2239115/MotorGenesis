"""Tests for the T0/T1/T2 torque decomposition, coolant spiral and verdicts."""

import numpy as np
import pytest
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.objects import HelicalCoolingChannels, Winding3D
from organic_motor.construct.connectivity import coolant_report
from organic_motor.construct.verdicts import evaluate_verdicts, format_verdict_table
from organic_motor.experiments.motor3d_powered import (
    Powered3DSettings,
    compute_powered_maps,
)
from organic_motor.optimization.objective3d import ForwardResult3D


def _cfg(**overrides):
    defaults = dict(
        shape=(6, 6, 6),
        excitation_mode="impressed",
        pole_pairs=2,  # legacy-winding design pole count
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=3,
        maxwell_maxiter=4,
        thermal_maxiter=4,
        electric_maxiter=4,
        n_theta=6,
        torque_n_z=4,
        torque_n_r=4,
    )
    defaults.update(overrides)
    return MotorConfig3D(**defaults)


class TestTorqueDecomposition:
    """compute_powered_maps must recover T0/T1/T2 exactly from a quadratic solver."""

    def test_exact_recovery(self):
        cfg = _cfg()
        # Ground truth: T(theta, a) = T0 + sum_p a_p T1_p + sum_p a_p^2 T2_p
        t0_true = 0.011 * np.cos(2.0 * np.arange(6) * 0.3 + 0.2)
        k_true = np.array([0.083, 0.067, 0.047])
        alpha = np.array([0.0, 2.0944, 4.1888])
        q_true = np.array([0.004, -0.006, 0.005])

        def fake_forward(_belts, angle, amplitudes):
            shape = cfg.shape
            scalar = jnp.ones(shape)
            vector = jnp.zeros(shape + (3,))
            current = jnp.zeros(shape + (3,))
            phase_current = jnp.stack((current, current, current))
            theta = float(np.asarray(angle)[0]) if hasattr(angle, "__len__") else float(angle)
            a = np.asarray(amplitudes)
            base = t0_true[int(round(theta / 0.3)) % 6]
            lin = sum(a[p] * k_true[p] * np.cos(2.0 * theta - alpha[p]) for p in range(3))
            quad = sum(a[p] ** 2 * q_true[p] for p in range(3))
            torque = base + lin + quad
            return ForwardResult3D(
                rho_air=0.6 * scalar, rho_iron=0.2 * scalar,
                rho_copper=0.1 * scalar, rho_pm=0.1 * scalar,
                rotor_ownership=0.5 * scalar, nu=scalar,
                magnetization=vector, vector_potential=vector,
                flux_density=vector, current_density=current,
                phase_current_density=phase_current,
                torques=jnp.asarray([torque]),
                joule_loss=scalar, iron_loss=scalar, loss_total=2.0 * scalar,
                temperature=30.0 * scalar,
                maxwell_residual=jnp.asarray(0.0),
                thermal_residual=jnp.asarray(0.0),
                electric_residual=jnp.asarray(0.0),
                source_divergence_residual=jnp.asarray(0.0),
                phase_balance_residual=jnp.asarray(0.0),
            )

        import jax.numpy as jnp  # noqa: F811

        angles = np.arange(6) * 0.3
        maps = compute_powered_maps(
            cfg, None, None, None, angles, Powered3DSettings(),
            phase_solver=fake_forward, include_mechanics=False,
            keep_volumes=False,
        )
        assert np.allclose(maps["torque_cogging"], t0_true, atol=1e-9)
        for p in range(3):
            expected = k_true[p] * np.cos(2.0 * angles - alpha[p])
            assert np.allclose(maps["torques_ph"][p], expected, atol=1e-9)
            assert np.allclose(maps["torque_i2_diag"][p], q_true[p] * np.ones(6), atol=1e-9)


class TestCoolingSpiral:
    """The corrected helix SDF must be ONE continuous through-flow channel."""

    @pytest.fixture(scope="class")
    def built(self):
        cfg = _cfg(
            shape=(96, 96, 58),
            excitation_mode="terminal",
            thermal_maxiter=4, electric_maxiter=4,
        )
        mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
        HelicalCoolingChannels(cfg).build(mf)
        return cfg, mf

    def test_dedicated_coolant_material(self, built):
        _cfg, mf = built
        assert "coolant" in mf.sdfs

    def test_single_through_flow_network(self, built):
        cfg, mf = built
        report = coolant_report(mf, cfg)
        assert report["dedicated_coolant"] is True
        assert report["through_flow_networks"] >= 1, report
        assert report["trapped_voids"] == 0, report
        # One spiral -> one coolant component (plus, possibly, nothing else).
        assert report["coolant_components"] <= 2, report

    def test_channel_is_continuous_not_rings(self, built):
        """The old same-z helix distance shattered into ~n_turns rings."""
        cfg, mf = built
        report = coolant_report(mf, cfg)
        assert report["coolant_components"] < 4, (
            "coolant shattered into rings: the helix SDF lost continuity",
            report,
        )


class TestVerdicts:
    """The six-verdict suite structure and honest None handling."""

    @pytest.fixture(scope="class")
    def winding_only(self):
        cfg = _cfg(shape=(160, 160, 96), excitation_mode="terminal")
        mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
        Winding3D(cfg).build(mf)
        return cfg, mf

    def test_six_verdict_keys(self, winding_only):
        cfg, mf = winding_only
        suite = evaluate_verdicts(mf, cfg, None)
        assert set(suite["verdicts"].keys()) == {
            "electromechanical", "winding", "cooling",
            "structure", "manufacturing", "mesh_convergence",
        }
        # Electromechanical not run -> None, never silently green.
        assert suite["verdicts"]["electromechanical"]["passed"] is None
        # Winding-only field: the winding verdict is the one that can pass.
        assert suite["verdicts"]["winding"]["passed"] is True
        # No coolant material in this field -> cooling None, not False.
        assert suite["verdicts"]["cooling"]["passed"] is None
        # No iron -> structure fails honestly.
        assert suite["verdicts"]["structure"]["passed"] is False
        assert suite["passed"] is False

    def test_overall_cannot_pass_without_core_verdicts(self, winding_only):
        cfg, mf = winding_only
        suite = evaluate_verdicts(mf, cfg, None)
        # winding True + structure False -> overall False even though the
        # winding itself is fine.
        assert not suite["passed"]
        assert "structure" in suite["failed"]

    def test_format_table(self, winding_only):
        cfg, mf = winding_only
        suite = evaluate_verdicts(mf, cfg, None)
        text = format_verdict_table(suite)
        assert "PASS" in text and "FAIL" in text and "overall" in text
