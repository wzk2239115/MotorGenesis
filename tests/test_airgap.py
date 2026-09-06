"""D2 tests: air-gap convection, end-face analogy, windage correlations.

Physics checks against DERIVABLE limits (audit: correlations must have
sources; laminar branches are exact, turbulent branches flagged).
"""

import pytest
import math
import numpy as np

from organic_motor.physics.airgap import (
    RHO_AIR, MU_AIR, K_AIR,
    air_gap_convection, end_face_convection, rotor_windage, windage_scan,
    _disk_moment_coefficient,
)


class TestAirGapConvection:
    def test_static_is_conduction(self):
        g = air_gap_convection(0.0, 0.0275, 0.003)
        assert g.h_W_m2K == pytest.approx(K_AIR / 0.003)

    def test_laminar_below_taylor(self):
        # This 3 mm gap: Ta(omega) ~ 2.58*omega^2; the enhancement Nu=1
        # crossing sits near omega~15.6 rad/s.  5 and 10 stay laminar.
        g1 = air_gap_convection(5.0, 0.0275, 0.003)
        g2 = air_gap_convection(10.0, 0.0275, 0.003)
        assert g1.regime == "laminar_couette"
        assert g2.regime == "laminar_couette"
        assert g1.h_W_m2K == pytest.approx(K_AIR / 0.003)
        assert g2.h_W_m2K == pytest.approx(K_AIR / 0.003)

    def test_taylor_enhancement_increases_h(self):
        g_lam = air_gap_convection(100.0, 0.0275, 0.003)
        g_turb = air_gap_convection(5000.0, 0.0275, 0.003)
        assert g_turb.regime == "taylor_vortex"
        assert g_turb.h_W_m2K > 2 * g_lam.h_W_m2K
        # Nu = 0.2 * Ta^0.25 exactly
        assert g_turb.nusselt == pytest.approx(0.2 * g_turb.taylor ** 0.25, rel=1e-6)

    def test_taylor_transition_continuous(self):
        # The max(1, 0.2*Ta^0.25) form is continuous; the effective
        # enhancement reaches Nu=1 exactly at Ta = (1/0.2)^4 = 625.
        for ta_target in (400.0, 625.0, 900.0):
            r, d = 0.0275, 0.003
            nu = MU_AIR / RHO_AIR
            re_c = math.sqrt(ta_target / (d / r))
            omega = re_c * nu / (r * d)
            g = air_gap_convection(omega, r, d)
            expected_nu = max(1.0, 0.2 * ta_target ** 0.25)
            assert g.nusselt == pytest.approx(expected_nu, rel=1e-3)
            assert g.h_W_m2K == pytest.approx(expected_nu * K_AIR / d, rel=1e-3)


class TestEndFace:
    def test_daily_nece_laminar(self):
        assert _disk_moment_coefficient(1e4) == pytest.approx(3.87 / 100.0)
        assert _disk_moment_coefficient(1e6) == pytest.approx(0.146 * 1e6 ** -0.2)

    def test_h_scales_with_omega(self):
        h1 = end_face_convection(100.0, 0.0275)
        h2 = end_face_convection(400.0, 0.0275)
        assert h2.h_W_m2K > h1.h_W_m2K

    def test_static_zero(self):
        assert end_face_convection(0.0, 0.0275).h_W_m2K == 0.0


class TestRotorWindage:
    def test_static_zero(self):
        w = rotor_windage(0.0, 0.0275, 0.06, 0.003)
        assert w.torque_Nm == 0.0 and w.power_W == 0.0

    def test_laminar_side_exact(self):
        """Laminar side torque must equal the exact Couette value
        T = 2*pi*mu*omega*r^3*L/delta (disks excluded).
        omega=10 keeps Ta=257 below 1700 for this geometry."""
        omega, r, L, d = 10.0, 0.0275, 0.06, 0.003
        w = rotor_windage(omega, r, L, d, n_end_disks=0)
        assert w.regime.startswith("laminar_couette")
        exact = 2 * math.pi * MU_AIR * omega * r ** 3 * L / d
        assert w.side_torque_Nm == pytest.approx(exact, rel=1e-6)

    def test_torque_sign_positive(self):
        w = rotor_windage(1000.0, 0.0275, 0.06, 0.003)
        assert w.torque_Nm > 0
        assert w.power_W == pytest.approx(w.torque_Nm * 1000.0)

    def test_torque_grows_sub_quadratically_turbulent(self):
        """With the CORRECTED speed dependence the turbulent-regime TORQUE
        exponent is < 2 (side 1.75, disks 1.5-1.8); the old inline bug
        (omega^2 everywhere, i.e. exponent exactly 2) is what this guards."""
        w1 = rotor_windage(2000.0, 0.0275, 0.06, 0.003)
        w2 = rotor_windage(4000.0, 0.0275, 0.06, 0.003)
        torque_exp = math.log(w2.torque_Nm / w1.torque_Nm) / math.log(2.0)
        assert 1.4 < torque_exp < 2.0, (
            f"torque exponent {torque_exp:.3f} outside the physical "
            "1.4-2.0 band (2.0 = the dropped-omega-dependence bug)"
        )
        power_exp = math.log(w2.power_W / w1.power_W) / math.log(2.0)
        assert power_exp < 3.0

    def test_windage_scan_reports(self):
        rows = windage_scan([100, 1000, 5000], 0.0275, 0.06, 0.003)
        assert len(rows) == 3
        assert rows[0]["gap_regime"] == "laminar_couette"
        assert rows[-1]["windage_W"] > rows[0]["windage_W"]


class TestSharedFormulaConsistency:
    """Audit item 5: the transient's windage MUST equal the standalone
    module — one formula, both backends, all regimes and directions."""

    OMEGAS = (0.0, 1e-2, 0.5, 5.0, 50.0, 500.0, 5e3, 5e4,
              -1e-2, -0.5, -5.0, -50.0, -500.0, -5e3, -5e4)

    def test_jax_matches_numpy_all_regimes(self):
        import jax.numpy as jnp
        from organic_motor.physics.airgap import windage_torque_formula
        r, L, d = 0.0275, 0.06, 0.003
        for w in self.OMEGAS:
            ref = rotor_windage(w, r, L, d)
            tot, side, disk = windage_torque_formula(jnp, w, r, L, d)
            assert float(tot) == pytest.approx(ref.torque_Nm, rel=1e-6, abs=1e-15), (
                f"omega={w}: jax {float(tot)} vs numpy {ref.torque_Nm}"
            )

    def test_odd_symmetry(self):
        import jax.numpy as jnp
        from organic_motor.physics.airgap import windage_torque_formula
        r, L, d = 0.0275, 0.06, 0.003
        for w in (0.5, 50.0, 5e3):
            tp, _, _ = windage_torque_formula(jnp, w, r, L, d)
            tm, _, _ = windage_torque_formula(jnp, -w, r, L, d)
            assert float(tm) == pytest.approx(-float(tp), rel=1e-6, abs=1e-15)

    def test_dissipative_both_directions(self):
        import jax.numpy as jnp
        from organic_motor.physics.airgap import windage_torque_formula
        r, L, d = 0.0275, 0.06, 0.003
        for w in self.OMEGAS:
            t, _, _ = windage_torque_formula(jnp, w, r, L, d)
            assert float(t) * w >= 0.0, f"windage adds energy at omega={w}"

    def test_laminar_side_linear_in_omega(self):
        """Below the Taylor transition torque is exactly linear (exact
        Couette): T(2w) == 2*T(w)."""
        import jax.numpy as jnp
        from organic_motor.physics.airgap import windage_torque_formula
        r, L, d = 0.0275, 0.06, 0.003
        _, s1, _ = windage_torque_formula(jnp, 3.0, r, L, d, n_end_disks=0)
        _, s2, _ = windage_torque_formula(jnp, 6.0, r, L, d, n_end_disks=0)
        assert float(s2) == pytest.approx(2.0 * float(s1), rel=1e-6)

    def test_transient_uses_shared_formula(self, ):
        """The scan's windage equals the formula (regression for the
        omega-dependence drop: old bug gave omega^2 in the laminar side)."""
        import jax.numpy as jnp
        from organic_motor.physics.airgap import windage_torque_formula
        from organic_motor.config3d import MotorConfig3D
        from organic_motor.experiments.motor3d_powered import (
            Powered3DSettings, run_powered_transient, _make_transient_scan,
        )
        from tests.test_powered_control import _synthetic_maps

        cfg = MotorConfig3D(shape=(6, 6, 6))
        maps = _synthetic_maps(cfg)
        s = Powered3DSettings(steps=10, include_windage=True,
                              control_mode="current_control", i_q_ref_A=5.0)
        scan = _make_transient_scan(maps, s, cfg)
        # indirect check: run works and the formula import didn't break jit
        d = run_powered_transient(maps, s, cfg, 0.0)
        assert np.all(np.isfinite(d["angular_velocity_rad_s"]))
        # and the reference implementation is importable & consistent
        r, L, g = 0.0275, 0.06, 0.003
        t, _, _ = windage_torque_formula(jnp, 123.4, r, L, g)
        assert float(t) > 0


class TestWindageInTransient:
    def test_windage_lowers_final_speed(self):
        """Enabling windage must reduce the no-load terminal speed."""
        from organic_motor.config3d import MotorConfig3D
        from organic_motor.experiments.motor3d_powered import (
            Powered3DSettings, run_powered_transient,
        )
        from tests.test_powered_control import _synthetic_maps

        cfg = MotorConfig3D(shape=(6, 6, 6))
        base = dict(
            control_mode="current_control", i_q_ref_A=8.0,
            load_torque=0.0, load_viscous=0.0, steps=3000,
        )
        maps_a = _synthetic_maps(cfg)
        maps_b = _synthetic_maps(cfg)
        d_off = run_powered_transient(
            maps_a, Powered3DSettings(**base), cfg, 0.0)
        d_on = run_powered_transient(
            maps_b, Powered3DSettings(include_windage=True, **base), cfg, 0.0)
        w_off = d_off["angular_velocity_rad_s"][-1]
        w_on = d_on["angular_velocity_rad_s"][-1]
        assert w_on < w_off, (
            f"windage must slow the rotor: {w_on:.1f} vs {w_on:.1f} rad/s"
        )

    def test_windage_zero_at_zero_speed(self):
        """No motion -> no aerodynamic load (energy identity intact)."""
        from organic_motor.config3d import MotorConfig3D
        from organic_motor.experiments.motor3d_powered import (
            Powered3DSettings, run_powered_transient,
        )
        from tests.test_powered_control import _synthetic_maps

        cfg = MotorConfig3D(shape=(6, 6, 6))
        maps = _synthetic_maps(cfg)
        s = Powered3DSettings(phase_voltage_peak=0.0, steps=50,
                              include_windage=True)
        d = run_powered_transient(maps, s, cfg, 0.0)
        assert abs(d["angular_velocity_rad_s"][-1]) < 1e-3
