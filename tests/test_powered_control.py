"""B3/B4/B5/B6/D4 tests: PI controller, energy balance, R(T), power-off, stability.

Uses SYNTHETIC maps (no Maxwell solves) so tests run in seconds. The maps
represent an ideal sinusoidal PM motor: T1_p(theta) = Kt*cos(p*theta + s_p),
which produces positive mean torque with comm=0 and balanced currents.
"""

import pytest
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.experiments.motor3d_powered import (
    Powered3DSettings,
    run_powered_transient,
    _make_transient_scan,
)

GRID = (6, 6, 6)
NA = 12
FLUX = 0.002  # small so back-emf stays below the bus in tests


def _synthetic_maps(cfg, i_nom=10.0):
    """Build minimal synthetic maps matching the scan's contract.

    Torque maps use Kt = p*psi*i_nom so the map-based conversion
    T_em*omega is INSTANTANEOUSLY equal to sum(emf*i) for balanced
    currents -- required for exact RL energy balance.
    """
    shifts = np.array((0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0))
    map_angles = np.arange(NA) * (2.0 * np.pi / cfg.pole_pairs) / NA
    elec = cfg.pole_pairs * map_angles
    kt = cfg.pole_pairs * FLUX * i_nom
    torques_ph = kt * np.cos(elec[None, :] + shifts[:, None])       # (3, na)
    torque_cogging = 1e-4 * np.cos(6.0 * elec)                      # (na,)
    torque_i2 = np.zeros((3, NA))                                    # (3, na)

    shape = cfg.shape
    j_maps = np.zeros((3, NA) + shape + (3,), dtype=np.float32)
    b_map = np.zeros((NA,) + shape + (3,), dtype=np.float32)

    fractions = np.stack([
        np.full(shape, 0.55), np.full(shape, 0.3),
        np.full(shape, 0.1), np.full(shape, 0.05),
    ]).astype(np.float32)
    k_mix = np.tensordot(
        np.array((0.026, 50.0, 400.0, 10.0)), fractions, axes=(0, 0)
    ).astype(np.float32)
    c_mix = np.tensordot(
        np.array((1.2e3, 3.6e6, 3.45e6, 3.4e6)), fractions, axes=(0, 0)
    ).astype(np.float32)

    return {
        "map_angles": map_angles,
        "period": 2.0 * np.pi / cfg.pole_pairs,
        "torques_ph": torques_ph,
        "torque_static": np.zeros_like(torques_ph),
        "torque_cogging": torque_cogging,
        "torque_i2_diag": torque_i2,
        "j_maps_ph": j_maps,
        "b_map": b_map,
        "temperature_init": np.full(shape, 25.0, dtype=np.float32),
        "materials": {
            "fractions": fractions,
            "thermal_conductivity": k_mix,
            "volumetric_heat_capacity": c_mix,
        },
        "masks": {"boundary": np.zeros(shape, dtype=np.float32)},
        "nominal_current": np.full(3, i_nom),
        "mechanics": None,
    }


def _base_settings(**ov):
    d = dict(
        steps=5000, dt=2.0e-5,
        phase_voltage_peak=24.0,
        phase_resistance=0.1,
        phase_inductance=1.0e-3,
        flux_linkage=FLUX,
        load_torque=1.0e-3,
        load_viscous=2.0e-3,   # keeps the speed modest so the PI tracks
        rotor_inertia=1.0e-4,
        commutation_offset=0.0,
    )
    d.update(ov)
    return Powered3DSettings(**d)


def _park_iq(currents, angles, pole_pairs):
    """Torque-axis (electrical-cos) current from history."""
    shifts = np.array((0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0))
    elec = pole_pairs * angles[:, None] + shifts[None, :]
    return 2.0 / 3.0 * np.sum(currents * np.cos(elec), axis=1)


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig3D(shape=GRID)


class TestPICurrentControl:
    def test_iq_tracks_reference(self, cfg):
        """Closed-loop q-current must track i_q_ref after settling."""
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=10.0)
        d = run_powered_transient(maps, s, cfg, 0.0)
        iq = _park_iq(d["currents_A"], d["rotor_angle_rad"], cfg.pole_pairs)
        settle = iq[4000:]  # last 20 ms
        assert np.mean(settle) == pytest.approx(10.0, rel=0.10), (
            f"mean i_q={np.mean(settle):.2f} A, want 10 A +/-10%"
        )

    def test_negative_iq_reverses(self, cfg):
        """Negative i_q_ref must produce negative current and torque."""
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=-10.0)
        d = run_powered_transient(maps, s, cfg, 0.0)
        iq = _park_iq(d["currents_A"], d["rotor_angle_rad"], cfg.pole_pairs)
        assert np.mean(iq[4000:]) == pytest.approx(-10.0, rel=0.10)
        assert d["angular_velocity_rad_s"][-1] < 0

    def test_voltage_saturation_limits_current(self, cfg):
        """With a small voltage limit, steady current must be below reference."""
        maps_hi = _synthetic_maps(cfg)
        maps_lo = _synthetic_maps(cfg)
        s_hi = _base_settings(control_mode="current_control", i_q_ref_A=40.0,
                              voltage_limit_V=1.0e9, overemf_trip_steps=0)
        s_lo = _base_settings(control_mode="current_control", i_q_ref_A=40.0,
                              voltage_limit_V=1.0, overemf_trip_steps=0)
        run_powered_transient(maps_hi, s_hi, cfg, 0.0)  # compile
        iq_hi_all = []
        for maps_x, s_x in ((maps_hi, s_hi), (maps_lo, s_lo)):
            maps_x.pop("_scan", None)
            d = run_powered_transient(maps_x, s_x, cfg, 0.0)
            iq = _park_iq(d["currents_A"], d["rotor_angle_rad"], cfg.pole_pairs)
            iq_hi_all.append(np.mean(np.abs(iq[4000:])))
        iq_hi, iq_lo = iq_hi_all
        assert iq_lo < iq_hi, (
            f"saturated i_q={iq_lo:.2f} must be < unsaturated {iq_hi:.2f}"
        )
        # 1 V across R=0.1 Ω (back-emf near zero at stall-ish speed) bounds i.
        assert iq_lo < 15.0, f"saturated i_q={iq_lo:.2f} exceeds 1V/0.1Ω bound"

    def test_bus_voltage_limits_speed(self, cfg):
        """Total-voltage saturation: at speed the back-EMF bounds the
        operating point, p*omega*psi cannot exceed v_max by much."""
        maps = _synthetic_maps(cfg)
        s = _base_settings(
            control_mode="current_control", i_q_ref_A=10.0,
            voltage_limit_V=2.0, load_torque=0.0, load_viscous=5.0e-3,
            steps=8000, overemf_trip_steps=0,
        )
        d = run_powered_transient(maps, s, cfg, 0.0)
        w_end = abs(d["angular_velocity_rad_s"][-1])
        ceiling = s.voltage_limit_V / (cfg.pole_pairs * FLUX)
        assert w_end < 1.15 * ceiling, (
            f"final speed {w_end:.1f} rad/s exceeds emf ceiling "
            f"{ceiling:.1f} rad/s — inverter generated voltage from nothing"
        )

    def test_overemf_trip_opens_contactor(self, cfg):
        """Sustained voltage saturation must latch the contactor open."""
        maps = _synthetic_maps(cfg)
        # flux bumped so the emf ceiling is low and quickly saturated
        s = _base_settings(
            control_mode="current_control", i_q_ref_A=10.0,
            voltage_limit_V=0.5, load_torque=0.0, load_viscous=2.0e-3,
            steps=6000, overemf_trip_steps=200,
        )
        d = run_powered_transient(maps, s, cfg, 0.0)
        cur = d["currents_A"]
        # after the trip window (200 steps) + margin, currents are zero
        assert np.max(np.abs(cur[1000:])) < 1e-6

    def test_comm_must_be_zero_in_current_control(self, cfg):
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=10.0,
                           commutation_offset=1.0)
        with pytest.raises(ValueError, match="commutation_offset"):
            _make_transient_scan(maps, s, cfg)


class TestEnergyBalance:
    def test_rl_energy_balance(self, cfg):
        """E_elec = copper loss + converted mechanical + stored magnetic.

        Circuit identity: sum(v*i) = sum(i^2 R) + sum(emf*i) + d/dt(0.5 L sum(i^2)).
        With map/emf consistency (Kt = p*psi*i_nom) the back-EMF conversion
        equals the map-based mechanical output, so:
        E_elec = E_joule + E_mech_conv + 0.5*L*sum(i_end^2).
        """
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=8.0)
        d = run_powered_transient(maps, s, cfg, 0.0)
        dt = s.dt
        R, L = s.phase_resistance, s.phase_inductance

        i = d["currents_A"][1:]           # (steps, 3)
        e_elec = np.sum(d["electrical_power_W"]) * dt
        e_joule = np.sum(np.sum(i ** 2, axis=1) * R) * dt
        e_mech = np.sum(d["mechanical_power_W"]) * dt
        mag = 0.5 * L * float(np.sum(i[-1] ** 2))

        balance = e_joule + e_mech + mag
        assert abs(e_elec) > 1e-6, "no energy flowed"
        rel_err = abs(balance - e_elec) / abs(e_elec)
        assert rel_err < 0.05, (
            f"energy mismatch {rel_err*100:.2f}%: "
            f"E_elec={e_elec*1e3:.3f} mJ vs components={balance*1e3:.3f} mJ"
        )

    def test_mechanical_conversion_matches_rotor_state(self, cfg):
        """E_mech_conv must equal final KE + load work (Newton consistency)."""
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=8.0)
        d = run_powered_transient(maps, s, cfg, 0.0)
        dt = s.dt
        ke = 0.5 * s.rotor_inertia * d["angular_velocity_rad_s"][-1] ** 2
        e_mech = np.sum(d["mechanical_power_W"]) * dt
        # Load work from the omega history (constant + viscous load).
        omega = d["angular_velocity_rad_s"][1:]
        load = s.load_torque + s.load_viscous * omega
        load_work = np.sum(load * omega) * dt
        assert e_mech == pytest.approx(ke + load_work, rel=0.05), (
            f"conversion {e_mech*1e3:.3f} mJ vs KE+load {(ke+load_work)*1e3:.3f} mJ"
        )

    def test_power_outputs_present(self, cfg):
        maps = _synthetic_maps(cfg)
        s = _base_settings(steps=100)
        d = run_powered_transient(maps, s, cfg, 0.0)
        assert "electrical_power_W" in d
        assert "mechanical_power_W" in d
        assert d["electrical_power_W"].shape == (100,)
        assert d["mechanical_power_W"].shape == (100,)


class TestPowerOff:
    def test_currents_decay_after_cutoff(self, cfg):
        """After physical disconnection, currents must decay toward zero."""
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=10.0,
                           power_off_at_s=0.05)
        d = run_powered_transient(maps, s, cfg, 0.0)
        currents = d["currents_A"]
        before = np.max(np.abs(currents[2400:2500]))   # t≈48-50 ms, powered
        after = np.max(np.abs(currents[4800:]))        # t≥96 ms, disconnected
        assert before > 5.0, f"powered current too small: {before:.2f} A"
        assert after < 0.5, (
            f"currents did not decay after power-off: {after:.3f} A"
        )

    def test_electrical_power_zero_after_cutoff(self, cfg):
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=10.0,
                           power_off_at_s=0.05)
        d = run_powered_transient(maps, s, cfg, 0.0)
        assert np.max(np.abs(d["electrical_power_W"][2600:])) < 1e-9


class TestScenariosB6:
    def test_zero_voltage_no_motion(self, cfg):
        maps = _synthetic_maps(cfg)
        s = _base_settings(phase_voltage_peak=0.0)
        d = run_powered_transient(maps, s, cfg, 0.0)
        # Zero voltage: near-zero current (tiny cogging drift only).
        assert np.max(np.abs(d["currents_A"])) < 1e-4
        assert abs(d["angular_velocity_rad_s"][-1]) < 1e-3

    def test_reversed_commutation_spins_backward(self, cfg):
        maps_a = _synthetic_maps(cfg)
        maps_b = _synthetic_maps(cfg)
        s_fwd = _base_settings(commutation_offset=0.0, load_torque=0.0)
        s_rev = _base_settings(commutation_offset=np.pi, load_torque=0.0)
        d_fwd = run_powered_transient(maps_a, s_fwd, cfg, 0.0)
        d_rev = run_powered_transient(maps_b, s_rev, cfg, 0.0)
        assert d_fwd["angular_velocity_rad_s"][-1] > 1.0
        assert d_rev["angular_velocity_rad_s"][-1] < -1.0

    def test_higher_load_lower_final_speed(self, cfg):
        maps_lo = _synthetic_maps(cfg)
        maps_hi = _synthetic_maps(cfg)
        s_lo = _base_settings(load_torque=2.0e-3)
        s_hi = _base_settings(load_torque=8.0e-3)
        d_lo = run_powered_transient(maps_lo, s_lo, cfg, 0.0)
        d_hi = run_powered_transient(maps_hi, s_hi, cfg, 0.0)
        assert d_hi["angular_velocity_rad_s"][-1] < d_lo["angular_velocity_rad_s"][-1]

    def test_dt_halving_consistent(self, cfg):
        """Half the time step, double the steps: same endpoint within 3%."""
        maps_a = _synthetic_maps(cfg)
        maps_b = _synthetic_maps(cfg)
        s_a = _base_settings(steps=5000, dt=2.0e-5)
        s_b = _base_settings(steps=10000, dt=1.0e-5)
        d_a = run_powered_transient(maps_a, s_a, cfg, 0.0)
        d_b = run_powered_transient(maps_b, s_b, cfg, 0.0)
        w_a = d_a["angular_velocity_rad_s"][-1]
        w_b = d_b["angular_velocity_rad_s"][-1]
        assert w_b == pytest.approx(w_a, rel=0.03), (
            f"dt halving changed final speed: {w_a:.3f} vs {w_b:.3f} rad/s"
        )


class TestResistanceTempFeedback:
    def test_hot_winding_limits_current(self, cfg):
        """Hot winding (higher R) must draw less current than cold.

        Locked rotor (huge J) keeps back-emf at zero so the steady state
        is set purely by V/R; the PI is saturated (v=v_max) throughout.
        """
        maps_cold = _synthetic_maps(cfg)
        maps_hot = _synthetic_maps(cfg)
        maps_hot["temperature_init"] = np.full(cfg.shape, 200.0, dtype=np.float32)
        s = dict(control_mode="current_control", i_q_ref_A=50.0,
                 voltage_limit_V=1.0, rotor_inertia=1.0, steps=6000,
                 overemf_trip_steps=0)
        d_cold = run_powered_transient(maps_cold, _base_settings(**s), cfg, 0.0)
        d_hot = run_powered_transient(maps_hot, _base_settings(**s), cfg, 0.0)
        # Steady state: i ~ V/R; hot R is ~1.7x cold R at 200 vs 25 degC.
        i_cold = np.mean(np.abs(d_cold["currents_A"][3000:]))
        i_hot = np.mean(np.abs(d_hot["currents_A"][3000:]))
        assert i_hot < i_cold * 0.75, (
            f"hot current {i_hot:.2f} A should be well below cold {i_cold:.2f} A"
        )

    def test_no_feedback_when_alpha_zero(self, cfg):
        """alpha=0 AND pm_coeff=0 must make response temperature-invariant."""
        maps_cold = _synthetic_maps(cfg)
        maps_hot = _synthetic_maps(cfg)
        maps_hot["temperature_init"] = np.full(cfg.shape, 120.0, dtype=np.float32)
        s = dict(control_mode="current_control", i_q_ref_A=5.0,
                 resistance_temp_coeff=0.0, pm_temp_coeff=0.0)
        d_cold = run_powered_transient(maps_cold, _base_settings(**s), cfg, 0.0)
        d_hot = run_powered_transient(maps_hot, _base_settings(**s), cfg, 0.0)
        assert np.allclose(d_cold["currents_A"], d_hot["currents_A"], atol=1e-4)


class TestPMTemperatureFeedback:
    def test_hot_magnet_less_torque(self, cfg):
        """Hot PM (lower remanence) must produce less torque/speed."""
        maps_cold = _synthetic_maps(cfg)
        maps_hot = _synthetic_maps(cfg)
        maps_hot["temperature_init"] = np.full(cfg.shape, 120.0, dtype=np.float32)
        s = dict(control_mode="current_control", i_q_ref_A=8.0,
                 resistance_temp_coeff=0.0)  # isolate the PM effect
        d_cold = run_powered_transient(maps_cold, _base_settings(**s), cfg, 0.0)
        d_hot = run_powered_transient(maps_hot, _base_settings(**s), cfg, 0.0)
        assert d_hot["angular_velocity_rad_s"][-1] < \
            d_cold["angular_velocity_rad_s"][-1]

    def test_zero_pm_coeff_no_effect(self, cfg):
        maps_cold = _synthetic_maps(cfg)
        maps_hot = _synthetic_maps(cfg)
        maps_hot["temperature_init"] = np.full(cfg.shape, 120.0, dtype=np.float32)
        s = dict(control_mode="current_control", i_q_ref_A=8.0,
                 resistance_temp_coeff=0.0, pm_temp_coeff=0.0)
        d_cold = run_powered_transient(maps_cold, _base_settings(**s), cfg, 0.0)
        d_hot = run_powered_transient(maps_hot, _base_settings(**s), cfg, 0.0)
        assert np.allclose(d_cold["currents_A"], d_hot["currents_A"], atol=1e-4)
        assert d_cold["angular_velocity_rad_s"][-1] == pytest.approx(
            d_hot["angular_velocity_rad_s"][-1], rel=1e-4)


class TestThermalStabilityCheckD4:
    def test_unstable_dt_rejected(self, cfg):
        """dt above the explicit-Euler diffusion limit must raise."""
        maps = _synthetic_maps(cfg)
        s = _base_settings(steps=10, dt=100.0)  # absurd dt
        with pytest.raises(ValueError, match="unstable"):
            run_powered_transient(maps, s, cfg, 0.0)

    def test_normal_dt_accepted(self, cfg):
        maps = _synthetic_maps(cfg)
        s = _base_settings(steps=50)
        d = run_powered_transient(maps, s, cfg, 0.0)
        assert np.all(np.isfinite(d["max_temperature_C"]))
