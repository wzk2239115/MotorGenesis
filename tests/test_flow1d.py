"""Tests for 1D flow network — physical correctness, not just non-empty."""

import math
import pytest

from organic_motor.physics.flow1d import (
    evaluate_channel,
    compare_channels,
    friction_factor,
    nusselt_number,
    FlowResult,
)


class TestFrictionFactor:
    def test_laminar(self):
        f = friction_factor(1000)
        assert abs(f - 64/1000) < 1e-10, f"laminar f={f}, expected {64/1000}"

    def test_turbulent(self):
        f = friction_factor(10000)
        expected = 0.316 / 10000**0.25
        assert abs(f - expected) / expected < 0.01

    def test_helical_increases_friction(self):
        f_straight = friction_factor(1000)
        f_helix = friction_factor(1000, dean=500)
        assert f_helix > f_straight, "helix must increase friction"


class TestNusselt:
    def test_laminar_constant(self):
        nu = nusselt_number(1000, 4.3)
        assert abs(nu - 3.66) < 1e-10, f"laminar Nu={nu}, expected 3.66"

    def test_turbulent_dittus_boelter(self):
        nu = nusselt_number(20000, 4.3)
        expected = 0.023 * 20000**0.8 * 4.3**0.4
        assert abs(nu - expected) / expected < 0.01

    def test_helical_increases_nu(self):
        nu_s = nusselt_number(1000, 4.3)
        nu_h = nusselt_number(1000, 4.3, dean=500)
        assert nu_h > nu_s, "helix must increase Nu"


class TestChannelEvaluation:
    def test_straight_channel(self):
        r = evaluate_channel("straight", 0.080, 0.003, 0.5)
        assert r.reynolds > 0
        assert r.pressure_drop_Pa > 0
        assert r.flow_rate_kg_s > 0
        assert r.temp_rise_K > 0

    def test_helical_channel(self):
        r = evaluate_channel("helical", 0.300, 0.003, 0.5,
                              helix_radius_m=0.045)
        assert r.reynolds > 0
        assert r.pressure_drop_Pa > 0
        assert r.flow_rate_kg_s > 0

    def test_helix_has_higher_pressure_drop(self):
        """At same diameter and pump power, helix should have lower flow
        due to longer path and helical friction."""
        r_s = evaluate_channel("straight", 0.080, 0.003, 0.5)
        r_h = evaluate_channel("helical", 0.300, 0.003, 0.5,
                                helix_radius_m=0.045)
        assert r_h.velocity_ms < r_s.velocity_ms, (
            "helix should have lower velocity at same pump power"
        )

    def test_pump_power_consistency(self):
        """Pump power = Δp × Q_vol."""
        r = evaluate_channel("straight", 0.080, 0.003, 0.5)
        Q_vol = r.flow_rate_kg_s / 992.0
        P_calc = r.pressure_drop_Pa * Q_vol
        assert abs(P_calc - r.pump_power_W) / r.pump_power_W < 0.05, (
            f"P={r.pump_power_W}, Δp×Q={P_calc}"
        )

    def test_energy_balance(self):
        """Q = m_dot × cp × ΔT."""
        r = evaluate_channel("straight", 0.080, 0.003, 0.5,
                              heat_load_W=10.0)
        Q_calc = r.flow_rate_kg_s * 4179.0 * r.temp_rise_K
        assert abs(Q_calc - r.heat_removed_W) / r.heat_removed_W < 0.05, (
            f"Q={r.heat_removed_W}W, m×cp×ΔT={Q_calc}W"
        )

    def test_applicability_flag(self):
        """Transitional flow must be flagged."""
        # Find a case in transitional range
        r = evaluate_channel("straight", 0.080, 0.003, 0.001)
        if 2300 < r.reynolds < 10000:
            assert not r.applicable, "transitional must be flagged"
        # And a case in laminar
        r2 = evaluate_channel("straight", 0.080, 0.003, 0.0001)
        if r2.reynolds < 2300:
            assert r2.applicable, "laminar should be applicable"


class TestComparison:
    def test_compare_returns_two(self):
        results = compare_channels()
        assert len(results) == 2
        assert results[0].channel_type == "straight"
        assert results[1].channel_type == "helical"

    def test_same_pump_power(self):
        results = compare_channels(pump_power_W=0.5)
        assert results[0].pump_power_W == results[1].pump_power_W
