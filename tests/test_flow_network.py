"""D1 tests: node/branch flow network — mass conservation, splitting, heat.

Verified properties:
  - series: identical branch flows, resistances add
  - parallel split by resistance ratio; node mass conservation exact
  - heat: series dT accumulates Q/(m*cp); energy out = heat in
  - geometry mapping: BranchingManifold -> 3 branches, helix -> 1 branch
"""

import pytest
import math

from organic_motor.physics.flow1d import (
    Branch, PumpCurve, solve_network,
    network_from_manifold, network_from_helix,
    CP_WATER,
)


class TestSeriesNetwork:
    def test_equal_flows(self):
        s = solve_network(
            [Branch("a", "source", "mid", 0.4, 0.003),
             Branch("b", "mid", "out", 0.4, 0.003)],
            {"source": 2.0e5, "out": 1.0e5},
        )
        assert s.converged
        assert s.branch_flows_kg_s["a"] == pytest.approx(
            s.branch_flows_kg_s["b"], rel=1e-6)

    def test_resistances_add(self):
        """Series pair must match one 2L pipe of the same diameter."""
        two = solve_network(
            [Branch("a", "source", "mid", 0.4, 0.003),
             Branch("b", "mid", "out", 0.4, 0.003)],
            {"source": 2.0e5, "out": 1.0e5},
        )
        one = solve_network(
            [Branch("ab", "source", "out", 0.8, 0.003)],
            {"source": 2.0e5, "out": 1.0e5},
        )
        assert two.branch_flows_kg_s["a"] == pytest.approx(
            one.branch_flows_kg_s["ab"], rel=0.02)


class TestParallelSplit:
    def test_symmetric_y_splits_evenly(self):
        s = solve_network(
            [Branch("in", "source", "fork", 0.05, 0.004),
             Branch("b1", "fork", "out1", 0.10, 0.003),
             Branch("b2", "fork", "out2", 0.10, 0.003)],
            {"source": 2.0e5, "out1": 1.0e5, "out2": 1.0e5},
        )
        assert s.converged
        f1, f2 = s.branch_flows_kg_s["b1"], s.branch_flows_kg_s["b2"]
        assert f1 == pytest.approx(f2, rel=1e-3)
        assert f1 > 0 and f2 > 0

    def test_mass_conservation_at_fork(self):
        s = solve_network(
            [Branch("in", "source", "fork", 0.05, 0.004),
             Branch("b1", "fork", "out1", 0.10, 0.003),
             Branch("b2", "fork", "out2", 0.30, 0.003)],
            {"source": 2.0e5, "out1": 1.0e5, "out2": 1.0e5},
        )
        fin = s.branch_flows_kg_s["in"]
        fout = s.branch_flows_kg_s["b1"] + s.branch_flows_kg_s["b2"]
        assert fin == pytest.approx(fout, abs=1e-8)

    def test_shorter_branch_takes_more_flow(self):
        s = solve_network(
            [Branch("in", "source", "fork", 0.05, 0.004),
             Branch("b1", "fork", "out1", 0.10, 0.003),
             Branch("b2", "fork", "out2", 0.30, 0.003)],
            {"source": 2.0e5, "out1": 1.0e5, "out2": 1.0e5},
        )
        assert s.branch_flows_kg_s["b1"] > s.branch_flows_kg_s["b2"]

    def test_parallel_beats_series_flow(self):
        """Two parallel paths carry more than one path at the same dp."""
        series = solve_network(
            [Branch("a", "source", "out", 0.2, 0.003)],
            {"source": 2.0e5, "out": 1.0e5},
        )
        parallel = solve_network(
            [Branch("a", "source", "out", 0.2, 0.003),
             Branch("b", "source", "out", 0.2, 0.003)],
            {"source": 2.0e5, "out": 1.0e5},
        )
        total_par = sum(abs(v) for v in parallel.branch_flows_kg_s.values())
        assert total_par > series.branch_flows_kg_s["a"]


class TestNetworkHeat:
    def test_series_temperature_accumulates(self):
        s = solve_network(
            [Branch("a", "source", "mid", 0.4, 0.003, heat_load_W=5.0),
             Branch("b", "mid", "out", 0.4, 0.003, heat_load_W=5.0)],
            {"source": 2.0e5, "out": 1.0e5}, inlet_temp_C=40.0,
        )
        m = s.branch_flows_kg_s["a"]
        expected_dT = 10.0 / (m * CP_WATER)
        assert s.outlet_temps_C["out"] - 40.0 == pytest.approx(expected_dT, rel=1e-3)

    def test_energy_balance(self):
        s = solve_network(
            [Branch("in", "source", "fork", 0.05, 0.004, heat_load_W=2.0),
             Branch("b1", "fork", "out1", 0.10, 0.003, heat_load_W=4.0),
             Branch("b2", "fork", "out2", 0.30, 0.003, heat_load_W=4.0)],
            {"source": 2.0e5, "out1": 1.0e5, "out2": 1.0e5},
            inlet_temp_C=40.0,
        )
        assert s.heat_removed_W == pytest.approx(10.0, rel=1e-3)

    def test_wall_temp_above_fluid(self):
        s = solve_network(
            [Branch("a", "source", "out", 0.4, 0.003, heat_load_W=5.0)],
            {"source": 2.0e5, "out": 1.0e5}, inlet_temp_C=40.0,
        )
        r = s.branch_results["a"]
        assert r["wall_temp_C"] > 40.0


class TestGeometryMapping:
    def test_manifold_maps_to_three_branches(self):
        from organic_motor.construct.morphology import BranchingManifold
        mf = BranchingManifold()
        branches = network_from_manifold(mf)
        assert [b.name for b in branches] == ["inlet", "branch1", "branch2"]
        assert branches[0].from_node == "source"
        assert branches[1].to_node == "out1"
        assert branches[2].to_node == "out2"
        for b in branches:
            assert b.length_m > 0
            assert b.diameter_m == pytest.approx(2 * mf.channel_radius)

    def test_manifold_network_solves(self):
        from organic_motor.construct.morphology import BranchingManifold
        branches = network_from_manifold(BranchingManifold())
        for b in branches:
            b.heat_load_W = 3.0
        s = solve_network(branches,
                          {"source": 2.5e5, "out1": 1.0e5, "out2": 1.0e5},
                          inlet_temp_C=40.0)
        assert s.converged
        assert s.branch_flows_kg_s["branch1"] == pytest.approx(
            s.branch_flows_kg_s["branch2"], rel=1e-3)
        assert s.heat_removed_W == pytest.approx(9.0, rel=1e-2)

    def test_helix_maps_to_single_branch(self):
        from organic_motor.construct.morphology import HelicalChannelGenerator
        hx = HelicalChannelGenerator()
        branches = network_from_helix(hx)
        assert len(branches) == 1
        b = branches[0]
        assert b.channel_type == "helical"
        assert b.helix_radius_m == pytest.approx(hx.radius)
        assert b.length_m == pytest.approx(hx.centerline_length(), rel=1e-6)


class TestPumpCurve:
    def test_pump_curve_shape(self):
        p = PumpCurve(dp_max_Pa=1.0e5, Q_max_m3s=1.0e-4)
        assert p.dp(0.0) == pytest.approx(1.0e5)
        assert p.dp(0.5e-4) == pytest.approx(0.75e5)
        assert p.dp(1.0e-4) == pytest.approx(0.0)

    def test_pump_reduces_flow_vs_fixed_dp(self):
        """A drooping curve delivers less than its dp_max fixed source.

        Fixed: source-out = dp_max.  Pump: same boundary pressures on both
        sides, all head from the curve (droops to zero at Q_max).
        """
        branches = [Branch("a", "source", "out", 0.4, 0.003)]
        fixed = solve_network(branches, {"source": 2.0e5, "out": 1.0e5})
        pumped = solve_network(
            branches, {"source": 1.0e5, "out": 1.0e5},
            pump=PumpCurve(dp_max_Pa=1.0e5, Q_max_m3s=0.5e-4),
        )
        assert pumped.branch_flows_kg_s["a"] < fixed.branch_flows_kg_s["a"]
        # operating point must sit on the curve side below Q_max
        q_op = pumped.branch_flows_kg_s["a"] / 992.0
        assert q_op < 0.5e-4
