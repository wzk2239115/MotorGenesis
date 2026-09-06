"""E1/E2/E3 tests: convergence summary, reference adapter registry, provenance."""

import pytest
import numpy as np

from organic_motor.reports.convergence_matrix import (
    summarize_convergence, format_matrix,
)
from organic_motor.physics.reference import (
    list_adapters, get_adapter, CrossCheck, ReferenceResult,
)
from organic_motor.reports.parameter_provenance import (
    format_provenance_table, confidence_counts, PROVENANCE,
)


class TestConvergenceSummary:
    def test_separates_channels(self):
        rows = [
            {"channel": "grid", "refinement": 96, "value": 0.020},
            {"channel": "grid", "refinement": 128, "value": 0.024},
            {"channel": "linear_solver", "refinement": 120, "value": 0.018},
            {"channel": "linear_solver", "refinement": 240, "value": 0.020},
        ]
        s = summarize_convergence(rows)
        assert set(s) == {"grid", "linear_solver"}
        assert s["grid"]["last_rel_change"] == pytest.approx(
            abs(0.024 - 0.020) / 0.020)
        assert s["linear_solver"]["last_rel_change"] == pytest.approx(
            abs(0.020 - 0.018) / 0.018)

    def test_single_point_no_change(self):
        s = summarize_convergence([
            {"channel": "grid", "refinement": 96, "value": 0.02}])
        assert s["grid"]["last_rel_change"] is None

    def test_sorted_by_refinement(self):
        rows = [
            {"channel": "c", "refinement": 2, "value": 1.0},
            {"channel": "c", "refinement": 1, "value": 0.5},
        ]
        s = summarize_convergence(rows)
        assert [e["refinement"] for e in s["c"]["series"]] == [1, 2]
        assert s["c"]["last_rel_change"] == pytest.approx(1.0)  # (1-0.5)/0.5

    def test_format_matrix_renders(self):
        s = summarize_convergence([
            {"channel": "grid", "refinement": 96, "value": 0.020},
            {"channel": "grid", "refinement": 128, "value": 0.024},
        ])
        md = format_matrix(s)
        assert "grid" in md and "%" in md


class TestReferenceAdapter:
    def test_registry_lists_unavailable(self):
        adapters = list_adapters()
        assert "femm" in adapters and adapters["femm"] is False
        assert "elmer" in adapters and adapters["elmer"] is False

    def test_unavailable_raises_loudly(self):
        a = get_adapter("femm")
        assert not a.available()
        with pytest.raises(RuntimeError, match="unavailable"):
            a.solve_magnetostatic(None)

    def test_unknown_adapter(self):
        with pytest.raises(KeyError):
            get_adapter("does-not-exist")

    def test_cross_check_math(self):
        c = CrossCheck(quantity="psi", ours=0.099, reference=0.100)
        assert c.rel_error == pytest.approx(0.01)
        c.tolerance = 0.05
        assert c.evaluate() is True
        c2 = CrossCheck(quantity="psi", ours=0.08, reference=0.10)
        assert c2.evaluate() is False

    def test_reference_result_fields(self):
        r = ReferenceResult(solver_name="x", solver_version="1")
        assert r.torque_Nm is None and r.notes == ""


class TestParameterProvenance:
    def test_every_entry_has_confidence(self):
        for p in PROVENANCE:
            assert p.confidence in ("verified", "estimate", "unverified")
            assert p.source and p.method

    def test_counts_consistent(self):
        counts = confidence_counts()
        assert counts["verified"] + counts["estimate"] + \
            counts["unverified"] == len(PROVENANCE)
        assert counts["unverified"] >= 2  # L_phase, J_rotor, iron-loss k

    def test_markdown_renders(self):
        md = format_provenance_table(markdown=True)
        assert "| 参数 |" in md or "symbol" in md
        assert "psi_pm" in md

    def test_tsv_renders(self):
        tsv = format_provenance_table(markdown=False)
        assert tsv.count("\n") >= len(PROVENANCE)
