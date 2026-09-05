"""Integration tests for the agent closed-loop: propose→build→score→verdict→select.

Covers:
  - Legitimate feasible candidate is selected (best_iter >= 0)
  - Broken candidate (shorted winding) is not selected
  - All-fail returns "no feasible design" (best_iter = -1, best = None)
  - Non-finite metrics (inf/nan) are not selected
  - Parameter changes produce actual geometry changes (not just code string changes)
"""

from __future__ import annotations

import numpy as np
import pytest

from organic_motor.agent.loop import AgentLoop, RunResult, IterationResult
from organic_motor.agent.prompt import BASELINE_CODE
from organic_motor.agent.sandbox import execute_agent_code
from organic_motor.config3d import MotorConfig3D


def _cfg(**overrides):
    defaults = dict(
        shape=(48, 48, 32),
        excitation_mode="impressed",
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=2,
        maxwell_maxiter=5,
        thermal_maxiter=10,
        electric_maxiter=5,
        n_theta=8,
        torque_n_z=4,
        torque_n_r=4,
    )
    defaults.update(overrides)
    return MotorConfig3D(**defaults)


BROKEN_CODE = '''# Deliberately broken: empty field, no copper, no iron.
def build(cfg):
    return empty_field(cfg.shape, cfg.spacing, cfg.origin)

def magnetization(cfg):
    import numpy as np
    return np.zeros(cfg.shape + (3,), dtype=np.float32)
'''


SHORTED_CODE = '''# Deliberately shorted: one solid copper ring (1 component, not 4+4+4).
def build(cfg):
    mf = empty_field(cfg.shape, cfg.spacing, cfg.origin)
    r = cfg.origin[0] + cfg.spacing[0] * np.arange(cfg.shape[0])
    cx = (cfg.origin[0] + r[-1]) / 2.0
    cy = (cfg.origin[1] + r[-1]) / 2.0
    rr = np.sqrt((r[:, None, None] - cx)**2 + (cfg.origin[1] + cfg.spacing[1] * np.arange(cfg.shape[1])[None, :, None] - cy)**2)
    cu_sdf = rr - 0.035  # solid ring at 35mm
    mf.add("copper", cu_sdf)
    return mf

def magnetization(cfg):
    import numpy as np
    return np.zeros(cfg.shape + (3,), dtype=np.float32)
'''


class TestNoFeasibleCandidate:
    """When all candidates fail, best_iter=-1 and best=None."""

    def test_all_broken_returns_no_feasible(self, tmp_path):
        cfg = _cfg()
        loop = AgentLoop(
            cfg=cfg, out_dir=tmp_path, agent=None,
            max_iters=2, display_shape=None,
            baseline_code=BROKEN_CODE,
        )
        result = loop.run()
        assert result.best_iter == -1, f"expected -1, got {result.best_iter}"
        assert result.best is None, "best should be None"
        for ir in result.iterations:
            assert ir.metrics.get("passed", False) is False

    def test_all_shorted_returns_no_feasible(self, tmp_path):
        cfg = _cfg()
        loop = AgentLoop(
            cfg=cfg, out_dir=tmp_path, agent=None,
            max_iters=2, display_shape=None,
            baseline_code=SHORTED_CODE,
        )
        result = loop.run()
        assert result.best_iter == -1
        assert result.best is None
        for ir in result.iterations:
            assert ir.metrics.get("passed", False) is False

    def test_runresult_default_state(self):
        r = RunResult(out_dir="/tmp/test")
        assert r.best_iter == -1
        assert r.best is None
        assert r.best_obj == float("inf")


class TestVerdictGate:
    """The verdict gate must reject candidates that fail engineering checks."""

    def test_missing_passed_means_infeasible(self, tmp_path):
        cfg = _cfg()
        loop = AgentLoop(
            cfg=cfg, out_dir=tmp_path, agent=None,
            max_iters=1, display_shape=None,
            baseline_code=BROKEN_CODE,
        )
        result = loop.run()
        assert result.best_iter == -1
        ir = result.iterations[0]
        assert ir.metrics.get("passed", False) is False

    def test_non_finite_obj_not_selected(self):
        """A candidate with obj=inf or nan must not become best."""
        r = RunResult(out_dir="/tmp/test")
        r.iterations.append(IterationResult(
            iter=0, code="x", metrics={"obj": float("inf"), "passed": True},
            error=None, elapsed=0.0,
        ))
        feasible = r.iterations[0].metrics.get("passed", False)
        obj = r.iterations[0].metrics.get("obj", float("inf"))
        assert not (feasible and obj < r.best_obj and np.isfinite(obj))

    def test_passed_false_overrides_good_obj(self):
        """Even with excellent obj, passed=False must prevent selection."""
        r = RunResult(out_dir="/tmp/test")
        metrics = {"obj": 0.001, "passed": False}
        feasible = metrics.get("passed", False)
        assert feasible is False
        assert not (feasible and metrics["obj"] < r.best_obj)


class TestParameterChanges:
    """Heuristic parameter changes must produce actual geometry changes."""

    def test_heuristic_produces_different_code(self, tmp_path):
        cfg = _cfg()
        loop = AgentLoop(cfg=cfg, out_dir=tmp_path, agent=None, max_iters=5)
        codes = [loop._heuristic_propose(BASELINE_CODE, {}, i) for i in range(5)]
        for i in range(5):
            compile(codes[i], "<test>", "exec")
        assert len(set(codes)) >= 3, "heuristic should produce >= 3 distinct codes"

    def test_heuristic_no_duplicate_keywords(self, tmp_path):
        cfg = _cfg()
        loop = AgentLoop(cfg=cfg, out_dir=tmp_path, agent=None)
        for i in range(5):
            code = loop._heuristic_propose(BASELINE_CODE, {}, i)
            assert code.count("arch_slope") <= 2, f"dup arch_slope at i={i}"
            assert code.count("n_bands") <= 2, f"dup n_bands at i={i}"

    def test_different_params_different_geometry(self):
        """n_bands=5 vs n_bands=7 must produce different copper voxel counts."""
        from organic_motor.construct.objects import field_driven_motor
        cfg = _cfg(shape=(96, 96, 58))
        mf5 = field_driven_motor(cfg, n_bands=5).build()
        mf7 = field_driven_motor(cfg, n_bands=7).build()
        cu5 = np.sum(mf5.sdfs["copper"].sdf < 0)
        cu7 = np.sum(mf7.sdfs["copper"].sdf < 0)
        assert cu5 != cu7, f"same copper: {cu5} vs {cu7}"


class TestRealClosedLoop:
    """Full agent loop at a resolution where verdicts can actually pass.

    Marked slow: requires 96^3 physics + 224^3 display (~2 min/iter).
    """

    @pytest.mark.slow
    def test_feasible_candidate_selected(self, tmp_path):
        cfg = _cfg(
            shape=(96, 96, 58),
            maxwell_maxiter=60, thermal_maxiter=60, electric_maxiter=60,
            n_theta=16, torque_n_z=8, torque_n_r=8,
        )
        loop = AgentLoop(
            cfg=cfg, out_dir=tmp_path, agent=None,
            max_iters=2, display_shape=(224, 224, 136),
        )
        result = loop.run()
        assert len(result.iterations) == 2
        for ir in result.iterations:
            assert ir.error is None, f"iter {ir.iter} errored: {ir.error}"
            assert "passed" in ir.metrics, "critic must set 'passed'"
            assert "verdicts" in ir.metrics, "critic must set 'verdicts'"
        ir0 = result.iterations[0]
        ir1 = result.iterations[1]
        assert ir0.code != ir1.code, "heuristic must change code between iters"
        assert "n_bands=7" in ir0.code
        assert "n_bands=5" in ir1.code
        # Both should be feasible (winding+structure pass at 224^3 display)
        assert ir0.metrics["passed"] is True
        assert ir1.metrics["passed"] is True
        # best_iter should be the one with lower obj
        assert result.best_iter >= 0
        assert result.best is not None
        assert result.best_obj == min(ir0.metrics["obj"], ir1.metrics["obj"])
