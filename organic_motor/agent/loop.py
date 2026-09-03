"""The agent loop: propose -> sandbox -> critic -> checkpoint -> feedback.

Each iteration an LLM (or fallback heuristic) writes construction code, the
sandbox builds a :class:`MaterialField` from it, the differentiable solver
scores it, and the result is written as a viewer-compatible checkpoint.  The
score and any execution error are fed back, so the agent corrects itself.

Run with::

    MOTORGENESIS_X64=0 python -m organic_motor.agent --iters 6
    motor-web --out organic_motor/out   # watch the agent evolve live

The loop is deliberately agnostic to the proposer: pass any callable that
returns a code string.  With no API key it falls back to a parametric
hill-climb so the infrastructure is exercised without an LLM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.critic import score_fields
from organic_motor.construct.export import save_checkpoint
from organic_motor.agent.prompt import BASELINE_CODE, FEEDBACK_TEMPLATE, SYSTEM_PROMPT
from organic_motor.agent.sandbox import execute_agent_code


@dataclass
class IterationResult:
    iter: int
    code: str
    metrics: dict
    error: str | None
    elapsed: float


@dataclass
class RunResult:
    out_dir: Path
    iterations: list[IterationResult] = field(default_factory=list)
    best_iter: int = 0
    best_obj: float = float("inf")

    @property
    def best(self) -> IterationResult:
        return self.iterations[self.best_iter] if self.iterations else None


def _metrics_table(metrics: dict) -> str:
    keys = (
        "obj", "torque", "|torque|", "torque_ripple", "copper_loss_W",
        "iron_loss_W", "loss_W", "temperature_max_C",
        "vol_iron", "vol_copper", "vol_pm",
        "maxwell_residual", "thermal_residual", "electric_residual",
        "copper_components", "copper_min_gap_mm",
        "air_gap_iron_bridge", "floating_islands",
        "rotor_anchored", "stator_anchored", "min_neck_mm",
        "trapped_voids", "housing_open_area_ratio", "end_face_occlusion",
        "through_flow_networks", "min_phase_gap_mm",
        "phase_a_components", "phase_b_components", "phase_c_components",
    )
    lines = []
    for k in keys:
        if k in metrics:
            val = metrics[k]
            if isinstance(val, bool):
                lines.append(f"  {k:<28} {val}")
            elif isinstance(val, float):
                lines.append(f"  {k:<28} {val:.4g}")
            else:
                lines.append(f"  {k:<28} {val}")
    return "\n".join(lines) if lines else "  (no metrics)"


def _diagnosis(metrics: dict, error: str | None) -> str:
    if error:
        return (
            "The code FAILED to run. Fix the error below and resubmit a "
            "complete code block. Common fixes: define build(cfg), use only "
            "the imported primitives, guard divisions with max(x, 1e-9).\n\n"
            "Traceback:\n" + error[-1500:]
        )
    parts = []
    obj = metrics.get("obj", float("inf"))
    torque = metrics.get("torque", 0.0)
    parts.append(f"Objective (lower=better): {obj:.4g}. Torque: {torque:.4g} N*m.")
    if torque <= 0:
        parts.append("Torque is near zero: the PM poles and winding are not coupling. "
                     "Check magnetisation alternates N/S and the winding annulus overlaps the phase belts.")
    if metrics.get("maxwell_residual", 1.0) > 0.5:
        parts.append("Maxwell solver did not converge well; the iron magnetic circuit may be broken "
                     "(discontinuities) or the air gap is mis-sized.")
    if metrics.get("temperature_max_C", 0.0) > 100:
        parts.append("Too hot: reduce copper loss (less current path resistance) or add cooling.")
    if metrics.get("vol_pm", 0.0) > 0.15:
        parts.append("PM volume is high (penalised); a thinner or narrower magnet may suffice.")
    cc = metrics.get("copper_components", -1)
    if cc == 1:
        parts.append("WARNING: copper is ONE connected component -- the winding is shorted into a ring. "
                     "Reduce wire radius or increase radial spacing between layers so coils are distinct.")
    elif cc > 12:
        parts.append(f"Copper has {cc} components -- winding is fragmented. Check end-turn connectivity.")
    if metrics.get("air_gap_iron_bridge", False):
        parts.append("WARNING: iron bridges the air gap -- rotor and stator are electrically/magnetically shorted. "
                     "Ensure FunctionalVoids is called to protect the gap.")
    fi = metrics.get("floating_islands", 0)
    if fi and fi > 0:
        parts.append(f"WARNING: {fi} floating structural islands -- metal with no path to shaft or housing. "
                     "Grow features from anchors with overlap; StructuralContinuity deletes the rest.")
    if not metrics.get("rotor_anchored", True):
        parts.append("WARNING: rotor is not anchored to the shaft -- no torque path. Add hub spokes.")
    if not metrics.get("stator_anchored", True):
        parts.append("WARNING: stator is not anchored to the housing -- add connecting walls/rim overlap.")
    neck = metrics.get("min_neck_mm", 0.0)
    if 0 < neck < 0.8:
        parts.append(f"Thin structural necks ({neck:.2f}mm) -- junctions may snap; "
                     "grow overlaps instead of tangent contact.")
    tv = metrics.get("trapped_voids", 0)
    if tv and tv > 0:
        parts.append(f"{tv} trapped coolant voids -- powder-removal/fill failure; vent them.")
    tfn = metrics.get("through_flow_networks", None)
    if tfn is not None and tfn < 1:
        parts.append("No through-flow coolant network (inlet->outlet) -- the channel is "
                     "dead-ended; connect it to both axial ends.")
    pg = metrics.get("min_phase_gap_mm", None)
    if pg is not None and pg <= 0.0:
        parts.append("Phase insulation gap is zero -- phases are shorted.")
    for phase in ("a", "b", "c"):
        comp = metrics.get(f"phase_{phase}_components", None)
        exp = (metrics.get("expected_components") or [None, None, None])[
            "abc".index(phase)]
        if comp is not None and exp is not None and comp != exp:
            parts.append(f"Phase {phase.upper()} has {comp} copper networks (expected {exp}) "
                         "-- breaks or fragmentation in the real winding; check end-turn continuity.")
    hoa = metrics.get("housing_open_area_ratio", 0.0)
    if hoa < 0.1:
        parts.append("Housing is nearly solid -- no windows visible. Add angular cutouts to the housing shell.")
    efo = metrics.get("end_face_occlusion", 0.0)
    if efo > 0.8:
        parts.append("End face is mostly solid iron -- housing rims block the interior view. "
                     "Segment end rings with angular windows.")
    return " ".join(parts) if parts else "Design is reasonable; refine for more torque or less loss."


class LLMAgent:
    """Calls the configured LLM to propose construction code each iteration."""

    def __init__(self, model: str | None = None, max_tokens: int = 16384, temperature: float = 0.4):
        from organic_motor.agent.llm import LLMClient

        # Default to gemini-3.6-flash: fast, non-reasoning, produces clean
        # code blocks.  Reasoning models (deepseek-v4, glm-5.3) eat the token
        # budget on chain-of-thought and emit empty content.
        self.client = LLMClient(model=model or "google/gemini-3.6-flash")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def propose(self, feedback: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": feedback},
        ]
        resp = self.client.complete(
            messages, max_tokens=self.max_tokens, temperature=self.temperature
        )
        from organic_motor.agent.llm import extract_code

        code = extract_code(resp.content)
        if code is None:
            raise RuntimeError(f"LLM returned no code block. Content: {resp.content[:300]}")
        return code

    def __call__(self, feedback: str) -> str:
        return self.propose(feedback)


class AgentLoop:
    """Orchestrates propose -> sandbox -> critic -> checkpoint -> feedback."""

    def __init__(
        self,
        cfg: MotorConfig3D,
        out_dir: str | Path,
        agent: Callable[[str], str] | None = None,
        max_iters: int = 5,
        baseline_code: str = BASELINE_CODE,
        display_shape: tuple[int, int, int] | None = (160, 160, 96),
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "checkpoints").mkdir(exist_ok=True)
        self.agent = agent
        self.max_iters = max_iters
        self.baseline_code = baseline_code
        from dataclasses import replace
        self.display_cfg = replace(cfg, shape=display_shape) if display_shape else None

    def _score(self, code: str):
        """Score code at physics resolution; geometry metrics at display resolution.

        Returns (metrics, error, mf, mag, mf_display).  The display rebuild
        serves double duty: geometric quality metrics (wires thinner than
        the physics voxel only resolve at display resolution) and the
        high-resolution checkpoint saved for the viewer.
        """
        mf, mag, err = execute_agent_code(code, self.cfg)
        if err is not None:
            return {"obj": float("inf"), "error": True}, err, None, None, None
        mf_display = None
        if self.display_cfg is not None:
            mf_display, _, err_d = execute_agent_code(code, self.display_cfg)
            if err_d is not None:
                mf_display = None
        metrics = score_fields(
            mf, self.cfg, mag,
            geometry_mf=mf_display, geometry_cfg=self.display_cfg,
        )
        metrics["materials"] = mf.materials_present()
        return metrics, None, mf, mag, mf_display

    def run(self) -> RunResult:
        result = RunResult(out_dir=self.out_dir)
        code = self.baseline_code
        print(f"[agent] loop start; {self.max_iters} iters; out={self.out_dir}")
        for i in range(self.max_iters):
            t0 = time.perf_counter()
            metrics, err, mf, mag, mf_display = self._score(code)
            elapsed = time.perf_counter() - t0
            if mf is not None:
                save_cfg = self.cfg
                save_mf = mf
                if mf_display is not None:
                    save_mf = mf_display
                    save_cfg = self.display_cfg
                save_checkpoint(
                    save_mf, save_cfg,
                    self.out_dir / "checkpoints" / f"step_{i:06d}.npz",
                    step=i, metrics=metrics, magnetization=mag,
                )
            ir = IterationResult(iter=i, code=code, metrics=metrics, error=err, elapsed=elapsed)
            result.iterations.append(ir)
            obj = metrics.get("obj", float("inf"))
            if obj < result.best_obj:
                result.best_obj = obj
                result.best_iter = i
            self._log(ir, result)
            (self.out_dir / "history.json").write_text(
                json.dumps(
                    {
                        "best_iter": result.best_iter,
                        "best_obj": result.best_obj,
                        "iterations": [
                            {
                                "iter": r.iter, "obj": r.metrics.get("obj"),
                                "torque": r.metrics.get("torque"),
                                "temperature_max_C": r.metrics.get("temperature_max_C"),
                                "error": r.error, "elapsed": r.elapsed,
                                "code": r.code,
                            }
                            for r in result.iterations
                        ],
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            if i == self.max_iters - 1:
                break
            feedback = FEEDBACK_TEMPLATE.format(
                iter=i + 1,
                code=code if len(code) < 4000 else code[:4000] + "    # ... (truncated)",
                metrics_table=_metrics_table(metrics),
                diagnosis=_diagnosis(metrics, err),
            )
            try:
                if self.agent is not None:
                    code = self.agent(feedback)
                else:
                    code = self._heuristic_propose(code, metrics, i)
            except Exception as exc:
                print(f"[agent] proposer failed ({exc}); keeping current code")
        print(
            f"[agent] loop done; best iter {result.best_iter} "
            f"obj={result.best_obj:.4g}"
        )
        return result

    def _log(self, ir: IterationResult, result: RunResult) -> None:
        m = ir.metrics
        if ir.error:
            print(f"  iter {ir.iter}: FAILED ({ir.elapsed:.0f}s)")
            print(f"    {ir.error.splitlines()[-1] if ir.error else ''}")
            return
        marker = " *" if ir.iter == result.best_iter else ""
        print(
            f"  iter {ir.iter}: obj={m.get('obj', float('nan')):.4g} "
            f"torque={m.get('torque', 0):.4g} "
            f"loss={m.get('loss_W', 0):.3g}W "
            f"Tmax={m.get('temperature_max_C', 0):.1f}C "
            f"({ir.elapsed:.0f}s){marker}"
        )

    def _heuristic_propose(self, code: str, metrics: dict, i: int) -> str:
        """Parametric fallback when no LLM is configured.

        Re-sends the baseline with a perturbed magnet thickness / pole
        fraction, so the loop infrastructure is exercised without an API key.
        """
        thickness = 0.0035 + 0.0008 * np.sin(i)
        frac = 0.72 + 0.06 * np.cos(i)
        return self.baseline_code.replace("0.0035", f"{thickness:.5f}").replace(
            "0.72", f"{frac:.4f}"
        )


def run_loop(
    cfg: MotorConfig3D | None = None,
    out_dir: str | Path = "organic_motor/out/agent",
    max_iters: int = 5,
    use_llm: bool = True,
    model: str | None = None,
) -> RunResult:
    """Convenience entry point for the CLI and tests."""
    cfg = cfg or MotorConfig3D(
        shape=(48, 48, 32), excitation_mode="terminal", filt_radius=0.0,
        projection_beta=0.0, mechanical_angles=3,
        maxwell_maxiter=120, thermal_maxiter=240, electric_maxiter=120,
        n_theta=32, torque_n_z=16, torque_n_r=16,
    )
    agent = LLMAgent(model=model) if use_llm else None
    loop = AgentLoop(cfg, out_dir, agent=agent, max_iters=max_iters)
    return loop.run()
