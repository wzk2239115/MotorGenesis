"""Six independent validation verdicts for a constructed motor.

A green "it spins" from the electromagnetic transient must NOT silently
cover broken winding connectivity, a dead-end coolant network, floating
structural metal, unmanufacturable walls or a result that changes by
orders of magnitude between grid resolutions.  Each aspect is its own
verdict:

  1. electromechanical  multi-angle startup transient (T0/T1/T2 maps)
  2. winding            per-phase electrical networks incl. end turns
  3. cooling            dedicated coolant: through-flow, no trapped voids
  4. structure          rotor->shaft and stator->housing load paths
  5. manufacturing      min wall / powder-removal subset (partial by
                        design: overhang and support constraints pending)
  6. mesh_convergence   topology + torque stability across resolutions

Verdict values: ``True`` (pass), ``False`` (fail), ``None`` (not
evaluated at the requested rigour -- shown, never hidden).

Resolution doctrine: topology verdicts are evaluated on the DISPLAY
grid (where the geometry is actually resolved) and cross-checked against
the physics grid; the discrepancy itself feeds the mesh-convergence
verdict, so a mixed-resolution report can never look uniformly green.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField


VERDICT_ORDER = (
    "electromechanical",
    "winding",
    "cooling",
    "structure",
    "manufacturing",
    "mesh_convergence",
)

VERDICT_LABELS = {
    "electromechanical": "电磁加速 Electromechanical",
    "winding": "绕组连通 Winding",
    "cooling": "冷却连通 Cooling",
    "structure": "结构连通 Structure",
    "manufacturing": "制造可行 Manufacturing",
    "mesh_convergence": "网格收敛 Mesh convergence",
}


def _structure_verdict(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    from organic_motor.construct.connectivity import structural_report

    report = structural_report(mf, cfg)
    passed = bool(
        report.get("floating_islands") == 0
        and report.get("rotor_anchored")
        and report.get("stator_anchored")
        and not report.get("rotor_stator_cross_bridge", False)
    )
    return {"passed": passed, "detail": report}


def _cooling_verdict(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    from organic_motor.construct.connectivity import coolant_report

    report = coolant_report(mf, cfg)
    if not report.get("dedicated_coolant", False):
        return {
            "passed": None,
            "detail": {**report, "reason": "no dedicated coolant network"},
        }
    passed = bool(
        report.get("through_flow_networks", 0) >= 1
        and report.get("trapped_voids", 0) == 0
    )
    return {"passed": passed, "detail": report}


def _winding_verdict(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    from organic_motor.construct.phase_verify import verify_phase_connectivity

    report = verify_phase_connectivity(mf, cfg)
    if "passed" not in report:
        return {"passed": None, "detail": report}
    return {"passed": bool(report["passed"]), "detail": report}


def _manufacturing_verdict(
    mf: MaterialField, cfg: MotorConfig3D, min_wall_mm: float = 0.4,
) -> dict:
    """Metal-AM process model: wall, powder escape, overhang.

    The computable subset of the additive-manufacturing constraints, on
    the stator print (the rotor assembly is a separately made part):
      - minimum wall / neck thickness (local feature size, 5th pct),
      - powder escape: every void connects to the external air,
      - overhang: down-facing faces classify into plate-supported (the
        print base), printable bores/thin ribs (inscribed width <= 5mm)
        and genuine unsupported SPANS (gated at 5%).
    Still not evaluated: machining allowance, magnet assembly route,
    dynamic balance, steel-copper interface thermal stress.
    """
    from organic_motor.construct.connectivity import (
        coolant_report,
        overhang_report,
        powder_report,
        structural_report,
    )

    structure = structural_report(mf, cfg)
    coolant = coolant_report(mf, cfg)
    powder = powder_report(mf, cfg)
    overhang = overhang_report(mf, cfg)
    neck = float(structure.get("min_neck_mm", 0.0))
    trapped_voids = int(coolant.get("trapped_voids", 0))
    pockets = int(powder.get("trapped_pockets", 0))
    span_fraction = float(overhang.get("span_fraction", 1.0))
    detail = {
        "min_neck_mm": neck,
        "trapped_voids": trapped_voids,
        "trapped_powder_pockets": pockets,
        "span_fraction": span_fraction,
        "bore_ceiling_fraction": overhang.get("bore_ceiling_fraction", 0.0),
        "plate_supported_fraction": overhang.get("plate_supported_fraction", 0.0),
        "min_wall_gate_mm": min_wall_mm,
        "not_evaluated": [
            "machining_allowance", "magnet_assembly_route",
            "dynamic_balance", "interface_thermal_stress",
        ],
    }
    passed = bool(
        neck >= min_wall_mm
        and trapped_voids == 0
        and pockets == 0
        and span_fraction <= 0.05
    )
    return {"passed": passed, "detail": detail}


def _topology_signature(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Resolution-independent topology fingerprint for convergence."""
    from scipy import ndimage

    from organic_motor.construct.phase_verify import verify_phase_connectivity
    from organic_motor.construct.connectivity import (
        coolant_report,
        structural_report,
    )

    copper = mf.sdfs.get("copper")
    copper_components = 0
    if copper is not None:
        mask = copper.sdf < 0.0
        if mask.any():
            structure = ndimage.generate_binary_structure(3, 1)
            _, copper_components = ndimage.label(mask, structure=structure)
    phase = verify_phase_connectivity(mf, cfg)
    structure = structural_report(mf, cfg)
    coolant = coolant_report(mf, cfg)
    return {
        "copper_components": int(copper_components),
        "phase_components": [
            int(phase.get(f"phase_{n}_components", -1)) for n in ("a", "b", "c")
        ],
        "phase_expected": phase.get("expected_components"),
        "structural_components": int(structure.get("structural_components", -1)),
        "floating_islands": int(structure.get("floating_islands", -1)),
        "trapped_voids": int(coolant.get("trapped_voids", -1)),
        "through_flow_networks": int(coolant.get("through_flow_networks", -1)),
    }


def evaluate_verdicts(
    mf: MaterialField,
    cfg: MotorConfig3D,
    startup_result=None,
    *,
    display_mf: MaterialField | None = None,
    display_cfg: MotorConfig3D | None = None,
    torque_convergence: dict | None = None,
    min_wall_mm: float = 0.4,
) -> dict:
    """Assemble the six verdicts.

    ``startup_result`` is a :class:`MultiAngleStartupResult` (or None when
    the electromechanical transient was not run).  ``display_mf``/
    ``display_cfg`` supply the display-resolution build: topology verdicts
    gate on it (that is where the geometry resolves) and the physics-grid
    numbers are reported alongside.  ``torque_convergence`` (optional)
    carries the quantitative 56-vs-96 grid torque comparison.
    """
    verdicts: dict[str, dict[str, Any]] = {}

    if startup_result is None:
        verdicts["electromechanical"] = {
            "passed": None, "detail": {"reason": "startup transient not run"},
        }
    else:
        verdicts["electromechanical"] = {
            "passed": bool(startup_result.passed),
            "detail": {
                "n_angles": startup_result.n_angles,
                "all_started": startup_result.all_started,
                "any_reversal": startup_result.any_reversal,
                "min_final_speed_rad_s": startup_result.min_final_speed_rad_s,
                "max_startup_current_A": startup_result.max_startup_current_A,
                "max_temperature_C": startup_result.max_temperature_C,
            },
        }

    topo_mf, topo_cfg = (display_mf, display_cfg) if display_mf is not None else (mf, cfg)
    grid_note = "display" if display_mf is not None else "physics"
    verdicts["winding"] = _winding_verdict(topo_mf, topo_cfg)
    verdicts["winding"]["detail"]["grid"] = grid_note
    if display_mf is not None:
        verdicts["winding"]["detail"]["physics_grid"] = _winding_verdict(mf, cfg)["detail"]

    verdicts["cooling"] = _cooling_verdict(topo_mf, topo_cfg)
    verdicts["cooling"]["detail"]["grid"] = grid_note
    if display_mf is not None:
        verdicts["cooling"]["detail"]["physics_grid"] = _cooling_verdict(mf, cfg)["detail"]

    verdicts["structure"] = _structure_verdict(topo_mf, topo_cfg)
    verdicts["structure"]["detail"]["grid"] = grid_note
    if display_mf is not None:
        verdicts["structure"]["detail"]["physics_grid"] = _structure_verdict(mf, cfg)["detail"]

    verdicts["manufacturing"] = _manufacturing_verdict(topo_mf, topo_cfg, min_wall_mm)

    # Mesh convergence: topology signature physics vs display, plus the
    # optional quantitative torque comparison between grid refinements.
    detail: dict[str, Any] = {
        "physics_shape": list(cfg.shape),
        "display_shape": list(display_cfg.shape) if display_cfg is not None else None,
    }
    if display_mf is not None:
        sig_phys = _topology_signature(mf, cfg)
        sig_disp = _topology_signature(display_mf, display_cfg)
        detail["topology_physics"] = sig_phys
        detail["topology_display"] = sig_disp
        # Cross-grid stability gates on the quantities BOTH grids must
        # resolve: phase networks (analytic netlist ownership), structural
        # bodies, floating islands, trapped voids.  Print-scale features
        # (the ~1.5mm coolant channels and 2.4mm coil side bands are
        # deliberately below the physics cell) are gated on the DISPLAY /
        # construction grid and only REPORTED for the physics grid, with a
        # fragmentation bound so a real sub-cell explosion still fails.
        copper_frag_ok = (
            sig_phys["copper_components"]
            <= 3 * max(sig_disp["copper_components"], 1)
        )
        topology_stable = (
            sig_phys["floating_islands"] == 0 and sig_disp["floating_islands"] == 0
            and sig_phys["trapped_voids"] == 0 and sig_disp["trapped_voids"] == 0
            and sig_disp["phase_components"] == (sig_disp["phase_expected"]
                                                 or sig_disp["phase_components"])
            and sig_phys["phase_components"] == sig_disp["phase_components"]
            and sig_phys["structural_components"] == sig_disp["structural_components"]
            and copper_frag_ok
            and sig_disp["through_flow_networks"] >= 1
        )
        detail["topology_stable"] = bool(topology_stable)
    else:
        topology_stable = None
        detail["reason"] = "single-resolution run; no display build available"
    if torque_convergence is not None:
        detail["torque"] = torque_convergence
        t1_change = abs(float(torque_convergence.get("t1_amplitude_change_pct", 100.0)))
        # T1 gates at 5% relative; T0 cogging additionally allows an
        # absolute floor (0.015 N*m) because it is a small difference of
        # large stresses with a measured localisation noise floor.
        t0_abs = abs(float(torque_convergence.get("t0_rms_fine_Nm", 1.0))
                     - float(torque_convergence.get("t0_rms_physics_Nm", 0.0)))
        t0_gate = max(0.05 * float(torque_convergence.get("t0_rms_physics_Nm", 1.0)), 0.015)
        detail["t0_gate_Nm"] = t0_gate
        detail["torque_stable"] = bool(t1_change <= 5.0 and t0_abs <= t0_gate)
        passed = bool(topology_stable and detail["torque_stable"])
    elif topology_stable is not None:
        # Topology-only convergence: honest "None" -- quantitative torque
        # convergence was not measured.
        passed = None if topology_stable else False
    else:
        passed = None
    verdicts["mesh_convergence"] = {"passed": passed, "detail": detail}

    evaluated = [k for k in VERDICT_ORDER if verdicts[k]["passed"] is not None]
    failed = [k for k in VERDICT_ORDER if verdicts[k]["passed"] is False]
    return {
        "verdicts": {k: verdicts[k] for k in VERDICT_ORDER},
        "labels": VERDICT_LABELS,
        "evaluated": len(evaluated),
        "failed": failed,
        # Overall green requires every evaluated verdict to pass AND the
        # three core verdicts (electromechanical, winding, structure) to be
        # actually evaluated and passing.  None verdicts never hide a fail.
        "passed": bool(
            not failed
            and verdicts["electromechanical"]["passed"] is True
            and verdicts["winding"]["passed"] is True
            and verdicts["structure"]["passed"] is True
        ),
    }


def format_verdict_table(suite: dict) -> str:
    """Human-readable six-verdict summary for console logs."""
    lines = []
    marks = {True: "PASS", False: "FAIL", None: "----"}
    for key in VERDICT_ORDER:
        v = suite["verdicts"][key]
        mark = marks[v["passed"]]
        extra = ""
        if key == "manufacturing":
            d = v["detail"]
            extra = (f"  (min wall {d.get('min_neck_mm', 0):.2f}mm, "
                     f"powder {d.get('trapped_powder_pockets', '?')}, "
                     f"span {100.0*d.get('span_fraction', 1.0):.1f}%)")
        elif key == "mesh_convergence":
            d = v["detail"]
            if "torque" in d:
                extra = (f"  (T1 Δ{d['torque'].get('t1_amplitude_change_pct', float('nan')):.1f}%, "
                         f"T0 Δ{d['torque'].get('t0_rms_change_pct', float('nan')):.1f}%)")
            elif d.get("topology_stable") is not None:
                extra = "  (仅拓扑对比, 转矩收敛未测)"
        lines.append(f"  [{mark}] {VERDICT_LABELS[key]}{extra}")
    core = suite["passed"]
    lines.append(f"  overall: {'PASS' if core else 'NOT YET'} "
                 f"({suite['evaluated']}/6 evaluated, failed: {suite['failed'] or 'none'})")
    return "\n".join(lines)
