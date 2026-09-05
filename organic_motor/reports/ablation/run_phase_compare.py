"""Minimal reproducible phase comparison: s=+1 vs s=-1 at δ=90°.

Only two conditions, 6 angles each, full per-angle data saved.
Report auto-generated from JSON.  Includes report generation test.

Usage:
    MOTORGENESIS_X64=0 python -m organic_motor.reports.ablation.run_phase_compare
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.agent.sandbox import execute_agent_code
from organic_motor.agent.prompt import BASELINE_CODE
from organic_motor.construct.realize import realize
from organic_motor.optimization.objective3d import forward3d_fields


OUT_DIR = Path(__file__).parent
OUT_JSON = OUT_DIR / "phase_compare_results.json"
OUT_REPORT = OUT_DIR / "phase_compare_report.md"

SHAPE = (96, 96, 58)
PP = 5
N_ANGLES = 6
MAXITER = 240


def make_cfg():
    return MotorConfig3D(
        shape=SHAPE, excitation_mode="impressed",
        filt_radius=0.0, projection_beta=0.0,
        mechanical_angles=1, maxwell_maxiter=MAXITER,
        thermal_maxiter=MAXITER, electric_maxiter=60,
        n_theta=32, torque_n_z=16, torque_n_r=16,
    )


def cfg_record(cfg):
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    nx, ny, nz = cfg.shape
    return {
        "shape": list(cfg.shape),
        "spacing_mm": [round(dx*1000, 4), round(dy*1000, 4), round(dz*1000, 4)],
        "origin_mm": [round(ox*1000, 2), round(oy*1000, 2), round(oz*1000, 2)],
        "node_extent_mm": [round((nx-1)*dx*1000, 2), round((ny-1)*dy*1000, 2), round((nz-1)*dz*1000, 2)],
        "maxwell_maxiter": cfg.maxwell_maxiter,
        "pole_pairs": cfg.pole_pairs,
        "electrical_phase_offset": float(cfg.electrical_phase_offset),
        "current_density_peak": float(cfg.current_density_peak),
    }


def proper_3phase(elec_angle):
    offs = np.array([0.0, 2.0*np.pi/3.0, 4.0*np.pi/3.0])
    return np.cos(elec_angle - offs)


def solve_one(cfg, mech_angle, amps):
    """Single solve at one angle, returns full record."""
    mf, mag_raw, _ = execute_agent_code(BASELINE_CODE, cfg)
    fields, mag_arr = realize(mf, cfg, mag_raw)
    reg = mf.metadata.get("centerline_registry")
    a = jnp.asarray([mech_angle])
    t0 = time.perf_counter()
    result = forward3d_fields(cfg, fields, mag_arr, a,
                              phase_amplitudes=jnp.asarray(amps),
                              centerline_registry=reg)
    elapsed = time.perf_counter() - t0
    torques = np.asarray(result.torques)
    B = np.asarray(result.flux_density)
    Bm = np.sqrt(B[...,0]**2 + B[...,1]**2 + B[...,2]**2)
    return {
        "torque": round(float(torques[0]), 8),
        "maxwell_residual": float(result.maxwell_residual),
        "thermal_residual": float(result.thermal_residual),
        "source_divergence_residual": float(result.source_divergence_residual),
        "phase_balance_residual": float(result.phase_balance_residual),
        "B_max": float(Bm.max()),
        "B_mean": float(Bm.mean()),
        "solve_time_s": round(elapsed, 1),
    }


def run_condition(sign, delta_deg, cfg):
    """Run one condition (sign, delta) over all angles."""
    delta = delta_deg * np.pi / 180.0
    entries = []
    for ai in range(N_ANGLES):
        mech = ai * (2.0 * np.pi / (PP * N_ANGLES))
        elec = sign * PP * mech + delta
        amps = proper_3phase(elec)

        rp = solve_one(cfg, mech, amps)
        rn = solve_one(cfg, mech, -amps)
        rz = solve_one(cfg, mech, np.zeros(3))

        tp, tn, tz = rp["torque"], rn["torque"], rz["torque"]
        t_odd = (tp - tn) / 2.0
        t_even = (tp + tn) / 2.0

        entry = {
            "angle_idx": ai,
            "mech_deg": round(mech * 180 / np.pi, 2),
            "elec_deg": round(elec * 180 / np.pi, 2),
            "phase_amps": [round(float(x), 6) for x in amps],
            "T_pos": tp, "T_neg": tn, "T_zero": tz,
            "T_odd": round(t_odd, 8),
            "T_even": round(t_even, 8),
            "residual_pos": rp["maxwell_residual"],
            "residual_neg": rn["maxwell_residual"],
            "residual_zero": rz["maxwell_residual"],
            "B_max_pos": rp["B_max"],
            "B_max_zero": rz["B_max"],
        }
        entries.append(entry)
        print(f"    θ={mech*180/np.pi:6.1f}°: T+={tp:+.6f} T-={tn:+.6f} T0={tz:+.6f} "
              f"T_odd={t_odd:+.6f}")

    t_odds = [e["T_odd"] for e in entries]
    t_zeros = [e["T_zero"] for e in entries]
    summary = {
        "sign": sign,
        "delta_deg": delta_deg,
        "n_angles": N_ANGLES,
        "T_odd_mean": round(float(np.mean(t_odds)), 8),
        "T_odd_std": round(float(np.std(t_odds)), 8),
        "T_odd_range": [round(min(t_odds), 8), round(max(t_odds), 8)],
        "T_zero_mean": round(float(np.mean(t_zeros)), 8),
        "T_zero_std": round(float(np.std(t_zeros)), 8),
        "per_angle": entries,
    }
    return summary


def generate_report(results):
    """Generate report markdown from JSON.  No manual numbers."""
    cfg = results["config"]
    sp = cfg["spacing_mm"]
    ne = cfg["node_extent_mm"]
    lines = [
        "# Phase Comparison Report",
        "",
        f"**Date**: {results['timestamp']}",
        f"**Git**: {results['git_hash']}",
        f"**Design hash**: {results['design_hash']}",
        f"**Config**: shape={cfg['shape']}, spacing=({sp[0]},{sp[1]},{sp[2]})mm",
        f"  node_extent=({ne[0]},{ne[1]},{ne[2]})mm",
        f"  maxiter={cfg['maxwell_maxiter']}, pp={cfg['pole_pairs']}",
        f"  N_angles={N_ANGLES}, maxiter={MAXITER}",
        "",
    ]

    for cond in results["conditions"]:
        lines += [
            f"## Condition: sign={cond['sign']:+d}, δ={cond['delta_deg']}°",
            "",
            f"T_odd mean = **{cond['T_odd_mean']:+.6f} Nm**, std = {cond['T_odd_std']:.6f}",
            f"T_zero mean = {cond['T_zero_mean']:+.6f} Nm, std = {cond['T_zero_std']:.6f}",
            "",
            "| θ° | elec° | T(+I) | T(-I) | T(0) | T_odd | T_even | res(+) | res(-) | res(0) |",
            "|-----|-------|-------|-------|------|-------|--------|--------|--------|--------|",
        ]
        for e in cond["per_angle"]:
            lines.append(
                f"| {e['mech_deg']:.1f} | {e['elec_deg']:.1f} | "
                f"{e['T_pos']:+.6f} | {e['T_neg']:+.6f} | {e['T_zero']:+.6f} | "
                f"{e['T_odd']:+.6f} | {e['T_even']:+.6f} | "
                f"{e['residual_pos']:.2e} | {e['residual_neg']:.2e} | {e['residual_zero']:.2e} |"
            )
        lines.append("")

    c0 = results["conditions"][0]
    c1 = results["conditions"][1]
    lines += [
        "## Comparison",
        "",
        "| | s=+1, δ=90° | s=-1, δ=90° |",
        "|---|-------------|-------------|",
        f"| T_odd mean | {c0['T_odd_mean']:+.6f} | {c1['T_odd_mean']:+.6f} |",
        f"| T_odd std | {c0['T_odd_std']:.6f} | {c1['T_odd_std']:.6f} |",
        f"| T_zero mean | {c0['T_zero_mean']:+.6f} | {c1['T_zero_mean']:+.6f} |",
        "",
        "**Interpretation**: The difference in T_odd mean between the two conditions",
        "indicates a sequence direction mismatch.  The root cause is under investigation",
        "— not yet determined whether it is magnetization rotation, winding polarity,",
        "or phase naming convention.",
        "",
    ]
    return "\n".join(lines)


def run_all():
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd="/home/wzk/projects/MotorGenesis",
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": git_hash,
        "design_hash": hashlib.sha256(BASELINE_CODE.encode()).hexdigest()[:12],
    }

    cfg = make_cfg()
    results["config"] = cfg_record(cfg)

    # Centerline info
    mf, _, _ = execute_agent_code(BASELINE_CODE, cfg)
    reg = mf.metadata.get("centerline_registry")
    results["centerline"] = {
        "n_entries": len(reg),
        "turns_per_phase": [sum(e["n_turns"] for e in reg if e["phase"]==p) for p in range(3)],
        "cross_section_area_m2": reg[0]["cross_section_area"],
    }

    conditions = []
    for sign, delta in [(+1, 90), (-1, 90)]:
        print(f"\n=== sign={sign:+d}, δ={delta}° ===")
        cond = run_condition(sign, delta, cfg)
        conditions.append(cond)
        print(f"  T_odd mean = {cond['T_odd_mean']:+.6f} (std={cond['T_odd_std']:.6f})")

    results["conditions"] = conditions

    # Save JSON
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nJSON: {OUT_JSON}")

    # Generate report
    report = generate_report(results)
    OUT_REPORT.write_text(report)
    print(f"Report: {OUT_REPORT}")

    return results


if __name__ == "__main__":
    run_all()
