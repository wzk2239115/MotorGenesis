"""Reproducible EM ablation experiment — torque error source isolation.

This script is the single source of truth for the ablation report.
It writes raw JSON output to ablation_results.json and prints summary
tables.  All configuration is explicit; no hidden defaults.

Usage:
    MOTORGENESIS_X64=0 python -m organic_motor.reports.ablation.run_ablation

Output:
    organic_motor/reports/ablation/ablation_results.json
    stdout summary tables
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.agent.sandbox import execute_agent_code
from organic_motor.agent.prompt import BASELINE_CODE
from organic_motor.construct.realize import realize
from organic_motor.optimization.objective3d import forward3d_fields


OUT_DIR = Path(__file__).parent
OUT_JSON = OUT_DIR / "ablation_results.json"


def make_cfg(shape=(96, 96, 58), *, maxwell_maxiter=120,
             thermal_maxiter=120, electric_maxiter=60,
             mechanical_angles=1, n_theta=32,
             torque_n_z=16, torque_n_r=16):
    cfg = MotorConfig3D(
        shape=shape,
        excitation_mode="impressed",
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=mechanical_angles,
        maxwell_maxiter=maxwell_maxiter,
        thermal_maxiter=thermal_maxiter,
        electric_maxiter=electric_maxiter,
        n_theta=n_theta,
        torque_n_z=torque_n_z,
        torque_n_r=torque_n_r,
    )
    return cfg


def cfg_record(cfg: MotorConfig3D) -> dict:
    """Auto-generate config record from actual object — no manual numbers."""
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    nx, ny, nz = cfg.shape
    return {
        "shape": list(cfg.shape),
        "spacing_mm": [dx * 1000, dy * 1000, dz * 1000],
        "origin_mm": [ox * 1000, oy * 1000, oz * 1000],
        "node_extent_mm": [(nx - 1) * dx * 1000, (ny - 1) * dy * 1000, (nz - 1) * dz * 1000],
        "maxwell_maxiter": cfg.maxwell_maxiter,
        "thermal_maxiter": cfg.thermal_maxiter,
        "electric_maxiter": cfg.electric_maxiter,
        "mechanical_angles": cfg.mechanical_angles,
        "n_theta": cfg.n_theta,
        "torque_n_z": cfg.torque_n_z,
        "torque_n_r": cfg.torque_n_r,
        "excitation_mode": cfg.excitation_mode,
        "pole_pairs": cfg.pole_pairs,
        "current_density_peak": float(cfg.current_density_peak),
    }


def build_and_solve(cfg, angles, phase_amplitudes=None, zero_copper=False,
                    zero_current=False):
    """Build geometry and solve forward, return full result dict."""
    mf, mag_raw, err = execute_agent_code(BASELINE_CODE, cfg)
    if err:
        return {"error": err}

    if zero_copper:
        import copy
        mf = copy.deepcopy(mf)
        if "copper" in mf.sdfs:
            sdf = mf.sdfs["copper"]
            mf.sdfs["copper"] = type(sdf)(
                np.full_like(sdf.sdf, 1.0), sdf.spacing, sdf.origin,
            )

    fields, mag_arr = realize(mf, cfg, mag_raw)
    reg = mf.metadata.get("centerline_registry")

    if zero_current:
        phase_amplitudes = jnp.asarray([0.0, 0.0, 0.0])

    t0 = time.perf_counter()
    result = forward3d_fields(
        cfg, fields, mag_arr, angles,
        phase_amplitudes=phase_amplitudes,
        centerline_registry=reg,
    )
    elapsed = time.perf_counter() - t0

    torques = np.asarray(result.torques)
    # Extract key B-field statistics
    B = np.asarray(result.flux_density)
    B_mag = np.sqrt(B[..., 0]**2 + B[..., 1]**2 + B[..., 2]**2)

    return {
        "torques": torques.tolist(),
        "torque_mean": float(torques.mean()),
        "torque_std": float(torques.std()),
        "maxwell_residual": float(result.maxwell_residual),
        "thermal_residual": float(result.thermal_residual),
        "electric_residual": float(result.electric_residual),
        "source_divergence_residual": float(result.source_divergence_residual),
        "phase_balance_residual": float(result.phase_balance_residual),
        "B_mag_max": float(B_mag.max()),
        "B_mag_mean": float(B_mag.mean()),
        "B_mag_p95": float(np.percentile(B_mag, 95)),
        "solve_time_s": elapsed,
        "phase_amplitudes": (np.asarray(phase_amplitudes).tolist()
                             if phase_amplitudes is not None
                             else "default_cos"),
    }


def proper_3phase(electrical_angle: float) -> np.ndarray:
    """Standard 3-phase cosine: [cos(ea), cos(ea-2π/3), cos(ea-4π/3)]."""
    offs = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    return np.cos(electrical_angle - offs)


def run_all():
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    cfg_base = make_cfg()
    results["config"] = cfg_record(cfg_base)

    # Build once for reference
    mf, mag_raw, _ = execute_agent_code(BASELINE_CODE, cfg_base)
    reg = mf.metadata.get("centerline_registry")
    results["centerline"] = {
        "n_entries": len(reg),
        "turns_per_phase": [sum(e["n_turns"] for e in reg if e["phase"] == p) for p in range(3)],
        "cross_section_area_m2": reg[0]["cross_section_area"],
        "band_radius_m": reg[0]["band_radius"],
    }

    # ============================================================
    # E1: Explain baseline torque discrepancy
    # ============================================================
    print("\n=== E1: Phase amplitude comparison (explaining torque discrepancy) ===")
    angles_0 = jnp.asarray([0.0])
    e1 = {}

    # Default: cos(ea - [0, 2π/3, 4π/3]) at ea=0 → [1, -0.5, -0.5]
    r_default = build_and_solve(cfg_base, angles_0, phase_amplitudes=None)
    e1["default_cos"] = r_default
    print(f"  default cos(0-[0,2π/3,4π/3]) = [1, -0.5, -0.5]: torque={r_default['torque_mean']:.6f}")

    # Explicit [1, -0.5, -0.5]
    r_proper = build_and_solve(cfg_base, angles_0,
                               phase_amplitudes=jnp.asarray([1.0, -0.5, -0.5]))
    e1["explicit_1_neg05_neg05"] = r_proper
    print(f"  explicit [1, -0.5, -0.5]:                     torque={r_proper['torque_mean']:.6f}")

    # [1, 1, 1] — what Ablation 3 used
    r_111 = build_and_solve(cfg_base, angles_0,
                            phase_amplitudes=jnp.asarray([1.0, 1.0, 1.0]))
    e1["explicit_1_1_1"] = r_111
    print(f"  explicit [1, 1, 1]:                            torque={r_111['torque_mean']:.6f}")
    print(f"  → discrepancy explained: [1,-0.5,-0.5] ≠ [1,1,1]")

    results["E1_phase_amplitude"] = e1

    # ============================================================
    # E2: Solver convergence at fixed grid/angle/current
    # ============================================================
    print("\n=== E2: Solver convergence (fixed 96×96×58, θ=0, proper 3-phase) ===")
    e2 = []
    amps_proper = jnp.asarray(proper_3phase(0.0))
    for mi in [5, 10, 20, 40, 60, 120, 240, 480]:
        cfg_e2 = make_cfg(maxwell_maxiter=mi, thermal_maxiter=480)
        r = build_and_solve(cfg_e2, angles_0, phase_amplitudes=amps_proper)
        r["maxwell_maxiter"] = mi
        e2.append(r)
        print(f"  maxiter={mi:4d}: torque={r['torque_mean']:.6f}  "
              f"max_res={r['maxwell_residual']:.6e}  "
              f"B_max={r['B_mag_max']:.4f}  "
              f"({r['solve_time_s']:.1f}s)")
    results["E2_solver_convergence"] = e2

    # ============================================================
    # E3: Integration sampling sweep
    # ============================================================
    print("\n=== E3: Torque integration sampling ===")
    e3 = {}
    for label, nz, nr in [("4x4", 4, 4), ("8x8", 8, 8), ("16x16", 16, 16),
                           ("32x32", 32, 32), ("64x64", 64, 64)]:
        cfg_e3 = make_cfg(maxwell_maxiter=480, thermal_maxiter=480,
                          torque_n_z=nz, torque_n_r=nr)
        r = build_and_solve(cfg_e3, angles_0, phase_amplitudes=amps_proper)
        e3[label] = r
        print(f"  nz={nz:2d},nr={nr:2d}: torque={r['torque_mean']:.6f}  "
              f"B_max={r['B_mag_max']:.4f}")
    results["E3_integration_sampling"] = e3

    # ============================================================
    # E4: Per-angle T(+I), T(-I), T(0) → T_odd
    # ============================================================
    print("\n=== E4: Per-angle T(+I), T(-I), T(0) → T_odd ===")
    pp = cfg_base.pole_pairs  # 5
    e4 = []
    n_angles = 12
    for ai in range(n_angles):
        mech_angle = ai * (2.0 * np.pi / (pp * n_angles))
        elec_angle = pp * mech_angle
        angles_ai = jnp.asarray([mech_angle])

        amps_pos = jnp.asarray(proper_3phase(elec_angle))
        amps_neg = jnp.asarray(-proper_3phase(elec_angle))
        amps_zero = jnp.asarray([0.0, 0.0, 0.0])

        cfg_e4 = make_cfg(maxwell_maxiter=480, thermal_maxiter=480,
                          mechanical_angles=1)

        r_pos = build_and_solve(cfg_e4, angles_ai, phase_amplitudes=amps_pos)
        r_neg = build_and_solve(cfg_e4, angles_ai, phase_amplitudes=amps_neg)
        r_zero = build_and_solve(cfg_e4, angles_ai, phase_amplitudes=amps_zero)

        t_pos = r_pos["torque_mean"]
        t_neg = r_neg["torque_mean"]
        t_zero = r_zero["torque_mean"]
        t_odd = (t_pos - t_neg) / 2.0
        t_even = (t_pos + t_neg) / 2.0

        entry = {
            "angle_idx": ai,
            "mech_angle_deg": mech_angle * 180 / np.pi,
            "elec_angle_deg": elec_angle * 180 / np.pi,
            "phase_amps_pos": amps_pos.tolist(),
            "T_pos": t_pos,
            "T_neg": t_neg,
            "T_zero": t_zero,
            "T_odd": t_odd,
            "T_even": t_even,
            "T_odd_minus_T_zero": t_odd - t_zero,
            "maxwell_residual": r_pos["maxwell_residual"],
        }
        e4.append(entry)
        print(f"  θ={mech_angle * 180 / np.pi:6.1f}°: "
              f"T+={t_pos:+.6f}  T-={t_neg:+.6f}  T0={t_zero:+.6f}  "
              f"T_odd={t_odd:+.6f}  T_even={t_even:+.6f}")

    t_odd_vals = [e["T_odd"] for e in e4]
    t_zero_vals = [e["T_zero"] for e in e4]
    print(f"\n  T_odd:  mean={np.mean(t_odd_vals):+.6f}  std={np.std(t_odd_vals):.6f}  "
          f"range=[{min(t_odd_vals):.6f}, {max(t_odd_vals):.6f}]")
    print(f"  T_zero: mean={np.mean(t_zero_vals):+.6f}  std={np.std(t_zero_vals):.6f}  "
          f"range=[{min(t_zero_vals):.6f}, {max(t_zero_vals):.6f}]")
    results["E4_per_angle_decomposition"] = e4

    # ============================================================
    # E5: Angle refinement at fixed high precision
    # ============================================================
    print("\n=== E5: Angle refinement (fixed 96×96×58, maxiter=480) ===")
    e5 = []
    for n_a in [3, 6, 12, 24, 36]:
        angles_e5 = jnp.asarray(
            np.arange(n_a) * (2.0 * np.pi / (pp * n_a))
        )
        # Build phase amplitudes per angle
        amps_list = [proper_3phase(pp * float(a)) for a in np.asarray(angles_e5)]
        # forward3d_fields handles per-angle amps internally when phase_amplitudes=None
        cfg_e5 = make_cfg(maxwell_maxiter=480, thermal_maxiter=480,
                          mechanical_angles=n_a)
        r = build_and_solve(cfg_e5, angles_e5, phase_amplitudes=None)
        r["n_angles"] = n_a
        e5.append(r)
        print(f"  n_angles={n_a:2d}: torque_mean={r['torque_mean']:+.6f}  "
              f"std={r['torque_std']:.6f}  "
              f"range=[{min(r['torques']):.6f}, {max(r['torques']):.6f}]")
    results["E5_angle_refinement"] = e5

    # ============================================================
    # E6: Grid refinement at fixed high precision
    # ============================================================
    print("\n=== E6: Grid refinement (maxiter=480, θ=0, proper 3-phase) ===")
    e6 = []
    for shape in [(96, 96, 58), (128, 128, 78), (160, 160, 96)]:
        cfg_e6 = make_cfg(shape=shape, maxwell_maxiter=480, thermal_maxiter=480)
        r = build_and_solve(cfg_e6, angles_0, phase_amplitudes=amps_proper)
        r["config"] = cfg_record(cfg_e6)
        e6.append(r)
        print(f"  {str(shape):<16}: torque={r['torque_mean']:+.6f}  "
              f"max_res={r['maxwell_residual']:.6e}  "
              f"B_max={r['B_mag_max']:.4f}  "
              f"({r['solve_time_s']:.1f}s)")
    results["E6_grid_refinement"] = e6

    # ============================================================
    # Save
    # ============================================================
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {OUT_JSON}")
    return results


if __name__ == "__main__":
    run_all()
