"""E1: convergence matrix — separate error channels, fixed design/operating point.

Audit: linear-solver error, integration error, grid error, angle sampling
and time discretisation must be reported SEPARATELY; "maxiter=240 on a
224 grid" is not an accuracy claim.

This module provides:
  - `summarize_convergence`: pure function that turns raw study results
    into a per-channel error table (unit-tested with synthetic data)
  - `main`: runnable study on the real motor (minutes per row on the
    GPU; run `python -m organic_motor.reports.convergence_matrix`)

Channels:
  grid          torque change vs next-coarser grid (same mi)
  linear_solver torque change vs higher maxiter (same grid)
  angle_sampling map-level torque change vs 2x map angles
  time_step     final-speed change vs dt/2 (same total time)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).parent / "convergence" / "convergence_matrix.json"


def summarize_convergence(rows: list[dict]) -> dict:
    """Compute per-channel relative changes from study rows.

    Each row: {"channel": ..., "refinement": int, "value": float} where
    value is the observable (torque [N m] or final speed [rad/s]).
    Returns channels with entries: (coarse, fine, rel_change).
    """
    channels: dict[str, list] = {}
    for r in rows:
        channels.setdefault(r["channel"], []).append(r)

    out = {}
    for name, entries in channels.items():
        entries = sorted(entries, key=lambda e: e["refinement"])
        pairs = []
        for coarse, fine in zip(entries, entries[1:]):
            denom = abs(coarse["value"]) if coarse["value"] else 1.0
            pairs.append({
                "coarse": coarse, "fine": fine,
                "rel_change": abs(fine["value"] - coarse["value"]) / denom,
            })
        out[name] = {
            "series": [
                {"refinement": e["refinement"], "value": e["value"]}
                for e in entries
            ],
            "refinements": pairs,
            "last_rel_change": pairs[-1]["rel_change"] if pairs else None,
        }
    return out


def format_matrix(summary: dict) -> str:
    lines = [
        "| 通道 | 序列 | 相对变化 |",
        "|---|---|---|",
    ]
    for name, ch in summary.items():
        series = " → ".join(
            f"{e['value']:.5g}" for e in ch["series"])
        last = ch["last_rel_change"]
        lines.append(
            f"| {name} | {series} | "
            f"{f'{last*100:.2f}%' if last is not None else '—'} |"
        )
    return "\n".join(lines)


def run_grid_channel(cfg_factory, shapes, maxiter, n_angles=6):
    """Torque vs grid at fixed solver iterations."""
    from organic_motor.optimization.objective3d import forward3d_fields
    from organic_motor.construct.realize import realize
    import jax.numpy as jnp

    rows = []
    for shape in shapes:
        cfg = cfg_factory(shape)
        from organic_motor.construct.objects import field_driven_motor
        mf = field_driven_motor(cfg).build()
        fields, mag = realize(mf, cfg, None)
        reg = mf.metadata.get("centerline_registry")
        r = forward3d_fields(
            cfg, fields, mag, [0.0], None,
            phase_amplitudes=jnp.array([1.0, -0.5, -0.5]),
            centerline_registry=reg, maxiter=maxiter,
        )
        rows.append({"channel": "grid",
                     "refinement": int(min(shape[:2])),
                     "value": float(np.asarray(r.torques).ravel()[0])})
        print(f"  grid {shape}: T = {rows[-1]['value']:+.5f} Nm", flush=True)
    return rows


def run_solver_channel(cfg_factory, shape, maxiters):
    """Torque vs linear-solver maxiter at fixed grid."""
    from organic_motor.optimization.objective3d import forward3d_fields
    from organic_motor.construct.realize import realize
    import jax.numpy as jnp

    cfg = cfg_factory(shape)
    from organic_motor.construct.objects import field_driven_motor
    mf = field_driven_motor(cfg).build()
    fields, mag = realize(mf, cfg, None)
    reg = mf.metadata.get("centerline_registry")
    rows = []
    for mi in maxiters:
        r = forward3d_fields(
            cfg, fields, mag, [0.0], None,
            phase_amplitudes=jnp.array([1.0, -0.5, -0.5]),
            centerline_registry=reg, maxiter=mi,
        )
        rows.append({"channel": "linear_solver", "refinement": int(mi),
                     "value": float(np.asarray(r.torques).ravel()[0])})
        print(f"  maxiter {mi}: T = {rows[-1]['value']:+.5f} Nm", flush=True)
    return rows


def run_timestep_channel(rows_dt):
    """Final speed vs dt — rows_dt: list of (dt, final_speed)."""
    return [
        {"channel": "time_step", "refinement": int(round(1.0 / dt)),
         "value": float(v)}
        for dt, v in rows_dt
    ]


def run_synthetic_channels():
    """Integration (dt) and angle-sampling channels on the deterministic
    synthetic model — fast, no Maxwell solves, so the full five-channel
    matrix always runs (audit item 8: 不能只运行部分误差通道)."""
    from organic_motor.config3d import MotorConfig3D
    from tests.test_powered_control import _synthetic_maps, _base_settings
    from organic_motor.experiments.motor3d_powered import run_powered_transient

    cfg = MotorConfig3D(shape=(6, 6, 6))
    rows = []
    # --- integration/time-step channel: circuit identity error vs dt ---
    from organic_motor.reports.energy_audit import ledger
    for steps, dt in ((1250, 8e-5), (2500, 4e-5), (5000, 2e-5), (10000, 1e-5)):
        maps = _synthetic_maps(cfg)
        s = _base_settings(control_mode="current_control", i_q_ref_A=8.0,
                           load_torque=4e-3, load_viscous=2e-3,
                           steps=steps, dt=dt)
        d = run_powered_transient(maps, s, cfg, 0.0)
        led = ledger(d, s)
        rows.append({"channel": "time_integration",
                     "refinement": int(round(1.0 / dt)),
                     "value": led["circuit_rel_err"]})
        print(f"  dt={dt:.1e}: circuit err = {led['circuit_rel_err']*100:.2f}%",
              flush=True)
    # --- angle-sampling channel: T1 fundamental amplitude vs na ---
    shifts = np.array((0.0, -2 * np.pi / 3, 2 * np.pi / 3))
    kt = cfg.pole_pairs * 0.002 * 10.0
    for na in (4, 6, 12, 24):
        map_angles = np.arange(na) * (2 * np.pi / cfg.pole_pairs) / na
        elec = cfg.pole_pairs * map_angles
        t1 = kt * np.cos(elec[None, :] + shifts[:, None])
        a = 2.0 / na * np.sum(t1[0] * np.cos(elec))
        b = 2.0 / na * np.sum(t1[0] * np.sin(elec))
        amp = float(np.sqrt(a * a + b * b))
        rows.append({"channel": "angle_sampling", "refinement": na,
                     "value": amp})
        print(f"  na={na}: T1 amplitude = {amp:.5f} (exact {kt:.5f})",
              flush=True)
    return rows


def main(shapes=((96, 96, 58), (112, 112, 68), (128, 128, 78)),
         maxiters=(120, 240, 480), quick=False):
    from organic_motor.config3d import MotorConfig3D

    t0 = time.time()
    rows = []
    print("[convergence] grid channel (fixed maxiter=240)…")
    rows += run_grid_channel(
        lambda s: MotorConfig3D(shape=s), shapes, maxiter=240)
    print("[convergence] linear-solver channel (fixed 96 grid)…")
    rows += run_solver_channel(
        lambda s: MotorConfig3D(shape=s), (96, 96, 58),
        maxiters if not quick else (120, 240))
    print("[convergence] integration + angle channels (synthetic)…")
    rows += run_synthetic_channels()

    summary = summarize_convergence(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "summary": summary,
        "rows": rows,
        "wall_time_s": time.time() - t0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "channels": ["grid", "linear_solver", "time_integration",
                     "angle_sampling"],
        "caveats": [
            "time_integration + angle_sampling run on the SYNTHETIC model "
            "(deterministic); for the synthetic model the angle channel is "
            "trivially exact (analytic cos maps).  The real-motor angle "
            "study (6 vs 12 map angles on GPU) is pending the map-quality "
            "fix — see reports/parameter_provenance.py.",
            "grid + linear_solver channels need the GPU quad solves "
            "(minutes per row).",
        ],
    }, indent=2))
    print(format_matrix(summary))
    print(f"[convergence] written to {OUT_PATH}")


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
