"""Real-motor angle-sampling study: T1 maps at na=6 vs na=12 (audit E1).

Runs in the background; writes reports/convergence/angle_study_real.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from organic_motor.construct.model_artifact import ModelArtifact
from organic_motor.config3d import MotorConfig3D
from organic_motor.optimization.objective3d import forward3d_fields
from organic_motor.experiments.motor3d_powered import (
    Powered3DSettings, compute_powered_maps,
)

OUT = Path(__file__).parent / "convergence" / "angle_study_real.json"


def main():
    artifact = ModelArtifact.load("organic_motor/out/assembly")
    import inspect
    valid = set(inspect.signature(MotorConfig3D.__init__).parameters) - {"self"}
    kw = {"shape": tuple(artifact.shape)}
    for k, v in artifact.config_dict.items():
        if k in valid:
            kw[k] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
    cfg = MotorConfig3D(**kw)
    fields, mag = artifact.solver_fields(cfg)
    registry = artifact.centerline_registry or None
    s = Powered3DSettings()  # excitation is fixed unit vectors

    def phase_solver(single, angle, amplitudes):
        return forward3d_fields(
            cfg, fields, mag, [angle], single,
            phase_amplitudes=amplitudes, centerline_registry=registry,
        )

    results = {}
    for na in (6, 12):
        t0 = time.time()
        angles = np.linspace(0, 2 * np.pi / cfg.pole_pairs, na, endpoint=False)
        maps = compute_powered_maps(
            cfg, None, None, mag, angles, s,
            phase_solver=phase_solver, include_mechanics=False,
            keep_volumes=False,
        )
        t1 = np.asarray(maps["torques_ph"])
        results[str(na)] = {
            "t1": t1.tolist(),
            "wall_s": time.time() - t0,
        }
        print(f"[angle] na={na} done in {time.time()-t0:.0f}s "
              f"T1_max={np.abs(t1).max():.4f}", flush=True)

    # Compare: interpolate na=12 onto na=6 sample points (fundamental
    # fit of each, plus direct sample comparison at shared angles).
    t1_6 = np.asarray(results["6"]["t1"])   # (3, 6)
    t1_12 = np.asarray(results["12"]["t1"])  # (3, 12)
    # shared angles: na=6 points are every 2nd na=12 point
    t1_12_on6 = t1_12[:, ::2]
    rel = np.abs(t1_12_on6 - t1_6) / np.maximum(np.abs(t1_6), 1e-9)
    # fundamental amplitude per phase
    def fund_amp(t1, na):
        amps = []
        for p in range(3):
            theta = np.arange(na) * (2 * np.pi / 5) / na
            a = 2 / na * np.sum(t1[p] * np.cos(5 * theta))
            b = 2 / na * np.sum(t1[p] * np.sin(5 * theta))
            amps.append(float(np.hypot(a, b)))
        return amps
    amp6 = fund_amp(t1_6, 6)
    amp12 = fund_amp(t1_12, 12)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "real 96^3 assembly artifact (design 6056e6cc), "
                 "solver_fields path",
        "sample_rel_change_max": float(rel.max()),
        "sample_rel_change_mean": float(rel.mean()),
        "fundamental_amplitude_Nm": {"na6": amp6, "na12": amp12},
        "fundamental_rel_change": [
            abs(a12 - a6) / max(abs(a6), 1e-9)
            for a6, a12 in zip(amp6, amp12)
        ],
        "wall_s": {k: v["wall_s"] for k, v in results.items()},
        "note": "T1 sampled at shared angles compares the discretisation; "
                "fundamental amplitudes compare the physical content.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[angle] sample rel change: max {rel.max()*100:.2f}% "
          f"mean {rel.mean()*100:.2f}%")
    print(f"[angle] fundamentals na6 {[f'{a:.5f}' for a in amp6]} vs "
          f"na12 {[f'{a:.5f}' for a in amp12]}")
    print(f"[angle] written {OUT}")


if __name__ == "__main__":
    main()
