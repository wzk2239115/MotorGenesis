"""Reproducible float precision and grid-convergence study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks, meshgrid
from organic_motor.optimization.objective import static_forward


def reference_design(cfg: MotorConfig):
    """Sample one continuous motor-like material design on any grid."""
    X, Y, _ = meshgrid(cfg)
    alpha = jnp.arctan2(Y, X)
    masks = domain_masks(cfg)
    rotor = masks["rotor_design"].astype(X.dtype)
    stator = masks["stator_design"].astype(X.dtype)
    winding = masks["winding"].astype(X.dtype)

    pm = rotor * (0.65 + 0.20 * jnp.cos(2 * cfg.pole_pairs * alpha) ** 2)
    copper = stator * winding * (0.28 + 0.08 * jnp.cos(6 * alpha) ** 2)
    iron = jnp.clip(0.72 * stator + rotor * (0.92 - pm), 0.0, 0.95)
    air = jnp.clip(1.0 - iron - copper - pm, 1e-5, 1.0)
    phases = jnp.stack([air, iron + 1e-5, copper + 1e-5, pm + 1e-5])
    phases = phases / phases.sum(axis=0, keepdims=True)
    z = jnp.log(phases)
    polarity = jnp.where(jnp.cos(cfg.pole_pairs * alpha) >= 0.0, 0.0, jnp.pi)
    theta = alpha + polarity
    return z, theta


def evaluate(n: int) -> dict:
    cfg = MotorConfig(N=n, filt_radius=0.0, projection_beta=0.0,
                      maxwell_maxiter=max(500, 5 * n),
                      thermal_maxiter=max(600, 12 * n))
    z, theta = reference_design(cfg)
    fr = static_forward(cfg, z, theta, temperature=1.0)
    bmag = jnp.hypot(fr.Bx, fr.By)
    return {
        "N": n,
        "h_mm": cfg.h * 1e3,
        "dtype": str(fr.az.dtype),
        "torque_Nm_per_m": float(fr.tau),
        "Bmax_T": float(jnp.max(bmag)),
        "Tmax_C": float(jnp.max(fr.temperature)),
        "maxwell_residual": float(fr.maxwell_residual),
        "thermal_residual": float(fr.thermal_residual),
    }


def _worker(n: int) -> None:
    print(json.dumps(evaluate(n)))


def _run_worker(n: int, x64: bool) -> dict:
    env = os.environ.copy()
    env["MOTORGENESIS_X64"] = "1" if x64 else "0"
    proc = subprocess.run(
        [sys.executable, "-m", "organic_motor.experiments.precision_study",
         "--worker", "--N", str(n)],
        check=True, capture_output=True, text=True, env=env,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_study(grids: list[int], output_dir: str | Path) -> tuple[Path, Path]:
    rows = []
    for n in grids:
        rows.append({"precision": "float32", **_run_worker(n, False)})
        rows.append({"precision": "float64", **_run_worker(n, True)})

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "precision_study.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="#090b10")
    metrics = [
        ("torque_Nm_per_m", "Torque N·m/m"),
        ("Bmax_T", "Peak |B| T"),
        ("Tmax_C", "Maximum temperature °C"),
        ("maxwell_residual", "Maxwell relative residual"),
    ]
    colors = {"float32": "#f59e42", "float64": "#55c2ff"}
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.set_facecolor("#0d1118")
        for precision in ("float32", "float64"):
            selected = [r for r in rows if r["precision"] == precision]
            ax.plot([r["h_mm"] for r in selected], [abs(r[key]) for r in selected],
                    "o-", color=colors[precision], label=precision)
        ax.set_title(title, color="white")
        ax.set_xlabel("grid spacing mm", color="white")
        ax.tick_params(colors="#c9ced8")
        ax.grid(alpha=0.18)
        if "residual" in key:
            ax.set_yscale("log")
        ax.invert_xaxis()
    axes[0, 0].legend(facecolor="#0d1118", labelcolor="white")
    fig.suptitle("MotorGenesis precision and grid convergence", color="white",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    image_path = root / "precision_study.png"
    fig.savefig(image_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return image_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--output-dir", default="organic_motor/out/precision_study")
    args = parser.parse_args()
    if args.worker:
        _worker(args.N)
        return
    image, data = run_study(args.grids, args.output_dir)
    print(image)
    print(data)


if __name__ == "__main__":
    main()
