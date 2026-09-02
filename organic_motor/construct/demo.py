"""Construct, score and export a baseline surface-PM motor end to end.

    python -m organic_motor.construct.demo --shape 48,48,32 --out organic_motor/out/constructed
    motor-web --out organic_motor/out   # then open the browser, pick "constructed"

The demo proves the LEAP 71-style constructive layer (SDF Booleans +
computational objects) drives the MotorGenesis differentiable critic
(forward3d_fields) and produces a viewer-ready checkpoint, with no gradient
tape involved in the construction step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct import baseline_motor, field_driven_motor, score
from organic_motor.construct.export import export_meshes, save_checkpoint


def _shape(text: str) -> tuple[int, int, int]:
    vals = tuple(int(v.strip()) for v in text.split(","))
    if len(vals) != 3 or any(v < 3 for v in vals):
        raise argparse.ArgumentTypeError("shape must be Nx,Ny,Nz >= 3")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", type=_shape, default=(48, 48, 32))
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--maxwell-iters", type=int, default=120)
    ap.add_argument("--thermal-iters", type=int, default=240)
    ap.add_argument("--electric-iters", type=int, default=120)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "constructed",
    )
    args = ap.parse_args()

    cfg = MotorConfig3D(
        shape=args.shape,
        excitation_mode="terminal",
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=args.angles,
        maxwell_maxiter=args.maxwell_iters,
        thermal_maxiter=args.thermal_iters,
        electric_maxiter=args.electric_iters,
        n_theta=32,
        torque_n_z=16,
        torque_n_r=16,
    )
    motor = field_driven_motor(cfg)

    print("[construct] building LEAP 71 field-driven motor from SDF Booleans...")
    mf = motor.build()
    densities = mf.to_densities()
    for m in ("air", "iron", "copper", "pm"):
        vol = float(densities[m].sum()) / np.prod(cfg.shape)
        print(f"  rho_{m:<6} volume fraction {vol:.3f}")

    print("[critic] running forward3d_fields (JIT compile + solve)...")
    metrics = score(motor, cfg)
    print("[critic] metrics:")
    for key in (
        "obj", "torque", "torque_ripple", "copper_loss_W", "iron_loss_W",
        "loss_W", "temperature_max_C", "vol_iron", "vol_copper", "vol_pm",
        "maxwell_residual", "thermal_residual", "electric_residual",
    ):
        if key in metrics:
            print(f"  {key:<22} {metrics[key]:.4g}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = save_checkpoint(
        mf, cfg, ckpt_dir / "step_000000.npz",
        step=0, metrics=metrics,
        magnetization=motor.magnetization(),
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"[export] checkpoint {ckpt}")
    print("[export] generating smooth GLB meshes...")
    meshes = export_meshes(mf, cfg, out / "meshes", basename="constructed")
    print(f"[export] meshes under {out / 'meshes'}: {list(meshes.values())}")
    print(f"\nNow run:  motor-web --out {out.parent}")
    print("and select the 'constructed' run in the browser.")


if __name__ == "__main__":
    main()
