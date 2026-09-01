"""Phase-1 extension: torque-ripple-aware free-topology motor.

Identical to ``motor2d_basic`` but the torque is evaluated at ``K`` rotor
positions (the rotor region is rotated while the stator stays fixed).  The
objective maximises the mean output torque and penalises its ripple, which
drives the optimiser toward periodic, motor-like structures without imposing
any slot/pole prior.

Usage:
    python -m organic_motor.experiments.motor2d_ripple [--steps 300] [--N 128] [--K 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.optimization.objective import make_ripple_loss, make_snapshot
from organic_motor.optimization.optimizer import optimize
from organic_motor.topology.density import assemble


def main(cfg: MotorConfig, K: int) -> None:
    out = Path(cfg.out_dir or (Path(__file__).parent.parent / "out" / "motor2d_ripple"))
    out.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = str(out)
    cfg.w_ripple = cfg.w_ripple if cfg.w_ripple > 0 else 0.5

    print(f"[motor2d_ripple] grid {cfg.N}x{cfg.N}, K={K} rotor positions, "
          f"{cfg.steps} steps")

    loss = make_ripple_loss(cfg, K)
    result = optimize(cfg, loss, jax.random.PRNGKey(cfg.seed),
                      snapshot_fn=make_snapshot(cfg), plot_every=20,
                      progress=True)

    mat = assemble(result.z, result.theta, cfg, temperature=cfg.sm_temp_final)
    jnp.savez(out / "final_design.npz",
              z=result.z, theta=result.theta,
              rho_iron=mat.rho_iron, rho_copper=mat.rho_copper,
              rho_pm=mat.rho_pm,
              Mx=mat.Mx, My=mat.My)

    from organic_motor.geometry.export import export_all
    from organic_motor.geometry.voxel import densities_to_voxels
    vol = densities_to_voxels(mat.rho_iron, mat.rho_pm, cfg, thickness=0.02)
    export_all(vol, str(out / "iron.stl"), str(out / "pm.stl"))

    if cfg.generate_growth_report:
        from organic_motor.visualization.growth_report import generate_growth_report
        generate_growth_report(out / "checkpoints", out / "growth_report",
                               cfg.growth_report_max_frames)

    print(f"[motor2d_ripple] done. outputs in {out}")
    print(f"  final mean |torque| : {result.history['|torque|'][-1]:.4f} N.m/m")
    print(f"  final relative ripple: {result.history['ripple'][-1]:.4f}")
    print(f"  PM volume fraction  : {result.history['vol_pm'][-1]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = MotorConfig(seed=args.seed)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.N is not None:
        cfg.N = args.N
    main(cfg, args.K)
