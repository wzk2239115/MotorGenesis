"""Phase-1 benchmark: 2-D free-topology permanent-magnet motor.

Runs the full loop:
    random init -> material field -> Maxwell solve -> B -> torque
    -> objective -> JAX gradient -> Adam update -> repeat

then renders the final topology, magnetic field, histories and exports the
extruded 3-D topology as STL.

Usage:
    python -m organic_motor.experiments.motor2d_basic [--steps 400] [--N 128]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.optimization.objective import make_snapshot, make_static_loss
from organic_motor.optimization.optimizer import optimize
from organic_motor.topology.density import assemble


def main(cfg: MotorConfig) -> None:
    out = Path(cfg.out_dir or (Path(__file__).parent.parent / "out" / "motor2d_basic"))
    out.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = str(out)

    print(f"[motor2d_basic] grid {cfg.N}x{cfg.N}, rotor < {cfg.R_rotor_outer} m, "
          f"air gap {cfg.R_rotor_outer}..{cfg.R_stator_inner} m, {cfg.steps} steps")

    loss = make_static_loss(cfg)
    result = optimize(cfg, loss, jax.random.PRNGKey(cfg.seed),
                      snapshot_fn=make_snapshot(cfg), plot_every=20,
                      progress=True)

    # --- final state ---
    mat = assemble(result.z, result.theta, cfg,
                   temperature=cfg.sm_temp_final)
    jnp.savez(out / "final_design.npz",
              z=result.z, theta=result.theta,
              rho_iron=mat.rho_iron, rho_copper=mat.rho_copper,
              rho_pm=mat.rho_pm, nu=mat.nu,
              Mx=mat.Mx, My=mat.My)

    # --- STL export ---
    from organic_motor.geometry.export import export_all
    from organic_motor.geometry.voxel import densities_to_voxels
    vol = densities_to_voxels(mat.rho_iron, mat.rho_pm, cfg, thickness=0.02)
    export_all(vol, str(out / "iron.stl"), str(out / "pm.stl"))

    if cfg.generate_growth_report:
        from organic_motor.visualization.growth_report import generate_growth_report
        generate_growth_report(out / "checkpoints", out / "growth_report",
                               cfg.growth_report_max_frames)

    print(f"[motor2d_basic] done. outputs in {out}")
    print(f"  final |torque|       : {result.history['|torque|'][-1]:.4f} N.m/m")
    print(f"  final torque/mass    : {result.history['torque/mass'][-1]:.4e}")
    print(f"  PM volume fraction   : {result.history['vol_pm'][-1]:.3f}")
    print(f"  iron volume fraction : {result.history['vol_iron'][-1]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = MotorConfig(seed=args.seed)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.N is not None:
        cfg.N = args.N
    main(cfg)
