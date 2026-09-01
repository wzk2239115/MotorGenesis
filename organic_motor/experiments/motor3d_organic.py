"""CPU-friendly entry point for the native 3-D organic motor loop.

Examples
--------
python -m organic_motor.experiments.motor3d_organic validate
python -m organic_motor.experiments.motor3d_organic simulate --shape 14,14,7
python -m organic_motor.experiments.motor3d_organic grow --levels 10,10,5:14,14,7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.export import export_materials3d
from organic_motor.geometry.voxel import density3d_to_volume
from organic_motor.optimization.objective3d import (
    forward3d,
    make_loss3d,
    objective3d,
    phase_terminal_masks3d,
    three_phase_impressed_source3d,
)
from organic_motor.optimization.optimizer3d import estimate_memory3d, optimize3d
from organic_motor.topology.density3d import assemble3d, random_init3d
from organic_motor.visualization.growth_report3d import generate_growth_report3d
from organic_motor.visualization.volume3d import (
    generate_powered_rotation_gif,
    generate_volume3d_views,
)


def _shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or any(value < 3 for value in values):
        raise argparse.ArgumentTypeError("shape must be Nx,Ny,Nz with values >= 3")
    return values


def _levels(text: str) -> list[tuple[int, int, int]]:
    return [_shape(item) for item in text.split(":")]


def _config(args: argparse.Namespace) -> MotorConfig3D:
    cfg = MotorConfig3D(
        shape=args.shape,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        filt_radius=args.filter_radius,
        maxwell_maxiter=args.maxwell_iters,
        thermal_maxiter=args.thermal_iters,
        checkpoint_every=args.checkpoint_every,
        out_dir=str(args.out),
        n_theta=args.torque_samples,
    )
    # Connectivity propagation is deliberately short on the low-grid CPU
    # default; users can increase it independently of the PDE iterations.
    cfg.connectivity_steps = args.connectivity_steps
    return cfg


def _angles(count: int, cfg: MotorConfig3D) -> jnp.ndarray:
    return jnp.arange(count) * (2.0 * jnp.pi / (cfg.pole_pairs * count))


def _initial_design(cfg: MotorConfig3D):
    key = jax.random.PRNGKey(cfg.seed)
    logits, rotor_logits = random_init3d(cfg, key)
    direction = jax.random.normal(
        jax.random.fold_in(key, 1), (3,) + cfg.shape
    )
    return logits, rotor_logits, direction


def _print_metrics(metrics: dict) -> None:
    print(json.dumps({key: float(value) for key, value in metrics.items()}, indent=2))


def _result_data(result) -> dict[str, np.ndarray]:
    return {
        "rho_air": np.asarray(result.rho_air),
        "rho_iron": np.asarray(result.rho_iron),
        "rho_copper": np.asarray(result.rho_copper),
        "rho_pm": np.asarray(result.rho_pm),
        "rotor_ownership": np.asarray(result.rotor_ownership),
        "nu": np.asarray(result.nu),
        "magnetization": np.asarray(result.magnetization),
        "vector_potential": np.asarray(result.vector_potential),
        "B": np.asarray(result.flux_density),
        "J": np.asarray(result.current_density),
        "phase_current_density": np.asarray(result.phase_current_density),
        "torques": np.asarray(result.torques),
        "joule_loss": np.asarray(result.joule_loss),
        "iron_loss": np.asarray(result.iron_loss),
        "temperature": np.asarray(result.temperature),
    }


def _write_3d_artifacts(
    data: dict[str, np.ndarray],
    cfg: MotorConfig3D,
    out: Path,
    *,
    basename: str,
    level: float = 0.35,
) -> None:
    volume = density3d_to_volume(data, cfg)
    export_materials3d(
        volume,
        out / "meshes",
        file_format="ply",
        level=level,
        smoothing="taubin",
        smoothing_iterations=4,
        basename=basename,
    )
    generate_volume3d_views(data, cfg, out / "views", level=level)
    generate_powered_rotation_gif(
        data,
        cfg,
        out / "views" / "powered_rotation3d.gif",
        frames=12,
        level=level,
    )


def validate(cfg: MotorConfig3D, angle_count: int) -> None:
    """Validate geometry/source contracts without running a PDE solve."""
    logits, rotor_logits, _ = _initial_design(cfg)
    fields = assemble3d(logits, rotor_logits, cfg, cfg.sm_temp_init)
    jz, phase_jz = three_phase_impressed_source3d(
        fields.rho_copper, cfg.electrical_phase_offset, cfg
    )
    phase_net = jnp.sum(phase_jz[:, :, :, 0], axis=(1, 2))
    phase_abs = jnp.sum(jnp.abs(phase_jz[:, :, :, 0]), axis=(1, 2))
    terminal_counts = jnp.sum(phase_terminal_masks3d(cfg), axis=(1, 2, 3))
    memory = estimate_memory3d(cfg.shape, angle_count)
    report = {
        **memory,
        "spacing_m": cfg.spacing,
        "radial_airgap_cells": (
            (cfg.R_stator_inner - cfg.R_rotor_outer) / max(cfg.dx, cfg.dy)
        ),
        "phase_terminal_nodes": [int(x) for x in terminal_counts],
        "phase_balance": [
            float(abs(net) / max(float(total), 1e-12))
            for net, total in zip(phase_net, phase_abs)
        ],
        "source_peak_A_per_m2": float(jnp.max(jnp.abs(jz))),
    }
    report["airgap_resolved"] = report["radial_airgap_cells"] >= 1.5
    print(json.dumps(report, indent=2))
    if min(report["phase_terminal_nodes"]) == 0:
        raise RuntimeError("grid is too coarse to represent all three phase terminals")


def simulate(cfg: MotorConfig3D, angle_count: int, out: Path) -> None:
    logits, rotor_logits, direction = _initial_design(cfg)
    result = forward3d(
        cfg,
        logits,
        rotor_logits,
        direction,
        _angles(angle_count, cfg),
        cfg.sm_temp_init,
    )
    _, metrics = objective3d(cfg, result)
    out.mkdir(parents=True, exist_ok=True)
    data = _result_data(result)
    np.savez_compressed(
        out / "simulation3d.npz",
        **data,
    )
    _write_3d_artifacts(data, cfg, out, basename="simulation")
    _print_metrics(metrics)
    print(f"saved {out / 'simulation3d.npz'}")


def grow(
    cfg: MotorConfig3D,
    angle_count: int,
    levels: list[tuple[int, int, int]],
    steps_per_level: int,
    out: Path,
    resume: Path | None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def factory(stage_cfg: MotorConfig3D):
        stage_cfg.connectivity_steps = cfg.connectivity_steps
        return make_loss3d(stage_cfg, _angles(angle_count, stage_cfg))

    result = optimize3d(
        cfg,
        factory,
        jax.random.PRNGKey(cfg.seed),
        levels=levels,
        steps_per_level=steps_per_level,
        checkpoint_dir=out / "checkpoints",
        resume=resume,
        progress=True,
    )
    fields = assemble3d(
        result.logits, result.rotor_logits, cfg, cfg.sm_temp_final
    )
    np.savez_compressed(
        out / "final_design3d.npz",
        logits=np.asarray(result.logits),
        rotor_logits=np.asarray(result.rotor_logits),
        magnetization_raw=np.asarray(result.magnetization_raw),
        rho_air=np.asarray(fields.rho_air),
        rho_iron=np.asarray(fields.rho_iron),
        rho_copper=np.asarray(fields.rho_copper),
        rho_pm=np.asarray(fields.rho_pm),
        history_json=np.asarray(json.dumps(result.history)),
    )
    final = forward3d(
        cfg,
        result.logits,
        result.rotor_logits,
        result.magnetization_raw,
        _angles(angle_count, cfg),
        cfg.sm_temp_final,
    )
    final_data = _result_data(final)
    np.savez_compressed(out / "final_simulation3d.npz", **final_data)
    _write_3d_artifacts(final_data, cfg, out, basename="grown_motor")
    generate_growth_report3d(
        out / "checkpoints",
        out / "growth_report3d",
        max_frames=cfg.growth_report_max_frames,
        level=0.35,
    )
    print(f"saved {out / 'final_design3d.npz'}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("validate", "simulate", "grow"))
    ap.add_argument("--shape", type=_shape, default=(48, 48, 32))
    ap.add_argument("--levels", type=_levels, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--steps-per-level", type=int, default=10)
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--filter-radius", type=float, default=0.0)
    ap.add_argument("--maxwell-iters", type=int, default=120)
    ap.add_argument("--thermal-iters", type=int, default=240)
    ap.add_argument("--torque-samples", type=int, default=32)
    ap.add_argument("--connectivity-steps", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--resume", type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "motor3d_organic",
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.angles < 1:
        raise SystemExit("--angles must be positive")
    cfg = _config(args)
    print(
        f"[motor3d_organic] command={args.command} shape={cfg.shape} "
        f"angles={args.angles} estimated={estimate_memory3d(cfg.shape, args.angles)['estimated_mib']:.1f} MiB"
    )
    if args.command == "validate":
        validate(cfg, args.angles)
    elif args.command == "simulate":
        simulate(cfg, args.angles, args.out)
    else:
        levels = args.levels or [cfg.shape]
        if levels[-1] != cfg.shape:
            levels.append(cfg.shape)
        grow(
            cfg,
            args.angles,
            levels,
            args.steps_per_level,
            args.out,
            args.resume,
        )


if __name__ == "__main__":
    main()
