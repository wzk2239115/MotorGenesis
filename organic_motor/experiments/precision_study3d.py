"""Native three-dimensional grid-convergence study for ``forward3d``.

Every grid samples the same continuous material and magnetisation definition
in physical coordinates.  In particular, both definitions vary axially; this
is not a study of an extruded two-dimensional field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.grid3d import meshgrid3d
from organic_motor.optimization.objective3d import forward3d
from organic_motor.optimization.optimizer3d import estimate_memory3d


Shape3D = tuple[int, int, int]


def parse_shape(text: str) -> Shape3D:
    """Parse ``Nx,Ny,Nz`` for the command line."""
    try:
        shape = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers") from exc
    if len(shape) != 3 or any(value < 3 for value in shape):
        raise argparse.ArgumentTypeError("shape must be Nx,Ny,Nz with values >= 3")
    return shape  # type: ignore[return-value]


def reference_design3d(
    cfg: MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample one continuous, genuinely 3-D reference definition.

    The returned arrays are phase logits, rotor-ownership logits, and raw
    magnetisation vectors.  Axial phase shifts and an axial magnetisation tilt
    make both the material and vector fields vary with ``z``.
    """
    x, y, z = meshgrid3d(cfg)
    cx, cy, cz = cfg.center
    alpha = jnp.arctan2(y - cy, x - cx)
    axial_scale = max(cfg.rotor_half_length, 1e-12)
    zeta = (z - cz) / axial_scale

    material_twist = 0.55 * jnp.pi * zeta
    pm = 0.50 + 0.16 * jnp.cos(
        2.0 * cfg.pole_pairs * alpha + material_twist
    ) + 0.05 * jnp.sin(jnp.pi * zeta)
    copper = 0.24 + 0.07 * jnp.cos(
        6.0 * alpha - 0.35 * jnp.pi * zeta
    )
    iron = 0.66 + 0.08 * jnp.cos(
        2.0 * alpha + 0.25 * jnp.pi * zeta
    )
    air = 0.10 + 0.03 * jnp.cos(jnp.pi * zeta)
    phases = jnp.stack((air, iron, copper, pm))
    phases = jnp.clip(phases, 1e-4, None)
    phases = phases / jnp.sum(phases, axis=0, keepdims=True)
    logits = jnp.log(phases)

    rotor_logits = 3.0 + 0.45 * jnp.sin(
        cfg.pole_pairs * alpha + 0.6 * jnp.pi * zeta
    )
    magnetic_angle = (
        alpha
        + jnp.where(jnp.cos(cfg.pole_pairs * alpha) >= 0.0, 0.0, jnp.pi)
        + 0.30 * jnp.pi * zeta
    )
    tilt = 0.28 * jnp.sin(jnp.pi * zeta)
    magnetisation = jnp.stack(
        (
            jnp.cos(magnetic_angle) * jnp.cos(tilt),
            jnp.sin(magnetic_angle) * jnp.cos(tilt),
            jnp.sin(tilt),
        )
    )
    return logits, rotor_logits, magnetisation


def _angles(count: int, cfg: MotorConfig3D) -> jnp.ndarray:
    return jnp.arange(count) * (2.0 * jnp.pi / (cfg.pole_pairs * count))


def evaluate_shape(
    shape: Shape3D,
    *,
    maxwell_iters: int = 120,
    thermal_iters: int = 240,
    electric_iters: int = 120,
    angles: int = 1,
    torque_samples: int = 32,
    torque_n_z: int = 16,
    torque_n_r: int = 16,
) -> dict:
    """Run terminal-driven native ``forward3d`` once on ``shape``."""
    cfg = MotorConfig3D(
        shape=shape,
        excitation_mode="terminal",
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=angles,
        maxwell_maxiter=maxwell_iters,
        thermal_maxiter=thermal_iters,
        electric_maxiter=electric_iters,
        n_theta=torque_samples,
        torque_n_z=torque_n_z,
        torque_n_r=torque_n_r,
    )
    logits, rotor_logits, magnetisation = reference_design3d(cfg)
    result = forward3d(
        cfg,
        logits,
        rotor_logits,
        magnetisation,
        _angles(angles, cfg),
        cfg.sm_temp_init,
    )

    torque_samples_nm = np.asarray(result.torques, dtype=float)
    cell_volume = cfg.cell_volume
    copper_loss = float(jnp.sum(result.joule_loss) * cell_volume)
    iron_loss = float(jnp.sum(result.iron_loss) * cell_volume)
    values = np.asarray(
        [
            *torque_samples_nm.ravel(),
            result.electric_residual,
            result.maxwell_residual,
            result.thermal_residual,
            copper_loss,
            iron_loss,
            jnp.max(result.temperature),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite forward3d result for shape {shape}")

    memory = estimate_memory3d(shape, angles, with_gradients=False)
    radial_gap = cfg.R_stator_inner - cfg.R_rotor_outer
    return {
        "shape": list(shape),
        "voxels": int(np.prod(shape)),
        "spacing_m": [float(value) for value in cfg.spacing],
        "torque_Nm": float(np.mean(torque_samples_nm)),
        "torque_samples_Nm": torque_samples_nm.tolist(),
        "torque_change_percent": None,
        "torque_change_below_1_percent": None,
        "residuals": {
            "electric": float(result.electric_residual),
            "maxwell": float(result.maxwell_residual),
            "thermal": float(result.thermal_residual),
        },
        "losses_W": {
            "copper": copper_loss,
            "iron": iron_loss,
            "total": copper_loss + iron_loss,
        },
        "max_temperature_C": float(jnp.max(result.temperature)),
        "memory_estimate": memory,
        "airgap_cells": {
            "radial": float(radial_gap / max(cfg.dx, cfg.dy)),
            "axial": float(cfg.axial_airgap / cfg.dz),
        },
        "excitation_mode": cfg.excitation_mode,
    }


def _add_convergence(rows: list[dict]) -> None:
    for previous, current in zip(rows, rows[1:]):
        scale = max(abs(float(previous["torque_Nm"])), 1e-12)
        change = 100.0 * abs(
            float(current["torque_Nm"]) - float(previous["torque_Nm"])
        ) / scale
        current["torque_change_percent"] = change
        current["torque_change_below_1_percent"] = change < 1.0


def _plot_report(report: dict, path: Path) -> None:
    rows = report["results"]
    labels = ["×".join(map(str, row["shape"])) for row in rows]
    index = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(index, [row["torque_Nm"] for row in rows], "o-")
    axes[0, 0].set_ylabel("Mean torque [N·m]")
    axes[0, 0].set_title("Native 3-D torque")

    for key in ("electric", "maxwell", "thermal"):
        axes[0, 1].plot(
            index,
            [max(row["residuals"][key], 1e-20) for row in rows],
            "o-",
            label=key,
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Solver residuals")
    axes[0, 1].legend()

    axes[1, 0].plot(
        index, [row["losses_W"]["total"] for row in rows], "o-", label="loss [W]"
    )
    temperature_axis = axes[1, 0].twinx()
    temperature_axis.plot(
        index,
        [row["max_temperature_C"] for row in rows],
        "s--",
        color="tab:red",
        label="Tmax [°C]",
    )
    axes[1, 0].set_title("Loss and temperature")
    axes[1, 0].set_ylabel("Total loss [W]")
    temperature_axis.set_ylabel("Maximum temperature [°C]")

    changes = [
        np.nan if row["torque_change_percent"] is None
        else row["torque_change_percent"]
        for row in rows
    ]
    axes[1, 1].plot(index, changes, "o-")
    axes[1, 1].axhline(1.0, color="tab:red", linestyle="--", label="1% threshold")
    axes[1, 1].set_ylabel("Change from previous grid [%]")
    axes[1, 1].set_title("Torque grid convergence")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xticks(index, labels, rotation=20)
        axis.set_xlabel("Nx×Ny×Nz")
        axis.grid(alpha=0.25)
    fig.suptitle("MotorGenesis native 3-D grid-convergence study")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_study(
    shapes: Sequence[Shape3D],
    output_dir: str | Path,
    **solver_options: int,
) -> tuple[Path, Path]:
    """Evaluate all grids and write a JSON data report plus PNG summary."""
    if not shapes:
        raise ValueError("at least one shape is required")
    rows = [evaluate_shape(shape, **solver_options) for shape in shapes]
    _add_convergence(rows)
    report = {
        "schema_version": 1,
        "study": {
            "name": "native_3d_grid_convergence",
            "analytic_reference": "axially_twisted_material_and_magnetisation",
            "excitation_mode": "terminal",
            "torque_unit": "N*m",
            "convergence_threshold_percent": 1.0,
        },
        "results": rows,
    }

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "precision_study3d.json"
    image_path = root / "precision_study3d.png"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _plot_report(report, image_path)
    return image_path, json_path


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(16, 16, 8), (24, 24, 12)],
        metavar="Nx,Ny,Nz",
    )
    ap.add_argument("--maxwell-iters", type=int, default=120)
    ap.add_argument("--thermal-iters", type=int, default=240)
    ap.add_argument("--electric-iters", type=int, default=120)
    ap.add_argument("--angles", type=int, default=1)
    ap.add_argument("--torque-samples", type=int, default=32)
    ap.add_argument("--torque-n-z", type=int, default=16)
    ap.add_argument("--torque-n-r", type=int, default=16)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("organic_motor/out/precision_study3d"),
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    positive = (
        args.maxwell_iters,
        args.thermal_iters,
        args.electric_iters,
        args.angles,
        args.torque_samples,
        args.torque_n_z,
        args.torque_n_r,
    )
    if any(value < 1 for value in positive):
        raise SystemExit("iteration, angle, and quadrature counts must be positive")
    image, data = run_study(
        args.shapes,
        args.output_dir,
        maxwell_iters=args.maxwell_iters,
        thermal_iters=args.thermal_iters,
        electric_iters=args.electric_iters,
        angles=args.angles,
        torque_samples=args.torque_samples,
        torque_n_z=args.torque_n_z,
        torque_n_r=args.torque_n_r,
    )
    print(data)
    print(image)


if __name__ == "__main__":
    main()
