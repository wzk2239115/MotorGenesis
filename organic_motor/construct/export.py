"""Export a constructed motor for the web viewer and direct mesh pipelines.

A constructed :class:`MaterialField` is saved in the same checkpoint NPZ
schema the browser viewer already understands, so a constructed motor appears
alongside grown designs with no viewer changes.  Direct PLY/GLB/STL meshes
use the existing Taubin-smoothed :mod:`geometry.export` pipeline so the
geometry is smooth rather than blocky.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.export import export_materials3d
from organic_motor.geometry.voxel import density3d_to_volume
from organic_motor.construct.material import MaterialField


def save_checkpoint(
    mf: MaterialField,
    cfg: MotorConfig3D,
    path: str | Path,
    *,
    step: int = 0,
    metrics: dict | None = None,
    magnetization: np.ndarray | None = None,
) -> Path:
    """Write a viewer-compatible checkpoint NPZ from a constructed motor.

    The file carries ``rho_air/iron/copper/pm``, ``spacing``, ``origin`` and
    optional ``metric__*`` arrays, matching what
    :mod:`organic_motor.web.builder` reads to generate a smooth GLB.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    densities = mf.to_densities()
    arrays = {
        "rho_air": densities["air"],
        "rho_iron": densities["iron"],
        "rho_copper": densities["copper"],
        "rho_pm": densities["pm"],
        "rotor_ownership": np.asarray(
            densities["iron"] > 0.0, dtype=np.float32
        ),
        "spacing": np.asarray(cfg.spacing, dtype=np.float32),
        "origin": np.asarray(cfg.origin, dtype=np.float32),
        "step": np.asarray(step, dtype=np.int32),
    }
    if magnetization is not None:
        arrays["magnetization"] = np.asarray(magnetization, dtype=np.float32)
    if metrics:
        for key, value in metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            arrays[f"metric__{key}"] = np.asarray([scalar], dtype=np.float32)
    np.savez_compressed(target, **arrays)
    return target


def export_meshes(
    mf: MaterialField,
    cfg: MotorConfig3D,
    output_dir: str | Path,
    *,
    level: float = 0.35,
    smoothing: str = "taubin",
    smoothing_iterations: int = 5,
    basename: str = "constructed",
) -> dict:
    """Export Taubin-smoothed watertight meshes (PLY/STL/GLB) per material."""
    volume = density3d_to_volume(
        {
            "rho_iron": mf.to_densities()["iron"],
            "rho_copper": mf.to_densities()["copper"],
            "rho_pm": mf.to_densities()["pm"],
            "rho_air": mf.to_densities()["air"],
        },
        cfg,
    )
    return export_materials3d(
        volume,
        output_dir,
        file_format="glb",
        level=level,
        smoothing=smoothing,
        smoothing_iterations=smoothing_iterations,
        basename=basename,
    )
