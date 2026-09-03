"""Build renderable GLB meshes and field slices from optimization checkpoints.

A checkpoint ``step_*.npz`` written by :mod:`organic_motor.optimization.optimizer3d`
stores the continuous material densities, spacing, origin and per-step metrics.
This module turns those densities into watertight, Taubin-smoothed GLB meshes
that a WebGL viewer can render directly, plus 2-D field slices and metric
histories for the timeline UI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from organic_motor.geometry.export import MATERIAL_COLORS, material_mesh
from organic_motor.geometry.voxel import VoxelVolume


MATERIALS = ("iron", "copper", "pm", "coolant")
MATERIAL_LABEL = {
    "iron": "铁 Iron",
    "copper": "铜 Copper",
    "pm": "磁钢 PM",
    "coolant": "冷却液 Coolant",
}


@dataclass
class CheckpointInfo:
    step: int
    path: Path
    mtime: float


def list_checkpoints(checkpoint_dir: Path) -> list[CheckpointInfo]:
    """Return all ``step_*.npz`` under a run's checkpoint directory."""
    root = Path(checkpoint_dir)
    if root.name != "checkpoints":
        root = root / "checkpoints"
    infos: list[CheckpointInfo] = []
    if not root.is_dir():
        return infos
    for path in sorted(root.glob("step_*.npz")):
        try:
            step = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        infos.append(CheckpointInfo(step=step, path=path, mtime=path.stat().st_mtime))
    return infos


def _load_volume(npz_path: Path) -> VoxelVolume:
    with np.load(npz_path, allow_pickle=False) as data:
        spacing = tuple(float(v) for v in np.asarray(data["spacing"]).ravel())
        origin = tuple(float(v) for v in np.asarray(data["origin"]).ravel())
        iron = np.asarray(data["rho_iron"], dtype=np.float32)
        pm = np.asarray(data["rho_pm"], dtype=np.float32)
        copper = (
            np.asarray(data["rho_copper"], dtype=np.float32)
            if "rho_copper" in data.files
            else None
        )
        air = (
            np.asarray(data["rho_air"], dtype=np.float32)
            if "rho_air" in data.files
            else None
        )
        coolant = (
            np.asarray(data["rho_coolant"], dtype=np.float32)
            if "rho_coolant" in data.files
            else None
        )
    return VoxelVolume(
        iron=iron, pm=pm, spacing=spacing, origin=origin, copper=copper, air=air,
        coolant=coolant,
    )


def checkpoint_to_glb(
    npz_path: Path,
    *,
    level: float = 0.35,
    smoothing: Literal["none", "taubin", "laplacian"] = "taubin",
    smoothing_iterations: int = 5,
    materials: Iterable[str] = MATERIALS,
) -> bytes:
    """Build a single GLB containing one smoothed mesh per requested material.

    The exported meshes are watertight isosurfaces with displacement-limited
    Taubin smoothing, so the viewer shows smooth motor geometry rather than
    voxel blocks even at modest grid resolution.
    """
    import trimesh

    vol = _load_volume(npz_path)
    scene = trimesh.Scene()
    for material in materials:
        if material not in vol.materials:
            continue
        mesh = material_mesh(
            vol,
            material,
            level=level,
            smoothing=smoothing,
            smoothing_iterations=smoothing_iterations,
        )
        if mesh is None:
            continue
        scene.add_geometry(mesh, node_name=material, geom_name=material)
    if len(scene.geometry) == 0:
        return trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=[[0, 1, 2]]).export(
            file_type="glb"
        )
    return scene.export(file_type="glb")


PHYSICS_FIELDS = {"temperature", "Bmag", "B", "Jmag", "J"}


def _load_fields(npz_path: Path) -> tuple[dict, tuple, tuple]:
    """Load all arrays plus spacing/origin from an NPZ checkpoint."""
    with np.load(npz_path, allow_pickle=False) as data:
        files = {key: np.asarray(data[key]) for key in data.files}
        spacing = tuple(float(v) for v in np.asarray(data["spacing"]).ravel()) \
            if "spacing" in data.files else ()
        origin = tuple(float(v) for v in np.asarray(data["origin"]).ravel()) \
            if "origin" in data.files else ()
    return files, spacing, origin


def field_slice(
    npz_path: Path,
    field: str,
    axis: int = 2,
    index: int | None = None,
    fallback_npz: Path | None = None,
) -> dict:
    """Return a 2-D slice of a scalar field plus its physical extents.

    Density and ownership fields live in every growth checkpoint.  Physics
    fields (temperature, |B|, |J|) are only produced by the final forward
    solve in ``final_simulation3d.npz``; pass that path as ``fallback_npz``
    and the spacing/origin are taken from the checkpoint so extents stay
    consistent across the timeline.
    """
    files, spacing, origin = _load_fields(npz_path)
    if field in PHYSICS_FIELDS and fallback_npz is not None and field not in _available_names(files):
        phys_files, _, _ = _load_fields(fallback_npz)
        files.update(phys_files)
    array = _select_field(files, field)
    if not spacing:
        spacing = (1.0, 1.0, 1.0)
    if not origin:
        origin = (0.0, 0.0, 0.0)
    shape = array.shape
    if index is None:
        index = shape[axis] // 2
    index = int(max(0, min(index, shape[axis] - 1)))
    slab = np.take(array, index, axis=axis)
    slab = np.swapaxes(slab, 0, 1) if axis == 2 else slab
    finite = slab[np.isfinite(slab)]
    if finite.size:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    extents = _slice_extents(shape, spacing, origin, axis)
    return {
        "field": field,
        "axis": axis,
        "index": index,
        "shape": list(slab.shape),
        "values": slab.astype(np.float32).tolist(),
        "vmin": vmin,
        "vmax": vmax,
        "extents": extents,
    }


def _available_names(files: dict) -> set[str]:
    """Names available in a loaded checkpoint mapping."""
    return set(files.keys())


def _select_field(files: dict, field: str) -> np.ndarray:
    direct = {
        "temperature": ("temperature", "T"),
        "rho_iron": ("rho_iron",),
        "rho_copper": ("rho_copper",),
        "rho_pm": ("rho_pm",),
        "rho_air": ("rho_air",),
        "rotor_ownership": ("rotor_ownership",),
    }
    if field in direct:
        for key in direct[field]:
            if key in files:
                return np.asarray(files[key], dtype=np.float32)
    if field in ("Bmag", "B") and "B" in files:
        b = np.asarray(files["B"], dtype=np.float32)
        return np.linalg.norm(b, axis=-1) if b.ndim == 4 and b.shape[-1] == 3 else b
    if field in ("Bmag", "B") and all(k in files for k in ("Bx", "By", "Bz")):
        comps = [np.asarray(files[k], dtype=np.float32) for k in ("Bx", "By", "Bz")]
        return np.sqrt(sum(c * c for c in comps))
    if field in ("Jmag", "J") and "J" in files:
        j = np.asarray(files["J"], dtype=np.float32)
        return np.linalg.norm(j, axis=-1) if j.ndim == 4 and j.shape[-1] == 3 else j
    if field in ("Jmag", "J") and all(k in files for k in ("Jx", "Jy", "Jz")):
        comps = [np.asarray(files[k], dtype=np.float32) for k in ("Jx", "Jy", "Jz")]
        return np.sqrt(sum(c * c for c in comps))
    raise KeyError(f"field {field!r} is not present in this checkpoint")


def _slice_extents(shape, spacing, origin, axis: int) -> dict:
    axes = [i for i in range(3) if i != axis]
    u, v = axes
    return {
        "u0": origin[u],
        "v0": origin[v],
        "u1": origin[u] + spacing[u] * (shape[u] - 1),
        "v1": origin[v] + spacing[v] * (shape[v] - 1),
        "w0": origin[axis],
        "w1": origin[axis] + spacing[axis] * (shape[axis] - 1),
        "w": float(origin[axis] + spacing[axis] * (shape[axis] // 2)),
        "spacing": list(spacing),
    }


def checkpoint_metrics(npz_path: Path) -> dict:
    """Return the latest scalar metrics and the full history stored in a checkpoint."""
    with np.load(npz_path, allow_pickle=False) as data:
        step = int(np.asarray(data["step"]).ravel()[0]) if "step" in data.files else -1
        metrics = {}
        for key in data.files:
            if key.startswith("metric__"):
                arr = np.asarray(data[key]).ravel()
                metrics[key[len("metric__"):]] = float(arr[-1]) if arr.size else 0.0
        history: dict[str, list[float]] = {}
        if "history_json" in data.files:
            try:
                history = json.loads(str(np.asarray(data["history_json"])))
            except (ValueError, TypeError):
                history = {}
        spacing = (
            [float(v) for v in np.asarray(data["spacing"]).ravel()]
            if "spacing" in data.files
            else []
        )
    return {"step": step, "metrics": metrics, "history": history, "spacing": spacing}


def run_summary(run_dir: Path) -> dict:
    """Summarize one run directory: checkpoints, meshes and a metrics timeline."""
    run_dir = Path(run_dir)
    checkpoints = list_checkpoints(run_dir)
    timeline: list[dict] = []
    if checkpoints:
        latest = checkpoint_metrics(checkpoints[-1].path)
        history = latest.get("history", {})
        steps = history.get("step", list(range(len(checkpoints))))
        for name in history:
            if name == "step":
                continue
            values = history[name]
            timeline.append(
                {
                    "name": name,
                    "values": [float(v) for v in values[-len(steps):]],
                }
            )
        timeline_steps = [int(s) for s in steps[-len(checkpoints):]]
    else:
        latest = {"step": -1, "metrics": {}, "history": {}, "spacing": []}
        timeline_steps = []
    meshes_dir = run_dir / "meshes"
    meshes = sorted(p.name for p in meshes_dir.glob("*")) if meshes_dir.is_dir() else []
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "checkpoint_steps": [c.step for c in checkpoints],
        "checkpoint_count": len(checkpoints),
        "latest_step": latest["step"],
        "latest_metrics": latest["metrics"],
        "timeline_steps": timeline_steps,
        "timeline": timeline,
        "meshes": meshes,
    }


def list_runs(out_root: Path) -> list[dict]:
    """List candidate run directories under an output root."""
    root = Path(out_root)
    runs: list[dict] = []
    if not root.is_dir():
        return runs
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        has_ckpts = (child / "checkpoints").is_dir() and any(
            (child / "checkpoints").glob("step_*.npz")
        )
        has_meshes = (child / "meshes").is_dir()
        has_npz = any(child.glob("*.npz"))
        if has_ckpts or has_meshes or has_npz:
            runs.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "has_checkpoints": has_ckpts,
                    "has_meshes": has_meshes,
                }
            )
    return runs
