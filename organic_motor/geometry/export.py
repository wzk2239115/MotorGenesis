"""Watertight mesh and metadata export for continuous 3-D material phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from skimage import measure

from organic_motor.geometry.voxel import VoxelVolume


MATERIAL_COLORS = {
    "iron": (92, 116, 138, 255),
    "copper": (214, 102, 43, 255),
    "pm": (205, 45, 72, 255),
    "coolant": (64, 140, 222, 255),
    "insulator": (235, 232, 224, 255),
}


def _marching_cubes(
    density: np.ndarray,
    spacing,
    origin,
    level: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a closed level set, padding the physical box with void.

    Padding is essential: skimage leaves an isosurface open when material
    reaches an array face.  The one-node void collar closes that surface at
    the represented domain boundary without repeating or extruding any axis.
    """
    field = np.asarray(density, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("density must have shape (Nx, Ny, Nz)")
    if not np.isfinite(field).all():
        raise ValueError("density contains NaN or infinite values")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if field.max(initial=0.0) < level:
        return None
    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
    verts, faces, _normals, _vals = measure.marching_cubes(
        padded, level=level, spacing=spacing, allow_degenerate=False
    )
    verts = verts + np.asarray(origin) - np.asarray(spacing)
    return verts, faces


def material_mesh(
    vol: VoxelVolume,
    material: str,
    *,
    level: float = 0.5,
    smoothing: Literal["none", "taubin", "laplacian"] = "none",
    smoothing_iterations: int = 0,
    smoothing_lambda: float = 0.35,
    max_displacement_fraction: float = 0.35,
):
    """Build one physical phase mesh with optional displacement-limited smoothing.

    Smoothing is off by default.  When enabled, each vertex displacement is
    clamped relative to the unsmoothed isosurface, preventing smoothing from
    inventing geometry farther than a controlled fraction of one grid spacing.
    """
    import trimesh

    if material not in vol.materials:
        raise ValueError(f"material {material!r} is not present")
    result = _marching_cubes(vol.materials[material], vol.spacing, vol.origin, level)
    if result is None:
        return None
    verts, faces = result
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if smoothing_iterations < 0:
        raise ValueError("smoothing_iterations must be non-negative")
    if smoothing not in {"none", "taubin", "laplacian"}:
        raise ValueError("smoothing must be 'none', 'taubin', or 'laplacian'")
    if smoothing != "none" and smoothing_iterations:
        original = mesh.vertices.copy()
        if smoothing == "taubin":
            trimesh.smoothing.filter_taubin(
                mesh, lamb=smoothing_lambda, iterations=smoothing_iterations
            )
        else:
            trimesh.smoothing.filter_laplacian(
                mesh, lamb=smoothing_lambda, iterations=smoothing_iterations
            )
        displacement = mesh.vertices - original
        distance = np.linalg.norm(displacement, axis=1)
        limit = max_displacement_fraction * min(vol.spacing)
        scale = np.minimum(1.0, limit / np.maximum(distance, np.finfo(float).eps))
        mesh.vertices = original + displacement * scale[:, None]
        mesh.fix_normals()
    mesh.visual.face_colors = MATERIAL_COLORS[material]
    return mesh


def export_stl(
    vol: VoxelVolume,
    path: str | Path,
    material: str = "iron",
    **mesh_options,
) -> None:
    """Export one material as STL (legacy signature retained)."""
    mesh = material_mesh(vol, material, **mesh_options)
    if mesh is None:
        _write_empty_stl(path, material)
        return
    mesh.export(str(path))


def _write_empty_stl(path: str | Path, material: str) -> None:
    """Write a valid zero-triangle binary STL (for an absent phase)."""
    import struct
    header = f"empty {material}".encode("ascii")
    header = (header + b"\0" * 80)[:80]
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", 0))


def export_all(vol: VoxelVolume, iron_path: str, pm_path: str) -> None:
    """Legacy iron/PM STL export API."""
    export_stl(vol, iron_path, "iron")
    export_stl(vol, pm_path, "pm")


def export_materials3d(
    vol: VoxelVolume,
    output_dir: str | Path,
    *,
    file_format: Literal["stl", "ply", "glb"] = "stl",
    level: float = 0.5,
    smoothing: Literal["none", "taubin", "laplacian"] = "none",
    smoothing_iterations: int = 0,
    smoothing_lambda: float = 0.35,
    max_displacement_fraction: float = 0.35,
    basename: str = "motor",
) -> dict[str, Path]:
    """Export native iron/copper/PM meshes plus NPZ and JSON metadata."""
    if file_format not in {"stl", "ply", "glb"}:
        raise ValueError("file_format must be stl, ply, or glb")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    mesh_records: dict[str, dict] = {}

    options = dict(
        level=level,
        smoothing=smoothing,
        smoothing_iterations=smoothing_iterations,
        smoothing_lambda=smoothing_lambda,
        max_displacement_fraction=max_displacement_fraction,
    )
    for material in ("iron", "copper", "pm"):
        if material not in vol.materials:
            continue
        mesh = material_mesh(vol, material, **options)
        if mesh is None:
            mesh_records[material] = {"present": False, "watertight": True}
            continue
        target = root / f"{basename}_{material}.{file_format}"
        mesh.export(str(target))
        outputs[material] = target
        mesh_records[material] = {
            "present": True,
            "path": target.name,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "volume_m3": float(abs(mesh.volume)),
        }

    density_path = root / f"{basename}_volume.npz"
    arrays = {f"rho_{key}": value for key, value in vol.materials.items()}
    if vol.air is not None:
        arrays["rho_air"] = np.asarray(vol.air)
    np.savez_compressed(
        density_path,
        **arrays,
        spacing=np.asarray(vol.spacing),
        origin=np.asarray(vol.origin),
        isovalue=np.asarray(level),
    )
    outputs["npz"] = density_path

    metadata = {
        "schema": "motor-genesis.volume3d/v1",
        "native_3d": True,
        "axis_order": ["x", "y", "z"],
        "shape": list(vol.shape),
        "spacing_m": list(vol.spacing),
        "origin_m": list(vol.origin),
        "isovalue": level,
        "smoothing": {
            "method": smoothing,
            "iterations": smoothing_iterations,
            "lambda": smoothing_lambda,
            "max_displacement_fraction": max_displacement_fraction,
        },
        "meshes": mesh_records,
        "densities": density_path.name,
    }
    metadata_path = root / f"{basename}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs["json"] = metadata_path
    return outputs


export_volume3d = export_materials3d
export_all3d = export_materials3d
