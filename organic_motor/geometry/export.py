"""STL export of the voxelised topology (marching cubes -> trimesh).

Each material phase is exported to its own STL (the STL format is uncoloured);
combine the two meshes downstream for multi-material printing.
"""

from __future__ import annotations

import numpy as np
from skimage import measure

from organic_motor.geometry.voxel import VoxelVolume


def _marching_cubes(mask: np.ndarray, spacing, origin) -> tuple | None:
    if mask.sum() == 0:
        return None
    verts, faces, _normals, _vals = measure.marching_cubes(mask, level=0.5,
                                                           spacing=spacing)
    verts = verts + np.array(origin)
    return verts, faces


def export_stl(vol: VoxelVolume, path: str, material: str = "iron") -> None:
    import trimesh

    mask = vol.iron if material == "iron" else vol.pm
    result = _marching_cubes(mask, vol.spacing, vol.origin)
    if result is None:
        _write_empty_stl(path, material)
        return
    verts, faces = result
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(path)


def _write_empty_stl(path: str, material: str) -> None:
    """Write a valid zero-triangle binary STL (for an absent phase)."""
    import struct
    header = f"empty {material}".encode("ascii")
    header = (header + b"\0" * 80)[:80]
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", 0))


def export_all(vol: VoxelVolume, iron_path: str, pm_path: str) -> None:
    export_stl(vol, iron_path, "iron")
    export_stl(vol, pm_path, "pm")
