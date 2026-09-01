"""Native 3-D material and multiphysics visualisation.

All arrays are interpreted in ``(Nx, Ny, Nz)`` order.  This module never
repeats a two-dimensional slice to manufacture a volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.export import MATERIAL_COLORS, _marching_cubes
from organic_motor.geometry.grid3d import rotate_owned_volume_z
from organic_motor.geometry.voxel import VoxelVolume, density3d_to_volume


def _as_arrays(data) -> dict[str, np.ndarray]:
    if isinstance(data, Mapping):
        return {str(key): np.asarray(value) for key, value in data.items()}
    keys = (
        "rho_air", "rho_iron", "rho_copper", "rho_pm",
        "rotor_ownership",
        "B", "Bx", "By", "Bz", "Bmag",
        "J", "Jx", "Jy", "Jz", "Jmag", "temperature", "T",
    )
    return {
        key: np.asarray(getattr(data, key))
        for key in keys if hasattr(data, key)
    }


def _volume(data, cfg: MotorConfig3D) -> VoxelVolume:
    arrays = _as_arrays(data)
    return density3d_to_volume(arrays, cfg)


def _validate_scalar(field: np.ndarray, shape: tuple[int, int, int], name: str) -> np.ndarray:
    field = np.asarray(field)
    if field.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {field.shape}")
    return field


def vector_magnitude(data, prefix: str) -> np.ndarray | None:
    """Return ``|B|`` or ``|J|`` from explicit native 3-D components."""
    arrays = _as_arrays(data)
    if prefix in arrays:
        vector = np.asarray(arrays[prefix], dtype=np.float64)
        if vector.ndim == 4 and vector.shape[-1] == 3:
            return np.linalg.norm(vector, axis=-1)
        if vector.ndim == 4 and vector.shape[0] == 3:
            return np.linalg.norm(vector, axis=0)
        if vector.ndim == 3:
            return vector
        raise ValueError(f"{prefix} must be a 3-D magnitude or 3-component volume")
    magnitude_key = f"{prefix}mag"
    if magnitude_key in arrays:
        return np.asarray(arrays[magnitude_key], dtype=np.float64)
    names = [f"{prefix}{axis}" for axis in "xyz"]
    if not all(name in arrays for name in names):
        return None
    components = [np.asarray(arrays[name], dtype=np.float64) for name in names]
    if len({component.shape for component in components}) != 1:
        raise ValueError(f"{prefix} components do not share one shape")
    return np.sqrt(sum(component * component for component in components))


def available_physics_fields(data) -> dict[str, np.ndarray]:
    """Collect display-ready ``|B|``, ``|J|`` and temperature volumes."""
    arrays = _as_arrays(data)
    result: dict[str, np.ndarray] = {}
    for prefix, label in (("B", "|B| [T]"), ("J", "|J| [A/m²]")):
        magnitude = vector_magnitude(arrays, prefix)
        if magnitude is not None:
            result[label] = magnitude
    for key in ("temperature", "T"):
        if key in arrays:
            result["T [°C]"] = np.asarray(arrays[key])
            break
    return result


def _slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis).T


def _material_rgb(vol: VoxelVolume, axis: int, index: int) -> np.ndarray:
    shape2 = np.take(np.asarray(vol.iron), index, axis=axis).shape
    rgb = np.full(shape2 + (3,), 0.018, dtype=np.float32)
    for name, density in vol.materials.items():
        color = np.asarray(MATERIAL_COLORS[name][:3], dtype=np.float32) / 255.0
        phase = np.clip(np.take(density, index, axis=axis), 0.0, 1.0)
        rgb += phase[..., None] * color
    return np.clip(np.swapaxes(rgb, 0, 1), 0.0, 1.0)


def render_engineering_cutaway(
    data,
    cfg: MotorConfig3D,
    output: str | Path,
    *,
    indices: Sequence[int] | None = None,
    dpi: int = 170,
) -> Path:
    """Render orthogonal material cuts and available B/J/T field slices."""
    vol = _volume(data, cfg)
    fields = available_physics_fields(data)
    cuts = tuple(indices) if indices is not None else tuple(n // 2 for n in vol.shape)
    if len(cuts) != 3 or any(i < 0 or i >= n for i, n in zip(cuts, vol.shape)):
        raise ValueError("indices must contain valid x, y and z indices")
    rows = 1 + len(fields)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows), squeeze=False)
    fig.patch.set_facecolor("#081018")
    plane_names = ("x cut (yz)", "y cut (xz)", "z cut (xy)")

    for axis, (index, title) in enumerate(zip(cuts, plane_names)):
        ax = axes[0, axis]
        ax.imshow(_material_rgb(vol, axis, index), origin="lower", interpolation="bilinear")
        ax.set_title(f"{title} · index {index}", color="white")
    for row, (name, field) in enumerate(fields.items(), start=1):
        field = _validate_scalar(field, vol.shape, name)
        finite = field[np.isfinite(field)]
        vmin, vmax = (
            (float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
            if finite.size else (0.0, 1.0)
        )
        if vmax <= vmin:
            vmax = vmin + 1.0
        for axis, index in enumerate(cuts):
            image = axes[row, axis].imshow(
                _slice(field, axis, index), origin="lower", cmap="turbo",
                vmin=vmin, vmax=vmax, interpolation="bilinear",
            )
            axes[row, axis].set_title(f"{name} · {plane_names[axis]}", color="white")
        bar = fig.colorbar(image, ax=list(axes[row]), fraction=0.018, pad=0.015)
        bar.ax.tick_params(colors="#d6e2ef", labelsize=7)
    for ax in axes.flat:
        ax.set_facecolor("#081018")
        ax.tick_params(colors="#9fb1c1", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#40566b")
    fig.suptitle("MotorGenesis · native 3-D engineering sections", color="white")
    fig.subplots_adjust(left=0.055, right=0.94, bottom=0.055, top=0.92,
                        wspace=0.20, hspace=0.30)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def render_organic_volume(
    data,
    cfg: MotorConfig3D,
    output: str | Path,
    *,
    level: float = 0.5,
    alpha: float = 0.82,
    elev: float = 24.0,
    azim: float = 38.0,
    dpi: int = 190,
) -> Path:
    """Render translucent continuous-phase isosurfaces with a wet tissue look."""
    vol = _volume(data, cfg)
    fig = plt.figure(figsize=(10, 8), facecolor="#05090d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05090d")
    for name in ("iron", "copper", "pm"):
        result = _marching_cubes(vol.materials[name], vol.spacing, vol.origin, level)
        if result is None:
            continue
        vertices, faces = result
        color = np.asarray(MATERIAL_COLORS[name][:3]) / 255.0
        surface = Poly3DCollection(
            vertices[faces], facecolor=color, edgecolor=color * 0.55,
            linewidth=0.035, alpha=alpha,
        )
        ax.add_collection3d(surface)
    end = np.asarray(vol.origin) + np.asarray(vol.spacing) * (np.asarray(vol.shape) - 1)
    ax.set_xlim(vol.origin[0], end[0])
    ax.set_ylim(vol.origin[1], end[1])
    ax.set_zlim(vol.origin[2], end[2])
    ax.set_box_aspect(np.asarray(cfg.box_size) / max(cfg.box_size))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [m]", color="#a7bdcc")
    ax.set_ylabel("y [m]", color="#a7bdcc")
    ax.set_zlabel("z [m]", color="#a7bdcc")
    ax.tick_params(colors="#7d94a4", labelsize=7)
    ax.grid(False)
    ax.set_title(
        f"Organic material differentiation · native {vol.shape} · ρ={level:g}",
        color="#e7f5ff", pad=18,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return target


def render_oblique_material_cut(
    data,
    cfg: MotorConfig3D,
    output: str | Path,
    *,
    normal: Sequence[float] = (1.0, -0.7, 0.55),
    offset_m: float = 0.0,
    samples: int | None = None,
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    """Remove one oblique half-space and expose the motor's material interior."""
    vol = _volume(data, cfg)
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0.0:
        raise ValueError("normal must be a finite, non-zero three-vector")
    n /= np.linalg.norm(n)
    origin = np.asarray(vol.origin)
    end = origin + np.asarray(vol.spacing) * (np.asarray(vol.shape) - 1)
    axes = [
        origin[index] + vol.spacing[index] * np.arange(vol.shape[index])
        for index in range(3)
    ]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    center = np.asarray(cfg.center) + offset_m * n
    signed_distance = (
        (X - center[0]) * n[0]
        + (Y - center[1]) * n[1]
        + (Z - center[2]) * n[2]
    )
    keep = signed_distance <= 0.0

    fig = plt.figure(figsize=(8.2, 7.2), facecolor="#05090d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05090d")
    cut_tolerance = 1.5 * max(vol.spacing)
    from organic_motor.geometry.domain3d import domain_masks3d
    render_materials = {
        "shaft": (
            np.asarray(domain_masks3d(cfg)["shaft"], dtype=float),
            np.asarray((0.48, 0.54, 0.60)),
        ),
        **{
            name: (
                vol.materials[name],
                np.asarray(MATERIAL_COLORS[name][:3], dtype=float) / 255.0,
            )
            for name in ("iron", "copper", "pm")
        },
    }
    for name, (density, base) in render_materials.items():
        clipped_density = np.where(keep, density, 0.0)
        result = _marching_cubes(
            clipped_density,
            vol.spacing,
            vol.origin,
            level=0.35,
        )
        if result is None:
            continue
        vertices, faces = result
        face_centers = vertices[faces].mean(axis=1)
        face_distance = np.abs(
            np.sum((face_centers - center[None, :]) * n[None, :], axis=1)
        )
        exposed = face_distance <= cut_tolerance
        colors = np.empty((len(faces), 4), dtype=float)
        colors[:, :3] = base * 0.72
        colors[:, 3] = 0.88
        colors[exposed, :3] = np.clip(base * 1.35 + 0.12, 0.0, 1.0)
        colors[exposed, 3] = 1.0
        surface = Poly3DCollection(
            vertices[faces],
            facecolors=colors,
            edgecolors=np.clip(colors[:, :3] * 0.55, 0.0, 1.0),
            linewidth=0.06,
        )
        ax.add_collection3d(surface)

    ax.plot(
        [cfg.center[0], cfg.center[0]],
        [cfg.center[1], cfg.center[1]],
        [origin[2], end[2]],
        color="#e3edf5",
        linewidth=1.2,
        alpha=0.8,
    )

    # Draw the physical simulation box so the removed half-space is obvious.
    corners = np.array(
        [
            [x, y, z]
            for x in (origin[0], end[0])
            for y in (origin[1], end[1])
            for z in (origin[2], end[2])
        ]
    )
    for i, first in enumerate(corners):
        for second in corners[i + 1:]:
            if np.count_nonzero(first != second) == 1:
                ax.plot(
                    *zip(first, second),
                    color="#718596",
                    linewidth=0.7,
                    alpha=0.65,
                )
    ax.set_xlim(origin[0], end[0])
    ax.set_ylim(origin[1], end[1])
    ax.set_zlim(origin[2], end[2])
    ax.set_box_aspect(np.asarray(cfg.box_size) / max(cfg.box_size))
    ax.view_init(elev=24.0, azim=38.0)
    ax.set_xlabel("x [m]", color="#a7bdcc")
    ax.set_ylabel("y [m]", color="#a7bdcc")
    ax.set_zlabel("z [m]", color="#a7bdcc")
    ax.tick_params(colors="#7d94a4", labelsize=7)
    ax.grid(False)
    ax.set_title(
        title or (
            "Native 3-D motor cutaway · oblique half-space removed"
            f"\nn={tuple(np.round(n, 2))}"
        ),
        color="#e7f5ff",
        pad=16,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return target


def generate_volume3d_views(
    data,
    cfg: MotorConfig3D,
    output_dir: str | Path,
    *,
    level: float = 0.5,
) -> dict[str, Path]:
    """Generate engineering and organic views for one native 3-D state."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "engineering": render_engineering_cutaway(
            data, cfg, root / "engineering_sections.png"
        ),
        "organic": render_organic_volume(
            data, cfg, root / "organic_volume.png", level=level
        ),
        "oblique": render_oblique_material_cut(
            data, cfg, root / "oblique_material_cut.png"
        ),
    }


def generate_powered_rotation_gif(
    data,
    cfg: MotorConfig3D,
    output: str | Path,
    *,
    frames: int = 12,
    level: float = 0.5,
    duration_ms: int = 120,
) -> Path:
    """Render a full rigid rotor revolution from native three-dimensional fields."""
    if frames < 2:
        raise ValueError("frames must be at least two")
    arrays = _as_arrays(data)
    ownership = arrays.get("rotor_ownership")
    if ownership is None:
        from organic_motor.geometry.domain3d import domain_masks3d
        ownership = np.asarray(domain_masks3d(cfg)["rotor_design"], dtype=float)
    frame_dir = Path(output).with_suffix("").parent / (
        Path(output).stem + "_frames"
    )
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index in range(frames):
        angle = 2.0 * np.pi * index / frames
        rotated = dict(arrays)
        for name in ("rho_iron", "rho_copper", "rho_pm"):
            field = arrays[name]
            # Copper is normally stator-owned.  Continuous ownership keeps the
            # animation valid for future freely assigned moving conductors.
            rotated[name] = np.asarray(
                rotate_owned_volume_z(field, ownership, angle, cfg)
            )
        rotated["rho_air"] = np.clip(
            1.0
            - rotated["rho_iron"]
            - rotated["rho_copper"]
            - rotated["rho_pm"],
            0.0,
            1.0,
        )
        frame = frame_dir / f"rotation_{index:03d}.png"
        render_organic_volume(rotated, cfg, frame, level=level, azim=38.0 + index)
        frame_paths.append(frame)

    from PIL import Image
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()
    return target


render_volume3d = generate_volume3d_views
render_engineering_sections = render_engineering_cutaway
