"""Growth timeline reports for native three-dimensional checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields as dataclass_fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.visualization.checkpoints import checkpoint_paths
from organic_motor.visualization.volume3d import (
    _material_rgb,
    render_engineering_cutaway,
    render_oblique_material_cut,
    render_organic_volume,
)
from organic_motor.geometry.voxel import density3d_to_volume


def _sample(paths: list[Path], maximum: int) -> list[Path]:
    if maximum < 1:
        raise ValueError("max_frames must be >= 1")
    if len(paths) <= maximum:
        return paths
    indices = np.unique(np.linspace(0, len(paths) - 1, maximum).round().astype(int))
    return [paths[index] for index in indices]


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("rho_air", "rho_iron", "rho_copper", "rho_pm")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} misses 3-D phase fields: {missing}")
    shape = data["rho_air"].shape
    if len(shape) != 3 or any(data[key].shape != shape for key in required):
        raise ValueError(f"{path} does not contain consistent native (Nx,Ny,Nz) phases")
    return data


def _config(checkpoint_dir: str | Path, data: dict[str, np.ndarray]) -> MotorConfig3D:
    shape = tuple(int(n) for n in data["rho_air"].shape)
    if "spacing" in data:
        spacing = tuple(float(v) for v in np.asarray(data["spacing"]).ravel())
        box_size = tuple(h * (n - 1) for h, n in zip(spacing, shape))
        origin = tuple(float(v) for v in np.asarray(data.get("origin", (0, 0, 0))).ravel())
        center = tuple(o + 0.5 * length for o, length in zip(origin, box_size))
        return MotorConfig3D(shape=shape, box_size=box_size, center=center)

    root = Path(checkpoint_dir)
    directory = root if root.name == "checkpoints" else root / "checkpoints"
    manifest = directory / "manifest.json"
    values: dict = {}
    if manifest.exists():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        config = document.get("config", {})
        allowed = {field.name for field in dataclass_fields(MotorConfig3D)}
        values = {key: value for key, value in config.items() if key in allowed}
    values["shape"] = shape
    return MotorConfig3D(**values)


def _step(data: dict[str, np.ndarray], fallback: int) -> int:
    return int(np.asarray(data.get("step", fallback)).reshape(()))


def _summary(data: dict[str, np.ndarray]) -> str:
    values = []
    for label, key in (
        ("Tq", "metric__|torque|"),
        ("mass", "metric__mass_kg"),
        ("Tmax", "metric__temperature_max_C"),
    ):
        if key in data and np.asarray(data[key]).size:
            values.append(f"{label} {float(np.asarray(data[key]).ravel()[-1]):.3g}")
    return " · ".join(values)


def _timeline_frame(data, cfg: MotorConfig3D, output: Path, number: int) -> Path:
    vol = density3d_to_volume(data, cfg)
    z = vol.shape[2] // 2
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="#071018")
    ax.imshow(_material_rgb(vol, 2, z), origin="lower", interpolation="bilinear")
    title = f"iteration {_step(data, number)} · native z={z}/{vol.shape[2] - 1}"
    detail = _summary(data)
    ax.set_title(title + (f"\n{detail}" if detail else ""), color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def _write_gif(frames: list[Path], target: Path, duration_ms: int) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("GIF output requires Pillow") from exc
    images = [Image.open(path).convert("RGB") for path in frames]
    images[0].save(
        target, save_all=True, append_images=images[1:], duration=duration_ms, loop=0
    )
    for image in images:
        image.close()
    return target


def generate_growth_report3d(
    checkpoint_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    max_frames: int = 12,
    make_gif: bool = False,
    gif_duration_ms: int = 450,
    level: float = 0.5,
) -> list[Path]:
    """Generate a 3-D checkpoint timeline, detailed sections and optional GIF."""
    paths = checkpoint_paths(checkpoint_dir)
    if not paths:
        raise ValueError(f"No checkpoints found under {checkpoint_dir}")
    selected = _sample(paths, max_frames)
    root = (
        Path(output_dir)
        if output_dir is not None
        else Path(checkpoint_dir).parent / "growth_report3d"
    )
    root.mkdir(parents=True, exist_ok=True)
    frame_dir = root / "frames"
    section_dir = root / "sections"
    oblique_dir = root / "oblique_cuts"

    loaded = [_load(path) for path in selected]
    configs = [_config(checkpoint_dir, data) for data in loaded]

    frames, sections, oblique_cuts = [], [], []
    for number, (path, data, cfg) in enumerate(zip(selected, loaded, configs)):
        stem = f"{number:03d}_{path.stem}"
        frames.append(_timeline_frame(data, cfg, frame_dir / f"{stem}.png", number))
        sections.append(
            render_engineering_cutaway(data, cfg, section_dir / f"{stem}.png")
        )
        detail = _summary(data)
        oblique_cuts.append(
            render_oblique_material_cut(
                data,
                cfg,
                oblique_dir / f"{stem}.png",
                title=(
                    f"Growth iteration {_step(data, number)} · 3-D motor cutaway"
                    + (f"\n{detail}" if detail else "")
                ),
            )
        )

    cols = min(4, len(frames))
    rows = math.ceil(len(frames) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4.25 * rows), squeeze=False,
        facecolor="#071018",
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, data, path, cfg in zip(axes.flat, loaded, selected, configs):
        ax.set_visible(True)
        vol = density3d_to_volume(data, cfg)
        ax.imshow(_material_rgb(vol, 2, vol.shape[2] // 2), origin="lower")
        ax.set_title(f"iteration {_step(data, 0)}", color="white")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("MotorGenesis · native 3-D growth timeline", color="white")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    contact = root / "growth_timeline3d.png"
    fig.savefig(contact, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    oblique_cols = min(4, math.ceil(math.sqrt(len(oblique_cuts))))
    oblique_rows = math.ceil(len(oblique_cuts) / oblique_cols)
    fig, axes = plt.subplots(
        oblique_rows,
        oblique_cols,
        figsize=(6.2 * oblique_cols, 5.4 * oblique_rows),
        squeeze=False,
        facecolor="#05090d",
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, image_path, data in zip(axes.flat, oblique_cuts, loaded):
        ax.set_visible(True)
        ax.imshow(plt.imread(image_path))
        ax.set_title(f"iteration {_step(data, 0)}", color="white")
        ax.set_axis_off()
    fig.suptitle(
        "MotorGenesis · native 3-D motor cutaway growth trajectory",
        color="white",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    oblique_contact = root / "oblique_growth_trajectory3d.png"
    fig.savefig(oblique_contact, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    final_cfg = configs[-1]
    organic = render_organic_volume(
        loaded[-1], final_cfg, root / "latest_organic3d.png", level=level
    )
    outputs = [contact, oblique_contact, organic, *sections, *oblique_cuts]
    gif_path = None
    if make_gif:
        gif_path = _write_gif(frames, root / "growth_timeline3d.gif", gif_duration_ms)
        outputs.append(gif_path)
    manifest = {
        "schema": "motor-genesis.growth-report3d/v1",
        "native_3d": True,
        "shape": list(final_cfg.shape),
        "spacing_m": list(final_cfg.spacing),
        "resolution_stages": [list(cfg.shape) for cfg in configs],
        "source_checkpoints": [str(path) for path in selected],
        "timeline": contact.name,
        "oblique_timeline": oblique_contact.name,
        "oblique_cuts": [str(path.relative_to(root)) for path in oblique_cuts],
        "latest_organic": organic.name,
        "sections": [str(path.relative_to(root)) for path in sections],
        "gif": gif_path.name if gif_path else None,
    }
    report = root / "report3d.json"
    report.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(report)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-duration-ms", type=int, default=450)
    parser.add_argument("--level", type=float, default=0.5)
    args = parser.parse_args()
    for path in generate_growth_report3d(
        args.checkpoint_dir,
        args.output_dir,
        max_frames=args.max_frames,
        make_gif=args.gif,
        gif_duration_ms=args.gif_duration_ms,
        level=args.level,
    ):
        print(path)


if __name__ == "__main__":
    main()
