"""Render checkpoint sequences into image-based morphogenesis acceptance reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from organic_motor.visualization.checkpoints import checkpoint_paths


MATERIAL_COLORS = {
    "rho_iron": np.array([0.42, 0.53, 0.64]),
    "rho_copper": np.array([0.95, 0.42, 0.10]),
    "rho_pm": np.array([0.90, 0.08, 0.28]),
}


def _sample_paths(paths: list[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return paths
    indices = np.linspace(0, len(paths) - 1, maximum).round().astype(int)
    return [paths[i] for i in np.unique(indices)]


def _material_rgb(data) -> np.ndarray:
    shape = data["rho_air"].shape
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    solid = np.zeros(shape, dtype=np.float32)
    for key, color in MATERIAL_COLORS.items():
        density = np.asarray(data[key], dtype=np.float32)
        rgb += density[..., None] * color
        solid += density
    # Dark tissue background, with continuous density shown as differentiation.
    return np.clip(rgb + (1.0 - np.clip(solid, 0, 1))[..., None] * 0.025, 0, 1)


def _metric(data, name: str, default: float = float("nan")) -> float:
    key = f"metric__{name}"
    return float(data[key][-1]) if key in data and len(data[key]) else default


def _field(data, name: str, shape, default: float = 0.0) -> np.ndarray:
    """Read a field while tolerating checkpoints from earlier schema revisions."""
    if name not in data:
        return np.full(shape, default, dtype=np.float32)
    value = np.asarray(data[name])
    if value.shape == ():
        return np.full(shape, float(value), dtype=np.float32)
    return value


def _summary(data) -> str:
    dtype = str(data["rho_air"].dtype)
    n = data["rho_air"].shape[0]
    return (
        f"Tq {_metric(data, '|torque|'):.3g} N·m/m   "
        f"m {_metric(data, 'mass_kg_per_m'):.2f} kg/m\n"
        f"Fe {_metric(data, 'vol_iron'):.1%}   "
        f"Cu {_metric(data, 'vol_copper'):.1%}   "
        f"PM {_metric(data, 'vol_pm'):.1%}\n"
        f"loss {_metric(data, 'loss_W_per_m'):.1f} W/m   "
        f"Tmax {_metric(data, 'temperature_max_C'):.1f} °C   "
        f"η* {_metric(data, 'efficiency_proxy'):.1%}\n"
        f"{dtype}  N={n}   rEM {_metric(data, 'maxwell_residual'):.1e}   "
        f"rTH {_metric(data, 'thermal_residual'):.1e}"
    )


def _clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#3b4352")
        spine.set_linewidth(0.7)


def render_detail(checkpoint: Path, output: Path) -> None:
    data = np.load(checkpoint)
    step = int(data["step"])
    shape = data["rho_air"].shape
    bmag = np.hypot(_field(data, "Bx", shape), _field(data, "By", shape))
    fields = [
        (_material_rgb(data), "material differentiation", None),
        (bmag, f"magnetic flux |B|  max {bmag.max():.3g} T", "viridis"),
        (_field(data, "temperature", shape, 25.0), "temperature °C", "inferno"),
        (_field(data, "loss_total", shape), "loss density W/m³", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor="#090b10")
    fig.suptitle(f"MotorGenesis morphogenesis · iteration {step}\n{_summary(data)}",
                 color="white", fontsize=12)
    for ax, (field, title, cmap) in zip(axes.flat, fields):
        ax.set_facecolor("#090b10")
        display = np.swapaxes(field, 0, 1) if field.ndim == 3 else field.T
        image = ax.imshow(display, origin="lower", cmap=cmap)
        ax.set_title(title, color="white", fontsize=10)
        _clean_axis(ax)
        if cmap:
            bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            bar.ax.tick_params(colors="#c9ced8", labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_growth_report(checkpoint_dir: str | Path, output_dir: str | Path | None = None,
                           max_frames: int = 12) -> list[Path]:
    """Generate a contact sheet plus detailed per-checkpoint acceptance images."""
    paths = checkpoint_paths(checkpoint_dir)
    if not paths:
        raise ValueError(f"No checkpoints found under {checkpoint_dir}")
    selected = _sample_paths(paths, max_frames)
    root = Path(output_dir) if output_dir else Path(checkpoint_dir).parent / "growth_report"
    details = root / "frames"
    details.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for path in selected:
        target = details / f"{path.stem}.png"
        render_detail(path, target)
        outputs.append(target)

    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 5.0 * rows),
                             squeeze=False, facecolor="#090b10")
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, path in zip(axes.flat, selected):
        ax.set_visible(True)
        data = np.load(path)
        ax.imshow(np.swapaxes(_material_rgb(data), 0, 1), origin="lower")
        ax.set_title(f"iteration {int(data['step'])}", color="white", fontsize=11)
        ax.text(0.02, 0.02, _summary(data), transform=ax.transAxes, color="white",
                fontsize=7.5, va="bottom", ha="left",
                bbox={"facecolor": "#090b10", "alpha": 0.78, "edgecolor": "none"})
        _clean_axis(ax)
    fig.suptitle("MotorGenesis · growth acceptance atlas", color="white", fontsize=15)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.93,
                        wspace=0.045, hspace=0.22)
    contact = root / "growth_contact_sheet.png"
    fig.savefig(contact, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    manifest = root / "report.json"
    manifest.write_text(json.dumps({
        "schema": "motor-genesis.growth-report/v1",
        "contact_sheet": contact.name,
        "source_checkpoints": [str(p) for p in selected],
        "detail_images": [str(p.relative_to(root)) for p in outputs],
    }, indent=2), encoding="utf-8")
    return [contact, *outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    for path in generate_growth_report(args.checkpoint_dir, args.output_dir,
                                       args.max_frames):
        print(path)


if __name__ == "__main__":
    main()
