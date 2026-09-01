"""Interactive PyVista checkpoint viewer for motor morphogenesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from organic_motor.visualization.checkpoints import checkpoint_paths


PHASES = {
    "rho_iron": ("iron", "#73808c", 0.90),
    "rho_copper": ("copper", "#d9772f", 0.62),
    "rho_pm": ("permanent magnet", "#b33a3a", 0.82),
}


def _volume(field: np.ndarray, nz: int = 12) -> np.ndarray:
    return np.repeat(field[:, :, None], nz, axis=2)


def launch(checkpoint_dir: str | Path) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit("Install viewer dependencies: pip install -e '.[morphogenesis]'") from exc

    paths = checkpoint_paths(checkpoint_dir)
    if not paths:
        raise SystemExit(f"No step_*.npz checkpoints found under {checkpoint_dir}")

    plotter = pv.Plotter()
    plotter.set_background("#090b10")
    actors = []

    def render(index: float) -> None:
        nonlocal actors
        for actor in actors:
            plotter.remove_actor(actor)
        actors = []
        data = np.load(paths[int(round(index))])
        for key, (label, color, opacity) in PHASES.items():
            vol = _volume(data[key])
            grid = pv.ImageData(dimensions=np.array(vol.shape) + 1)
            grid.cell_data[label] = vol.flatten(order="F")
            actor = plotter.add_volume(
                grid, scalars=label, cmap=[color], clim=(0.15, 1.0),
                opacity=[0.0, 0.05, opacity], shade=True,
            )
            actors.append(actor)
        thermal = _volume(data["temperature"])
        thermal_grid = pv.ImageData(dimensions=np.array(thermal.shape) + 1)
        thermal_grid.cell_data["temperature_C"] = thermal.flatten(order="F")
        thermal_slice = thermal_grid.slice(normal="z", origin=(0, 0, thermal.shape[2] / 2))
        actors.append(plotter.add_mesh(
            thermal_slice, scalars="temperature_C", cmap="inferno", opacity=0.48,
            show_scalar_bar=True, scalar_bar_args={"title": "Temperature °C"},
        ))
        bx, by = data["Bx"], data["By"]
        bmax = float(np.sqrt(bx * bx + by * by).max())
        tmax = float(data["temperature"].max())
        step = int(data["step"])
        metric = lambda name: float(data[f"metric__{name}"][-1])
        plotter.add_text(
            f"MOTOR ORGAN  |  iteration {step}\n"
            f"torque {metric('|torque|'):.3g} N·m/m   "
            f"mass {metric('mass_kg_per_m'):.3g} kg/m\n"
            f"Fe {metric('vol_iron'):.1%}   Cu {metric('vol_copper'):.1%}   "
            f"PM {metric('vol_pm'):.1%}   |B|max {bmax:.3g} T\n"
            f"loss {metric('loss_W_per_m'):.3g} W/m   "
            f"Tmax {tmax:.1f} °C   η* {metric('efficiency_proxy'):.1%}",
            name="metrics", position="upper_left", color="white", font_size=10,
        )
        plotter.render()

    plotter.add_slider_widget(render, (0, len(paths) - 1), value=0,
                              title="morphogenesis time", interaction_event="always")
    plotter.add_axes()
    render(0)
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()
    launch(args.checkpoint_dir)


if __name__ == "__main__":
    main()
