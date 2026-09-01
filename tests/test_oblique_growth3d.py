"""Native three-dimensional oblique growth trajectory rendering."""

import json

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.visualization.growth_report3d import generate_growth_report3d
from organic_motor.visualization.volume3d import render_oblique_material_cut


def _state(cfg: MotorConfig3D, shift: float) -> dict[str, np.ndarray]:
    x, y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, cfg.Nx),
        np.linspace(-1.0, 1.0, cfg.Ny),
        np.linspace(-1.0, 1.0, cfg.Nz),
        indexing="ij",
    )
    iron = np.exp(-5.0 * ((x - shift) ** 2 + y**2 + 0.5 * z**2))
    copper = np.exp(-12.0 * (x**2 + (y + 0.45) ** 2 + (z - 0.2) ** 2))
    pm = np.exp(-10.0 * ((x + 0.35) ** 2 + (y - 0.2) ** 2 + z**2))
    solids = np.clip(iron + copper + pm, 0.0, 1.0)
    return {
        "rho_air": 1.0 - solids,
        "rho_iron": iron,
        "rho_copper": copper,
        "rho_pm": pm,
    }


def test_oblique_growth_report_writes_per_step_cuts_and_contact_sheet(tmp_path):
    cfg = MotorConfig3D(shape=(8, 9, 6))
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step, shift in enumerate((-0.2, 0.2), start=1):
        np.savez_compressed(
            checkpoint_dir / f"step_{step:06d}.npz",
            **_state(cfg, shift),
            spacing=np.asarray(cfg.spacing),
            origin=np.asarray(cfg.origin),
            step=np.asarray(step),
        )

    direct = render_oblique_material_cut(
        _state(cfg, 0.0),
        cfg,
        tmp_path / "direct_oblique.png",
        samples=24,
    )
    outputs = generate_growth_report3d(
        checkpoint_dir,
        tmp_path / "report",
        max_frames=2,
        level=0.35,
    )
    manifest = json.loads(
        (tmp_path / "report" / "report3d.json").read_text(encoding="utf-8")
    )

    assert direct.is_file() and direct.stat().st_size > 0
    assert manifest["oblique_timeline"] == "oblique_growth_trajectory3d.png"
    assert len(manifest["oblique_cuts"]) == 2
    assert all((tmp_path / "report" / path).is_file()
               for path in manifest["oblique_cuts"])
    assert (tmp_path / "report" / manifest["oblique_timeline"]).is_file()
    assert outputs
