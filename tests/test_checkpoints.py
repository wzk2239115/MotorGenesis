"""Morphogenesis checkpoint schema tests."""

import json

import numpy as np

from organic_motor.config import MotorConfig
from organic_motor.visualization.checkpoints import CheckpointWriter
from organic_motor.visualization.growth_report import generate_growth_report


class _Frame:
    pass


def test_checkpoint_preserves_full_cell_and_field_state(tmp_path):
    cfg = MotorConfig(N=8)
    frame = _Frame()
    for name in ("rho_air", "rho_iron", "rho_copper", "rho_pm", "Mx", "My",
                 "az", "Bx", "By", "Jz", "loss_copper", "loss_iron",
                 "loss_total", "temperature"):
        setattr(frame, name, np.ones((8, 8), dtype=np.float32))
    writer = CheckpointWriter(tmp_path, cfg)
    path = writer.write(12, 0.5, frame, {"obj": [2.0, 1.0]})
    saved = np.load(path)
    assert saved["rho_copper"].shape == (8, 8)
    assert saved["temperature"].shape == (8, 8)
    assert saved["softmax_temperature"].shape == ()
    assert saved["metric__obj"].tolist() == [2.0, 1.0]
    manifest = json.loads((tmp_path / "checkpoints" / "manifest.json").read_text())
    assert manifest["schema"] == "motor-genesis.morphogenesis/v1"
    assert manifest["materials"] == ["air", "iron", "copper", "pm"]


def test_growth_report_renders_contact_sheet_and_details(tmp_path):
    cfg = MotorConfig(N=8)
    writer = CheckpointWriter(tmp_path, cfg)
    history = {
        "|torque|": [], "mass_kg_per_m": [], "vol_iron": [],
        "vol_copper": [], "vol_pm": [], "loss_W_per_m": [],
        "temperature_max_C": [], "efficiency_proxy": [],
        "maxwell_residual": [], "thermal_residual": [],
    }
    for step in range(3):
        frame = _Frame()
        x = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        for name in ("rho_air", "rho_iron", "rho_copper", "rho_pm", "Mx", "My",
                     "az", "Bx", "By", "Jz", "loss_copper", "loss_iron",
                     "loss_total", "temperature"):
            setattr(frame, name, x + step)
        for values in history.values():
            values.append(float(step + 1))
        writer.write(step, 1.0, frame, history)

    outputs = generate_growth_report(tmp_path / "checkpoints",
                                     tmp_path / "report", max_frames=2)
    assert len(outputs) == 3
    assert all(path.exists() and path.stat().st_size > 1000 for path in outputs)
    assert (tmp_path / "report" / "report.json").exists()
