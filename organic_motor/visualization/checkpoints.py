"""Stable, viewer-independent optimization checkpoint format."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from organic_motor.config import MotorConfig


FIELD_NAMES = (
    "rho_air", "rho_iron", "rho_copper", "rho_pm",
    "Mx", "My", "az", "Bx", "By", "Jz", "loss_copper", "loss_iron",
    "loss_total", "temperature",
)


class CheckpointWriter:
    """Write complete material, electromagnetic and metric history snapshots."""

    def __init__(self, output_dir: str | Path, cfg: MotorConfig):
        self.root = Path(output_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        config = asdict(cfg)
        (self.root / "manifest.json").write_text(json.dumps({
            "schema": "motor-genesis.morphogenesis/v1",
            "materials": ["air", "iron", "copper", "pm"],
            "field_names": list(FIELD_NAMES),
            "config": config,
        }, indent=2), encoding="utf-8")

    def write(self, step: int, temperature: float, frame, history: dict) -> Path:
        arrays = {name: np.asarray(getattr(frame, name)) for name in FIELD_NAMES}
        arrays["step"] = np.asarray(step, dtype=np.int32)
        # Keep solver temperature field and density-softmax continuation
        # temperature distinct in the serialized schema.
        arrays["softmax_temperature"] = np.asarray(temperature, dtype=np.float32)
        for key, values in history.items():
            arrays[f"metric__{key}"] = np.asarray(values, dtype=np.float32)
        path = self.root / f"step_{step:06d}.npz"
        np.savez_compressed(path, **arrays)
        return path


def checkpoint_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    directory = root if root.name == "checkpoints" else root / "checkpoints"
    return sorted(directory.glob("step_*.npz"))
