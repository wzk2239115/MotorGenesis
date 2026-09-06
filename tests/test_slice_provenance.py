"""C8 tests: field slice provenance — source labels, grid guard, not-computed."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from organic_motor.web.builder import field_slice


def _write_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture
def run_dir(tmp_path):
    ckpt = tmp_path / "checkpoints" / "step_000000.npz"
    _write_npz(ckpt, {
        "spacing": np.array([1e-3, 1e-3, 1e-3], dtype=np.float32),
        "origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "rho_iron": np.random.rand(8, 8, 6).astype(np.float32),
    })
    return tmp_path, ckpt


class TestSliceProvenance:
    def test_checkpoint_field_labeled(self, run_dir):
        d, ckpt = run_dir
        out = field_slice(ckpt, "rho_iron")
        assert out["source"] == "checkpoint"
        assert "本步数据" in out["source_label"]

    def test_physics_fallback_labeled(self, run_dir):
        d, ckpt = run_dir
        final = d / "final_simulation3d.npz"
        _write_npz(final, {
            "spacing": np.array([1e-3, 1e-3, 1e-3], dtype=np.float32),
            "temperature": np.random.rand(8, 8, 6).astype(np.float32) * 80,
        })
        out = field_slice(ckpt, "temperature", fallback_npz=final)
        assert out["source"] == "final_simulation3d"
        assert "参考场" in out["source_label"]

    def test_missing_field_raises_not_computed(self, run_dir):
        d, ckpt = run_dir
        with pytest.raises(ValueError, match="not computed"):
            field_slice(ckpt, "temperature")

    def test_grid_mismatch_refused(self, run_dir):
        """Fallback on a different grid must be refused (no mixed grids)."""
        d, ckpt = run_dir
        final = d / "final_simulation3d.npz"
        _write_npz(final, {
            "spacing": np.array([1e-3, 1e-3, 1e-3], dtype=np.float32),
            "temperature": np.random.rand(16, 16, 12).astype(np.float32) * 80,
        })
        with pytest.raises(ValueError, match="grid"):
            field_slice(ckpt, "temperature", fallback_npz=final)
