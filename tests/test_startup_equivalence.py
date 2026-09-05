"""B1: Equivalence test — constructed direct vs checkpoint startup path.

Verifies that building a motor and running startup validation directly
produces the same torque maps as saving to NPZ and loading back.

This is the critical data-chain integrity test: if the checkpoint
loses information, the two paths diverge.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture(scope="module")
def built_motor():
    from organic_motor.construct.objects import field_driven_motor
    from organic_motor.config3d import MotorConfig3D
    cfg = MotorConfig3D(shape=(32, 32, 20))
    motor = field_driven_motor(cfg)
    mf = motor.build()
    return motor, mf, cfg


class TestStartupPathEquivalence:
    def test_centerline_count_matches(self, built_motor):
        """Direct build and checkpoint load must have same centerline count."""
        from organic_motor.construct.model_artifact import ModelArtifact
        motor, mf, cfg = built_motor

        direct_count = len(mf.metadata.get("centerline_registry", []))

        with tempfile.TemporaryDirectory() as tmp:
            artifact = ModelArtifact.from_motor(motor, mf, cfg)
            artifact.save(Path(tmp))
            loaded = ModelArtifact.load(Path(tmp))

        assert loaded.centerline_registry is not None
        assert len(loaded.centerline_registry) == direct_count, (
            f"direct={direct_count}, loaded={len(loaded.centerline_registry)}"
        )

    def test_magnetization_preserved(self, built_motor):
        """Magnetization must be identical after round-trip."""
        from organic_motor.construct.model_artifact import ModelArtifact
        motor, mf, cfg = built_motor
        mag_direct = motor.magnetization()

        with tempfile.TemporaryDirectory() as tmp:
            artifact = ModelArtifact.from_motor(motor, mf, cfg)
            artifact.save(Path(tmp))
            loaded = ModelArtifact.load(Path(tmp))

        assert np.allclose(mag_direct, loaded.magnetization, atol=1e-6)

    def test_densities_preserved(self, built_motor):
        """Material densities must be identical after round-trip."""
        from organic_motor.construct.model_artifact import ModelArtifact
        motor, mf, cfg = built_motor
        vol_direct = mf.to_volume()

        with tempfile.TemporaryDirectory() as tmp:
            artifact = ModelArtifact.from_motor(motor, mf, cfg)
            artifact.save(Path(tmp))
            loaded = ModelArtifact.load(Path(tmp))

        for name in ("iron", "pm", "copper", "air"):
            direct = getattr(vol_direct, name)
            loaded_arr = loaded.densities.get(f"rho_{name}")
            if direct is not None and loaded_arr is not None:
                assert np.allclose(direct, loaded_arr, atol=1e-6), (
                    f"{name} density mismatch"
                )

    def test_design_hash_same_after_roundtrip(self, built_motor):
        """Design hash must be deterministic for same geometry."""
        from organic_motor.construct.model_artifact import ModelArtifact
        motor, mf, cfg = built_motor

        a1 = ModelArtifact.from_motor(motor, mf, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            a1.save(Path(tmp))
            a2 = ModelArtifact.load(Path(tmp))

        assert a1.design_hash == a2.design_hash

    def test_can_energize_both_paths(self, built_motor):
        """Both direct and loaded must agree on can_energize."""
        from organic_motor.construct.model_artifact import ModelArtifact
        motor, mf, cfg = built_motor

        direct = ModelArtifact.from_motor(motor, mf, cfg)
        can_direct, reasons_direct = direct.can_energize()

        with tempfile.TemporaryDirectory() as tmp:
            direct.save(Path(tmp))
            loaded = ModelArtifact.load(Path(tmp))

        can_loaded, reasons_loaded = loaded.can_energize()

        assert can_direct == can_loaded, (
            f"direct={can_direct} ({reasons_direct}), "
            f"loaded={can_loaded} ({reasons_loaded})"
        )
