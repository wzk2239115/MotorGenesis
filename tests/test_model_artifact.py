"""Tests for ModelArtifact: save/load round-trip, energization gate, centerline recovery.

A5: Serialization round-trip — densities, magnetization, centerlines, volumes preserved.
A6: validate_from_checkpoint — centerline_registry recovered from NPZ, not silently zeroed.
A7: can_energize() gate — missing magnetization/centerlines rejected, preview allowed.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.model_artifact import ModelArtifact


@pytest.fixture(scope="module")
def built_motor():
    """Build a real motor with magnetization and centerlines."""
    from organic_motor.construct.objects import field_driven_motor
    cfg = MotorConfig3D(shape=(32, 32, 20))
    motor = field_driven_motor(cfg)
    mf = motor.build()
    return motor, mf, cfg


@pytest.fixture(scope="module")
def saved_artifact(built_motor):
    """Save motor to a temp dir and return (artifact, path)."""
    motor, mf, cfg = built_motor
    with tempfile.TemporaryDirectory() as tmp:
        artifact = ModelArtifact.from_motor(motor, mf, cfg)
        artifact.save(Path(tmp))
        yield artifact, Path(tmp)


class TestModelArtifactSave:
    def test_creates_npz_and_json(self, saved_artifact):
        artifact, path = saved_artifact
        assert (path / "checkpoints" / "step_000000.npz").exists()
        assert (path / "model_meta.json").exists()

    def test_densities_saved(self, saved_artifact):
        artifact, path = saved_artifact
        assert "rho_iron" in artifact.densities
        assert "rho_copper" in artifact.densities
        assert "rho_pm" in artifact.densities

    def test_magnetization_nonzero(self, saved_artifact):
        """A real PM motor must have non-zero magnetization."""
        artifact, _ = saved_artifact
        assert artifact.has_magnetization, "PM motor should have magnetization"
        assert artifact.magnetization.max() > 0

    def test_centerlines_present(self, saved_artifact):
        """A real winding must have centerline entries."""
        artifact, _ = saved_artifact
        assert len(artifact.centerline_registry) > 0

    def test_design_hash_stable(self, saved_artifact):
        artifact, _ = saved_artifact
        assert len(artifact.design_hash) > 0

    def test_motion_groups(self, saved_artifact):
        artifact, _ = saved_artifact
        assert "rotor" in artifact.motion_groups
        assert "stator" in artifact.motion_groups
        assert len(artifact.motion_groups["rotor"]) > 0


class TestModelArtifactLoad:
    def test_round_trip_densities(self, saved_artifact):
        """Loaded densities must match original."""
        artifact, path = saved_artifact
        loaded = ModelArtifact.load(path)
        for name in artifact.densities:
            assert np.allclose(loaded.densities[name], artifact.densities[name]), \
                f"{name} mismatch after round-trip"

    def test_round_trip_magnetization(self, saved_artifact):
        """Loaded magnetization must match original."""
        artifact, path = saved_artifact
        loaded = ModelArtifact.load(path)
        assert np.allclose(loaded.magnetization, artifact.magnetization)

    def test_round_trip_centerlines(self, saved_artifact):
        """Loaded centerline registry must have same count and phases."""
        artifact, path = saved_artifact
        loaded = ModelArtifact.load(path)
        assert len(loaded.centerline_registry) == len(artifact.centerline_registry)
        for orig, loaded_cl in zip(artifact.centerline_registry, loaded.centerline_registry):
            assert orig["phase"] == loaded_cl["phase"]
            assert np.allclose(orig["points"], loaded_cl["points"])

    def test_round_trip_design_hash(self, saved_artifact):
        """Design hash must be identical after round-trip."""
        artifact, path = saved_artifact
        loaded = ModelArtifact.load(path)
        assert loaded.design_hash == artifact.design_hash

    def test_round_trip_spacing_origin(self, saved_artifact):
        artifact, path = saved_artifact
        loaded = ModelArtifact.load(path)
        assert np.allclose(loaded.spacing, artifact.spacing, atol=1e-6)
        assert np.allclose(loaded.origin, artifact.origin, atol=1e-6)

    def test_round_trip_motion_groups(self, saved_artifact):
        artifact, path = saved_artifact
        loaded = ModelArtifact.load(path)
        assert loaded.motion_groups == artifact.motion_groups

    def test_winding_change_invalidates_hash(self, built_motor):
        """Audit item 4: different winding connection -> different hash
        (cache keyed on the hash must not be reused)."""
        motor, mf, cfg = built_motor
        a1 = ModelArtifact.from_motor(motor, mf, cfg)
        # Perturb ONE centerline (e.g. reversed polarity = different
        # electrical connection, identical densities).
        mf2_metadata = dict(mf.metadata)
        import copy
        reg2 = copy.deepcopy(mf.metadata["centerline_registry"])
        reg2[0]["polarity"] = -reg2[0]["polarity"]
        mf.metadata["centerline_registry"] = reg2
        a2 = ModelArtifact.from_motor(motor, mf, cfg)
        mf.metadata["centerline_metadata"] = mf2_metadata.get("centerline_metadata")
        mf.metadata["centerline_registry"] = mf2_metadata["centerline_registry"]
        assert a1.design_hash != a2.design_hash, (
            "winding change must invalidate the design hash"
        )
        assert np.allclose(a1.densities["rho_iron"], a2.densities["rho_iron"])


class TestEnergizationGate:
    def test_complete_motor_can_energize(self, saved_artifact):
        """A real PM motor with centerlines must pass the gate."""
        artifact, _ = saved_artifact
        can, reasons = artifact.can_energize()
        assert can, f"Should be energizable but: {reasons}"

    def test_missing_magnetization_rejected(self):
        """Zero magnetization must fail energization."""
        from organic_motor.construct.material import MaterialField
        cfg = MotorConfig3D(shape=(16, 16, 10))
        mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
        artifact = ModelArtifact.from_material_field(mf, cfg, magnetization=None)
        can, reasons = artifact.can_energize()
        assert not can
        assert any("magnetization" in r for r in reasons)

    def test_missing_centerlines_rejected(self, built_motor):
        """Motor without centerlines must fail energization."""
        motor, mf, cfg = built_motor
        mag = motor.magnetization()
        if mag is None:
            mag = np.zeros((3,) + cfg.shape, dtype=np.float32)
        artifact = ModelArtifact(
            densities={"rho_iron": np.zeros(cfg.shape, np.float32),
                       "rho_pm": np.zeros(cfg.shape, np.float32),
                       "rho_copper": np.zeros(cfg.shape, np.float32)},
            spacing=cfg.spacing, origin=cfg.origin, shape=cfg.shape,
            magnetization=mag, centerline_registry=[],
            motion_groups={"rotor": [], "stator": []}, config_dict={},
            design_hash="test", has_magnetization=True, has_centerlines=False,
        )
        can, reasons = artifact.can_energize()
        assert not can
        assert any("centerline" in r for r in reasons)

    def test_preview_allowed_without_energization(self):
        """Missing data allows preview mode (load geometry only)."""
        cfg = MotorConfig3D(shape=(16, 16, 10))
        artifact = ModelArtifact(
            densities={"rho_iron": np.zeros(cfg.shape, np.float32),
                       "rho_pm": np.zeros(cfg.shape, np.float32)},
            spacing=cfg.spacing, origin=cfg.origin, shape=cfg.shape,
            magnetization=np.zeros((3,) + cfg.shape, np.float32),
            centerline_registry=[],
            motion_groups={"rotor": [], "stator": []}, config_dict={},
            design_hash="test", has_magnetization=False, has_centerlines=False,
        )
        can, reasons = artifact.can_energize()
        assert not can
        assert artifact.to_volume() is not None


class TestCheckpointCompatibility:
    def test_npz_loadable_by_web_builder(self, saved_artifact):
        """The saved NPZ must be loadable by the existing web builder._load_volume."""
        from organic_motor.web.builder import _load_volume
        artifact, path = saved_artifact
        npz = path / "checkpoints" / "step_000000.npz"
        vol = _load_volume(npz)
        assert vol.iron is not None
        assert vol.pm is not None

    def test_npz_has_step_key(self, saved_artifact):
        """NPZ must have a step key for the web viewer timeline."""
        artifact, path = saved_artifact
        npz = path / "checkpoints" / "step_000000.npz"
        with np.load(npz) as data:
            assert "step" in data.files
