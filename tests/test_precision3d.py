"""Fast schema smoke test for the native 3-D precision study."""

import json

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.experiments.precision_study3d import (
    reference_design3d,
    run_study,
)


def test_reference_is_axially_varying_and_report_schema_is_complete(tmp_path):
    cfg = MotorConfig3D(shape=(8, 8, 5), filt_radius=0.0)
    logits, _, magnetisation = reference_design3d(cfg)

    assert not np.allclose(np.asarray(logits[..., 0]), np.asarray(logits[..., -1]))
    assert not np.allclose(
        np.asarray(magnetisation[..., 0]), np.asarray(magnetisation[..., -1])
    )

    image_path, json_path = run_study(
        [(7, 7, 4), (8, 8, 5)],
        tmp_path,
        maxwell_iters=2,
        thermal_iters=2,
        electric_iters=2,
        angles=1,
        torque_samples=4,
        torque_n_z=2,
        torque_n_r=2,
    )
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["study"]["excitation_mode"] == "terminal"
    assert report["study"]["torque_unit"] == "N*m"

    row = report["results"][1]
    assert row["shape"] == [8, 8, 5]
    assert row["excitation_mode"] == "terminal"
    assert row["torque_change_percent"] >= 0.0
    assert isinstance(row["torque_change_below_1_percent"], bool)
    assert set(row["residuals"]) == {"electric", "maxwell", "thermal"}
    assert set(row["losses_W"]) == {"copper", "iron", "total"}
    assert set(row["airgap_cells"]) == {"radial", "axial"}
    assert row["memory_estimate"]["voxels"] == 8 * 8 * 5
    assert np.isfinite(row["torque_Nm"])
    assert np.isfinite(row["max_temperature_C"])
    assert image_path.is_file() and image_path.stat().st_size > 0
