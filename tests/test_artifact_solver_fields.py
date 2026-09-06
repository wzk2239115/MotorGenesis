"""Regression: artifact-loaded models must solve at ROTATED angles.

Root cause this guards against (found 2026-09-06): building solver
fields by re-deriving an impostor SDF (0.5 - rho) from artifact
densities and re-smoothing stalls the Maxwell CG at every non-zero
rotor angle (residual 0.3-1.0, garbage torque up to 4935 Nm).  Building
TopologyFields3D DIRECTLY from the saved densities converges (1.3e-5).

The 96^3 reproduction is recorded in reports/diag_angle_sweep.py; this
test checks the construction-level invariant on a small artifact plus
solve-level convergence at a rotated angle.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import field_driven_motor
from organic_motor.construct.model_artifact import ModelArtifact


@pytest.fixture(scope="module")
def artifact_dir():
    cfg = MotorConfig3D(shape=(32, 32, 20), excitation_mode="impressed",
                        filt_radius=0.0, projection_beta=0.0)
    motor = field_driven_motor(cfg)
    mf = motor.build()
    with tempfile.TemporaryDirectory() as tmp:
        art = ModelArtifact.from_motor(motor, mf, cfg)
        art.save(Path(tmp))
        yield Path(tmp), cfg


class TestSolverFields:
    def test_fields_match_saved_densities(self, artifact_dir):
        """solver_fields must carry the artifact densities VERBATIM."""
        d, cfg = artifact_dir
        art = ModelArtifact.load(d)
        fields, mag = art.solver_fields(cfg)
        assert np.allclose(np.asarray(fields.rho_iron),
                           art.densities["rho_iron"], atol=1e-6)
        assert np.allclose(np.asarray(fields.rho_pm),
                           art.densities["rho_pm"], atol=1e-6)
        assert np.allclose(np.asarray(fields.rho_copper),
                           art.densities["rho_copper"], atol=1e-6)
        assert np.allclose(np.asarray(mag), art.magnetization, atol=1e-6)

    def test_rotated_angle_solves_converge(self, artifact_dir):
        """PM-only solve at a NON-ZERO rotor angle must converge (the
        impostor-SDF path stalled at 0.3-1.0 residual on the 96^3 motor;
        on this small grid we assert the healthy <5e-3 band)."""
        from organic_motor.optimization.objective3d import forward3d_fields
        d, cfg = artifact_dir
        art = ModelArtifact.load(d)
        fields, mag = art.solver_fields(cfg)
        zero = jnp.zeros(3)
        for angle_deg in (0.0, 24.0, 60.0):
            r = forward3d_fields(
                cfg, fields, mag, [np.radians(angle_deg)], None,
                phase_amplitudes=zero,
                centerline_registry=art.centerline_registry,
            )
            res = float(np.asarray(r.maxwell_residual).ravel()[0])
            assert res < 5e-3, (
                f"rotated solve at {angle_deg} deg did not converge "
                f"(residual {res:.3e}) — impostor-SDF regression?"
            )

    def test_no_fake_sdf_in_solve_path(self, artifact_dir):
        """The old broken path (realize on a 0.5-rho impostor) must NOT
        be used by validate_from_checkpoint: solver_fields is wired in."""
        import inspect
        from organic_motor.construct import startup_validation as sv
        src = inspect.getsource(sv.validate_from_checkpoint)
        assert "solver_fields" in src and "_mf_from_artifact" not in src
        from organic_motor.web import server as web_server
        src2 = inspect.getsource(web_server)
        assert "solver_fields" in src2
