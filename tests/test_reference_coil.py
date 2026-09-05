"""F3: Reference coil validation test — analytical Biot-Savart comparison."""

import numpy as np
import pytest
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.field import SDFVoxelField
from organic_motor.construct.realize import realize
from organic_motor.optimization.objective3d import forward3d_fields


class TestReferenceCoil:
    """Validate current deposit against analytical Biot-Savart for a circular loop.

    B_z on axis = μ0 * I * R^2 / (2 * (R^2 + z^2)^(3/2))
    """

    @classmethod
    def setup_class(cls):
        cls.R = 0.020
        cls.mu0 = 4e-7 * np.pi
        cls.cfg = MotorConfig3D(
            shape=(64, 64, 64), excitation_mode="impressed",
            filt_radius=0.0, projection_beta=0.0,
            maxwell_maxiter=240, thermal_maxiter=10,
            electric_maxiter=10, n_theta=16,
            torque_n_z=4, torque_n_r=4, mechanical_angles=1,
        )
        dx, dy, dz = cls.cfg.spacing
        ox, oy, oz = cls.cfg.origin
        cx, cy, cz = cls.cfg.center
        cls.I_per_turn = cls.cfg.current_density_peak * 1e-6

        n_pts = 64
        angles = np.linspace(0, 2 * np.pi, n_pts + 1)
        pts = np.column_stack([
            cx + cls.R * np.cos(angles),
            cy + cls.R * np.sin(angles),
            np.full(n_pts + 1, cz),
        ])
        cls.reg = [{
            "points": pts, "physical_points": pts[:-1],
            "turn_map": np.ones(n_pts + 1, dtype=int),
            "phase": 0, "polarity": 1, "n_turns": 1, "tooth": 0,
            "cross_section_area": 1e-6, "band_radius": 0.001,
            "solver_closure": False,
        }]

        mf = MaterialField(shape=cls.cfg.shape, spacing=cls.cfg.spacing,
                           origin=cls.cfg.origin)
        mf.add(SDFVoxelField(np.full(cls.cfg.shape, 1.0, dtype=np.float32),
                            cls.cfg.spacing, cls.cfg.origin), "air")
        cls.mag = np.zeros((3,) + cls.cfg.shape, dtype=np.float32)
        cls.fields, _ = realize(mf, cls.cfg, cls.mag)
        cls.a0 = jnp.asarray([0.0])

        cls.result = forward3d_fields(
            cls.cfg, cls.fields, cls.mag, cls.a0,
            centerline_registry=cls.reg,
        )
        cls.Bz = np.asarray(cls.result.flux_density)[..., 2]

    def _analytical_bz(self, z):
        return self.mu0 * self.I_per_turn * self.R**2 / (
            2 * (self.R**2 + z**2)**1.5
        )

    def _numerical_bz(self, z_mm):
        dz = self.cfg.spacing[2]
        oz = self.cfg.origin[2]
        cz = self.cfg.center[2]
        z_vox = int((cz + z_mm * 1e-3 - oz) / dz)
        ix, iy = self.cfg.shape[0] // 2, self.cfg.shape[1] // 2
        if 0 <= z_vox < self.cfg.shape[2]:
            return float(self.Bz[ix, iy, z_vox])
        return None

    def test_magnitude_near_loop(self):
        """B_z at z=10mm must match analytical within 5%."""
        B_num = self._numerical_bz(10)
        B_ana = self._analytical_bz(0.010)
        err = abs(B_num - B_ana) / abs(B_ana) * 100
        assert err < 5.0, f"B_z error {err:.1f}% > 5%"

    def test_current_reversal(self):
        """Reversing current must negate B."""
        reg_neg = [dict(self.reg[0])]
        reg_neg[0]["polarity"] = -1
        r_neg = forward3d_fields(
            self.cfg, self.fields, self.mag, self.a0,
            centerline_registry=reg_neg,
        )
        B_neg = float(np.asarray(r_neg.flux_density)[..., 2][
            self.cfg.shape[0]//2, self.cfg.shape[1]//2,
            int((self.cfg.center[2] + 0.010 - self.cfg.origin[2]) / self.cfg.spacing[2])
        ])
        B_pos = self._numerical_bz(10)
        ratio = B_neg / max(abs(B_pos), 1e-15)
        assert abs(ratio + 1.0) < 0.05, f"reversal ratio {ratio:.3f} ≠ -1"

    def test_current_scaling(self):
        """Doubling current must double B."""
        r2x = forward3d_fields(
            self.cfg, self.fields, self.mag, self.a0,
            phase_amplitudes=jnp.asarray([2.0, 0.0, 0.0]),
            centerline_registry=self.reg,
        )
        B_2x = float(np.asarray(r2x.flux_density)[..., 2][
            self.cfg.shape[0]//2, self.cfg.shape[1]//2,
            int((self.cfg.center[2] + 0.010 - self.cfg.origin[2]) / self.cfg.spacing[2])
        ])
        B_1x = self._numerical_bz(10)
        ratio = B_2x / max(B_1x, 1e-15)
        assert abs(ratio - 2.0) < 0.05, f"scaling ratio {ratio:.3f} ≠ 2"
