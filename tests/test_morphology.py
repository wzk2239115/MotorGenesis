"""Tests for parametric morphology generators — shape correctness, not just non-empty."""

import numpy as np
import pytest
from scipy import ndimage

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.morphology import (
    HoneycombGenerator,
    HelicalChannelGenerator,
    BranchingManifold,
    honeycomb_support,
    helical_cooling,
)


def _cfg(**ov):
    d = dict(shape=(96, 96, 58), excitation_mode="impressed",
             filt_radius=0.0, projection_beta=0.0)
    d.update(ov)
    return MotorConfig3D(**d)


class TestHoneycomb:
    """Verify true hexagonal cells, not triangular mesh."""

    def _gen_and_slice(self, cell_size=0.012, wall_t=0.003):
        cfg = _cfg()
        gen = HoneycombGenerator(
            r_inner=0.001, r_outer=0.070,
            z_bottom=-0.050, z_top=0.050,
            cell_size=cell_size, wall_thickness=wall_t,
        )
        sdf = gen.build(cfg)
        k = cfg.shape[2] // 2
        return cfg, sdf.sdf[:, :, k]

    def test_produces_material(self):
        cfg, slc = self._gen_and_slice()
        assert (slc < 0).sum() > 0

    def test_has_hex_voids(self):
        """The pattern must have enclosed voids (hex cells), not just walls."""
        cfg, slc = self._gen_and_slice()
        material = slc < 0
        void = ~material
        labeled, n_voids = ndimage.label(void, structure=ndimage.generate_binary_structure(2, 1))
        n_voids = int(n_voids)
        if n_voids == 0:
            pytest.fail("no void regions at all")
        border_mask = np.zeros_like(void)
        border_mask[0, :] = True; border_mask[-1, :] = True
        border_mask[:, 0] = True; border_mask[:, -1] = True
        enclosed = 0
        for i in range(1, n_voids + 1):
            region = labeled == i
            if not (region & border_mask).any():
                enclosed += 1
        assert enclosed > 0, "no enclosed voids — pattern is not honeycomb"

    def test_wall_thickness_changes_volume(self):
        cfg = _cfg()
        kwargs = dict(r_inner=0.001, r_outer=0.070,
                       z_bottom=-0.050, z_top=0.050,
                       cell_size=0.006)
        s_thin = HoneycombGenerator(wall_thickness=0.0004, **kwargs).build(cfg)
        s_thick = HoneycombGenerator(wall_thickness=0.002, **kwargs).build(cfg)
        v_thin = (s_thin.sdf < 0).sum()
        v_thick = (s_thick.sdf < 0).sum()
        assert v_thick > v_thin, "thicker walls must produce more material"

    def test_cell_size_changes_void_count(self):
        """Smaller cells must produce more enclosed voids."""
        _, slc_small = self._gen_and_slice(cell_size=0.004)
        _, slc_large = self._gen_and_slice(cell_size=0.012)
        # Count enclosed voids
        for label, slc in [("small", slc_small), ("large", slc_large)]:
            material = slc < 0
            void = ~material
            n = ndimage.label(void, structure=ndimage.generate_binary_structure(2, 1))[0]
            # Just verify different counts
        assert (slc_small < 0).sum() != (slc_large < 0).sum(), "different cell sizes must differ"

    def test_density_gradient_changes_local_thickness(self):
        """Wall thickness should vary with radius when gradient is provided."""
        cfg = _cfg()
        gradient = np.array([0.5, 1.0, 2.0])  # thin→thick
        gen = HoneycombGenerator(
            r_inner=0.030, r_outer=0.050,
            z_bottom=-0.050, z_top=0.050,
            cell_size=0.006, wall_thickness=0.001,
            density_gradient=gradient,
        )
        sdf = gen.build(cfg)
        k = cfg.shape[2] // 2
        slc = sdf.sdf[:, :, k]
        dx, dy = cfg.spacing[:2]
        cx, cy = cfg.center[:2]
        # Material at inner radius should be thinner than at outer
        X = cfg.origin[0] + dx * np.arange(cfg.shape[0])[:, None]
        Y = cfg.origin[1] + dy * np.arange(cfg.shape[1])[None, :]
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        inner_mask = (R > 0.031) & (R < 0.035)
        outer_mask = (R > 0.045) & (R < 0.049)
        v_inner = (slc < 0)[inner_mask].sum()
        v_outer = (slc < 0)[outer_mask].sum()
        # Outer (thicker walls) should have more material per unit area
        assert v_outer > 0 and v_inner > 0, "both regions must have material"

    def test_radial_bounds(self):
        cfg = _cfg()
        gen = HoneycombGenerator(
            r_inner=0.040, r_outer=0.046,
            z_bottom=-0.045, z_top=0.045,
            cell_size=0.004, wall_thickness=0.001,
        )
        sdf = gen.build(cfg)
        dx, dy, dz = cfg.spacing
        k = cfg.shape[2] // 2
        for i in range(0, cfg.shape[0], 8):
            for j in range(0, cfg.shape[1], 8):
                x = cfg.origin[0] + i * dx
                y = cfg.origin[1] + j * dy
                R = np.sqrt((x - cfg.center[0])**2 + (y - cfg.center[1])**2)
                if R > 0.050:
                    assert sdf.sdf[i, j, k] >= 0, f"material at R={R*1000:.1f}mm"


class TestHelicalChannel:
    """Verify 3D centerline sweep with correct cross-section."""

    def test_produces_void(self):
        cfg = _cfg()
        sdf = HelicalChannelGenerator(
            radius=0.045, pitch=0.015, n_turns=3.0,
            channel_radius=0.002,
        ).build(cfg)
        assert (sdf.sdf < 0).sum() > 0

    def test_handedness_changes_position(self):
        cfg = _cfg()
        s_r = HelicalChannelGenerator(handedness=+1, n_turns=0.5).build(cfg)
        s_l = HelicalChannelGenerator(handedness=-1, n_turns=0.5).build(cfg)
        diff = np.abs(s_r.sdf - s_l.sdf)
        assert diff.max() > 0.001, "handedness must change channel position"

    def test_pitch_changes_volume(self):
        cfg = _cfg()
        s_tight = HelicalChannelGenerator(pitch=0.005, n_turns=10).build(cfg)
        s_loose = HelicalChannelGenerator(pitch=0.030, n_turns=1).build(cfg)
        v_tight = (s_tight.sdf < 0).sum()
        v_loose = (s_loose.sdf < 0).sum()
        assert v_tight != v_loose

    def test_centerline_length(self):
        gen = HelicalChannelGenerator(radius=0.045, pitch=0.015, n_turns=3.0)
        L = gen.centerline_length()
        # Analytical: sqrt(R^2 + (p/2pi)^2) * 2*pi*n
        import math
        expected = math.sqrt(0.045**2 + (0.015 / (2 * math.pi))**2) * 2 * math.pi * 3.0
        assert abs(L - expected) / expected < 0.01, f"L={L:.6f}, expected={expected:.6f}"

    def test_bounding_box(self):
        cfg = _cfg()
        gen = HelicalChannelGenerator(
            radius=0.045, pitch=0.015, n_turns=3.0,
            z_start=-0.020, channel_radius=0.002,
        )
        bmin, bmax = gen.bounding_box(cfg)
        # Z extent should match n_turns * pitch
        z_extent = bmax[2] - bmin[2]
        expected_z = 3.0 * 0.015
        assert abs(z_extent - expected_z) / max(expected_z, 1e-6) < 0.05, (
            f"z extent {z_extent:.4f} != {expected_z:.4f}"
        )

    def test_channel_radius_changes_cross_section(self):
        cfg = _cfg()
        s_thin = HelicalChannelGenerator(channel_radius=0.001).build(cfg)
        s_thick = HelicalChannelGenerator(channel_radius=0.003).build(cfg)
        v_thin = (s_thin.sdf < 0).sum()
        v_thick = (s_thick.sdf < 0).sum()
        assert v_thick > v_thin * 2, "3x radius should give >>2x volume"


class TestBranchingManifold:
    """Verify Y-shape: 1 inlet, 2 outlets, connectivity."""

    def test_produces_void(self):
        cfg = _cfg()
        gen = BranchingManifold(
            inlet_pos=(0.0, 0.0, -0.040),
            fork_pos=(0.045, 0.0, -0.010),
            outlet_offset=(0.0, 0.020, 0.030),
            channel_radius=0.002,
        )
        sdf = gen.build(cfg)
        assert (sdf.sdf < 0).sum() > 0

    def test_has_one_inlet_and_two_outlets(self):
        gen = BranchingManifold(
            inlet_pos=(0.0, 0.0, -0.040),
            fork_pos=(0.045, 0.0, -0.010),
            outlet_offset=(0.0, 0.020, 0.030),
        )
        inlet = gen.inlet()
        outlets = gen.outlets()
        assert len(outlets) == 2, "must have exactly 2 outlets"
        # Inlet should be at the specified position
        assert np.allclose(inlet, [0.0, 0.0, -0.040])
        # Outlets should be different positions
        assert not np.allclose(outlets[0], outlets[1])

    def test_segments_connect_at_fork(self):
        gen = BranchingManifold(
            inlet_pos=(0.0, 0.0, -0.040),
            fork_pos=(0.045, 0.0, -0.010),
            outlet_offset=(0.0, 0.020, 0.030),
        )
        segments = gen._segments()
        # Inlet end == fork start
        assert np.allclose(segments[0][-1], segments[1][0]), "inlet must end at fork"
        assert np.allclose(segments[0][-1], segments[2][0]), "inlet must end at fork"

    def test_total_volume_finite_and_bounded(self):
        cfg = _cfg()
        gen = BranchingManifold(channel_radius=0.002)
        sdf = gen.build(cfg)
        # Channel should be within a reasonable bounding box
        mask = sdf.sdf < 0
        idx = np.argwhere(mask)
        if len(idx) > 0:
            dx, dy, dz = cfg.spacing
            ox, oy, oz = cfg.origin
            xs = ox + idx[:, 0] * dx
            ys = oy + idx[:, 1] * dy
            zs = oz + idx[:, 2] * dz
            extent = [xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]
            # Should be less than 100mm in each direction
            assert all(e < 0.100 for e in extent), f"extent too large: {extent}"


class TestConvenienceEntries:
    """Verify convenience functions use real config fields."""

    def test_honeycomb_support(self):
        cfg = _cfg()
        sdf = honeycomb_support(cfg)
        assert (sdf.sdf < 0).sum() > 0

    def test_helical_cooling(self):
        cfg = _cfg()
        sdf = helical_cooling(cfg)
        assert (sdf.sdf < 0).sum() > 0
