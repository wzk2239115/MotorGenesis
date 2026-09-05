"""Tests for parametric morphology generators."""

import numpy as np
import pytest

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.morphology import (
    HoneycombGenerator,
    HelicalChannelGenerator,
    honeycomb_support,
    helical_cooling,
)


def _cfg(**ov):
    d = dict(shape=(96, 96, 58), excitation_mode="impressed",
             filt_radius=0.0, projection_beta=0.0)
    d.update(ov)
    return MotorConfig3D(**d)


class TestHoneycomb:
    def test_produces_material(self):
        cfg = _cfg()
        sdf = HoneycombGenerator(
            r_inner=0.044, r_outer=0.046,
            z_bottom=-0.045, z_top=0.045,
            cell_size=0.003, wall_thickness=0.001,
        ).build(cfg)
        inside = (sdf.sdf < 0).sum()
        assert inside > 0, "honeycomb must produce material"

    def test_radial_bounds(self):
        cfg = _cfg()
        gen = HoneycombGenerator(
            r_inner=0.044, r_outer=0.046,
            z_bottom=-0.045, z_top=0.045,
            cell_size=0.003, wall_thickness=0.001,
        )
        sdf = gen.build(cfg)
        dx, dy, dz = cfg.spacing
        ox, oy, oz = cfg.origin
        cz = cfg.center[2]
        # Check at z=center slice only
        k_mid = cfg.shape[2] // 2
        for i in range(0, cfg.shape[0], 8):
            for j in range(0, cfg.shape[1], 8):
                x = ox + i * dx
                y = oy + j * dy
                R = np.sqrt((x - cfg.center[0])**2 + (y - cfg.center[1])**2)
                val = sdf.sdf[i, j, k_mid]
                if R > 0.052:
                    assert val >= 0, f"material at R={R*1000:.1f}mm (outside bounds)"

    def test_wall_thickness_changes_volume(self):
        cfg = _cfg()
        s_thin = HoneycombGenerator(
            r_inner=0.044, r_outer=0.046,
            z_bottom=-0.045, z_top=0.045,
            cell_size=0.003, wall_thickness=0.0004,
        ).build(cfg)
        s_thick = HoneycombGenerator(
            r_inner=0.044, r_outer=0.046,
            z_bottom=-0.045, z_top=0.045,
            cell_size=0.003, wall_thickness=0.0015,
        ).build(cfg)
        v_thin = (s_thin.sdf < 0).sum()
        v_thick = (s_thick.sdf < 0).sum()
        assert v_thick > v_thin, "thicker walls must produce more material"


class TestHelicalChannel:
    def test_produces_void(self):
        cfg = _cfg()
        sdf = HelicalChannelGenerator(
            radius=0.045, pitch=0.015, n_turns=3.0,
        ).build(cfg)
        void = (sdf.sdf < 0).sum()
        assert void > 0, "helix must produce channel void"

    def test_handedness_changes_position(self):
        cfg = _cfg()
        s_r = HelicalChannelGenerator(handedness=+1).build(cfg)
        s_l = HelicalChannelGenerator(handedness=-1).build(cfg)
        # The voids should be at different positions
        diff = np.abs(s_r.sdf - s_l.sdf)
        assert diff.max() > 0.001, "handedness must change channel position"

    def test_pitch_changes_volume(self):
        cfg = _cfg()
        s_tight = HelicalChannelGenerator(pitch=0.005, n_turns=10).build(cfg)
        s_loose = HelicalChannelGenerator(pitch=0.030, n_turns=1).build(cfg)
        v_tight = (s_tight.sdf < 0).sum()
        v_loose = (s_loose.sdf < 0).sum()
        assert v_tight != v_loose, "different pitch must produce different volume"
