"""Tests for assembly features: end caps, bearing seats, bolt holes, wire exit."""

import pytest
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.assembly_features import EndCapGenerator, MountingFlangeGenerator
from organic_motor.construct.field import SDFVoxelField
from organic_motor.construct.material import MaterialField


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig3D(shape=(56, 56, 36))


class TestEndCap:
    def test_endcap_present(self, cfg):
        """End cap generator produces non-zero solid voxels."""
        cap = EndCapGenerator().build(cfg)
        n_voxels = int((cap.sdf < 0).sum())
        assert n_voxels > 0, "end cap has zero voxels"

    def test_bearing_bore_exists(self, cfg):
        """Bearing bore: no material well inside bearing OD radius."""
        cap = EndCapGenerator(bearing_od=0.028).build(cfg)
        cx, cy, cz = cfg.center
        z_half = cfg.stator_half_length
        z_check = z_half + 0.003  # mid-cap
        nx, ny, nz = cfg.shape
        dx, dy, dz = cfg.spacing
        x0, y0, z0 = cfg.origin
        iz = int(round((z_check - z0) / dz))
        iz = max(0, min(nz - 1, iz))
        # Check at R = 0.011 (well inside bearing bore 0.014,
        # but outside shaft 0.008)
        r_check = 0.011
        count_in_bore = 0
        for i in range(nx):
            for j in range(ny):
                px = x0 + i * dx
                py = y0 + j * dy
                r = np.sqrt((px - cx)**2 + (py - cy)**2)
                if abs(r - r_check) < dx:
                    if cap.sdf[i, j, iz] < 0:
                        count_in_bore += 1
        assert count_in_bore == 0, (
            f"bearing bore not hollow at R={r_check}: "
            f"{count_in_bore} solid voxels"
        )

    def test_bolt_holes_exist(self, cfg):
        """Bolt holes: voids at bolt circle positions."""
        gen = EndCapGenerator(n_bolts=6, bolt_circle_radius=0.042,
                              bolt_hole_radius=0.0022)
        cap = gen.build(cfg)
        cx, cy, cz = cfg.center
        z_half = cfg.stator_half_length
        z_check = z_half + 0.003  # mid-cap
        nx, ny, nz = cfg.shape
        dx, dy, dz = cfg.spacing
        x0, y0, z0 = cfg.origin
        iz = int(round((z_check - z0) / dz))
        iz = max(0, min(nz - 1, iz))
        for i in range(gen.n_bolts):
            angle = 2.0 * np.pi * i / gen.n_bolts + (np.pi / gen.n_bolts)
            bx = gen.bolt_circle_radius * np.cos(angle)
            by = gen.bolt_circle_radius * np.sin(angle)
            # Find nearest voxel
            ix = int(round((bx + cx - x0) / dx))
            iy = int(round((by + cy - y0) / dy))
            ix = max(0, min(nx - 1, ix))
            iy = max(0, min(ny - 1, iy))
            assert cap.sdf[ix, iy, iz] > 0, (
                f"bolt hole {i} at ({bx:.4f}, {by:.4f}) is not void"
            )

    def test_both_caps_generated(self, cfg):
        """Both front (z+) and rear (z-) caps are present."""
        cap = EndCapGenerator().build(cfg)
        z_half = cfg.stator_half_length
        dx, dy, dz = cfg.spacing
        x0, y0, z0 = cfg.origin
        # Front cap
        z_front = z_half + 0.003
        iz_f = int(round((z_front - z0) / dz))
        # Rear cap
        z_rear = -(z_half + 0.003)
        iz_r = int(round((z_rear - z0) / dz))
        iz_f = max(0, min(cfg.shape[2] - 1, iz_f))
        iz_r = max(0, min(cfg.shape[2] - 1, iz_r))
        front_voxels = int((cap.sdf[:, :, iz_f] < 0).sum())
        rear_voxels = int((cap.sdf[:, :, iz_r] < 0).sum())
        assert front_voxels > 0, "front cap missing"
        assert rear_voxels > 0, "rear cap missing"

    def test_wire_exit_only_rear(self, cfg):
        """Wire exit slot exists on rear cap (z-), not on front (z+)."""
        gen = EndCapGenerator(include_wire_exit=True,
                             wire_exit_angle=-np.pi/2)
        cap = gen.build(cfg)
        # Wire exit is at bottom (y < 0), rear cap
        # Check that rear cap has less material at bottom than front
        z_half = cfg.stator_half_length
        dx, dy, dz = cfg.spacing
        x0, y0, z0 = cfg.origin
        iz_f = int(round((z_half + 0.003 - z0) / dz))
        iz_r = int(round((-(z_half + 0.003) - z0) / dz))
        iz_f = max(0, min(cfg.shape[2] - 1, iz_f))
        iz_r = max(0, min(cfg.shape[2] - 1, iz_r))
        # Bottom quadrant voxels
        front_bottom = int((cap.sdf[:, :cfg.shape[1]//2, iz_f] < 0).sum())
        rear_bottom = int((cap.sdf[:, :cfg.shape[1]//2, iz_r] < 0).sum())
        assert rear_bottom <= front_bottom, (
            f"wire exit should reduce rear-cap bottom material: "
            f"front={front_bottom} rear={rear_bottom}"
        )

    def test_endcap_merges_with_motor(self, cfg):
        """End cap can be added to a MaterialField without error."""
        mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
        cap = EndCapGenerator().build(cfg)
        mf.add(cap, "iron", priority=True)
        assert "iron" in mf.sdfs
        assert int((mf.sdfs["iron"].sdf < 0).sum()) > 0


class TestMountingFlange:
    def test_flange_present(self, cfg):
        flange = MountingFlangeGenerator().build(cfg)
        assert int((flange.sdf < 0).sum()) > 0

    def test_mount_holes_voids(self, cfg):
        gen = MountingFlangeGenerator(n_mount_holes=4)
        flange = gen.build(cfg)
        cx, cy, cz = cfg.center
        dx, dy, dz = cfg.spacing
        x0, y0, z0 = cfg.origin
        z_c = cz + gen.mount_offset_z
        iz = int(round((z_c - z0) / dz))
        iz = max(0, min(cfg.shape[2] - 1, iz))
        hole_spacing = gen.flange_width * 0.7
        for i in range(gen.n_mount_holes):
            hx = -hole_spacing/2 + i * (hole_spacing / (gen.n_mount_holes - 1))
            ix = int(round((hx + cx - x0) / dx))
            ix = max(0, min(cfg.shape[0] - 1, ix))
            # Find y position of holes
            r_base = float(cfg.R_design) + 0.002
            r_tip = r_base + gen.flange_depth
            hy = -(r_base + r_tip) / 2
            iy = int(round((hy + cy - y0) / dy))
            iy = max(0, min(cfg.shape[1] - 1, iy))
            assert flange.sdf[ix, iy, iz] > 0, (
                f"mount hole {i} at ({hx:.4f}, {hy:.4f}) is not void"
            )


def test_bearing_dimension_contract():
    cfg=MotorConfig3D(shape=(24,24,24))
    with pytest.raises(ValueError,match="直径"):
        EndCapGenerator(bearing_od=.014).build(cfg)


def test_register_protrudes_toward_housing():
    # A point inside the lip, but outside the original cap disk.
    from types import SimpleNamespace
    gen=EndCapGenerator(include_wire_exit=False)
    z_inner=.030+gen.cap_gap
    cfg=SimpleNamespace(shape=(3,3,3),origin=(.0484,-.0001,z_inner-.0011),
                        spacing=(.0001,)*3,center=(0,0,0),stator_half_length=.030,
                        R_shaft=.008,R_design=.05,R_winding_outer=.043)
    cap=gen.build(cfg)
    assert cap.sdf[1,1,1]<0


def test_endcaps_translate_with_center():
    from types import SimpleNamespace
    cfg=MotorConfig3D(shape=(28,28,28))
    shift=np.array([.004,-.003,.007])
    moved=SimpleNamespace(shape=cfg.shape,origin=np.array(cfg.origin)+shift,
        spacing=cfg.spacing,center=np.array(cfg.center)+shift,
        stator_half_length=cfg.stator_half_length,R_shaft=cfg.R_shaft,
        R_design=cfg.R_design,R_winding_outer=cfg.R_winding_outer)
    assert np.allclose(EndCapGenerator().build(cfg).sdf,EndCapGenerator().build(moved).sdf,atol=1e-7)
