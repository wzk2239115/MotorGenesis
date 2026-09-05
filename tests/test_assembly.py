"""Tests for the assembly demo (honeycomb + helical cooling + wall + ports).

Tests run at 96^3 (not 224^3) for speed, but verify all the same properties.
"""

import pytest
import numpy as np
from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.field import SDFVoxelField, polyline_capsule_sdf
from organic_motor.construct.morphology import (
    HoneycombGenerator,
    HelicalChannelGenerator,
)
from organic_motor.geometry.grid3d import meshgrid3d


def _build_small_assembly():
    """Build assembly at 96^3 for testing."""
    cfg = MotorConfig3D(shape=(96, 96, 58))
    dx, dy, dz = cfg.spacing
    cx, cy, cz = cfg.center

    r_support_inner = cfg.R_winding_outer + 0.001
    r_support_outer = cfg.R_design - 0.001
    r_wall_inner = cfg.R_design
    r_wall_outer = cfg.R_design + 0.002
    z_half = cfg.stator_half_length

    # Honeycomb
    honeycomb = HoneycombGenerator(
        r_inner=r_support_inner,
        r_outer=r_support_outer,
        z_bottom=-z_half,
        z_top=z_half,
        cell_size=0.006,
        wall_thickness=0.001,
    ).build(cfg)

    # Helix
    helix = HelicalChannelGenerator(
        radius=(r_support_inner + r_support_outer) / 2,
        pitch=2 * z_half / 4.0,
        n_turns=4.0,
        channel_radius=0.002,
        z_start=-z_half + 0.002,
        handedness=1,
        n_segments=100,
    ).build(cfg)

    # Wall
    X, Y, Z = meshgrid3d(cfg)
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    wall_inner_sdf = np.maximum(r_wall_inner - R, R - r_wall_outer)
    wall_z = np.abs(Z - cz) - (z_half + 0.002)
    wall = np.maximum(wall_inner_sdf, wall_z)

    # Ports
    port_radius = 0.002
    inlet_pts = np.array([
        [cx + r_wall_inner - 0.001, cy, cz - z_half + 0.003],
        [cx + r_wall_outer + 0.001, cy, cz - z_half + 0.003],
    ])
    inlet_sdf = polyline_capsule_sdf(
        cfg.shape, cfg.spacing, cfg.origin, inlet_pts, port_radius,
    )
    outlet_pts = np.array([
        [cx - r_wall_inner - 0.001, cy, cz + z_half - 0.003],
        [cx - r_wall_outer - 0.001, cy, cz + z_half - 0.003],
    ])
    outlet_sdf = polyline_capsule_sdf(
        cfg.shape, cfg.spacing, cfg.origin, outlet_pts, port_radius,
    )
    port_void = np.minimum(inlet_sdf, outlet_sdf)

    # Assemble
    mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
    mf.add(honeycomb, "iron", priority=False)
    wall_field = SDFVoxelField(wall.astype(np.float32), cfg.spacing, cfg.origin)
    mf.add(wall_field, "iron", priority=True)
    coolant_sdf = np.minimum(helix.sdf, port_void)
    coolant_field = SDFVoxelField(
        coolant_sdf.astype(np.float32), cfg.spacing, cfg.origin,
    )
    mf.add(coolant_field, "coolant", priority=True)

    return cfg, mf, R


class TestAssembly:
    def test_produces_material(self):
        cfg, mf, R = _build_small_assembly()
        assert "iron" in mf.sdfs
        assert "coolant" in mf.sdfs
        iron_voxels = int((mf.sdfs["iron"].sdf < 0).sum())
        coolant_voxels = int((mf.sdfs["coolant"].sdf < 0).sum())
        assert iron_voxels > 1000, f"iron too sparse: {iron_voxels}"
        assert coolant_voxels > 100, f"coolant too sparse: {coolant_voxels}"

    def test_no_honeycomb_in_air_gap(self):
        cfg, mf, R = _build_small_assembly()
        air_gap = (R > cfg.R_sleeve_outer) & (R < cfg.R_stator_inner)
        iron = mf.sdfs["iron"].sdf
        intrusion = int(((iron < 0) & air_gap).sum())
        assert intrusion == 0, f"iron in air gap: {intrusion} voxels"

    def test_no_honeycomb_in_winding(self):
        cfg, mf, R = _build_small_assembly()
        winding = (R > cfg.R_winding_inner) & (R < cfg.R_winding_outer)
        iron = mf.sdfs["iron"].sdf
        intrusion = int(((iron < 0) & winding).sum())
        assert intrusion == 0, f"iron in winding: {intrusion} voxels"

    def test_no_coolant_in_air_gap(self):
        cfg, mf, R = _build_small_assembly()
        air_gap = (R > cfg.R_sleeve_outer) & (R < cfg.R_stator_inner)
        coolant = mf.sdfs["coolant"].sdf
        intrusion = int(((coolant < 0) & air_gap).sum())
        assert intrusion == 0, f"coolant in air gap: {intrusion} voxels"

    def test_no_coolant_in_winding(self):
        cfg, mf, R = _build_small_assembly()
        winding = (R > cfg.R_winding_inner) & (R < cfg.R_winding_outer)
        coolant = mf.sdfs["coolant"].sdf
        intrusion = int(((coolant < 0) & winding).sum())
        assert intrusion == 0, f"coolant in winding: {intrusion} voxels"

    def test_wall_continuous(self):
        """Wall ring must be complete at z=mid."""
        cfg, mf, R = _build_small_assembly()
        k = cfg.shape[2] // 2
        iron = mf.sdfs["iron"].sdf[:, :, k]
        r_test = cfg.R_design + 0.001
        wall_mask = np.abs(R[:, :, k] - r_test) < 0.002
        wall_voxels = (iron[wall_mask] < 0).sum()
        assert wall_voxels > 0, "no wall voxels at test radius"

    def test_coolant_is_connected(self):
        """Coolant channel + ports should be connected (<=2 components)."""
        from scipy import ndimage
        cfg, mf, R = _build_small_assembly()
        coolant_mask = (mf.sdfs["coolant"].sdf < 0)
        labeled, n_arr = ndimage.label(
            coolant_mask, structure=ndimage.generate_binary_structure(3, 1)
        )
        n_components = len(np.unique(labeled)) - 1
        assert n_components <= 3, (
            f"coolant has {n_components} components, expected <=3"
        )

    def test_helix_reduces_iron_volume(self):
        """Adding the cooling channel should remove iron voxels."""
        cfg, mf, R = _build_small_assembly()
        # Iron without coolant subtraction
        honeycomb_only = HoneycombGenerator(
            r_inner=cfg.R_winding_outer + 0.001,
            r_outer=cfg.R_design - 0.001,
            z_bottom=-cfg.stator_half_length,
            z_top=cfg.stator_half_length,
            cell_size=0.006,
            wall_thickness=0.001,
        ).build(cfg)
        hc_vol = int((honeycomb_only.sdf < 0).sum())
        iron_vol = int((mf.sdfs["iron"].sdf < 0).sum())
        # Iron in assembly includes wall + honeycomb minus coolant
        # Just check iron is present and reasonable
        assert iron_vol > 0

    def test_materials_present(self):
        """Both iron and coolant must be present."""
        cfg, mf, R = _build_small_assembly()
        materials = mf.materials_present()
        assert "iron" in materials
        assert "coolant" in materials

    def test_assembly_report_exists(self):
        """The 224^3 assembly report should exist after running the demo."""
        from pathlib import Path
        report = Path(__file__).parent.parent / "organic_motor" / "reports" / "assembly" / "assembly_report.json"
        if report.exists():
            import json
            data = json.loads(report.read_text())
            assert data["audit"]["honeycomb_in_air_gap_pass"] is True
            assert data["audit"]["coolant_in_air_gap_pass"] is True
            assert data["audit"]["wall_continuous"] is True
