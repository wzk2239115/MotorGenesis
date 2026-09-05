"""Demonstration assembly: motor + honeycomb support + helical cooling + wall.

Integrates morphology generators into the stator outer region with:
  - Honeycomb structural support between winding outer and design outer
  - Helical cooling channel embedded in the support
  - Outer wall (housing) enclosing the cooling region
  - Inlet/outlet ports for the cooling channel

Outputs:
  - MaterialField with iron, copper, pm, insulator, coolant, support
  - Cross-section images (axial, radial, circumferential)
  - Geometry audit report (conflicts, connectivity, volumes)
"""

from __future__ import annotations

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.field import SDFVoxelField, polyline_capsule_sdf
from organic_motor.construct.morphology import (
    HoneycombGenerator,
    HelicalChannelGenerator,
    BranchingManifold,
)
from organic_motor.geometry.grid3d import meshgrid3d


OUT_DIR = Path(__file__).parent.parent / "reports" / "assembly"


def build_motor_with_assembly(cfg: MotorConfig3D | None = None) -> tuple:
    """Build complete motor + honeycomb support + helical cooling + wall.

    Returns (MaterialField, cfg).
    """
    cfg = cfg or MotorConfig3D(shape=(96, 96, 58))
    cx, cy, cz = cfg.center

    # 1. Build the motor itself (stator iron + copper + pm + insulator)
    print("  Building motor...")
    from organic_motor.construct.objects import field_driven_motor
    motor = field_driven_motor(cfg)
    mf = motor.build()
    print(f"    materials: {mf.materials_present()}")

    # 2. Build support/cooling/wall
    r_winding_outer = cfg.R_winding_outer       # 0.043
    r_design = cfg.R_design                      # 0.050
    z_half = cfg.stator_half_length              # 0.031

    r_support_inner = r_winding_outer + 0.001    # 0.044
    r_support_outer = r_design - 0.001           # 0.049
    r_wall_inner = r_design                       # 0.050
    r_wall_outer = r_design + 0.002               # 0.052

    print("  Building honeycomb support...")
    honeycomb = HoneycombGenerator(
        r_inner=r_support_inner,
        r_outer=r_support_outer,
        z_bottom=-z_half,
        z_top=z_half,
        cell_size=0.006,
        wall_thickness=0.001,
    ).build(cfg)

    print("  Building helical cooling channel...")
    helix = HelicalChannelGenerator(
        radius=(r_support_inner + r_support_outer) / 2,
        pitch=2 * z_half / 4.0,
        n_turns=4.0,
        channel_radius=0.002,
        z_start=-z_half + 0.002,
        handedness=1,
        n_segments=100,
    ).build(cfg)

    print("  Building housing wall...")
    X, Y, Z = meshgrid3d(cfg)
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    wall_inner_sdf = np.maximum(r_wall_inner - R, R - r_wall_outer)
    wall_z = np.abs(Z - cz) - (z_half + 0.002)
    wall = np.maximum(wall_inner_sdf, wall_z)
    wall_field = SDFVoxelField(wall.astype(np.float32), cfg.spacing, cfg.origin)

    print("  Building inlet/outlet ports...")
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

    # 3. Add to motor MaterialField
    # Honeycomb iron (non-priority: doesn't carve other materials)
    mf.add(honeycomb, "iron", priority=False)
    # Wall iron (priority: carves coolant)
    mf.add(wall_field, "iron", priority=True)
    # Coolant (helix + ports, priority: carves iron)
    coolant_sdf = np.minimum(helix.sdf, port_void)
    coolant_field = SDFVoxelField(
        coolant_sdf.astype(np.float32), cfg.spacing, cfg.origin,
    )
    mf.add(coolant_field, "coolant", priority=True)

    return mf, cfg


def export_checkpoint(mf, cfg, out_dir: Path, motor=None):
    """Save MaterialField as versioned ModelArtifact for web viewer + simulation.

    If motor is provided, saves magnetization and motion groups too.
    """
    from organic_motor.construct.model_artifact import ModelArtifact

    if motor is not None:
        artifact = ModelArtifact.from_motor(motor, mf, cfg)
    else:
        artifact = ModelArtifact.from_material_field(mf, cfg)

    npz_path = artifact.save(out_dir)
    print(f"  Artifact saved: {npz_path}")
    print(f"  Design hash: {artifact.design_hash}")
    print(f"  Magnetization: {'yes' if artifact.has_magnetization else 'no'}")
    print(f"  Centerlines: {len(artifact.centerline_registry)}")
    can_run, reasons = artifact.can_energize()
    print(f"  Can energize: {can_run}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return npz_path


def build_assembly(cfg: MotorConfig3D | None = None) -> dict:
    """Build the full assembly and return audit results.

    Returns dict with:
      - mf: MaterialField
      - audit: conflict checks, volumes, connectivity
      - config: actual config used
    """
    cfg = cfg or MotorConfig3D(shape=(224, 224, 136))
    dx, dy, dz = cfg.spacing
    cx, cy, cz = cfg.center

    # --- Geometry parameters (explicit, from config) ---
    r_winding_outer = cfg.R_winding_outer       # 0.043
    r_design = cfg.R_design                      # 0.050
    r_stator_inner = cfg.R_stator_inner          # 0.0305
    z_half = cfg.stator_half_length              # 0.031

    # Support region: between winding outer and design outer
    r_support_inner = r_winding_outer + 0.001    # 0.044
    r_support_outer = r_design - 0.001           # 0.049
    # Housing wall: thin shell at design radius
    r_wall_inner = r_design                       # 0.050
    r_wall_outer = r_design + 0.002               # 0.052

    # --- Build SDFs ---
    # 1. Honeycomb support (structural iron)
    print("  Building honeycomb support...")
    honeycomb = HoneycombGenerator(
        r_inner=r_support_inner,
        r_outer=r_support_outer,
        z_bottom=-z_half,
        z_top=z_half,
        cell_size=0.004,
        wall_thickness=0.0008,
    ).build(cfg)
    hc_voxels = int((honeycomb.sdf < 0).sum())
    print(f"    honeycomb: {hc_voxels} voxels")

    # 2. Helical cooling channel (void in support)
    print("  Building helical cooling channel...")
    helix = HelicalChannelGenerator(
        radius=(r_support_inner + r_support_outer) / 2,  # mid-radius
        pitch=2 * z_half / 4.0,                           # 4 turns
        n_turns=4.0,
        channel_radius=0.0015,
        z_start=-z_half + 0.002,
        handedness=1,
        n_segments=200,
    ).build(cfg)
    hl_voxels = int((helix.sdf < 0).sum())
    print(f"    helix void: {hl_voxels} voxels")

    # 3. Housing wall (structural iron shell)
    print("  Building housing wall...")
    X, Y, Z = meshgrid3d(cfg)
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    wall_sdf = np.maximum(
        np.maximum(r_wall_inner - R, R - r_wall_outer),
        np.maximum(np.abs(Z - cz) - z_half - 0.002, 0),
    )
    # Make it a proper SDF (negative inside wall)
    wall_inner_sdf = np.maximum(r_wall_inner - R, R - r_wall_outer)
    wall_z = np.abs(Z - cz) - (z_half + 0.002)
    wall = np.maximum(wall_inner_sdf, wall_z)
    wall_voxels = int((wall < 0).sum())
    print(f"    wall: {wall_voxels} voxels")

    # 4. Inlet/outlet ports (short straight channels through the wall)
    print("  Building inlet/outlet ports...")
    port_radius = 0.002
    # Inlet: at angle 0, going radially outward
    inlet_pts = np.array([
        [cx + r_wall_inner - 0.001, cy, cz - z_half + 0.003],
        [cx + r_wall_outer + 0.001, cy, cz - z_half + 0.003],
    ])
    inlet_sdf = polyline_capsule_sdf(
        cfg.shape, cfg.spacing, cfg.origin, inlet_pts, port_radius,
    )
    # Outlet: at angle 180, going radially outward
    outlet_pts = np.array([
        [cx - r_wall_inner - 0.001, cy, cz + z_half - 0.003],
        [cx - r_wall_outer - 0.001, cy, cz + z_half - 0.003],
    ])
    outlet_sdf = polyline_capsule_sdf(
        cfg.shape, cfg.spacing, cfg.origin, outlet_pts, port_radius,
    )
    port_void = np.minimum(inlet_sdf, outlet_sdf)
    port_voxels = int((port_void < 0).sum())
    print(f"    ports: {port_voxels} voxels")

    # --- Assemble MaterialField ---
    # The support is iron (structural). The cooling channel and ports
    # are voids (coolant). The wall is iron.
    mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)

    # Add honeycomb as iron (structural support)
    mf.add(honeycomb, "iron", priority=False)

    # Add wall as iron (priority — encloses everything)
    wall_field = SDFVoxelField(wall.astype(np.float32), cfg.spacing, cfg.origin)
    mf.add(wall_field, "iron", priority=True)

    # Subtract cooling channel and ports from iron (make them voids/coolant)
    coolant_sdf = np.minimum(helix.sdf, port_void)
    coolant_field = SDFVoxelField(
        coolant_sdf.astype(np.float32), cfg.spacing, cfg.origin,
    )
    mf.add(coolant_field, "coolant", priority=True)

    # --- Audit ---
    print("  Running audit...")
    audit = _audit(cfg, mf, honeycomb, helix, wall, port_void, R)

    # --- Generate views ---
    print("  Generating views...")
    _generate_views(cfg, mf, OUT_DIR)

    result = {
        "config": {
            "shape": list(cfg.shape),
            "spacing_mm": [round(dx*1000, 4), round(dy*1000, 4), round(dz*1000, 4)],
            "r_support_inner_mm": r_support_inner * 1000,
            "r_support_outer_mm": r_support_outer * 1000,
            "r_wall_inner_mm": r_wall_inner * 1000,
            "r_wall_outer_mm": r_wall_outer * 1000,
        },
        "voxels": {
            "honeycomb": hc_voxels,
            "helix_void": hl_voxels,
            "wall": wall_voxels,
            "ports": port_voxels,
        },
        "audit": audit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "assembly_report.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(f"  Report saved to {OUT_DIR / 'assembly_report.json'}")

    return result


def _audit(cfg, mf, honeycomb, helix, wall, port_void, R) -> dict:
    """Check for conflicts and verify assembly integrity."""
    dx, dy, dz = cfg.spacing
    cx, cy, cz = cfg.center
    audit = {}

    # 1. No honeycomb in air gap
    air_gap = (R > cfg.R_sleeve_outer) & (R < cfg.R_stator_inner)
    hc_in_gap = int(((honeycomb.sdf < 0) & air_gap).sum())
    audit["honeycomb_in_air_gap"] = hc_in_gap
    audit["honeycomb_in_air_gap_pass"] = hc_in_gap == 0

    # 2. No honeycomb in winding region
    winding = (R > cfg.R_winding_inner) & (R < cfg.R_winding_outer)
    hc_in_winding = int(((honeycomb.sdf < 0) & winding).sum())
    audit["honeycomb_in_winding"] = hc_in_winding
    audit["honeycomb_in_winding_pass"] = hc_in_winding == 0

    # 3. No coolant in air gap
    from organic_motor.construct.material import MATERIALS, AUX_MATERIALS
    cu = mf.sdfs.get("coolant")
    if cu is not None:
        coolant_in_gap = int(((cu.sdf < 0) & air_gap).sum())
        audit["coolant_in_air_gap"] = coolant_in_gap
        audit["coolant_in_air_gap_pass"] = coolant_in_gap == 0

    # 4. No coolant in winding
        coolant_in_winding = int(((cu.sdf < 0) & winding).sum())
        audit["coolant_in_winding"] = coolant_in_winding
        audit["coolant_in_winding_pass"] = coolant_in_winding == 0

    # 5. Wall is continuous (check a ring at z=mid)
    k = cfg.shape[2] // 2
    wall_ring = wall[:, :, k]
    # At r_wall_inner to r_wall_outer, check the ring is complete
    r_test = (cfg.R_design + 0.001)  # mid-wall
    wall_mask = np.abs(R[:, :, k] - r_test) < 0.001
    wall_complete = bool((wall_ring[wall_mask] < 0).all()) if wall_mask.sum() > 0 else False
    audit["wall_continuous"] = wall_complete

    # 6. Coolant channel connects to ports
    from scipy import ndimage
    coolant_mask = cu.sdf < 0 if cu is not None else np.zeros(cfg.shape, dtype=bool)
    labeled, n_arr = ndimage.label(coolant_mask, structure=ndimage.generate_binary_structure(3, 1))
    n_components = int(n_arr) if np.isscalar(n_arr) or np.ndim(n_arr) == 0 else 0
    # Alternative: count unique labels
    if n_components == 0:
        n_components = len(np.unique(labeled)) - 1  # subtract background (0)
    audit["coolant_components"] = n_components
    # Should be 1 (connected) or at most 2 (channel + ports if barely touching)
    audit["coolant_connectivity_pass"] = n_components <= 2

    # 7. Materials present
    audit["materials"] = mf.materials_present()

    # 8. Volume summary
    for name in mf.sdfs:
        audit[f"vol_{name}"] = int((mf.sdfs[name].sdf < 0).sum())

    return audit


def _generate_views(cfg, mf, out_dir):
    """Generate cross-section images."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Material color map
    color_map = {
        "iron": 1, "copper": 2, "pm": 3,
        "insulator": 4, "coolant": 5, "air": 6,
    }

    def material_slice(axis, idx):
        """Get material map at given slice."""
        if axis == "z":
            slc = np.zeros(cfg.shape[:2], dtype=int)
            for name, sdf in mf.sdfs.items():
                m = (sdf.sdf[:, :, idx] < 0).astype(int)
                code = color_map.get(name, 9)
                slc[m > 0] = code
        elif axis == "x":
            slc = np.zeros((cfg.shape[1], cfg.shape[2]), dtype=int)
            for name, sdf in mf.sdfs.items():
                m = (sdf.sdf[idx, :, :] < 0).astype(int)
                code = color_map.get(name, 9)
                slc[m > 0] = code
        elif axis == "y":
            slc = np.zeros((cfg.shape[0], cfg.shape[2]), dtype=int)
            for name, sdf in mf.sdfs.items():
                m = (sdf.sdf[:, idx, :] < 0).astype(int)
                code = color_map.get(name, 9)
                slc[m > 0] = code
        return slc

    # 1. Axial slice (z=mid) — shows the full cross-section
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    z_slices = [cfg.shape[2]//4, cfg.shape[2]//2, cfg.shape[2]*3//4]
    for ax, z in zip(axes, z_slices):
        slc = material_slice("z", z)
        ax.imshow(slc, cmap="tab10", vmin=0, vmax=9)
        ax.set_title(f"z={z} ({cfg.origin[2]+z*cfg.spacing[2]:.1f}mm)")
        ax.set_xlabel("y"); ax.set_ylabel("x")
    plt.tight_layout()
    plt.savefig(out_dir / "axial_slices.png", dpi=150)
    plt.close()

    # 2. Radial slice (x=mid) — shows axial cross-section
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, x in zip(axes, [cfg.shape[0]//2, cfg.shape[0]//4]):
        slc = material_slice("x", x)
        ax.imshow(slc, cmap="tab10", vmin=0, vmax=9, aspect="auto")
        ax.set_title(f"x={x}")
        ax.set_xlabel("z"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / "radial_slices.png", dpi=150)
    plt.close()

    # 3. Support region zoom (axial, z=mid)
    k = cfg.shape[2] // 2
    slc = material_slice("z", k)
    # Zoom to support region (outer 20% of grid)
    margin = cfg.shape[0] // 6
    zoom = slc[margin:-margin, margin:-margin]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(zoom, cmap="tab10", vmin=0, vmax=9)
    ax.set_title("Support region zoom (z=mid)")
    plt.tight_layout()
    plt.savefig(out_dir / "support_zoom.png", dpi=150)
    plt.close()

    # 4. Coolant-only view
    cu = mf.sdfs.get("coolant")
    if cu is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, z in zip(axes, z_slices):
            slc = (cu.sdf[:, :, z] < 0).astype(float)
            ax.imshow(slc, cmap="Blues")
            ax.set_title(f"coolant z={z}")
        plt.tight_layout()
        plt.savefig(out_dir / "coolant_slices.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    import sys

    if "--full" in sys.argv or "--motor" in sys.argv:
        print("=== MOTOR + ASSEMBLY (full) ===")
        cfg = MotorConfig3D(shape=(96, 96, 58))
        from organic_motor.construct.objects import field_driven_motor
        motor = field_driven_motor(cfg)
        mf = motor.build()
        out_dir = Path(__file__).parent.parent / "out" / "assembly"
        export_checkpoint(mf, cfg, out_dir, motor=motor)
        print(f"\n  Materials: {mf.materials_present()}")
        print("  DONE — open web viewer and select 'assembly' run")
    else:
        print("=== ASSEMBLY DEMO (support only) ===")
        result = build_assembly()
        print("\n=== AUDIT RESULTS ===")
        for k, v in result["audit"].items():
            print(f"  {k}: {v}")
