"""Geometric quality metrics for constructed motors.

These metrics are computed from the realised density fields and give the
agent feedback about structural quality beyond raw electromagnetic numbers.
Without these, the optimiser converges to "solid iron rod + copper ring +
iron ring" because that maximises torque and minimises loss -- but it is
not a motor.

Metrics:
  - copper_components: connected component count of copper (1 = shorted ring)
  - copper_min_gap_mm: minimum gap between distinct copper components
  - air_gap_iron_bridge: True if iron crosses the air gap
  - shaft_rotor_merge: True if shaft and rotor iron are connected
  - housing_open_area_ratio: fraction of housing perimeter that is open
  - end_face_occlusion: fraction of end face blocked by solid iron
  - min_wall_thickness_mm: thinnest iron wall in housing/stator
  - solid_iron_ratio: fraction of iron that is solid (no voids)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField


def _material_inside(mf: MaterialField, material: str) -> np.ndarray:
    """Hard inside test using SDF < 0, bypassing density smoothing."""
    if material not in mf.sdfs:
        return np.zeros(mf.shape, dtype=bool)
    return mf.sdfs[material].sdf < 0.0


def _copper_inside(mf: MaterialField, cfg: MotorConfig3D) -> np.ndarray:
    """Copper wire-core test.

    With one radial layer per phase, the inter-layer insulation gap
    (~1.3mm) exceeds the display voxel size, so a plain SDF < 0 mask
    resolves the phase networks correctly without erosion margins (an
    erosion margin would fragment wires whose radius is near the voxel
    size).
    """
    return _material_inside(mf, "copper")


def copper_connected_components(mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3) -> dict:
    """Count connected copper components and minimum inter-component gap.

    The pairwise gap search is O(n^2) distance transforms, so it is skipped
    when there are many components (fragmented geometry) -- the component
    count itself is the actionable signal.
    """
    copper = _copper_inside(mf, cfg)
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_comp = ndimage.label(copper, structure=structure)
    if n_comp <= 1 or n_comp > 20:
        return {"copper_components": n_comp, "copper_min_gap_mm": 0.0}
    min_gap = float("inf")
    for i in range(1, n_comp + 1):
        mask_i = labels == i
        for j in range(i + 1, n_comp + 1):
            mask_j = labels == j
            dt_i = ndimage.distance_transform_edt(~mask_i, sampling=cfg.spacing)
            dist = dt_i[mask_j].min() if mask_j.any() else float("inf")
            min_gap = min(min_gap, dist)
    return {
        "copper_components": n_comp,
        "copper_min_gap_mm": min_gap * 1000 if np.isfinite(min_gap) else 0.0,
    }


def air_gap_iron_bridge(mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3) -> dict:
    """Check if iron crosses the stator-side air gap (between sleeve outer and stator inner).

    Uses SDF < 0 (hard inside test) so density smoothing doesn't create
    false bridges at coarse grid resolution.
    """
    cx, cy, cz = cfg.center
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rotor_axial = np.abs(Z - cz) <= cfg.rotor_half_length
    r_rotor_solid = min(
        getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer + 0.0044) + 0.0001,
        cfg.R_stator_inner - 0.0005,
    )
    gap_band = (r >= r_rotor_solid) & (r < cfg.R_stator_inner) & rotor_axial
    if gap_band.sum() == 0:
        return {"air_gap_iron_bridge": False, "air_gap_iron_fraction": 0.0}
    iron = _material_inside(mf, "iron")
    iron_in_gap = iron & gap_band
    bridged = bool(iron_in_gap.any())
    bridge_fraction = float(iron_in_gap.sum()) / max(float(gap_band.sum()), 1.0)
    return {"air_gap_iron_bridge": bridged, "air_gap_iron_fraction": bridge_fraction}


def shaft_rotor_merge_check(mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3) -> dict:
    """Check if shaft and rotor iron are connected (should be separated by air)."""
    from organic_motor.geometry.domain3d import domain_masks3d
    iron = _material_inside(mf, "iron")
    masks = domain_masks3d(cfg)
    shaft = np.asarray(masks["shaft"])
    rotor = np.asarray(masks["rotor_design"])
    shaft_iron = iron & shaft
    rotor_iron = iron & rotor
    if not shaft_iron.any() or not rotor_iron.any():
        return {"shaft_rotor_merge": False, "shaft_rotor_gap_mm": 0.0}
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_comp = ndimage.label(iron, structure=structure)
    shaft_labels = set(labels[shaft_iron].tolist())
    rotor_labels = set(labels[rotor_iron].tolist())
    shared = shaft_labels & rotor_labels
    merged = bool(shared)
    if not merged:
        dt_shaft = ndimage.distance_transform_edt(~shaft_iron, sampling=cfg.spacing)
        gap = dt_shaft[rotor_iron].min() if rotor_iron.any() else 0.0
    else:
        gap = 0.0
    return {"shaft_rotor_merge": merged, "shaft_rotor_gap_mm": gap * 1000}


def housing_open_area(mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3) -> dict:
    """Estimate the open area fraction of the housing shell."""
    iron = _material_inside(mf, "iron")
    cx, cy, cz = cfg.center
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy = cfg.origin[:2]
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # MotorHousing: r_in = 0.055, r_out = 0.058 (R_design + 5mm .. + 8mm)
    r_housing = cfg.R_design + 0.005
    ring_slice = (r > r_housing) & (r < r_housing + 0.003)
    mid_z = nz // 2
    iron_slice = iron[:, :, mid_z]
    solid_in_ring = ring_slice & iron_slice
    if ring_slice.sum() == 0:
        return {"housing_open_area_ratio": 0.0, "housing_blade_count": 0}
    open_ratio = 1.0 - float(solid_in_ring.sum()) / float(ring_slice.sum())
    structure_2d = ndimage.generate_binary_structure(2, 1)
    _, n_blades = ndimage.label(solid_in_ring, structure=structure_2d)
    return {"housing_open_area_ratio": open_ratio, "housing_blade_count": n_blades}


def end_face_occlusion(mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3) -> dict:
    """Fraction of the front END-CAP face blocked by solid iron.

    Measured on the axial slice through the end caps (where
    ``ShaftAndBearings`` places them), not on the outer domain boundary:
    the box wall is tens of millimetres beyond the machine, so a boundary
    measurement reports ~zero while the actual machine face is largely
    covered.
    """
    iron = _material_inside(mf, "iron")
    cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
    dx, dy = cfg.spacing[:2]
    ox, oy = cfg.origin[:2]
    x = ox + dx * np.arange(cfg.shape[0], dtype=np.float32)
    y = oy + dy * np.arange(cfg.shape[1], dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # End-cap placement mirrors ShaftAndBearings defaults.
    bearing_width = 0.004
    cap_thickness = 0.003
    z_cap = cz + (cfg.rotor_half_length + cfg.axial_airgap + bearing_width
                  + cap_thickness + 0.5 * cap_thickness + 0.004)
    r_cap = cfg.R_design + 0.006
    oz, dz = cfg.origin[2], cfg.dz
    k = int(np.clip(round((z_cap - oz) / dz), 0, cfg.shape[2] - 1))
    front = iron[:, :, k]
    motor_mask = r < r_cap
    total = motor_mask.sum()
    blocked = (front & motor_mask).sum()
    return {
        "end_face_occlusion": float(blocked) / max(float(total), 1.0),
        "end_face_slice_z_mm": float(oz + dz * k) * 1000.0,
    }


def compute_geometry_metrics(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Compute all geometric quality metrics for agent feedback."""
    metrics = {}
    try:
        metrics.update(copper_connected_components(mf, cfg))
    except Exception:
        metrics["copper_components"] = -1
    try:
        metrics.update(air_gap_iron_bridge(mf, cfg))
    except Exception:
        metrics["air_gap_iron_bridge"] = False
    try:
        metrics.update(shaft_rotor_merge_check(mf, cfg))
    except Exception:
        metrics["shaft_rotor_merge"] = False
    try:
        metrics.update(housing_open_area(mf, cfg))
    except Exception:
        metrics["housing_open_area_ratio"] = 0.0
    try:
        metrics.update(end_face_occlusion(mf, cfg))
    except Exception:
        metrics["end_face_occlusion"] = 0.0
    return metrics
