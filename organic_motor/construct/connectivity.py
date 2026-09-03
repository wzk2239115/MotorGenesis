"""Connectivity graphs for constructed motors.

LEAP 71's deepest lesson (ShapeKernel / HelixHeatX): connectivity should
be ENCODED INTO THE GROWTH TOPOLOGY -- every new solid grows from an
anchor with deliberate overlap and voxel-union -- not checked afterwards
and hoped for.  This module provides the audit layer that matches that
doctrine: separate connectivity graphs per physical network, because a
motor deliberately wants DIFFERENT connectivity in each:

  STRUCTURAL  every load-bearing solid must trace back to an anchor
              (rotor side: the shaft; stator side: the housing ring).
              Floating islands are pruned automatically.

  ELECTRICAL  conductors must follow the circuit topology: each phase
              one connected network, phases mutually insulated
              (see phase_verify).

  COOLANT     every coolant region must run inlet -> outlet with no
              trapped voids, separated from solids by a wall.

  MAGNETIC    connectivity follows flux paths; deliberate segmentation
              (eddy-current slits) is allowed, so no connectivity is
              enforced here -- only reported.

Also implements the minimum-connection-thickness audit (local feature
size via distance transform), because geometric connection does not
imply structural connection: a 0.2mm neck is connected but snaps.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField


def _grids(cfg: MotorConfig3D):
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt((X - cfg.center[0]) ** 2 + (Y - cfg.center[1]) ** 2)
    return X, Y, Z, r


def anchor_masks(cfg: MotorConfig3D):
    """Structural anchors: the shaft (rotor side) and the housing ring."""
    _X, _Y, Z, r = _grids(cfg)
    shaft = r < (cfg.R_shaft + 0.001)
    housing = r > (cfg.R_design + 0.003)
    return shaft, housing


def structural_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Structural connectivity: anchor tracing, floating islands, necks.

    Two ownership graphs with separate anchors:
      - ROTOR side (rotor iron + PM + sleeve): must trace to the SHAFT.
      - STATOR side (yoke + housing + caps + jacket walls): must trace to
        the HOUSING ring.

    A component counts as anchored only if the component that actually
    contains rotor-iron (resp. stator-iron) touches the corresponding
    anchor -- merely "some component somewhere touches the shaft" is not a
    load path.  Components touching neither anchor are floating islands.
    """
    iron = mf.sdfs.get("iron")
    pm = mf.sdfs.get("pm")
    solid = np.zeros(cfg.shape, dtype=bool)
    if iron is not None:
        solid |= iron.sdf < 0.0
    if pm is not None:
        solid |= pm.sdf < 0.0
    if not solid.any():
        return {
            "structural_components": 0, "floating_islands": 0,
            "rotor_anchored": False, "stator_anchored": False,
            "anchored_fraction": 0.0, "min_neck_mm": 0.0,
        }

    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_comp = ndimage.label(solid, structure=structure)
    _X, _Y, Z, r = _grids(cfg)
    shaft, housing = anchor_masks(cfg)
    shaft_anchor = shaft & solid
    housing_anchor = housing & solid

    # Component identity by BODY, not by per-voxel radius: end caps and
    # bearing housings legitimately span the air-gap radius at the machine
    # ends, so a radial split misclassifies them.  A component is "rotor"
    # iff it contains rotor-body iron (inside the gap radius, within the
    # rotor stack), "stator" iff it contains yoke-body iron.
    r_split = 0.5 * (getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer) + cfg.R_stator_inner)
    z = np.asarray(Z)
    rotor_body = solid & (r < r_split) & (np.abs(z - cfg.center[2]) <= cfg.rotor_half_length + 2.0 * cfg.dz)
    yoke_body = solid & (r >= cfg.R_stator_inner) & (np.abs(z - cfg.center[2]) <= cfg.stator_half_length + 2.0 * cfg.dz)

    shaft_labels = set(labels[shaft_anchor].tolist()) - {0} if shaft_anchor.any() else set()
    housing_labels = set(labels[housing_anchor].tolist()) - {0} if housing_anchor.any() else set()
    rotor_labels = set(labels[rotor_body].tolist()) - {0} if rotor_body.any() else set()
    yoke_labels = set(labels[yoke_body].tolist()) - {0} if yoke_body.any() else set()

    rotor_anchored = bool(rotor_labels & shaft_labels) if rotor_labels else False
    stator_anchored = bool(yoke_labels & housing_labels) if yoke_labels else False
    # A single component containing BOTH the rotor body and the yoke is a
    # genuine air-gap bridge (rotor and stator fused = crash).  Likewise a
    # rotor-body component reaching the housing ring (or a yoke component
    # reaching the shaft) is a structural short-circuit, not an anchor.
    gap_bridge = bool(rotor_labels & yoke_labels)
    cross_bridge = bool(
        gap_bridge
        or (rotor_labels & housing_labels)
        or (yoke_labels & shaft_labels)
    )

    anchored = shaft_labels | housing_labels
    island_labels = set(range(1, n_comp + 1)) - anchored

    total = int(solid.sum())
    island_volume = int(sum((labels == i).sum() for i in island_labels))

    # Minimum connection thickness: local feature size (2x inscribed
    # radius) over the anchored solid.  The 5th percentile proxies the
    # narrowest necks without full skeletonisation.
    anchored_mask = np.isin(labels, list(anchored)) if anchored else np.zeros_like(solid)
    neck = 0.0
    if anchored_mask.any():
        edt = ndimage.distance_transform_edt(anchored_mask, sampling=cfg.spacing)
        thickness = 2.0 * edt[anchored_mask]
        neck = float(np.percentile(thickness, 5))

    return {
        "structural_components": int(n_comp),
        "floating_islands": int(len(island_labels)),
        "floating_island_fraction": island_volume / max(total, 1),
        "rotor_anchored": rotor_anchored,
        "stator_anchored": stator_anchored,
        "rotor_stator_cross_bridge": cross_bridge,
        "air_gap_solid_bridge": gap_bridge,
        "anchored_fraction": (total - island_volume) / max(total, 1),
        "min_neck_mm": neck * 1000.0,
    }


def prune_floating_islands(mf: MaterialField, cfg: MotorConfig3D) -> MaterialField:
    """Delete structural solids that trace to no anchor (in-place edit).

    The LEAP 71 growth doctrine makes this a no-op when geometry is
    generated from anchors; it exists so the agent cannot ship a design
    with disconnected metal.  Copper is untouched (the electrical graph
    owns it) and the removal keeps a positive SDF margin so the density
    smoothing cannot resurrect the island.
    """
    from organic_motor.construct.field import SDFVoxelField

    iron = mf.sdfs.get("iron")
    pm = mf.sdfs.get("pm")
    solid = np.zeros(cfg.shape, dtype=bool)
    if iron is not None:
        solid |= iron.sdf < 0.0
    if pm is not None:
        solid |= pm.sdf < 0.0
    if not solid.any():
        return mf

    structure = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(solid, structure=structure)
    shaft, housing = anchor_masks(cfg)
    anchored = set()
    for anchor in (shaft & solid, housing & solid):
        if anchor.any():
            anchored |= set(labels[anchor].tolist())
    anchored.discard(0)
    island = solid & ~np.isin(labels, list(anchored)) if anchored else solid
    if not island.any():
        return mf

    margin = 2.0 * min(cfg.spacing)
    for material in ("iron", "pm"):
        sdf_field = mf.sdfs.get(material)
        if sdf_field is None:
            continue
        removed = island & (sdf_field.sdf < 0.0)
        sdf = sdf_field.sdf.copy()
        sdf[removed] = np.maximum(sdf[removed], margin)
        mf.sdfs[material] = SDFVoxelField(
            sdf=sdf.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin
        )
    return mf


def coolant_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Coolant connectivity on the DEDICATED coolant material.

    The coolant network must run inlet -> outlet: a component qualifies as
    a through-flow path only if it reaches axially beyond BOTH ends of the
    stator stack (where the inlet/outlet stubs live).  Components fully
    enclosed by solid that reach neither end are trapped voids -- powder-
    removal and pressure-fill failure in manufacture.

    Falls back to the "air" material (with a flag) only when no dedicated
    coolant network exists, e.g. for legacy fields where all voids were
    one material.
    """
    coolant = mf.sdfs.get("coolant")
    if coolant is not None:
        void = coolant.sdf < 0.0
        dedicated = True
    else:
        air = mf.sdfs.get("air")
        if air is None:
            return {"trapped_voids": 0, "coolant_components": 0,
                    "through_flow_networks": 0, "dedicated_coolant": False}
        void = air.sdf < 0.0
        dedicated = False
    if not void.any():
        return {"trapped_voids": 0, "coolant_components": 0,
                "through_flow_networks": 0, "dedicated_coolant": dedicated}

    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_comp = ndimage.label(void, structure=structure)
    _X, _Y, Z, _r = _grids(cfg)
    cz = cfg.center[2]
    hz = cfg.stator_half_length
    low_end = Z <= (cz - hz + 2.0 * cfg.dz)
    high_end = Z >= (cz + hz - 2.0 * cfg.dz)
    boundary = np.zeros(cfg.shape, dtype=bool)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True

    trapped = 0
    through = 0
    open_components = 0
    for i in range(1, n_comp + 1):
        comp = labels == i
        if (comp & boundary).any():
            open_components += 1
        has_low = bool((comp & low_end).any())
        has_high = bool((comp & high_end).any())
        if has_low and has_high:
            through += 1
        elif not (has_low or has_high) and comp.sum() >= 100:
            # ~85 mm^3: print-blocking pockets only
            trapped += 1
    return {
        "trapped_voids": int(trapped),
        "coolant_components": int(n_comp),
        "coolant_open_components": int(open_components),
        "through_flow_networks": int(through),
        "dedicated_coolant": bool(dedicated),
    }


def connectivity_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Full four-graph connectivity audit (magnetic graph is reported,
    not enforced: segmentation is deliberate)."""
    report = {}
    try:
        report.update(structural_report(mf, cfg))
    except Exception:
        report.update({"structural_components": -1, "floating_islands": -1})
    try:
        report.update(coolant_report(mf, cfg))
    except Exception:
        report.update({"trapped_voids": -1})
    return report
