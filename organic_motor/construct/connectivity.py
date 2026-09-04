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
    """Structural anchors: the shaft (rotor side) and the yoke OUTER band.

    The stator anchor is the outer 2mm ring of the iron core
    (R_design-2mm .. R_design+0.5mm): the exoskeleton's base collar, walls
    and yoke all fuse into that band, and NOTHING exists beyond R_design
    any more (the old 55-58mm barrel is gone).  An over-wide anchor lets
    any detached fleck near the outside count as a load path, which is
    exactly the false-anchor failure the audit exists to catch.
    """
    _X, _Y, Z, r = _grids(cfg)
    shaft = r < (cfg.R_shaft + 0.001)
    housing = (r >= cfg.R_design - 0.002) & (r <= cfg.R_design + 0.0005)
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


def _external_air(cfg: MotorConfig3D, solid: np.ndarray) -> np.ndarray:
    """Dilated mask of the air that reaches the domain boundary.

    A port/vent "opens to the outside" iff it touches the SAME connected
    air body the domain boundary is made of -- not merely any interior
    void.  Used by both the coolant and the powder-escape audits.
    """
    from scipy import ndimage as _ndi

    air = ~solid
    structure = _ndi.generate_binary_structure(3, 1)
    labels, _ = _ndi.label(air, structure=structure)
    boundary = np.zeros(cfg.shape, dtype=bool)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    ext_labels = set(labels[boundary].tolist()) - {0} if (air & boundary).any() else set()
    ext = np.isin(labels, list(ext_labels)) if ext_labels else np.zeros_like(air)
    return _ndi.binary_dilation(ext, structure=structure, iterations=2)


def coolant_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Coolant connectivity on the DEDICATED coolant material.

    The printed cooling network (per-coil channels + supply/return rings +
    two ports) qualifies as through-flow when it has at least TWO distinct
    openings to the EXTERNAL air (in and out port) -- or, for legacy
    jacket designs, when one component spans both axial ends of the
    stator stack.  Components fully enclosed by solid that reach neither
    are trapped voids: pressure-fill and powder-removal failure in
    manufacture.

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

    solid = np.zeros(cfg.shape, dtype=bool)
    for name in ("iron", "copper", "pm", "insulator"):
        field = mf.sdfs.get(name)
        if field is not None:
            solid |= field.sdf < 0.0
    ext_air = _external_air(cfg, solid)

    min_opening_voxels = max(4, int((0.0015 / min(cfg.spacing)) ** 2))
    trapped = 0
    through = 0
    openings_total = 0
    for i in range(1, n_comp + 1):
        comp = labels == i
        contact = comp & ext_air
        if contact.any():
            cl, n_contact = ndimage.label(contact, structure=structure)
            sizes = ndimage.sum(contact, cl, range(1, n_contact + 1))
            openings = int((sizes >= min_opening_voxels).sum())
        else:
            openings = 0
        openings_total += openings
        has_low = bool((comp & low_end).any())
        has_high = bool((comp & high_end).any())
        if openings >= 2 or (has_low and has_high):
            through += 1
        elif openings == 0 and not (has_low or has_high) and comp.sum() >= 100:
            # ~85 mm^3: print-blocking pockets only
            trapped += 1
    return {
        "trapped_voids": int(trapped),
        "coolant_components": int(n_comp),
        "coolant_openings": int(openings_total),
        "through_flow_networks": int(through),
        "dedicated_coolant": bool(dedicated),
    }


def _am_solid(mf: MaterialField, cfg: MotorConfig3D) -> np.ndarray:
    """Solid mask for the additive-manufacture audits (stator print only).

    The rotor assembly (rotor iron, magnets, sleeve, hub spokes, shaft,
    bearing races -- everything inside the air-gap split radius, at any z)
    is a SEPARATELY printed / machined part assembled afterwards: its down
    faces are irrelevant to the stator print and would drown the audit
    (the magnets alone contribute ~6000 mm^2).
    """
    solid = np.zeros(cfg.shape, dtype=bool)
    for name in ("iron", "copper", "pm", "insulator"):
        field = mf.sdfs.get(name)
        if field is not None:
            solid |= field.sdf < 0.0
    _X, _Y, Z, r = _grids(cfg)
    r_split = 0.5 * (getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer) + cfg.R_stator_inner)
    return solid & (r >= r_split)


def powder_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Powder-escape audit for additive manufacture.

    EVERY void (air + coolant) must connect to the external air: in metal
    powder-bed printing, an enclosed pocket traps loose powder -- extra
    mass, contamination, and no way to clean it.  The coil channels and
    manifolds escape through their ports; the petal windows vent upwards;
    the slot-back air columns vent through the machine ends.
    """
    solid = _am_solid(mf, cfg)
    void = ~solid
    if not void.any():
        return {"trapped_pockets": 0, "escaped_fraction": 1.0}
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_comp = ndimage.label(void, structure=structure)
    ext_air = _external_air(cfg, solid)
    min_pocket = max(50, int((0.8 / min(cfg.spacing)) ** 3))
    trapped = 0
    trapped_volume = 0
    total_void = 0
    for i in range(1, n_comp + 1):
        comp = labels == i
        n_vox = int(comp.sum())
        total_void += n_vox
        if not (comp & ext_air).any() and n_vox >= min_pocket:
            trapped += 1
            trapped_volume += n_vox
    return {
        "trapped_pockets": int(trapped),
        "trapped_void_fraction": trapped_volume / max(total_void, 1),
        "escaped_fraction": 1.0 - trapped_volume / max(total_void, 1),
    }


def overhang_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Downward-facing surface audit (print direction +z, stator print).

    Classification of every down-facing face:
      - PLATE   the first solid layers rest on the build plate (the base
                collar's bottom IS the print start, not an overhang);
      - BORE    narrow ceilings whose local inscribed width <= ~5 mm --
                the coolant channel loops and manifold rings: printable
                self-supporting bores in metal AM;
      - WALL    thin-rib bridges: the crown's radial walls and the hub,
                spanning between supported regions (thin-wall strategy);
      - SPAN    everything else -- genuine support-less failures.

    The audit reports each class separately; the SPAN fraction gates.
    """
    solid = _am_solid(mf, cfg)
    if not solid.any():
        return {"down_faces": 0, "unsupported_fraction": 0.0,
                "unsupported_clusters": 0, "span_fraction": 0.0}

    below = np.zeros_like(solid)
    below[:, :, 1:] = solid[:, :, :-1]
    down_faces = solid & ~below

    nb = np.zeros_like(solid)
    nb[:, :, 1:] = solid[:, :, :-1]
    lateral = np.zeros_like(solid)
    lateral[1:, :, :] |= nb[:-1, :, :]   # neighbour -x solid one cell below
    lateral[:-1, :, :] |= nb[1:, :, :]   # neighbour +x
    lateral[:, 1:, :] |= nb[:, :-1, :]   # neighbour -y
    lateral[:, :-1, :] |= nb[:, 1:, :]   # neighbour +y
    unsupported = down_faces & ~lateral

    # Plate rows: the print's lowest solid layers sit on the build plate.
    ks = np.where(solid.any(axis=(0, 1)))[0]
    plate = np.zeros(cfg.shape, dtype=bool)
    if ks.size:
        plate[:, :, : ks[0] + 2] = True
    unsupported = unsupported & ~plate

    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_clusters = ndimage.label(unsupported, structure=structure)
    cell_mm2 = (cfg.dx * 1000.0) * (cfg.dz * 1000.0)
    bore_width = max(3, int(round(0.0025 / min(cfg.dx, cfg.dz))))
    span_cells = 0
    bore_cells = 0
    max_span = 0.0
    big = 0
    if n_clusters:
        sizes = ndimage.sum(unsupported, labels, range(1, n_clusters + 1))
        edt = ndimage.distance_transform_edt(unsupported, sampling=(cfg.dx, cfg.dy, cfg.dz))
        for i in range(1, n_clusters + 1):
            comp = labels == i
            width = float(edt[comp].max())
            if width <= 0.0025:
                bore_cells += int(sizes[i - 1])
            else:
                span_cells += int(sizes[i - 1])
                max_span = max(max_span, float(sizes[i - 1]) * cell_mm2)
                if sizes[i - 1] * cell_mm2 > 10.0:
                    big += 1
    total = max(int(down_faces.sum()), 1)
    return {
        "down_faces": int(down_faces.sum()),
        "plate_supported_fraction": float((down_faces & plate).sum()) / total,
        "bore_ceiling_fraction": bore_cells / total,
        "span_fraction": span_cells / total,
        "unsupported_fraction": (bore_cells + span_cells) / total,
        "unsupported_clusters": int(n_clusters),
        "large_span_clusters": big,
        "max_span_mm2": max_span,
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
