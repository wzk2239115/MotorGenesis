"""Phase connectivity verification for constructed motor windings.

Verifies the electrical topology of a constructed winding on the REAL
copper network -- including end turns and terminals, not a slot-sector
clip:

  - Each phase (A, B, C) is internally connected (exact component count
    matching the netlist's expected ring-per-layer-group topology)
  - Phases are mutually insulated (no voxel shared or adjacent between
    phases at the inspected resolution; the measured insulation gap is
    reported in mm)
  - Each phase has copper in the end regions (terminal availability)

This is the first validation gate before claiming a motor can rotate:
"rho_copper exists" does not prove "current can flow along a closed path,"
and a slot-sector clip that deletes end turns proves nothing about the
real circuit.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.winding_netlist import CoilNetlist


def _min_gap_mm(mask_a: np.ndarray, mask_b: np.ndarray, spacing) -> float:
    """Minimum physical gap [mm] between two disjoint boolean masks."""
    if not mask_a.any() or not mask_b.any():
        return float("inf")
    dt = ndimage.distance_transform_edt(~mask_a, sampling=spacing)
    return float(dt[mask_b].min()) * 1000.0


def verify_phase_connectivity(
    mf: MaterialField, cfg: MotorConfig3D, threshold: float = 0.3,
) -> dict:
    """Verify A/B/C phase connectivity and mutual insulation.

    Uses the exact per-phase voxel ownership recorded during winding
    construction (``winding_phase_sdf``) when available -- it covers slot
    conductors, end-turn arcs and strand bundles alike.  The legacy
    slot-sector belt clip is kept only as a fallback for old fields, and
    its result is flagged as such.

    Returns a dict with per-phase component counts (actual and expected),
    the measured minimum inter-phase insulation gap, cross-short flags and
    the overall ``passed`` verdict.
    """
    netlist = mf.metadata.get("winding_netlist") if hasattr(mf, "metadata") else None
    if not isinstance(netlist, CoilNetlist):
        return {
            "phase_a_components": -1, "phase_b_components": -1, "phase_c_components": -1,
            "phase_cross_short": False, "phase_terminals": [False] * 3,
            "passed": False, "reason": "no winding netlist in metadata",
        }

    phase_sdf = mf.metadata.get("winding_phase_sdf")
    structure = ndimage.generate_binary_structure(3, 1)
    phase_names = ["a", "b", "c"]
    result: dict = {"method": "phase_sdf_ownership" if phase_sdf is not None
                    else "belt_sector_clip_fallback"}
    phase_masks = []

    for i, name in enumerate(phase_names):
        if phase_sdf is not None:
            phase_copper = np.asarray(phase_sdf[i]) < 0.0
        else:
            belts = netlist.phase_belts_3d(cfg)
            copper_sdf = mf.sdfs.get("copper")
            densities = mf.to_densities()
            belt = belts[i] != 0
            phase_copper = (
                (copper_sdf.sdf < 0.0) if copper_sdf is not None
                else (densities["copper"] > threshold)
            ) & belt
        phase_masks.append(phase_copper)
        if not phase_copper.any():
            result[f"phase_{name}_components"] = 0
            result[f"phase_{name}_voxels"] = 0
            continue
        labels, n_comp = ndimage.label(phase_copper, structure=structure)
        sizes = ndimage.sum(phase_copper, labels, range(1, n_comp + 1))
        result[f"phase_{name}_components"] = int(n_comp)
        result[f"phase_{name}_voxels"] = int(phase_copper.sum())
        result[f"phase_{name}_largest_fraction"] = float(max(sizes)) / float(phase_copper.sum())

    # Cross-phase short: any voxel inside two phases' copper (only exact
    # with phase_sdf ownership; the fallback clip is disjoint by design).
    cross_short = False
    min_gap = float("inf")
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = phase_masks[i] & phase_masks[j]
            if overlap.any():
                cross_short = True
                result[f"overlap_{phase_names[i]}_{phase_names[j]}"] = int(overlap.sum())
            elif phase_masks[i].any() and phase_masks[j].any():
                min_gap = min(min_gap, _min_gap_mm(phase_masks[i], phase_masks[j], cfg.spacing))
    result["phase_cross_short"] = cross_short
    result["min_phase_gap_mm"] = min_gap if np.isfinite(min_gap) else 0.0

    # Terminals: phase copper present in the axial end slabs of the stack.
    cz = cfg.center[2]
    oz = cfg.origin[2]
    dz = cfg.dz
    nz = cfg.shape[2]
    z = oz + dz * np.arange(nz)
    end_slab = np.abs(np.abs(z - cz) - cfg.stator_half_length) <= 1.5 * dz
    for i, name in enumerate(phase_names):
        has_end = bool(phase_masks[i][:, :, end_slab].any()) if phase_masks[i].any() else False
        result[f"phase_{name}_has_terminal"] = has_end
    result["phase_terminals"] = [result.get(f"phase_{p}_has_terminal", False)
                                 for p in phase_names]

    expected = netlist.expected_phase_components()
    # P5 serpentine: one continuous path per tooth (all turns in series),
    # so expected = 1 component per tooth = 4 per phase.
    # (Legacy format had n_turns independent loops per tooth.)
    if hasattr(mf, "metadata"):
        reg = mf.metadata.get("centerline_registry")
        if reg and "turn_map" in reg[0]:
            pass  # serpentine: 1 path per tooth, expected stays [4,4,4]
        elif reg:
            n_turns = len(set(e["turn"] for e in reg))
            expected = expected * n_turns
    result["expected_components"] = [int(e) for e in expected]
    # Insulation must be resolvable: min_gap is a voxel-CENTRE distance, so
    # phases separated by less than ~one cell measure at ~1 cell -- a true
    # insulation gap must push the centre distance clearly beyond it.
    cell_mm = min(cfg.spacing) * 1000.0
    result["insulation_resolved"] = bool(min_gap > 1.5 * cell_mm) if not cross_short else False
    all_connected = all(
        result.get(f"phase_{n}_components", 0) == int(expected[i])
        for i, n in enumerate(phase_names)
    )
    has_terminals = all(result.get(f"phase_{n}_has_terminal", False) for n in phase_names)
    result["passed"] = (
        all_connected and not cross_short and has_terminals
        and result["insulation_resolved"]
    )
    return result
