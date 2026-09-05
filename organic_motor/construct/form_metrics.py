"""Reference-form metrics: does the geometry look like a printed stator cell?

The connectivity verdicts prove the electrical/structural networks are
consistent -- they do NOT prove the form resembles a LEAP 71 printed
stator (12 visible electromagnetic cells, ~6-8 copper bands per cell,
exposed copper, large negative space, a domed envelope).  These metrics
measure the FORM itself, against the reference morphology, so a
"fixed-annular-sector CSG that happens to pass connectivity" cannot hide.

All metrics are evaluated on the CONSTRUCTION grid (where the geometry is
actually resolved), the same grid the topology verdicts use.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField


def _stator_zone(cfg: MotorConfig3D) -> np.ndarray:
    """Boolean mask of the stator envelope (r in the winding/yoke band, 3-D)."""
    nx, ny, nz = cfg.shape
    dx, dy, _dz = cfg.spacing
    ox, oy, _oz = cfg.origin
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    zone2d = (r >= cfg.R_stator_inner) & (r <= cfg.R_design + 0.002)
    return np.broadcast_to(zone2d[..., None], cfg.shape).copy()


def cell_report(mf: MaterialField, cfg: MotorConfig3D, n_cells_expected: int = 12) -> dict:
    """Count visible electromagnetic cells by counting iron teeth.

    The iron teeth ARE the cells: each tooth + its copper coil is one
    electromagnetic cell.  We count distinct angular clusters of iron at
    the tooth radius (between R_stator_inner and R_winding_inner) — this
    is robust to the copper topology (monolithic frame vs swept bands)
    because it counts the magnetic structure, not the electrical.
    """
    iron = mf.sdfs.get("iron")
    if iron is None:
        return {"form_cells": 0, "n_cells_expected": n_cells_expected}
    nx, ny, nz = cfg.shape
    dx, dy = cfg.spacing[0], cfg.spacing[1]
    ox, oy = cfg.origin[0], cfg.origin[1]
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    # Tooth band: between stator inner and winding outer
    tooth_band = (r >= cfg.R_stator_inner) & (r <= cfg.R_winding_inner + 0.002)
    # Take a z-slice through the stack
    cz = cfg.center[2]
    kz = int(round((cz - cfg.origin[2]) / cfg.spacing[2]))
    iron_slice = (iron.sdf[:, :, kz] < 0.0) & tooth_band
    if not iron_slice.any():
        return {"form_cells": 0, "n_cells_expected": n_cells_expected}
    # Angular histogram of iron teeth
    hist, _ = np.histogram(theta[iron_slice], bins=n_cells_expected * 6,
                           range=(-np.pi, np.pi))
    # Teeth are tall narrow peaks; use a high threshold to find gaps
    thr = 0.15 * hist.max() if hist.max() > 0 else 1
    above = hist > thr
    peaks = int(np.sum(above[1:] & ~above[:-1]))
    # If all above (monolithic ring), fallback: count via angular profile
    if peaks == 0 and hist.max() > 0:
        peaks = 1 if above.all() else 0
    return {"form_cells": peaks, "n_cells_expected": n_cells_expected}


def copper_band_report(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Count distinct copper bands (turns) per cell from the VISIBLE grid.

    Counts connected components in the end-region above the stack at
    a tooth cross-section.  This is grid-dependent (requires the band
    thickness to be resolved), unlike reading from the registry — but
    it catches ribbon SDF fragmentation that registry-only counting
    would miss.
    """
    copper = mf.sdfs.get("copper")
    if copper is None:
        return {"bands_per_cell": 0}
    ny = cfg.shape[1]
    y = cfg.origin[1] + cfg.spacing[1] * np.arange(ny, dtype=np.float32)
    j0 = int(np.argmin(np.abs(y)))
    cz = cfg.center[2]
    hz = cfg.stator_half_length
    z = cfg.origin[2] + cfg.spacing[2] * np.arange(cfg.shape[2], dtype=np.float32)
    above = z > (cz + hz)
    slab = (copper.sdf[:, j0, :][:, above]) < 0.0
    if not slab.any():
        return {"bands_per_cell": 0, "copper_clusters_raw": 0}
    structure = ndimage.generate_binary_structure(2, 1)
    labels, n = ndimage.label(slab, structure=structure)
    sizes = ndimage.sum(slab, labels, range(1, n + 1)) if n else np.array([])
    # Count only bands with at least 2 voxels (filter noise)
    min_band = 2
    bands = int((sizes >= min_band).sum()) if n else 0
    return {"bands_per_cell": bands, "copper_clusters_raw": int(n)}


def exposed_copper_fraction(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Fraction of copper surface NOT buried under insulator.

    LEAP images show copper as the PRIMARY visible feature; an insulator
    sock wrapping the whole coil hides it.  We measure the copper voxels
    with no insulator within one cell -- the "exposed" copper.
    """
    copper = mf.sdfs.get("copper")
    insulator = mf.sdfs.get("insulator")
    if copper is None:
        return {"exposed_copper_fraction": 0.0}
    cu = copper.sdf < 0.0
    if not cu.any():
        return {"exposed_copper_fraction": 0.0}
    if insulator is not None:
        ins_near = ndimage.binary_dilation(insulator.sdf < 0.0,
                                           structure=ndimage.generate_binary_structure(3, 1),
                                           iterations=1)
        exposed = cu & ~ins_near
    else:
        exposed = cu
    return {"exposed_copper_fraction": float(exposed.sum()) / float(cu.sum())}


def open_area_fraction(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Fraction of the stator envelope that is NEGATIVE space (void).

    A solid barrel reads ~0; a petal/cell stator with large inter-cell
    windows reads 0.3-0.6 (the LEAP look).
    """
    zone = _stator_zone(cfg)
    solid = np.zeros(cfg.shape, dtype=bool)
    for name in ("iron", "copper", "pm", "insulator"):
        field = mf.sdfs.get(name)
        if field is not None:
            solid |= field.sdf < 0.0
    void = zone & ~solid
    return {"open_area_fraction": float(void.sum()) / max(float(zone.sum()), 1.0)}


def dome_ratio(mf: MaterialField, cfg: MotorConfig3D) -> dict:
    """Height of the copper dome above the stack / stator outer radius.

    A flat end-turn reads ~0; an arched multi-band dome reads 0.2-0.6
    (the LEAP "dome taller than it is thick" look).
    """
    copper = mf.sdfs.get("copper")
    if copper is None:
        return {"dome_ratio": 0.0}
    cu = copper.sdf < 0.0
    if not cu.any():
        return {"dome_ratio": 0.0}
    _X, _Y, Z, _r = _grids(cfg)
    cz = cfg.center[2]
    hz = cfg.stator_half_length
    above = cu & (Z > cz + hz)
    if not above.any():
        return {"dome_ratio": 0.0}
    dome_h = float(Z[above].max() - (cz + hz))
    return {"dome_ratio": dome_h / max(cfg.R_design, 1e-9)}


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


def form_metrics(mf: MaterialField, cfg: MotorConfig3D, n_cells_expected: int = 12) -> dict:
    """All reference-form metrics in one report."""
    report = {}
    report.update(cell_report(mf, cfg, n_cells_expected))
    report.update(copper_band_report(mf, cfg))
    report.update(exposed_copper_fraction(mf, cfg))
    report.update(open_area_fraction(mf, cfg))
    report.update(dome_ratio(mf, cfg))
    # Honest first-pass targets (LEAP reference morphology):
    #   cells = n_expected, bands 6-8, exposed copper > 0.5,
    #   open area > 0.25, dome ratio > 0.15.
    report["targets"] = {
        "cells": n_cells_expected,
        "bands_min": 6, "bands_max": 8,
        "exposed_copper_min": 0.5,
        "open_area_min": 0.25,
        "dome_ratio_min": 0.15,
    }
    return report
