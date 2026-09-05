"""Conservative line-current deposition from centerline geometry.

The expert insight: for a motor whose copper is swept tubes (0.7mm
radius), the electromagnetic solver does NOT need the tubes to be
resolved on the coarse physics grid.  The solver needs the correct
CURRENT PATH and AMPERE-TURNS.  This module deposits line currents
from the high-resolution centerlines directly onto the coarse Maxwell
grid, using a compact-support tent kernel that spreads each segment's
current over ~2-3 coarse cells.

The centerline registry (stored in ``mf.metadata["centerline_registry"]``
by :class:`StatorCellArray`) is the single source of truth: each entry
carries the 3-D polyline, phase, polarity, turn index, cross-section
area, and band radius.

For a closed loop carrying current ``I``, the deposition is::

    J(x) = sum_segments integral I * tangent(s) * kernel3D(x - x(s)) ds

where ``kernel3D`` is a separable 1-D tent that integrates to 1 over
space, so the cross-section integral of ``J`` equals ``I`` exactly
(in the continuum; discretization introduces small error).

Optionally, a Hodge projection ``J <- J - grad(phi)`` where
``nabla^2 phi = div(J)`` can clean up residual divergence from
discretization.  For closed loops with a symmetric kernel, ``div(J)``
is already small.
"""

from __future__ import annotations

import numpy as np

from organic_motor.config3d import MotorConfig3D


def _tent_kernel_1d(dist: float, h: float) -> float:
    """1-D tent function: (1/h)(1 - |d|/h) for |d| < h, else 0."""
    ad = abs(dist)
    if ad >= h:
        return 0.0
    return (1.0 / h) * (1.0 - ad / h)


def deposit_centerline_currents(
    cfg: MotorConfig3D,
    registry: list[dict],
    current_per_turn: float,
    phase_amplitudes: np.ndarray | None = None,
    kernel_half_voxels: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Deposit line currents from all centerlines onto the grid.

    Uses a **face-flux deposition** that is exactly divergence-free for
    closed loops: each segment is traced through the voxel grid (3-D
    DDA), and the current ``I`` is added to every face it crosses.  For
    a closed loop, every voxel the path enters it also exits, so
    ``div(J) = 0`` exactly (up to floating-point round-off).

    The face fluxes are then converted to cell-centred current density
    by dividing by the face area, and optionally smoothed with a
    compact tent kernel to spread the current over ~2-3 cells (the
    Maxwell solver needs a smooth source, not a delta on one face).

    Returns ``(J_total, phase_J)`` with shapes ``(Nx,Ny,Nz,3)`` and
    ``(3,Nx,Ny,Nz,3)``.
    """
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin

    # Face fluxes: Jx at (i+1/2,j,k), Jy at (i,j+1/2,k), Jz at (i,j,k+1/2)
    # Stored as (nx+1,ny,nz), (nx,ny+1,nz), (nx,ny,nz+1) per phase
    flux_x = [np.zeros((nx + 1, ny, nz), dtype=np.float64) for _ in range(3)]
    flux_y = [np.zeros((nx, ny + 1, nz), dtype=np.float64) for _ in range(3)]
    flux_z = [np.zeros((nx, ny, nz + 1), dtype=np.float64) for _ in range(3)]

    for entry in registry:
        phase = entry["phase"]
        polarity = entry["polarity"]
        pts = entry["points"]

        if phase_amplitudes is not None:
            amp = float(phase_amplitudes[phase])
        else:
            amp = 1.0
        I = current_per_turn * amp * polarity

        for seg in range(len(pts) - 1):
            p1 = pts[seg]
            p2 = pts[seg + 1]
            _trace_segment_faces(
                p1, p2, I,
                ox, oy, oz, dx, dy, dz,
                nx, ny, nz,
                flux_x[phase], flux_y[phase], flux_z[phase],
            )

    # Convert face fluxes to cell-centred J with divergence-preserving smoothing
    phase_J = np.zeros((3, nx, ny, nz, 3), dtype=np.float32)
    for p in range(3):
        # Smooth each flux ONLY in transverse directions (perpendicular
        # to the face normal) — this preserves div(J)=0 exactly because
        # the transverse smoothing commutes with the face-normal difference.
        fx = flux_x[p]
        fy = flux_y[p]
        fz = flux_z[p]
        if kernel_half_voxels > 0:
            # flux_x: smooth along y, z (not x)
            fx = _smooth_axis(fx, axis=1)
            fx = _smooth_axis(fx, axis=2)
            # flux_y: smooth along x, z (not y)
            fy = _smooth_axis(fy, axis=0)
            fy = _smooth_axis(fy, axis=2)
            # flux_z: smooth along x, y (not z)
            fz = _smooth_axis(fz, axis=0)
            fz = _smooth_axis(fz, axis=1)
        # Face-to-cell-centre: average adjacent faces, divide by face area
        jx = 0.5 * (fx[1:] + fx[:-1]) / (dy * dz)
        jy = 0.5 * (fy[:, 1:] + fy[:, :-1]) / (dx * dz)
        jz = 0.5 * (fz[:, :, 1:] + fz[:, :, :-1]) / (dx * dy)
        phase_J[p, ..., 0] = jx
        phase_J[p, ..., 1] = jy
        phase_J[p, ..., 2] = jz

    J_total = np.sum(phase_J, axis=0)
    return J_total.astype(np.float32), phase_J.astype(np.float32)


def _trace_segment_faces(
    p1, p2, I,
    ox, oy, oz, dx, dy, dz,
    nx, ny, nz,
    flux_x, flux_y, flux_z,
):
    """Trace a segment through the voxel grid, depositing I on each face crossed.

    Uses the Amanatides & Woo DDA algorithm.  For each face crossing,
    adds ``I`` to the face flux — the sign is determined by the crossing
    direction (positive when entering a new voxel in +direction).
    """
    # Convert to voxel coordinates
    x0 = (p1[0] - ox) / dx
    y0 = (p1[1] - oy) / dy
    z0 = (p1[2] - oz) / dz
    x1 = (p2[0] - ox) / dx
    y1 = (p2[1] - oy) / dy
    z1 = (p2[2] - oz) / dz

    direction = np.array([x1 - x0, y1 - y0, z1 - z0])
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return

    # Current voxel
    i = int(np.floor(x0))
    j = int(np.floor(y0))
    k = int(np.floor(z0))

    # Step directions
    step_x = 1 if direction[0] > 0 else (-1 if direction[0] < 0 else 0)
    step_y = 1 if direction[1] > 0 else (-1 if direction[1] < 0 else 0)
    step_z = 1 if direction[2] > 0 else (-1 if direction[2] < 0 else 0)

    # tMax: parametric value at next boundary
    # tDelta: parametric value between boundaries
    tMax = [1e30, 1e30, 1e30]
    tDelta = [1e30, 1e30, 1e30]

    if step_x != 0:
        boundary = (i + 1) if step_x > 0 else i
        tMax[0] = (boundary - x0) / direction[0] if direction[0] != 0 else 1e30
        tDelta[0] = abs(1.0 / direction[0])
    if step_y != 0:
        boundary = (j + 1) if step_y > 0 else j
        tMax[1] = (boundary - y0) / direction[1] if direction[1] != 0 else 1e30
        tDelta[1] = abs(1.0 / direction[1])
    if step_z != 0:
        boundary = (k + 1) if step_z > 0 else k
        tMax[2] = (boundary - z0) / direction[2] if direction[2] != 0 else 1e30
        tDelta[2] = abs(1.0 / direction[2])

    t_end = 1.0  # parametric end of segment
    max_iter = int(nx + ny + nz + 10)
    tol = 1e-9

    for _ in range(max_iter):
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            break

        # Find next face crossing(s) — handle ties (edge/corner crossings)
        # by processing ALL axes with tMax within tol of the minimum.
        t_min = min(tMax)
        if t_min > t_end:
            break

        for axis in range(3):
            if tMax[axis] > t_min + tol:
                continue
            if tMax[axis] > t_end:
                continue

            if axis == 0:  # x-face
                face_i = i + 1 if step_x > 0 else i
                if 0 <= face_i <= nx and 0 <= j < ny and 0 <= k < nz:
                    flux_x[face_i, j, k] += I * step_x
                i += step_x
            elif axis == 1:  # y-face
                face_j = j + 1 if step_y > 0 else j
                if 0 <= i < nx and 0 <= face_j <= ny and 0 <= k < nz:
                    flux_y[i, face_j, k] += I * step_y
                j += step_y
            else:  # z-face
                face_k = k + 1 if step_z > 0 else k
                if 0 <= i < nx and 0 <= j < ny and 0 <= face_k <= nz:
                    flux_z[i, j, face_k] += I * step_z
                k += step_z

            tMax[axis] += tDelta[axis]


def _smooth_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """One pass of 3-point average along a single axis (preserves div(J)=0
    when applied transverse to the face normal)."""
    out = arr.copy()
    s = [1] * arr.ndim
    s[axis] = 3
    kernel = np.array([0.25, 0.5, 0.25]).reshape(s)
    # Convolve along the chosen axis using slicing
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(1, -1)
    sl_prev = [slice(None)] * arr.ndim
    sl_prev[axis] = slice(0, -2)
    sl_next = [slice(None)] * arr.ndim
    sl_next[axis] = slice(2, None)
    out[tuple(sl)] = 0.25 * arr[tuple(sl_prev)] + 0.5 * arr[tuple(sl)] + 0.25 * arr[tuple(sl_next)]
    return out


def face_flux_divergence(
    cfg: MotorConfig3D,
    registry: list[dict],
    current_per_turn: float,
    phase_amplitudes: np.ndarray,
) -> float:
    """Max relative divergence of the face-flux field.

    For a closed loop, the face-flux DDA is exactly div-free.
    Returns max|div(flux)| / max|flux| as a fraction.
    """
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin

    flux_x = [np.zeros((nx + 1, ny, nz), dtype=np.float64) for _ in range(3)]
    flux_y = [np.zeros((nx, ny + 1, nz), dtype=np.float64) for _ in range(3)]
    flux_z = [np.zeros((nx, ny, nz + 1), dtype=np.float64) for _ in range(3)]

    for entry in registry:
        phase = entry["phase"]
        polarity = entry["polarity"]
        pts = entry["points"]
        if phase_amplitudes is not None:
            amp = float(phase_amplitudes[phase])
        else:
            amp = 1.0
        I = current_per_turn * amp * polarity
        n_pts = len(pts)
        for seg in range(n_pts):
            p1 = pts[seg]
            p2 = pts[(seg + 1) % n_pts]
            _trace_segment_faces(
                p1, p2, I,
                ox, oy, oz, dx, dy, dz,
                nx, ny, nz,
                flux_x[phase], flux_y[phase], flux_z[phase],
            )

    # Compute face-flux divergence: div = net outflow / cell_vol
    # For a closed loop, current conservation gives exactly zero.
    cell_vol = dx * dy * dz
    max_div = 0.0
    max_flux = 0.0
    for p in range(3):
        fx, fy, fz = flux_x[p], flux_y[p], flux_z[p]
        div = ((fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1]) +
               (fz[:, :, 1:] - fz[:, :, :-1])) / cell_vol
        max_div = max(max_div, float(np.max(np.abs(div))))
        max_flux = max(max_flux, float(np.max(np.abs(fx))),
                       float(np.max(np.abs(fy))), float(np.max(np.abs(fz))))
    return max_div / max(max_flux / cell_vol, 1e-12)


def centerline_resistance(registry: list[dict], rho_e: float = 1.68e-8) -> dict:
    """Analytical resistance from centerline geometry.

    For the serpentine format: each registry entry is ONE continuous
    path through all n_bands turns of a tooth.  The total length is the
    polyline length, and R = rho_e * L / A gives the cell resistance
    (all turns in series).  Per-phase R = sum of 4 cells (series).

    Returns dict with per-cell, per-phase resistance.
    """
    from collections import defaultdict

    cell_R = defaultdict(list)

    for entry in registry:
        pts = entry["points"]
        phase = entry["phase"]
        tooth = entry["tooth"]
        A = entry["cross_section_area"]

        L = 0.0
        for seg in range(len(pts) - 1):
            d = pts[seg + 1] - pts[seg]
            L += float(np.sqrt(d @ d))
        R = rho_e * L / A
        cell_R[phase].append(R)

    per_phase_R = {}
    for phase, Rs in cell_R.items():
        per_phase_R[phase] = sum(Rs)  # 4 cells in series

    total_R = sum(per_phase_R.values()) / 3.0  # average per phase
    n_turns = sum(e.get("n_turns", 7) for e in registry)
    return {
        "per_cell_R": dict(cell_R),
        "per_phase_R": per_phase_R,
        "avg_phase_R": total_R,
        "n_turns_total": n_turns,
        "n_cells": len(registry),
    }


def hodge_project(J: np.ndarray, cfg: MotorConfig3D,
                  max_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """Project J onto divergence-free space: J <- J - grad(phi).

    Solves nabla^2 phi = div(J) with homogeneous Neumann BCs, then
    subtracts grad(phi) from J.  Uses Gauss-Seidel for the Poisson
    solve.  The residual div(J) after projection is ~tol.
    """
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing

    # Compute div(J) using central differences
    divJ = np.zeros(cfg.shape, dtype=np.float32)
    divJ[1:-1, :, :] += (J[2:, :, :, 0] - J[:-2, :, :, 0]) / (2 * dx)
    divJ[:, 1:-1, :] += (J[:, 2:, :, 1] - J[:, :-2, :, 1]) / (2 * dy)
    divJ[:, :, 1:-1] += (J[:, :, 2:, 2] - J[:, :, :-2, 2]) / (2 * dz)

    # Solve nabla^2 phi = divJ (Neumann BC: zero gradient at boundary)
    phi = np.zeros(cfg.shape, dtype=np.float32)
    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
    for _ in range(max_iter):
        phi_old = phi.copy()
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    ip = min(i + 1, nx - 1)
                    im = max(i - 1, 0)
                    jp = min(j + 1, ny - 1)
                    jm = max(j - 1, 0)
                    kp = min(k + 1, nz - 1)
                    km = max(k - 1, 0)
                    denom = 2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2
                    rhs = divJ[i, j, k]
                    phi[i, j, k] = (
                        (phi[ip, j, k] + phi[im, j, k]) / dx2
                        + (phi[i, jp, k] + phi[i, jm, k]) / dy2
                        + (phi[i, j, kp] + phi[i, j, km]) / dz2
                        - rhs
                    ) / denom
        if np.max(np.abs(phi - phi_old)) < tol:
            break

    # Subtract grad(phi)
    J_proj = J.copy()
    J_proj[1:-1, :, :, 0] -= (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
    J_proj[:, 1:-1, :, 1] -= (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
    J_proj[:, :, 1:-1, 2] -= (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)

    return J_proj


def _deposit_joule_heat(
    cfg: MotorConfig3D,
    registry: list[dict],
    I_per_turn: float,
    amps: np.ndarray,
) -> np.ndarray:
    """Analytical copper loss I²ρL/A deposited as heat along centreline.

    For each centreline segment, the power dissipated is::

        P_seg = I² * ρ * L_seg / A

    where ``I`` is the phase current times the polarity, ``ρ`` is the
    copper resistivity, ``L_seg`` is the segment length, and ``A`` is
    the cross-section area.  This is grid-independent and exact for the
    printed conductor topology.

    The power is deposited into the nearest coarse cell (no spreading
    needed — heat diffusion smooths it on the thermal grid).

    Returns a 3-D array of volumetric heat density [W/m³].
    """
    rho_e = 1.0 / cfg.sigma_copper
    nx, ny, nz = cfg.shape
    q = np.zeros(cfg.shape, dtype=np.float32)
    cell_vol = cfg.cell_volume

    for entry in registry:
        pts = entry["points"]
        phase = entry["phase"]
        polarity = entry["polarity"]
        A = entry["cross_section_area"]
        turn_map = entry.get("turn_map")
        I_phase = float(amps[phase]) * polarity * I_per_turn

        if turn_map is not None:
            # Serpentine: single continuous path, all turns in series
            # carry the same current
            for seg in range(len(pts) - 1):
                d = pts[seg + 1] - pts[seg]
                L_seg = float(np.sqrt(d @ d))
                P_seg = I_phase ** 2 * rho_e * L_seg / A
                # Deposit at midpoint
                mid = 0.5 * (pts[seg] + pts[seg + 1])
                idx = ((mid - cfg.origin) / cfg.spacing).astype(int)
                i = np.clip(idx[0], 0, nx - 1)
                j = np.clip(idx[1], 0, ny - 1)
                k = np.clip(idx[2], 0, nz - 1)
                q[i, j, k] += P_seg / cell_vol
        else:
            # Legacy: independent closed loops per turn
            n_pts = len(pts)
            for seg in range(n_pts):
                p0 = pts[seg]
                p1 = pts[(seg + 1) % n_pts]
                d = p1 - p0
                L_seg = float(np.sqrt(d @ d))
                P_seg = I_phase ** 2 * rho_e * L_seg / A
                mid = 0.5 * (p0 + p1)
                idx = ((mid - cfg.origin) / cfg.spacing).astype(int)
                i = np.clip(idx[0], 0, nx - 1)
                j = np.clip(idx[1], 0, ny - 1)
                k = np.clip(idx[2], 0, nz - 1)
                q[i, j, k] += P_seg / cell_vol

    return q
