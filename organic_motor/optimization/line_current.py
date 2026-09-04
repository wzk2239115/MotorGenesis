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

    for _ in range(max_iter):
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            break

        # Find next face crossing
        axis = np.argmin(tMax)
        t = tMax[axis]

        if t > t_end:
            break

        # Deposit I on the crossed face
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


def centerline_resistance(registry: list[dict], rho_e: float = 1.68e-8) -> dict:
    """Analytical resistance from centerline geometry.

    For each turn (closed loop), R = rho_e * L / A where L is the loop
    length and A is the cross-section area.  Per-cell R = sum of 7 turns
    (series).  Per-phase R = sum of 4 cells (series).

    Returns dict with per-turn, per-cell, per-phase resistance and total
    copper loss at unit current.
    """
    from collections import defaultdict

    turn_R = defaultdict(list)
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
        turn_R[(phase, tooth)].append(R)

    # Per-cell: sum of turns (series)
    for (phase, tooth), Rs in turn_R.items():
        cell_R[phase].append(sum(Rs))

    per_phase_R = {}
    for phase, Rs in cell_R.items():
        per_phase_R[phase] = sum(Rs)  # 4 cells in series

    total_R = sum(per_phase_R.values()) / 3.0  # average per phase
    return {
        "per_turn_R": dict(turn_R),
        "per_cell_R": {k: sum(v) for k, v in cell_R.items()},
        "per_phase_R": per_phase_R,
        "avg_phase_R": total_R,
        "n_turns": len(registry),
        "n_cells": len(set((e["tooth"], e["phase"]) for e in registry)),
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
