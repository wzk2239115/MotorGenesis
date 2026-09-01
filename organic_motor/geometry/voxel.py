"""Voxelisation of the optimised 2-D topology (extrusion to 3-D).

The continuous density fields are thresholded into a binary voxel grid and
extruded along z to obtain a simple manufacturable 3-D representation (the
phase-1 placeholder for a full 3-D voxel design).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from organic_motor.config import MotorConfig


@dataclass
class VoxelVolume:
    iron: np.ndarray      # (N, N, Nz) bool
    pm: np.ndarray        # (N, N, Nz) bool
    spacing: tuple[float, float, float]   # (dx, dy, dz) [m]
    origin: tuple[float, float, float]    # (x0, y0, z0) [m]


def densities_to_voxels(rho_iron, rho_pm, cfg: MotorConfig,
                        thickness: float = 0.02, nz: int = 8,
                        threshold: float = 0.5) -> VoxelVolume:
    ri = np.asarray(rho_iron) > threshold
    rp = np.asarray(rho_pm) > threshold
    ri = ri & ~rp          # disambiguate overlapping voxels (iron wins over air)

    iron = np.repeat(ri[..., None], nz, axis=2)
    pm = np.repeat(rp[..., None], nz, axis=2)

    spacing = (cfg.h, cfg.h, thickness / nz)
    origin = (-cfg.L, -cfg.L, 0.0)
    return VoxelVolume(iron=iron, pm=pm, spacing=spacing, origin=origin)