"""Material volumes used by geometry export.

``density3d_to_volume`` preserves native ``(Nx, Ny, Nz)`` continuous phase
fields.  ``densities_to_voxels`` is retained only as the legacy 2-D extrusion
API; it must not be used to claim a native three-dimensional result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from organic_motor.config import MotorConfig
from organic_motor.config3d import MotorConfig3D

if TYPE_CHECKING:
    from organic_motor.topology.density3d import TopologyFields3D


@dataclass
class VoxelVolume:
    iron: np.ndarray      # (Nx, Ny, Nz), bool or continuous [0, 1]
    pm: np.ndarray        # (Nx, Ny, Nz), bool or continuous [0, 1]
    spacing: tuple[float, float, float]   # (dx, dy, dz) [m]
    origin: tuple[float, float, float]    # (x0, y0, z0) [m]
    copper: np.ndarray | None = None
    air: np.ndarray | None = None
    coolant: np.ndarray | None = None
    insulator: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = np.asarray(self.iron).shape
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError("material volumes must have non-empty shape (Nx, Ny, Nz)")
        for name in ("pm", "copper", "air", "coolant", "insulator"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != shape:
                raise ValueError(f"{name} shape must match iron shape {shape}")
        if len(self.spacing) != 3 or any(h <= 0 for h in self.spacing):
            raise ValueError("spacing must contain three positive values")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(np.asarray(self.iron).shape)

    @property
    def materials(self) -> dict[str, np.ndarray]:
        result = {"iron": np.asarray(self.iron)}
        if self.copper is not None:
            result["copper"] = np.asarray(self.copper)
        result["pm"] = np.asarray(self.pm)
        if self.insulator is not None:
            result["insulator"] = np.asarray(self.insulator)
        if self.coolant is not None:
            result["coolant"] = np.asarray(self.coolant)
        return result


def density3d_to_volume(
    fields: TopologyFields3D | Mapping[str, np.ndarray],
    cfg: MotorConfig3D,
    *,
    clip: bool = True,
) -> VoxelVolume:
    """Create an export volume from genuine continuous 3-D phase densities.

    ``fields`` may be :class:`TopologyFields3D` or a mapping with either
    ``rho_*`` keys or short material names.  No axis is repeated or extruded.
    """
    def read(name: str) -> np.ndarray:
        if isinstance(fields, Mapping):
            key = f"rho_{name}" if f"rho_{name}" in fields else name
            if key not in fields:
                raise KeyError(f"missing three-dimensional density {key!r}")
            value = fields[key]
        else:
            value = getattr(fields, f"rho_{name}")
        array = np.asarray(value, dtype=np.float32)
        if array.shape != cfg.shape:
            raise ValueError(
                f"rho_{name} must have native shape {cfg.shape}, got {array.shape}"
            )
        return np.clip(array, 0.0, 1.0) if clip else array

    return VoxelVolume(
        iron=read("iron"),
        pm=read("pm"),
        spacing=tuple(float(v) for v in cfg.spacing),
        origin=tuple(float(v) for v in cfg.origin),
        copper=read("copper"),
        air=read("air"),
    )


def densities_to_voxels(rho_iron, rho_pm, cfg: MotorConfig,
                        thickness: float = 0.02, nz: int = 8,
                        threshold: float = 0.5) -> VoxelVolume:
    """Legacy 2-D-to-3-D extrusion kept for backwards compatibility."""
    ri = np.asarray(rho_iron) > threshold
    rp = np.asarray(rho_pm) > threshold
    ri = ri & ~rp          # disambiguate overlapping voxels (iron wins over air)

    iron = np.repeat(ri[..., None], nz, axis=2)
    pm = np.repeat(rp[..., None], nz, axis=2)

    spacing = (cfg.h, cfg.h, thickness / nz)
    origin = (-cfg.L, -cfg.L, 0.0)
    return VoxelVolume(iron=iron, pm=pm, spacing=spacing, origin=origin)


densities3d_to_voxels = density3d_to_volume
native_density_volume = density3d_to_volume