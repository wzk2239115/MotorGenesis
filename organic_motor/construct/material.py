"""Multi-material voxel container with priority-resolved Booleans.

A :class:`MaterialField` holds one signed-distance field per material, all on
a shared grid.  Adding geometry with ``priority=True`` subtracts it from
every other material first, so the stored solids are disjoint by construction
-- the LEAP 71 ``each voxel carries a material`` invariant, realised as a set
of non-overlapping SDFs rather than a label array.

The container is the bridge between the constructive layer and the
differentiable critic: ``to_densities`` produces the four continuous phase
fields that ``forward3d`` consumes, and ``to_volume`` produces the
:class:`VoxelVolume` the existing export pipeline expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from organic_motor.construct.field import (
    SDFVoxelField,
    boolean_add,
    boolean_subtract,
)


MATERIALS = ("iron", "copper", "pm")
LABELS = {**{"air": 0}, **{m: i + 1 for i, m in enumerate(MATERIALS)}}


@dataclass
class MaterialField:
    """A disjoint union of per-material signed-distance voxel fields."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    sdfs: dict[str, SDFVoxelField] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(n < 2 for n in self.shape):
            raise ValueError("shape must be three integers >= 2")

    def add(self, geometry: SDFVoxelField, material: str, priority: bool = True) -> "MaterialField":
        """Add ``geometry`` as ``material``, winning overlaps when ``priority``.

        With ``priority=True`` the geometry is subtracted from every other
        material first, so this material owns the voxels it occupies.  This is
        the operation that makes a motor assemble cleanly: the shaft bore
        subtracts from the rotor iron, the magnet poles subtract from the
        stator, the cooling channels subtract from the housing.
        """
        self._check_grid(geometry)
        if priority:
            for other, sdf in list(self.sdfs.items()):
                if other != material:
                    self.sdfs[other] = boolean_subtract(sdf, geometry)
        if material in self.sdfs:
            self.sdfs[material] = boolean_add(self.sdfs[material], geometry)
        else:
            self.sdfs[material] = SDFVoxelField(
                sdf=geometry.sdf.copy(), spacing=self.spacing, origin=self.origin
            )
        return self

    def subtract(self, geometry: SDFVoxelField, material: str) -> "MaterialField":
        """Remove ``geometry`` from one material only."""
        self._check_grid(geometry)
        if material in self.sdfs:
            self.sdfs[material] = boolean_subtract(self.sdfs[material], geometry)
        return self

    def _check_grid(self, geometry: SDFVoxelField) -> None:
        if geometry.shape != self.shape or geometry.spacing != self.spacing:
            raise ValueError(
                f"geometry grid {geometry.shape}@{geometry.spacing} does not "
                f"match container {self.shape}@{self.spacing}"
            )

    def to_densities(self, bandwidth: float | None = None) -> dict[str, np.ndarray]:
        """Continuous ``[0, 1]`` phase densities, partitioned with air as complement.

        Because priority Booleans keep the solids disjoint, each material's
        density is an independent smoothstep of its SDF and the air fraction is
        the residual.  These are exactly the ``rho_*`` fields the differentiable
        solver consumes.
        """
        solids: dict[str, np.ndarray] = {}
        for material in MATERIALS:
            if material in self.sdfs:
                solids[material] = self.sdfs[material].to_density(bandwidth)
            else:
                solids[material] = np.zeros(self.shape, dtype=np.float32)
        total = sum(solids.values())
        scale = np.maximum(1.0, total)
        for material in MATERIALS:
            solids[material] = (solids[material] / scale).astype(np.float32)
        solids["air"] = np.clip(1.0 - sum(solids[m] for m in MATERIALS), 0.0, 1.0).astype(np.float32)
        return solids

    def label(self) -> np.ndarray:
        """Integer material labels: 0 air, 1 iron, 2 copper, 3 PM."""
        label = np.zeros(self.shape, dtype=np.int8)
        best = np.full(self.shape, 0.0, dtype=np.float32)
        for material in MATERIALS:
            if material not in self.sdfs:
                continue
            inside = self.sdfs[material].sdf < 0.0
            deeper = inside & (self.sdfs[material].sdf < best)
            label = np.where(deeper, LABELS[material], label)
            best = np.where(inside, np.minimum(best, self.sdfs[material].sdf), best)
        return label

    def to_volume(self):
        """Build a :class:`VoxelVolume` for the existing export pipeline."""
        from organic_motor.geometry.voxel import VoxelVolume

        densities = self.to_densities()
        return VoxelVolume(
            iron=densities["iron"],
            pm=densities["pm"],
            spacing=self.spacing,
            origin=self.origin,
            copper=densities["copper"],
            air=densities["air"],
        )

    def materials_present(self) -> list[str]:
        return [m for m in MATERIALS if m in self.sdfs]
