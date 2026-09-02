"""Explicit three-phase winding netlist for constructed motors.

A real motor winding is not just "copper exists."  It has an electrical
topology: slots, poles, phases, coils, turns, series/parallel connections,
terminals and a star/delta wiring.  This module generates that topology
deterministically from slot/pole counts, so every copper voxel can be
attributed to a specific phase and coil side.

The netlist is the single source of truth shared between:
  - ``Winding3D`` (geometry: where each phase's copper goes)
  - ``objective3d`` (solver: which voxels carry which phase current)
  - ``verify_phase_connectivity`` (audit: A/B/C each connected, mutually insulated)
  - ``powered_transient`` (circuit: R, L, back-EMF from actual geometry)

This replaces the analytic ``_phase_belts`` cosine assignment with a netlist
that matches the physical coil geometry, so the visible winding and the
solved winding are the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from organic_motor.config3d import MotorConfig3D


@dataclass(frozen=True)
class CoilEntry:
    """One coil side pair in the netlist."""

    coil_id: int
    phase: int  # 0=A, 1=B, 2=C
    slot_pos: int  # positive coil side slot index
    slot_neg: int  # negative coil side slot index
    layer: int  # radial layer (0=innermost)
    turns: int
    polarity: int  # +1 or -1 (direction of current flow)


@dataclass
class CoilNetlist:
    """Deterministic three-phase winding topology from slot/pole counts.

    The winding layout follows the standard integral-slot concentrated or
    distributed winding rules:

    - ``n_slots`` must be divisible by 3 for a balanced three-phase winding.
    - Each slot hosts ``n_layers`` concentric conductors (layers).
    - Coils are assigned to phases in the pattern A, C', B, A', C, B', ...
      (the standard 60-degree phase belt rotation for ``pole_pairs`` poles).
    - ``coil_span`` is the slot pitch between the two sides of one coil.
    - Layers alternate polarity to fill the slot without overlap.

    The netlist is independent of geometry resolution: it only knows slot
    indices, phases and turns.  The geometry layer maps slots to angular
    positions and the solver maps voxels to slots.
    """

    n_slots: int = 12
    pole_pairs: int = 2
    n_phases: int = 3
    coil_span: int = 3
    n_layers: int = 4
    turns_per_coil: int = 1
    connection: str = "star"  # "star" or "delta"

    def __post_init__(self) -> None:
        if self.n_slots % self.n_phases != 0:
            raise ValueError(
                f"n_slots={self.n_slots} must be divisible by n_phases={self.n_phases}"
            )
        if self.pole_pairs < 1:
            raise ValueError("pole_pairs must be >= 1")

    @property
    def slot_pitch(self) -> float:
        return 2.0 * np.pi / self.n_slots

    @property
    def n_coils(self) -> int:
        return self.n_slots * self.n_layers

    @property
    def coils(self) -> list[CoilEntry]:
        coils: list[CoilEntry] = []
        cid = 0
        for layer in range(self.n_layers):
            for slot in range(self.n_slots):
                phase = self._slot_phase(slot)
                pol = self._slot_polarity(slot, layer)
                neg_slot = (slot + self.coil_span) % self.n_slots
                if pol > 0:
                    pos_slot, neg_slot = slot, neg_slot
                else:
                    pos_slot, neg_slot = neg_slot, slot
                coils.append(CoilEntry(
                    coil_id=cid, phase=phase,
                    slot_pos=pos_slot, slot_neg=neg_slot,
                    layer=layer, turns=self.turns_per_coil,
                    polarity=pol,
                ))
                cid += 1
        return coils

    def _slot_phase(self, slot: int) -> int:
        poles_total = 2 * self.pole_pairs
        slots_per_pole = self.n_slots / poles_total
        slots_per_phase = slots_per_pole / self.n_phases
        belt = int(slot / slots_per_phase) % self.n_phases
        return belt

    def _slot_polarity(self, slot: int, layer: int) -> int:
        """Coil-side sign at (slot, layer): the standard 12s4p winding table.

        Phases occupy slots in sequence (A, B, C, A, ...) but each phase's
        entry polarity follows a phase-shifted cosine, so the three layers'
        spatial MMF fundamentals sum to a PURE forward-rotating wave
        (forward phasors {0,0,0} elec, backward {0,+120,-120} summing to
        zero).  With naive pole-parity signs the backward wave dominates
        and the synchronous torque collapses to a zero-mean oscillation.
        """
        theta = slot * 2.0 * np.pi / self.n_slots
        phase = self._slot_phase(slot)
        # A -> 0, B -> 4*pi/3, C -> 2*pi/3 (the winding-table entry angles)
        psi = ((2 * phase) % self.n_phases) * 2.0 * np.pi / self.n_phases
        c = np.cos(self.pole_pairs * theta - psi)
        return 1 if c >= 0.0 else -1

    def slot_phase_assignment(self) -> np.ndarray:
        """Return ``(n_slots,)`` int array: phase index per slot (0=A,1=B,2=C)."""
        return np.array([self._slot_phase(s) for s in range(self.n_slots)], dtype=np.int32)

    def slot_polarity_assignment(self) -> np.ndarray:
        """Return ``(n_layers, n_slots)`` int array: +1/-1 polarity per (layer, slot)."""
        return np.array([
            [self._slot_polarity(s, l) for s in range(self.n_slots)]
            for l in range(self.n_layers)
        ], dtype=np.int32)

    def phase_slot_mask(self, phase: int) -> np.ndarray:
        """Boolean ``(n_slots,)`` mask: which slots have this phase's conductors."""
        return self.slot_phase_assignment() == phase

    def phase_belts_3d(self, cfg: MotorConfig3D) -> np.ndarray:
        """Return ``(3, Nx, Ny, Nz)`` phase assignment matching Winding3D geometry.

        Exclusive nearest-slot / nearest-layer assignment: every voxel in the
        winding annulus belongs to exactly one (slot, layer) cell, whose phase
        and polarity come from the netlist tables.  Unlike band-overlap
        construction this can never double-count a boundary voxel, so the
        three phases are guaranteed disjoint.
        """
        nx, ny, nz = cfg.shape
        cx, cy = cfg.center[0], cfg.center[1]
        dx, dy = cfg.spacing[0], cfg.spacing[1]
        ox, oy = cfg.origin[0], cfg.origin[1]
        x = ox + dx * np.arange(nx, dtype=np.float32)
        y = oy + dy * np.arange(ny, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="ij")
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        theta = np.arctan2(Y - cy, X - cx)

        slot_pitch = 2.0 * np.pi / self.n_slots
        slot_idx = np.mod(np.round(theta / slot_pitch).astype(np.int32), self.n_slots)

        r_wi = cfg.R_winding_inner
        r_wo = cfg.R_winding_outer
        dr = (r_wo - r_wi) / max(self.n_layers, 1)
        layer_idx = np.clip(((r - r_wi) / dr).astype(np.int32), 0, self.n_layers - 1)

        in_annulus = (r >= r_wi) & (r < r_wo)
        phase_table = self.slot_phase_assignment()
        pol_table = self.slot_polarity_assignment()
        phase_grid = phase_table[slot_idx]
        pol_grid = pol_table[layer_idx, slot_idx].astype(np.float32)

        belts = np.zeros((3, nx, ny, nz), dtype=np.float32)
        for ph in range(3):
            # A radial layer hosts conductors of exactly one phase
            # (Winding3D._slot_layers), so the belt is that annular layer.
            # The z-range is NOT restricted here: terminal conduction needs
            # the full copper network, and the impressed source applies its
            # own slot-region mask where axial currents are physical.
            layer_owns = (layer_idx % self.n_phases) == ph
            mask2d = in_annulus & layer_owns
            belts[ph] = np.broadcast_to(
                np.where(mask2d, pol_grid, 0.0)[..., None], (nx, ny, nz)
            )
        return belts

    def phase_resistance(self, cfg: MotorConfig3D, rho_copper: np.ndarray) -> float:
        """Estimate per-phase resistance from copper volume and mean path length.

        R = sigma^-1 * L_wire / A_wire, where:
        - L_wire = n_turns * (2 * stack_length + 2 * end_turn_arc)
        - A_wire = copper_cross_section / n_conductors_per_phase
        - copper_cross_section estimated from voxel count * cell_area
        """
        sigma = cfg.sigma_copper
        stack_len = cfg.stack_length
        end_turn_arc = self.coil_span * self.slot_pitch * cfg.R_winding_inner
        mean_path = 2 * stack_len + 2 * end_turn_arc
        n_turns_total = self.turns_per_coil * self.n_layers
        slots_per_phase = self.n_slots // self.n_phases
        L_wire = n_turns_total * slots_per_phase * mean_path
        copper_vol = float(np.sum(rho_copper)) * cfg.cell_volume
        copper_area = copper_vol / max(mean_path * slots_per_phase * self.n_layers, 1e-9)
        return 1.0 / (sigma * max(copper_area, 1e-9)) * L_wire / max(slots_per_phase, 1)

    def phase_inductance(self, cfg: MotorConfig3D, flux_linkage_per_ampere: float = 0.03) -> float:
        """Estimate per-phase inductance (simplified, to be refined by field solve)."""
        n_turns_total = self.turns_per_coil * self.n_layers
        slots_per_phase = self.n_slots // self.n_phases
        return flux_linkage_per_ampere * n_turns_total * slots_per_phase

    def flux_linkage_estimate(self, cfg: MotorConfig3D, b_gap: float = 1.0) -> float:
        """Estimate PM flux linkage from air-gap flux density and winding factor."""
        n_turns_total = self.turns_per_coil * self.n_layers
        slots_per_phase = self.n_slots // self.n_phases
        kw = 0.9  # winding factor (simplified)
        area = cfg.stack_length * cfg.R_stator_inner * np.pi / max(self.n_slots, 1)
        return kw * b_gap * area * n_turns_total * slots_per_phase

    def summary(self) -> dict:
        return {
            "n_slots": self.n_slots,
            "pole_pairs": self.pole_pairs,
            "n_phases": self.n_phases,
            "coil_span": self.coil_span,
            "n_layers": self.n_layers,
            "turns_per_coil": self.turns_per_coil,
            "connection": self.connection,
            "n_coils": self.n_coils,
            "slots_per_phase": self.n_slots // self.n_phases,
        }


def default_netlist(cfg: MotorConfig3D) -> CoilNetlist:
    """Construct the netlist matching the default Winding3D parameters."""
    return CoilNetlist(
        n_slots=12,
        pole_pairs=cfg.pole_pairs,
        n_phases=3,
        coil_span=3,
        n_layers=4,
        turns_per_coil=1,
        connection="star",
    )
