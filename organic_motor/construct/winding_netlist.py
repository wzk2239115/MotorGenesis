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
        """Phase owning a slot: 60-degree belt on EXACT integer arithmetic.

        The electrical angle of slot ``s`` is ``360 * p * s / n_slots``
        degrees.  For 12s10p it lands exactly BETWEEN two phase axes for
        every other slot, and a float ``argmax |cos|`` lets last-ulp noise
        decide the tie -- which silently shredded one phase of the winding
        (measured T1: A 0.024 vs B/C 0.285 N*m).  The belt index is
        instead quantised with exact integer round-half-up,
        ``k = (6*p*s + n_slots/2) // n_slots  (mod 6)`` -- deterministic
        and balanced for every slot/pole combination, and identical to
        the documented A, C', B, A', C, B' rotation for integral-slot
        windings (n_slots = 6*k).
        """
        num = 6 * self.pole_pairs * slot
        den = self.n_slots
        k = ((num + den // 2) // den) % 6
        return (0, 2, 1, 0, 2, 1)[k]

    def _slot_polarity(self, slot: int, layer: int) -> int:
        """Coil-side sign at (slot, layer): the belt's own cosine sign.

        All layers of a slot share the polarity: they are parallel paths
        of the same coil.  Sign from the same integer belt ``k`` as the
        phase (positive on the axis, negative between -- exactly the
        sign of cos(alpha - phi_p) at the quantised belt).
        """
        num = 6 * self.pole_pairs * slot
        den = self.n_slots
        k = ((num + den // 2) // den) % 6
        return 1 if k in (0, 2, 4) else -1

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

    def expected_phase_components(self) -> np.ndarray:
        """Expected connected-component count per phase, from the topology.

        With phase insulation by radial layer (phase p on layers p, p+3,
        ...), the coils of one phase chain through shared slot conductors
        into one ring per occupied layer, so a REALISED winding must have
        exactly this many components per phase: fewer means a break, more
        means fragmentation.
        """
        return np.array(
            [len(range(p, self.n_layers, self.n_phases)) for p in range(self.n_phases)],
            dtype=np.int32,
        )

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
            # Discrete SLOT SECTORS of this phase on its own radial layer:
            # (a) end-turn arc copper lives BETWEEN slots and belongs to no
            #     sector, so the z-averaged conductor cannot pollute other
            #     angular positions with spurious axial current;
            # (b) the sectors are z-UNIFORM, so the impressed Jz columns
            #     stay divergence-free (div J = dJz/dz = 0);
            # (c) the sector combs reproduce the analytic phase belts at
            #     slot centres, giving the forward-rotating MMF.
            layer_owns = (layer_idx % self.n_phases) == ph
            mask2d = in_annulus & layer_owns & (phase_grid == ph)
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


# ---------------------------------------------------------------------------
# Printed concentrated winding (12-slot / 10-pole)
# ---------------------------------------------------------------------------

# Angular geometry of one printed stator cell, in radians, measured from the
# TOOTH centre line.  One slot pitch is 30 deg; the cell is
#
#     tooth flank | clad | copper side | liner | wall/separator | ...
#
#   -TOOTH_HALF .. +TOOTH_HALF          iron tooth wedge
#   +/-CLAD_HALF                         insulator cladding on the flanks
#   +/-FRAME_HALF                        the copper frame around the tooth
#   +/-SLOT_HALF (= pitch/2)             cell boundary at the slot centre
PRINTED_TOOTH_HALF = np.deg2rad(7.5)
PRINTED_CLAD_HALF = np.deg2rad(8.44)     # tooth + 0.6mm cladding at r ~ 36.75mm
PRINTED_FRAME_HALF = np.deg2rad(13.8)    # copper frame; wall gap 0.6 deg to slot centre
PRINTED_SLOT_HALF = np.deg2rad(15.0)     # slot centre = boundary to the next cell
PRINTED_END_BAND = 0.0035                # coil bridge axial thickness beyond the stack


@dataclass
class PrintedCoilNetlist(CoilNetlist):
    """Concentrated 12s10p winding for the printed stator: one coil per tooth.

    Electrical topology (the classic 12-slot 10-pole concentrated winding,
    winding factor 0.933): every tooth carries ONE printed coil loop whose
    two sides occupy the HALF-SLOTS flanking it, so all coil sides sit at
    the same radii and the three phases are geometrically IDENTICAL -- the
    radial-layer asymmetry of the distributed winding dies here.  Each coil
    is an independent printed loop (hollow conductor with an internal
    cooling channel); the four coils of a phase are wired externally.

    Phase/sign table per tooth n (standard 60-degree-belt rule, identical
    to ``CoilNetlist`` for 12s10p):  A+ B+ B- C- C+ A+ A- B- B+ C+ C- A-.
    """

    coil_span: int = 1
    n_layers: int = 1
    turns_per_coil: int = 1

    def coil_table(self) -> list[tuple[int, int, int]]:
        """``[(tooth, phase, polarity), ...]`` for all 12 coils."""
        return [
            (n, int(self._slot_phase(n)), int(self._slot_polarity(n, 0)))
            for n in range(self.n_slots)
        ]

    def expected_phase_components(self) -> np.ndarray:
        """Four independent printed loops per phase (externally wired).

        Each tooth has one printed coil loop (coil_span=1, n_layers=1,
        turns_per_coil=1 in the P4 frame topology).  The P5 swept-band
        design carries ``n_turns_per_cell`` independent closed loops per
        tooth (7 bands); when the centerline registry is present the
        expected count is multiplied by the turn count (28 per phase, not
        4).  Real jumpers between bands are not yet modelled in the
        geometry — the impressed line-current source treats all turns as
        carrying the same ``I`` in the same direction (series).
        """
        per_phase = self.n_slots // self.n_phases
        return np.full(self.n_phases, per_phase, dtype=np.int32)

    def coil_zc(self, cfg: MotorConfig3D) -> float:
        """Half the axial extent of the copper frame (stack + end band)."""
        return cfg.stator_half_length + PRINTED_END_BAND

    def phase_belts_3d(self, cfg: MotorConfig3D) -> np.ndarray:
        """``(3, Nx, Ny, Nz)`` belts: +/-1 in the frame's two side bands.

        Every voxel is attributed to the NEAREST tooth (slot-pitch cells,
        exactly the printed geometry), the side band between the tooth
        cladding and the frame edge carries the coil current: the ``u < 0``
        side is the GO side (+polarity), the ``u > 0`` side the RETURN.
        Belts are z-UNIFORM (divergence-free columns); the end-turn bridges
        are supplied by the printed end-closure currents in
        :func:`_printed_end_closure_currents`.
        """
        nx, ny, nz = cfg.shape
        _X, _Y, _Z, r, theta = _polar_grid(cfg)
        pitch = 2.0 * np.pi / self.n_slots
        n_tooth = np.mod(np.round(theta / pitch).astype(np.int32), self.n_slots)
        u = np.mod(theta - n_tooth * pitch + np.pi, 2.0 * np.pi) - np.pi

        r_wi = cfg.R_winding_inner
        r_wo = cfg.R_winding_outer - 0.0005  # frame outer edge leaves an air slot-back
        in_side = (np.abs(u) >= PRINTED_CLAD_HALF) & (np.abs(u) <= PRINTED_FRAME_HALF)
        in_radial = (r >= r_wi) & (r <= r_wo)

        phase = np.zeros((nx, ny), dtype=np.int8)
        sign = np.zeros((nx, ny), dtype=np.float32)
        table = {tooth: (ph, pol) for tooth, ph, pol in self.coil_table()}
        for tooth in range(self.n_slots):
            sel = n_tooth == tooth
            ph, pol = table[tooth]
            phase[sel] = ph
            sign[sel] = pol
        # u < 0 -> go side (+), u > 0 -> return side (-)
        dirn = np.where(u < 0.0, 1.0, -1.0).astype(np.float32)

        belts = np.zeros((3, nx, ny, nz), dtype=np.float32)
        mask2d = (in_side & in_radial).astype(np.float32)
        for p in range(3):
            sel = (mask2d * (phase == p)).astype(np.float32)
            belts[p] = np.broadcast_to((sel * sign * dirn)[..., None], (nx, ny, nz))
        return belts

    def summary(self) -> dict:
        base = super().summary()
        base["style"] = "printed_concentrated"
        return base


def _polar_grid(cfg: MotorConfig3D):
    nx, ny, nz = cfg.shape
    cx, cy = cfg.center[0], cfg.center[1]
    dx, dy = cfg.spacing[0], cfg.spacing[1]
    ox, oy = cfg.origin[0], cfg.origin[1]
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = np.zeros(1, dtype=np.float32)
    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    theta = np.arctan2(Y - cy, X - cx)
    return X[..., 0], Y[..., 0], np.zeros((nx, ny, 1)), r[..., 0], theta[..., 0]


def printed_netlist(cfg: MotorConfig3D) -> PrintedCoilNetlist:
    """The netlist of the printed concentrated stator for ``cfg``."""
    return PrintedCoilNetlist(
        n_slots=12,
        pole_pairs=cfg.pole_pairs,
        n_phases=3,
        coil_span=1,
        n_layers=1,
        turns_per_coil=1,
        connection="star",
    )
