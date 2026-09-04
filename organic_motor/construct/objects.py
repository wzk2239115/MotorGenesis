"""Computational motor objects: self-constructing components.

Each object is a small bundle of parameters plus a ``build(material_field)``
method that renders its implicit primitives into the shared
:class:`MaterialField` using priority Booleans.  The decomposition mirrors a
domain expert's mental model of an inner-rotor PM machine -- shaft bore,
rotor back-iron, surface magnets, stator yoke, winding, cooling jacket -- and
is exactly the granularity at which a code-generating agent can compose and
re-design a motor.

Geometry anchors (radii, stack length, pole count) come from a
:class:`MotorConfig3D` so a constructed motor is always consistent with the
solver's domain.  The objects never touch the differentiable solver; they
only produce geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.field import (
    SDFVoxelField,
    boolean_subtract,
    empty_field,
    from_implicit,
    offset,
    shell,
)
from organic_motor.construct.implicit import (
    annular_sector,
    cylinder_z,
    plane,
)
from organic_motor.construct.lattice import gyroid_sheet
from organic_motor.construct.material import MaterialField


def _grid_of(cfg: MotorConfig3D) -> MaterialField:
    return MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)


def _annulus(
    cfg: MotorConfig3D,
    r_inner: float,
    r_outer: float,
    half_z: float,
) -> SDFVoxelField:
    """A full annular cylinder about z, built as a Boolean difference of two cylinders."""
    cx, cy = cfg.center[0], cfg.center[1]
    outer = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                           cylinder_z((cx, cy), r_outer, half_z))
    inner = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                          cylinder_z((cx, cy), r_inner, half_z))
    return boolean_subtract(outer, inner)


def _angles_of(cfg: MotorConfig3D):
    cx, cy, _ = cfg.center
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    a = np.arctan2(Y - cy, X - cx)
    return X, Y, Z, r, a


# ---------------------------------------------------------------------------
# Rotor
# ---------------------------------------------------------------------------

@dataclass
class RotorCore:
    """Laminated rotor back-iron annulus with a shaft bore."""

    cfg: MotorConfig3D
    clearance: float = 0.004  # 4mm gap so shaft and rotor don't merge at any resolution

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.rotor_half_length
        outer = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                               cylinder_z((cfg.center[0], cfg.center[1]),
                                          cfg.R_rotor_outer, hz))
        bore = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                             cylinder_z((cfg.center[0], cfg.center[1]),
                                        cfg.R_shaft + self.clearance, hz))
        iron = boolean_subtract(outer, bore)
        mf.add(iron, "iron", priority=True)
        return mf


@dataclass
class SurfaceMagnets:
    """Surface-mounted PM poles on the rotor outer radius.

    ``pole_fraction`` is the fraction of one pole pitch each magnet occupies;
    ``thickness`` is the radial magnet thickness.  ``skew_angle`` twists the
    pole pattern helically over the stack (one slot pitch by default) --
    the classical skew that cancels cogging torque in a slotted stator.
    Magnetisation alternates radially between poles; :meth:`magnetization`
    returns the per-voxel magnetisation vector the critic needs.
    """

    cfg: MotorConfig3D
    thickness: float = 0.0015
    pole_fraction: float = 0.72
    skew_angle: float = 0.5236  # one slot pitch (30 deg) over the stack

    def _pole_specs(self):
        cfg = self.cfg
        poles = 2 * cfg.pole_pairs
        pitch = 2.0 * np.pi / poles
        r0 = cfg.R_rotor_outer + 0.0002
        r1 = r0 + self.thickness
        specs = []
        for p in range(poles):
            centre = p * pitch
            span = pitch * self.pole_fraction
            a0 = centre - span * 0.5
            a1 = centre + span * 0.5
            sign = 1.0 if p % 2 == 0 else -1.0
            specs.append((a0, a1, r0, r1, sign, centre))
        return specs

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        poles = 2 * cfg.pole_pairs
        pitch = 2.0 * np.pi / poles
        hz = cfg.rotor_half_length - 0.0002
        cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
        _X, _Y, Z, r, theta = _angles_of(cfg)
        z_norm = (Z - cz) / max(2.0 * hz, 1e-9)
        half_span = pitch * self.pole_fraction * 0.5

        sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        r0 = cfg.R_rotor_outer + 0.0002
        r1 = r0 + self.thickness
        for p in range(poles):
            centre = p * pitch + self.skew_angle * z_norm
            ang = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
            band = np.abs(ang) - half_span
            radial = np.maximum(r - r1, r0 - r)
            pole = np.maximum(np.maximum(band, radial), axial)
            sdf = np.minimum(sdf, pole.astype(np.float32))
        mf.add(SDFVoxelField(sdf=sdf, spacing=cfg.spacing, origin=cfg.origin), "pm", priority=True)
        return mf

    def magnetization(self) -> np.ndarray:
        """Per-voxel unit magnetisation vector, alternating radially per pole.

        Radial outward for even poles, inward for odd -- the canonical
        surface-PM magnetisation pattern, following the same helical skew
        as the geometry so poles and magnetisation always agree.
        """
        cfg = self.cfg
        _X, _Y, Z, _r, angle = _angles_of(cfg)
        cz = cfg.center[2]
        poles = 2 * cfg.pole_pairs
        pitch = 2.0 * np.pi / poles
        hz = cfg.rotor_half_length
        z_norm = (Z - cz) / max(2.0 * hz, 1e-9)
        skewed = angle - self.skew_angle * z_norm
        pole_index = np.rint(skewed / pitch)
        sign = np.where((pole_index.astype(int) % 2) == 0, 1.0, -1.0)
        mx = sign * np.cos(angle)
        my = sign * np.sin(angle)
        mz = np.zeros_like(mx)
        return np.stack([mx, my, mz], axis=0).astype(np.float32)


@dataclass
class FieldDrivenMagnets:
    """Surface PM poles whose radial thickness follows a physics field.

    The magnet thickness is sampled per-voxel from a :class:`ScalarField`
    (typically the air-gap |B| demand): where the field says more flux is
    needed the magnet thickens, where less is needed it thins.  This is the
    LEAP 71 ``thickness = f(position, physics)`` pattern for the magnetic
    ``bone``, realised as one union implicit whose outer radius varies with
    the sampled field.
    """

    cfg: MotorConfig3D
    base_thickness: float = 0.0035
    min_thickness: float = 0.0030
    max_thickness: float = 0.0040
    pole_fraction: float = 0.72
    skew_angle: float = 0.5236  # helical pole twist over the stack (rad)
    thickness_field: object | None = None  # ScalarField; default = airgap_B

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.fields_motor import airgap_B
        from organic_motor.construct.field import ScalarField

        cfg = self.cfg
        poles = 2 * cfg.pole_pairs
        pitch = 2.0 * np.pi / poles
        r0 = cfg.R_rotor_outer + 0.0002
        hz = cfg.rotor_half_length - 0.0002
        cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]

        field = self.thickness_field
        if field is None:
            field = airgap_B(cfg)
        fmag = np.abs(field.data)
        fmax = max(float(fmag.max()), 1e-9)

        X, Y, Z, r, theta = _angles_of(cfg)
        z_norm = np.clip((Z - cz + hz) / (2.0 * hz), 0.0, 1.0)
        # Barrel floor 0.7: the axial ends keep >= 70% of the pole
        # thickness so the magnet stays >= 2 cells at the physics grid
        # (the old 0.4 floor thinned the ends to sub-cell).
        axial_factor = 0.7 + 0.3 * np.sin(np.pi * z_norm)
        # SAME skew normalisation as SurfaceMagnets.magnetization: the pole
        # pattern twists by exactly ``skew_angle`` over the FULL stack
        # (z_norm in [-1, 1] -> offset in [-skew/2, +skew/2]).
        z_skew = (Z - cz) / max(2.0 * cfg.rotor_half_length, 1e-9)

        sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        for p in range(poles):
            centre = p * pitch + self.skew_angle * z_skew
            ang = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
            in_pole = np.abs(ang) < pitch * self.pole_fraction * 0.5
            b_avg = float(fmag[in_pole].mean()) / fmax if in_pole.any() else 0.0
            base_t = self.min_thickness + (self.max_thickness - self.min_thickness) * b_avg
            r1_z = r0 + base_t * axial_factor
            rm = 0.5 * (r0 + r1_z)
            dr = 0.5 * (r1_z - r0)
            radial = np.abs(r - rm) - dr
            axial = np.abs(Z - cz) - hz
            band = np.abs(ang) - pitch * self.pole_fraction * 0.5
            pole_sdf = np.maximum(np.maximum(radial, axial), band)
            sdf = np.minimum(sdf, pole_sdf.astype(np.float32))
        from organic_motor.construct.field import SDFVoxelField

        mf.add(SDFVoxelField(sdf=sdf, spacing=cfg.spacing, origin=cfg.origin), "pm", priority=True)
        return mf

    def magnetization(self) -> np.ndarray:
        return SurfaceMagnets(self.cfg, skew_angle=self.skew_angle).magnetization()


# ---------------------------------------------------------------------------
# Stator
# ---------------------------------------------------------------------------

@dataclass
class StatorCore:
    """Stator iron yoke with slot openings between teeth."""

    cfg: MotorConfig3D
    slots: int = 12
    slot_opening: float = 0.0028

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.stator_half_length
        cx, cy = cfg.center[0], cfg.center[1]
        yoke = _annulus(cfg, cfg.R_stator_inner, cfg.R_design, hz)
        # Subtract evenly spaced slot openings from the inner radius so the
        # stator reads as a toothed ring rather than a plain annulus.
        pitch = 2.0 * np.pi / self.slots
        slot_r0 = cfg.R_stator_inner
        slot_r1 = cfg.R_winding_inner + 0.5 * (cfg.R_winding_outer - cfg.R_winding_inner)
        for s in range(self.slots):
            centre = s * pitch
            span = self.slot_opening / max(cfg.R_stator_inner, 1e-6)
            opening = from_implicit(
                cfg.shape, cfg.spacing, cfg.origin,
                annular_sector((cx, cy), slot_r0, slot_r1,
                               centre - span * 0.5, centre + span * 0.5, hz),
            )
            yoke = boolean_subtract(yoke, opening)
        mf.add(yoke, "iron", priority=True)
        return mf


@dataclass
class FieldDrivenStatorYoke:
    """Stator iron yoke whose radial thickness follows the local flux density.

    The yoke thickens where the air-gap |B| is high (under magnet poles,
    where the magnetic circuit carries the most flux) and thins in the
    interpole gaps.  Built as a union of angular slices, each with an inner
    radius sampled from the |B| field -- so the iron ``bone`` grows where
    the physics demands a lower-reluctance return path.
    """

    cfg: MotorConfig3D
    slots: int = 12
    slot_opening: float = 0.005
    min_yoke_thickness: float = 0.006
    max_yoke_thickness: float = 0.018
    angular_slices: int = 48
    flux_field: object | None = None  # ScalarField; default = airgap_B

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.fields_motor import airgap_B
        from organic_motor.construct.field import SDFVoxelField, smooth_boolean_subtract

        cfg = self.cfg
        cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
        hz = cfg.stator_half_length
        r_inner = cfg.R_stator_inner
        max_available = cfg.R_design - cfg.R_stator_inner
        max_thickness = min(self.max_yoke_thickness, max_available)

        field = self.flux_field
        if field is None:
            field = airgap_B(cfg)
        fmag = np.abs(field.data)
        fmax = max(float(fmag.max()), 1e-9)
        b_norm = np.clip(fmag / fmax, 0.0, 1.0).astype(np.float32)

        X, Y, Z, r, theta = _angles_of(cfg)
        r_outer = r_inner + self.min_yoke_thickness + (max_thickness - self.min_yoke_thickness) * b_norm
        rm = 0.5 * (r_inner + r_outer)
        dr = 0.5 * (r_outer - r_inner)
        radial = np.abs(r - rm) - dr
        axial = np.abs(Z - cz) - hz
        yoke_sdf = np.maximum(radial, axial).astype(np.float32)

        yoke = SDFVoxelField(sdf=yoke_sdf, spacing=cfg.spacing, origin=cfg.origin)
        pitch = 2.0 * np.pi / self.slots
        slot_r1 = cfg.R_winding_outer
        for s in range(self.slots):
            centre = s * pitch
            span = self.slot_opening / max(cfg.R_stator_inner, 1e-6)
            opening = from_implicit(
                cfg.shape, cfg.spacing, cfg.origin,
                annular_sector((cx, cy), cfg.R_stator_inner, slot_r1,
                               centre - span * 0.5, centre + span * 0.5, hz),
            )
            yoke = smooth_boolean_subtract(yoke, opening, blend=0.001)
        mf.add(yoke, "iron", priority=True)
        return mf


@dataclass
class DistributedWinding:
    """Three-phase copper occupying the winding annulus.

    The solver derives phase belts analytically from position, so a
    continuous copper annulus is sufficient for a faithful critic run; the
    slot openings cut by :class:`StatorCore` give it the toothed appearance.
    """

    cfg: MotorConfig3D

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.stator_half_length
        copper = _annulus(cfg, cfg.R_winding_inner, cfg.R_winding_outer, hz)
        mf.add(copper, "copper", priority=True)
        return mf


@dataclass
class Winding3D:
    """Real 3D distributed winding with phase-separated radial layers.

    Each coil is a continuous copper loop: two straight axial sections in
    slots on opposite sides of a pole, connected by end-turn arcs at the
    stator ends.  The arcs use a CLOSED-FORM arc-tube SDF (distance to a
    circular arc segment), not sampled bead capsules -- beads alias into
    disconnected rings when the bead spacing approaches the cell size.

    Electrical topology (phase insulation by construction): slots are
    assigned to phases A/C'/B rotationally, and each slot's conductors live
    ONLY on radial layers belonging to that phase (layers p, p+3, p+6, ...
    for phase p).  Because every layer is a distinct radius band, arcs and
    conductors of different phases can never touch -- phase-to-phase
    insulation is geometric, not hoped for.  Coils of one phase chain
    through shared slot conductors, so each phase forms one connected
    network per layer group (parallel paths), exactly like a real lap
    winding.

    During construction every copper voxel is tagged with its phase in
    ``mf.metadata["winding_phase_owner"]`` (int8, -1 = none, 0/1/2 = A/B/C)
    INCLUDING end turns and terminals, so the connectivity audit checks the
    real electrical network instead of a slot-sector clip that would delete
    exactly the copper that connects the coil sides.
    """

    cfg: MotorConfig3D
    n_slots: int = 12
    coil_span: int = 3
    n_layers: int = 3  # one radial layer per phase (phase insulation by radius)
    strands_per_slot: int = 3  # touching wires per slot: raises slot fill ~3x
    wire_radius: float = 0.0  # 0 = auto from layer spacing (35% fill)
    end_turn_rise: float = 0.0005
    end_turn_gap: float = 0.001  # kept for API compat

    def _slot_layers(self, phase: int) -> list[int]:
        """Radial layers hosting conductors of ``phase`` (every 3rd layer)."""
        return list(range(phase, self.n_layers, 3))

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField
        from organic_motor.construct.winding_netlist import CoilNetlist

        cfg = self.cfg
        cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
        hz = cfg.stator_half_length
        r_wi = cfg.R_winding_inner
        r_wo = cfg.R_winding_outer
        slot_pitch = 2.0 * np.pi / self.n_slots
        if self.wire_radius > 0.0:
            wr = self.wire_radius
        else:
            # 35% of layer spacing: inter-layer insulation gap is 30% of
            # the spacing (~1.3mm), resolvable at display resolution and
            # realistic for real phase insulation.
            wr = 0.35 * (r_wo - r_wi) / self.n_layers

        netlist = CoilNetlist(
            n_slots=self.n_slots, pole_pairs=cfg.pole_pairs,
            n_phases=3, coil_span=self.coil_span,
            n_layers=self.n_layers, turns_per_coil=1,
            connection="star",
        )
        phase_of_slot = netlist.slot_phase_assignment()

        X, Y, Z, r, theta = _angles_of(cfg)
        copper_sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        phase_sdf = np.full((3,) + cfg.shape, 1e9, dtype=np.float32)
        S = max(1, self.strands_per_slot)

        for coil in range(self.n_slots):
            slot_a = coil
            slot_b = (coil + self.coil_span) % self.n_slots
            theta_a = slot_a * slot_pitch
            theta_b = slot_b * slot_pitch
            d_ab = np.mod(theta_b - theta_a + np.pi, 2 * np.pi) - np.pi
            phase = int(phase_of_slot[coil])
            coil_sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
            # Arc geometry (shared by every strand of this coil): a tube of
            # radius wr around the circular arc from theta_as to theta_bs at
            # radius r_mid, in the plane z = cz +/- (hz + 0.5*wr) so it
            # overlaps the axial bars by half a wire radius.
            theta_mid = theta_a + 0.5 * d_ab
            half_span = 0.5 * abs(d_ab)

            for layer in self._slot_layers(phase):
                r_mid = r_wi + (layer + 0.5) * (r_wo - r_wi) / self.n_layers
                # Strands sit side by side across the slot, touching (same
                # phase, so contact is fine): a form-wound conductor bundle.
                d_theta_strand = 2.0 * wr / r_mid

                for strand in range(S):
                    off = (strand - 0.5 * (S - 1)) * d_theta_strand
                    theta_as = theta_a + off
                    theta_bs = theta_b + off

                    d_ta = np.mod(theta - theta_as + np.pi, 2 * np.pi) - np.pi
                    d_tb = np.mod(theta - theta_bs + np.pi, 2 * np.pi) - np.pi
                    radial_a = np.sqrt((r - r_mid) ** 2 + (r_mid * d_ta) ** 2)
                    radial_b = np.sqrt((r - r_mid) ** 2 + (r_mid * d_tb) ** 2)
                    axial_in = np.maximum(np.abs(Z - cz) - hz, 0.0)
                    coil_sdf = np.minimum(coil_sdf, np.sqrt(radial_a ** 2 + axial_in ** 2) - wr)
                    coil_sdf = np.minimum(coil_sdf, np.sqrt(radial_b ** 2 + axial_in ** 2) - wr)

                    # Closed-form end-turn arc tube: distance to the arc
                    # segment between the two coil sides at radius r_mid.
                    # d_arc = 0 inside the angular span, otherwise the
                    # arc-length distance to the nearest span end.
                    d_mid = np.mod(theta - (theta_mid + off) + np.pi, 2 * np.pi) - np.pi
                    arc_gap = np.maximum(np.abs(d_mid) - half_span, 0.0) * r_mid
                    for sign in (+1, -1):
                        z_arc = cz + sign * (hz + 0.5 * wr)
                        arc_dist = np.sqrt(
                            (r - r_mid) ** 2 + (Z - z_arc) ** 2 + arc_gap ** 2
                        )
                        coil_sdf = np.minimum(coil_sdf, (arc_dist - wr).astype(np.float32))

            copper_sdf = np.minimum(copper_sdf, coil_sdf)
            phase_sdf[phase] = np.minimum(phase_sdf[phase], coil_sdf)

        # Exact per-phase voxel ownership INCLUDING end turns and terminals:
        # the audit checks the real electrical network, not a slot-sector
        # clip (which would delete exactly the copper that connects the
        # coil sides).  phase_sdf also gives an exact cross-short test.
        phase_owner = np.argmin(phase_sdf, axis=0).astype(np.int8)
        phase_owner[copper_sdf >= 0.0] = -1

        mf.add(SDFVoxelField(sdf=copper_sdf, spacing=cfg.spacing, origin=cfg.origin), "copper", priority=True)
        mf.metadata["winding_netlist"] = netlist
        mf.metadata["winding_phase_owner"] = phase_owner
        mf.metadata["winding_phase_sdf"] = phase_sdf
        return mf


@dataclass
class CoolingJacket:
    """A gyroid-sheet coolant wall around the stator outer radius.

    Demonstrates the lattice ``flesh`` LEAP 71 is known for: a triply
    periodic minimal surface sheet filling the housing annulus, providing
    structural stiffness and a convoluted coolant channel.  Built as iron so
    it is visible; it sits outside the design region and so does not perturb
    the magnetic solve.
    """

    cfg: MotorConfig3D
    outer_radius: float = 0.058
    wall_thickness: float = 0.004
    scale: float = 280.0

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.stator_half_length + 0.002
        annulus = _annulus(cfg, cfg.R_design + 0.001, self.outer_radius, hz)
        gsheet = gyroid_sheet(self.scale, self.wall_thickness, cfg.shape,
                              cfg.spacing, cfg.origin)
        from organic_motor.construct.field import boolean_intersect

        jacket = boolean_intersect(annulus, gsheet)
        mf.add(jacket, "iron", priority=True)
        return mf


@dataclass
class FieldDrivenCoolingJacket:
    """A gyroid coolant wall whose thickness follows the local Joule-heat field.

    This is the LEAP 71 ``blood-vessel`` pattern made explicit: the wall
    thickens where the winding runs hot (high I^2 R) and thins where it is
    cool, so the cooling capacity tracks the local transport demand.  The
    wall thickness is ``f(position, physics)`` -- the signature of true
    computational engineering rather than a uniform parametric lattice.
    """

    cfg: MotorConfig3D
    outer_radius: float = 0.058
    min_wall: float = 0.0008
    max_wall: float = 0.0035
    scale: float = 280.0
    heat_field: object | None = None  # ScalarField; built from cfg if None

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.implicit import gyroid as gyroid_implicit
        from organic_motor.construct.implicit_modular import field_modulated_surface
        from organic_motor.construct.field import ScalarField, boolean_intersect
        from organic_motor.construct.fields_motor import joule_heat

        cfg = self.cfg
        hz = cfg.stator_half_length + 0.002
        annulus = _annulus(cfg, cfg.R_design + 0.001, self.outer_radius, hz)

        # Reduced-physics heat field; normalise to [0,1] and map to a wall
        # thickness in [min_wall, max_wall].  This is the field->geometry step.
        q = self.heat_field
        if q is None:
            q = joule_heat(cfg)
        qmax = max(float(q.data.max()), 1e-9)
        qnorm = np.clip(q.data / qmax, 0.0, 1.0)
        wall_grid = (self.min_wall + (self.max_wall - self.min_wall) * qnorm).astype(np.float32)
        wall_field = ScalarField(data=wall_grid, spacing=cfg.spacing, origin=cfg.origin)

        gy = gyroid_implicit(self.scale)
        gsheet = field_modulated_surface(gy, wall_field, cfg.shape, cfg.spacing, cfg.origin)
        jacket = boolean_intersect(annulus, gsheet)
        mf.add(jacket, "iron", priority=True)
        return mf


@dataclass
class HelicalCoolingChannels:
    """A CONTINUOUS spiral coolant channel with explicit inlet and outlet.

    The old per-z-slice "distance to the helix at the same z" was wrong by
    a factor of sqrt(1+(R*k)^2) ~ 30 for a real cooling pitch: consecutive
    slices jumped a full channel diameter circumferentially, so the void
    shattered into the disconnected rings seen in the display, and the
    wall around it into hundreds of structural fragments.  The true 3-D
    distance to a helix of slope k = dtheta/dz at radius R is

        d_min = R * |wrap(theta - k z)| / sqrt(1 + (R k)^2)

    (closed form of the minimisation over the axial coordinate), which
    makes the channel a single continuous screw thread.

    The channel runs bottom inlet -> top outlet through ``n_turns`` turns
    (pitch derived from the channel diameter plus wall), with axial stubs
    at both ends so the coolant network is open to the outside at TWO
    distinct points.  The void is stored as a dedicated ``coolant``
    material (not "air"), so the coolant graph is auditable on its own.
    """

    cfg: MotorConfig3D
    n_channels: int = 1  # kept for API compat; one spiral is the design
    n_turns: float = 0.0  # 0 = auto from pitch
    channel_radius: float = 0.004
    min_wall: float = 0.0015
    max_wall: float = 0.0035
    jacket_radius: float = 0.0  # default: R_design + 0.004
    stub_length: float = 0.006
    heat_field: object | None = None

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.fields_motor import joule_heat
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        cz = cfg.center[2]
        hz = cfg.stator_half_length - 0.001
        R = self.jacket_radius or (cfg.R_design + 0.004)

        X, Y, Z, r, theta = _angles_of(cfg)
        z_bot = cz - hz
        z_top = cz + hz
        span = z_top - z_bot
        if self.n_turns > 0.0:
            n_turns = self.n_turns
        else:
            pitch = 2.0 * self.channel_radius + 2.0 * self.min_wall
            n_turns = float(np.clip(round(span / pitch), 2.0, 8.0))
        k = 2.0 * np.pi * n_turns / span  # helix slope dtheta/dz
        theta_in = 0.0
        theta_helix = k * (Z - z_bot) + theta_in

        # True 3-D distance to the helical spine (see class docstring).
        d_theta = np.mod(theta - theta_helix + np.pi, 2 * np.pi) - np.pi
        d_arc = R * np.abs(d_theta) / np.sqrt(1.0 + (R * k) ** 2)
        dist_to_spine = np.sqrt((r - R) ** 2 + d_arc ** 2)
        # Axial bound: the spiral ends at the stub outlets (stack half
        # length + stub) -- without it the helix keeps wrapping to the
        # domain ends and its wall aliases into detached fragments there.
        axial_bound = np.abs(Z - cz) - (hz + self.stub_length)
        void_sdf = np.maximum(dist_to_spine - self.channel_radius, axial_bound).astype(np.float32)

        # Axial inlet (bottom, at the spiral start angle) and outlet (top,
        # at the angle the spiral reaches after n_turns turns).
        theta_out = theta_in + 2.0 * np.pi * n_turns
        for sign, z_open, z_stub in ((+1, z_top, z_top + 0.5 * self.stub_length),
                                     (-1, z_bot, z_bot - 0.5 * self.stub_length)):
            th = theta_out if sign > 0 else theta_in
            d_th = np.mod(theta - th + np.pi, 2 * np.pi) - np.pi
            stub_radial = np.sqrt((r - R) ** 2 + (R * d_th) ** 2)
            stub_axial = np.abs(Z - z_stub) - 0.5 * self.stub_length
            stub_sdf = np.maximum(stub_radial - self.channel_radius, stub_axial)
            void_sdf = np.minimum(void_sdf, stub_sdf.astype(np.float32))

        # Field-driven wall: shell(void, t) with t = f(local Joule heat).
        q = self.heat_field
        if q is None:
            q = joule_heat(cfg)
        qmax = max(float(np.abs(q.data).max()), 1e-9)
        qnorm = np.clip(np.abs(q.data) / qmax, 0.0, 1.0)
        wall_t = (self.min_wall + (self.max_wall - self.min_wall) * qnorm).astype(np.float32)

        wall_sdf = np.maximum(void_sdf - wall_t, -void_sdf)

        mf.add(SDFVoxelField(sdf=wall_sdf, spacing=cfg.spacing, origin=cfg.origin), "iron", priority=True)
        mf.add(SDFVoxelField(sdf=void_sdf, spacing=cfg.spacing, origin=cfg.origin), "coolant", priority=True)
        return mf


# ---------------------------------------------------------------------------
# Printed multi-material stator (P4 topology)
# ---------------------------------------------------------------------------


def _sector_sdf(
    cfg: MotorConfig3D,
    r0: float,
    r1: float,
    a_centre: float,
    a_half: float,
    z0: float,
    z1: float,
    _angles_cache=None,
) -> np.ndarray:
    """Approximate SDF of an annular-sector prism (correct inside, conservative outside).

    ``a_half >= pi`` degenerates to a full annulus with a z window.  The
    angular distance uses the arc length at the sector's mid radius -- the
    same approximation the hub-spoke webs already use.
    """
    if _angles_cache is not None:
        _X, _Y, Z, r, theta = _angles_cache
    else:
        _X, _Y, Z, r, theta = _angles_of(cfg)
    r_mid = 0.5 * (r0 + r1)
    d_r = np.maximum(r - r1, r0 - r)
    if a_half >= np.pi:
        band = np.full_like(r, -1.0)
    else:
        d_ang = np.mod(theta - a_centre + np.pi, 2.0 * np.pi) - np.pi
        band = np.abs(d_ang) - a_half
    cz = 0.5 * (z0 + z1)
    d_z = np.abs(Z - cz) - 0.5 * (z1 - z0)
    sdf = np.maximum(np.maximum(d_r, band * r_mid), d_z)
    return sdf.astype(np.float32)


def _printed_frame_sdf(
    cfg: MotorConfig3D, margin: float = 0.0,
    netlist=None, cache=None,
) -> np.ndarray:
    """Union SDF of the 12 copper frames around the teeth.

    Each frame is a "picture frame" in the (theta, z) plane, extruded over
    the winding radial band: the outer sector spans the full coil pitch up
    to the end bands, the window (tooth + cladding, stack z-range) is
    subtracted -- so copper wraps each tooth flanks + bridges over both
    tooth ends, exactly the printed concentrated-coil topology.  ``margin``
    dilates the frame (insulator pocket lining).
    """
    from organic_motor.construct.winding_netlist import (
        PRINTED_CLAD_HALF,
        PRINTED_FRAME_HALF,
        printed_netlist,
    )

    if netlist is None:
        netlist = printed_netlist(cfg)
    angles = cache or _angles_of(cfg)
    _X, _Y, Z, r, theta = angles
    cz = cfg.center[2]
    zh = cfg.stator_half_length
    zc = netlist.coil_zc(cfg)
    r_wi = cfg.R_winding_inner
    r_wo = cfg.R_winding_outer - 0.0005  # air slot-back to the yoke
    pitch = 2.0 * np.pi / netlist.n_slots

    copper = np.full(cfg.shape, 1e9, dtype=np.float32)
    for n in range(netlist.n_slots):
        theta_n = n * pitch
        outer = _sector_sdf(cfg, r_wi, r_wo, theta_n, PRINTED_FRAME_HALF,
                            cz - zc, cz + zc, _angles_cache=angles)
        window = _sector_sdf(cfg, r_wi, r_wo, theta_n, PRINTED_CLAD_HALF,
                             cz - zh, cz + zh, _angles_cache=angles)
        frame = np.maximum(outer, -window)
        copper = np.minimum(copper, frame)
    return (copper - margin).astype(np.float32)


def _segment_capsule_sdf(cfg: MotorConfig3D, p1, p2, radius: float,
                         cache=None) -> np.ndarray:
    """SDF of a capsule (capped cylinder) from ``p1`` to ``p2``."""
    _X, Y, Z, _r, _t = cache or _angles_of(cfg)
    X = _X
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    d = p2 - p1
    L2 = float(d @ d)
    t = np.clip(((X - p1[0]) * d[0] + (Y - p1[1]) * d[1] + (Z - p1[2]) * d[2]) / L2, 0.0, 1.0)
    px = p1[0] + t * d[0]
    py = p1[1] + t * d[1]
    pz = p1[2] + t * d[2]
    dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2)
    return (dist - radius).astype(np.float32)


@dataclass
class PrintedStatorCore:
    """Iron yoke ring plus 12 tooth wedges of the printed stator.

    Replaces the field-driven yoke: the toothed core is the multi-material
    print's magnetic ``bone`` -- yoke annulus behind the winding band, one
    wedge per tooth inside it, grown with 0.2mm overlap into the yoke so
    the iron is one connected body by construction.
    """

    cfg: MotorConfig3D

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import PRINTED_TOOTH_HALF, printed_netlist

        cfg = self.cfg
        netlist = printed_netlist(cfg)
        angles = _angles_of(cfg)
        cz = cfg.center[2]
        zh = cfg.stator_half_length
        pitch = 2.0 * np.pi / netlist.n_slots

        yoke = _sector_sdf(cfg, cfg.R_winding_outer, cfg.R_design,
                           0.0, 2.0 * np.pi, cz - zh, cz + zh, _angles_cache=angles)
        core = yoke
        for n in range(netlist.n_slots):
            tooth = _sector_sdf(cfg, cfg.R_stator_inner, cfg.R_winding_outer + 0.0002,
                                n * pitch, PRINTED_TOOTH_HALF, cz - zh, cz + zh,
                                _angles_cache=angles)
            core = np.minimum(core, tooth)
        mf.add(SDFVoxelField(sdf=core, spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)
        return mf


@dataclass
class PrintedStatorWinding:
    """Twelve hollow printed coil loops, one around every tooth.

    The topology change that makes the machine a PRINTED stator: each coil
    is a single continuous copper "picture frame" around its tooth -- two
    side bands in the half-slots (all coil sides at the SAME radii, so the
    three phases are geometrically identical and the radial-layer torque
    asymmetry of the distributed winding dies) plus bridges over both tooth
    ends that are the real, physical end turns.  Phase insulation is by
    construction: adjacent frames never touch (tooth + cladding inside,
    slot-centre separator/wall outside).

    Every copper voxel is tagged with its phase exactly (analytic per-coil
    frame SDFs), and the netlist in metadata is the single source of truth
    for the solver's impressed source AND the phase-connectivity audit.
    """

    cfg: MotorConfig3D

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import (
            PRINTED_CLAD_HALF,
            PRINTED_FRAME_HALF,
            printed_netlist,
        )

        cfg = self.cfg
        netlist = printed_netlist(cfg)
        angles = _angles_of(cfg)
        _X, _Y, Z, r, theta = angles
        cz = cfg.center[2]
        zh = cfg.stator_half_length
        zc = netlist.coil_zc(cfg)
        r_wi = cfg.R_winding_inner
        r_wo = cfg.R_winding_outer - 0.0005
        pitch = 2.0 * np.pi / netlist.n_slots

        copper_sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        phase_sdf = np.full((3,) + cfg.shape, 1e9, dtype=np.float32)
        table = {tooth: ph for tooth, ph, _pol in netlist.coil_table()}

        for n in range(netlist.n_slots):
            outer = _sector_sdf(cfg, r_wi, r_wo, n * pitch, PRINTED_FRAME_HALF,
                                cz - zc, cz + zc, _angles_cache=angles)
            window = _sector_sdf(cfg, r_wi, r_wo, n * pitch, PRINTED_CLAD_HALF,
                                 cz - zh, cz + zh, _angles_cache=angles)
            frame = np.maximum(outer, -window).astype(np.float32)
            copper_sdf = np.minimum(copper_sdf, frame)
            phase_sdf[table[n]] = np.minimum(phase_sdf[table[n]], frame)

        phase_owner = np.argmin(phase_sdf, axis=0).astype(np.int8)
        phase_owner[copper_sdf >= 0.0] = -1

        mf.add(SDFVoxelField(sdf=copper_sdf, spacing=cfg.spacing, origin=cfg.origin),
               "copper", priority=True)
        mf.metadata["winding_netlist"] = netlist
        mf.metadata["winding_phase_owner"] = phase_owner
        mf.metadata["winding_phase_sdf"] = phase_sdf
        mf.metadata["winding_style"] = "printed"
        return mf


@dataclass
class WindingInsulation:
    """Explicit dielectric kit: no copper-iron coplanar contact, anywhere.

    Four printed insulator features (the ceramic/dielectric material of the
    multi-material print):
      1. tooth-flank cladding -- the coil sides face the tooth across a
         0.6mm insulator shell, never bare iron;
      2. tooth end caps -- the copper bridges rest on insulator caps, so
         the end turns never touch the tooth iron;
      3. slot-centre separators -- phase-to-phase barrier between adjacent
         frames, continuous into the exoskeleton wall plane;
      4. the coil pocket liner -- a 0.8mm sock around every frame where it
         sits inside the exoskeleton shell, i.e. copper-iron separation on
         the load path too.
    """

    cfg: MotorConfig3D
    liner_margin: float = 0.0008

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import (
            PRINTED_CLAD_HALF,
            PRINTED_FRAME_HALF,
            PRINTED_TOOTH_HALF,
            printed_netlist,
        )

        cfg = self.cfg
        netlist = printed_netlist(cfg)
        angles = _angles_of(cfg)
        cz = cfg.center[2]
        zh = cfg.stator_half_length
        zc = netlist.coil_zc(cfg)
        pitch = 2.0 * np.pi / netlist.n_slots

        insulator = np.full(cfg.shape, 1e9, dtype=np.float32)

        for n in range(netlist.n_slots):
            theta_n = n * pitch
            # (1) tooth-flank cladding: shell between tooth and cladding angles
            clad_outer = _sector_sdf(cfg, cfg.R_stator_inner, cfg.R_winding_outer,
                                     theta_n, PRINTED_CLAD_HALF, cz - zh, cz + zh,
                                     _angles_cache=angles)
            clad_inner = _sector_sdf(cfg, cfg.R_stator_inner + 0.0002, cfg.R_winding_outer,
                                     theta_n, PRINTED_TOOTH_HALF, cz - zh + 0.001, cz + zh - 0.001,
                                     _angles_cache=angles)
            insulator = np.minimum(insulator, np.maximum(clad_outer, -clad_inner))
            # (2) tooth end caps (overlap the tooth 0.1mm so the cap is anchored)
            for sign in (+1, -1):
                cap = _sector_sdf(cfg, cfg.R_stator_inner, cfg.R_winding_outer,
                                  theta_n, PRINTED_TOOTH_HALF,
                                  cz + sign * zh - 0.0001, cz + sign * (zh + 0.0006),
                                  _angles_cache=angles)
                insulator = np.minimum(insulator, cap)
            # (3) slot-centre separator (stops at the exoskeleton wall bottom)
            sep = _sector_sdf(cfg, cfg.R_stator_inner, cfg.R_winding_outer,
                              theta_n + 0.5 * pitch, np.deg2rad(1.1),
                              cz - zc - self.liner_margin, cz + 0.0305,
                              _angles_cache=angles)
            insulator = np.minimum(insulator, sep)

        # (4) pocket liner: the shell pocket wall around every frame
        frame = _printed_frame_sdf(cfg, netlist=netlist, cache=angles)
        liner = np.maximum(frame - self.liner_margin, -frame)
        insulator = np.minimum(insulator, liner)

        mf.add(SDFVoxelField(sdf=insulator.astype(np.float32),
                             spacing=cfg.spacing, origin=cfg.origin),
               "insulator", priority=True)
        return mf


@dataclass
class StatorExoskeleton:
    """Structural exoskeleton grown around the 12 electromagnetic cells.

    Replaces the cylindrical barrel housing + flat spoked end caps (both
    forbidden forms): the load path is

        central bearing crown (hub ring + 12 thin radial walls descending
        to the yoke between the coil bridges)
        -> toothed core
        -> solid base collar that carries the winding pockets
        -> bottom plate with the coolant ports.

    Between the walls the machine keeps LARGE negative space (petal
    windows): the coil bridges are directly visible from above, exactly
    like the printed reference stators.  The bearing race itself stays a
    separate ring with an assembly gap to the crown (a pressed fit, not a
    weld -- the rotor must stay electrically and mechanically isolated
    from the stator structure).
    """

    cfg: MotorConfig3D
    hub_inner: float = 0.0155
    hub_outer: float = 0.020
    wall_half_angle: float = np.deg2rad(0.6)
    base_z0: float = -0.0415
    base_z1: float = -0.031
    plate_z0: float = -0.0435
    plate_z1: float = -0.0412
    pocket_margin: float = 0.0008

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import printed_netlist

        cfg = self.cfg
        netlist = printed_netlist(cfg)
        angles = _angles_of(cfg)
        cz = cfg.center[2]
        pitch = 2.0 * np.pi / netlist.n_slots

        parts = []
        # bearing crown hub ring
        parts.append(_sector_sdf(cfg, self.hub_inner, self.hub_outer,
                                 0.0, 2.0 * np.pi, cz + 0.033, cz + 0.037,
                                 _angles_cache=angles))
        # 12 radial walls at slot centres: from the hub down/out to the yoke
        for n in range(netlist.n_slots):
            parts.append(_sector_sdf(cfg, self.hub_inner, cfg.R_design,
                                     (n + 0.5) * pitch, self.wall_half_angle,
                                     cz + 0.0305, cz + 0.037,
                                     _angles_cache=angles))
        # solid base collar around the winding pockets
        frame = _printed_frame_sdf(cfg, margin=self.pocket_margin,
                                   netlist=netlist, cache=angles)
        shell = _sector_sdf(cfg, cfg.R_stator_inner, cfg.R_design,
                            0.0, 2.0 * np.pi, cz + self.base_z0, cz + self.base_z1,
                            _angles_cache=angles)
        parts.append(np.maximum(shell, -frame))
        # bottom plate (ports pierce it later as coolant)
        parts.append(_sector_sdf(cfg, cfg.R_stator_inner, cfg.R_design,
                                 0.0, 2.0 * np.pi, cz + self.plate_z0, cz + self.plate_z1,
                                 _angles_cache=angles))

        exo = parts[0]
        for p in parts[1:]:
            exo = np.minimum(exo, p)
        mf.add(SDFVoxelField(sdf=exo.astype(np.float32),
                             spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)
        return mf


@dataclass
class CoilCoolingNetwork:
    """Coolant INSIDE the copper: per-coil channel loops + base manifolds.

    The cooling topology the printed stator exists for: every coil frame
    carries an internal channel (elliptical tube ~1.5 x 5 mm following the
    frame loop: up one side, over the bridge, down the other side, under
    the bottom bridge), fed from a supply ring and drained into a return
    ring embedded in the exoskeleton base collar, with two ports through
    the bottom plate.  The coolant is a dedicated material so the network
    is auditable: one connected network, two openings, no trapped voids --
    and the channel walls are the heat-transfer surface directly at the
    Joule heat source.
    """

    cfg: MotorConfig3D
    channel_tangential: float = 0.00075
    channel_radial: float = 0.0025
    corner_radius: float = 0.002
    ring_supply_r: float = 0.0335
    ring_return_r: float = 0.0400
    ring_z: float = -0.0385
    ring_radius: float = 0.0013
    port_radius: float = 0.0010

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import (
            PRINTED_CLAD_HALF,
            PRINTED_FRAME_HALF,
            printed_netlist,
        )

        cfg = self.cfg
        netlist = printed_netlist(cfg)
        _X, _Y, Z, r, theta = angles = _angles_of(cfg)
        cz = cfg.center[2]
        zh = cfg.stator_half_length
        zc = netlist.coil_zc(cfg)
        r_c = 0.5 * (cfg.R_winding_inner + cfg.R_winding_outer - 0.0005)
        u_c = 0.5 * (PRINTED_CLAD_HALF + PRINTED_FRAME_HALF)
        A = u_c * r_c                       # loop half-width (arc length)
        B = zc - 0.0012                     # loop half-height
        zm = 0.5 * (zh + zc)

        coolant = np.full(cfg.shape, 1e9, dtype=np.float32)
        pitch = 2.0 * np.pi / netlist.n_slots
        for n in range(netlist.n_slots):
            d_ang = np.mod(theta - n * pitch + np.pi, 2.0 * np.pi) - np.pi
            a = d_ang * r_c
            qa = np.abs(a) - (A - self.corner_radius)
            qb = np.abs(Z - cz) - (B - self.corner_radius)
            qa_c = np.maximum(qa, 0.0)
            qb_c = np.maximum(qb, 0.0)
            d2 = (np.sqrt(qa_c ** 2 + qb_c ** 2)
                  + np.minimum(np.maximum(qa, qb), 0.0) - self.corner_radius)
            dr = r - r_c
            ell = np.sqrt((d2 / self.channel_tangential) ** 2
                          + (dr / self.channel_radial) ** 2) - 1.0
            coolant = np.minimum(coolant, ell * self.channel_tangential)

            # supply stub (go side) and return stub (return side)
            for side, r_ring in ((-1.0, self.ring_supply_r), (+1.0, self.ring_return_r)):
                ang = n * pitch + side * u_c
                p1 = (r_c * np.cos(ang), r_c * np.sin(ang), cz - zm)
                p2 = (r_ring * np.cos(ang), r_ring * np.sin(ang), cz + self.ring_z)
                stub = _segment_capsule_sdf(cfg, p1, p2, self.channel_tangential,
                                            cache=angles)
                coolant = np.minimum(coolant, stub)

        # supply and return manifold rings in the base collar
        for r_ring in (self.ring_supply_r, self.ring_return_r):
            ring = np.sqrt((r - r_ring) ** 2 + (Z - (cz + self.ring_z)) ** 2) - self.ring_radius
            coolant = np.minimum(coolant, ring.astype(np.float32))

        # two ports through the bottom plate (open to the outside air)
        for ang_deg, r_port in ((15.0, self.ring_supply_r), (195.0, self.ring_return_r)):
            ang = np.deg2rad(ang_deg)
            p1 = (r_port * np.cos(ang), r_port * np.sin(ang), cz + self.ring_z)
            p2 = (r_port * np.cos(ang), r_port * np.sin(ang), cz + self.plate_port_z())
            port = _segment_capsule_sdf(cfg, p1, p2, self.port_radius, cache=angles)
            coolant = np.minimum(coolant, port)

        mf.add(SDFVoxelField(sdf=coolant.astype(np.float32),
                             spacing=cfg.spacing, origin=cfg.origin),
               "coolant", priority=True)
        return mf

    def plate_port_z(self) -> float:
        return self.ring_z - 0.006


@dataclass
class RotorSleeve:
    """Thin retaining sleeve over surface magnets.

    Multi-functional (LEAP 71 ``one material, many jobs``):
    - Contains PM against centrifugal stress at speed
    - Provides a smooth outer surface for the air gap
    - Adds a small structural path for torque transfer
    """

    cfg: MotorConfig3D
    thickness: float = 0.0026
    clearance: float = 0.0001

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.rotor_half_length - 0.0001
        # Sleeve = the shell between (R_sleeve_outer - thickness) and
        # R_sleeve_outer.  The radius budget in config3d guarantees the
        # inner edge clears the magnets (which peak at R_rotor_outer +
        # 0.0002 + max magnet thickness) with a small assembly clearance,
        # and that the open air gap beyond R_sleeve_outer stays resolvable
        # at the physics grid.
        r_outer = cfg.R_sleeve_outer
        r_inner = r_outer - self.thickness
        if r_outer <= r_inner:
            return mf
        sleeve = _annulus(cfg, r_inner, r_outer, hz)
        mf.add(sleeve, "iron", priority=True)
        # The sleeve is STRUCTURAL iron but must be NON-MAGNETIC, like the
        # real Inconel / carbon-fibre retaining sleeve: a mu_r = 2000 sleeve
        # shunts the pole flux circumferentially, and the shunt resolves
        # ever better with grid refinement (the monotonic fine-grid torque
        # decline the convergence ladder caught).  realize() subtracts the
        # listed regions from rho_iron (they become air magnetically)
        # while everything else about them stays iron.
        mf.metadata.setdefault("nonmagnetic_regions", []).append(sleeve)
        return mf


@dataclass
class StatorSegmentation:
    """Radial slits through the stator yoke to suppress eddy currents.

    The 3D equivalent of laminated silicon steel: thin angular cuts break the
    circumferential current loop, so high-frequency AC losses are reduced.
    This is the LEAP 71 ``manufacturing constraint drives geometry`` pattern
    applied to electromagnetic efficiency.
    """

    cfg: MotorConfig3D
    n_slits: int = 8
    slit_width: float = 0.002

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.stator_half_length
        cx, cy = cfg.center[0], cfg.center[1]
        r0 = cfg.R_stator_inner
        r1 = cfg.R_design
        pitch = 2.0 * np.pi / self.n_slits
        span = self.slit_width / max(r0, 1e-6)
        for i in range(self.n_slits):
            centre = i * pitch + pitch * 0.5
            slit = from_implicit(
                cfg.shape, cfg.spacing, cfg.origin,
                annular_sector((cx, cy), r0, r1,
                               centre - span * 0.5, centre + span * 0.5, hz),
            )
            mf.subtract(slit, "iron")
        return mf


@dataclass
class ShaftAndBearings:
    """Central shaft, rotor hub spokes and the two bearing races.

    The end caps are GONE: their flat spoked discs were the old housing
    mother's closing plates, replaced by the exoskeleton's bearing crown.
    The bearing races stay separate rings with a deliberate assembly gap
    to the crown hub (a press fit at assembly time -- never a weld, or the
    rotor would be structurally shorted to the stator and the
    connectivity audit would rightly flag a cross-bridge).
    """

    cfg: MotorConfig3D
    shaft_extension: float = 0.012
    bearing_width: float = 0.004
    bearing_radius: float = 0.0
    end_cap_thickness: float = 0.0  # kept for API compat; caps removed (P4)
    n_cap_spokes: int = 6
    spoke_fraction: float = 0.35
    n_hub_spokes: int = 6
    hub_spoke_width: float = 0.004

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        cx, cy, cz = cfg.center
        hz = cfg.stator_half_length
        r_shaft = cfg.R_shaft
        r_bearing = self.bearing_radius or (r_shaft + 0.006)

        X, Y, Z, r, theta = _angles_of(cfg)
        shaft_half = hz + self.shaft_extension
        shaft_sdf = np.maximum(r - r_shaft, np.abs(Z - cz) - shaft_half)

        # Hub spokes: the structural anchor that ties the rotor iron to the
        # shaft (rotor -> hub -> shaft load path).  A web of spokes at each
        # rotor end bridges the bore clearance with deliberate overlap into
        # both the shaft and the rotor body -- connectivity by construction,
        # not by contact.
        r_rotor_bore = cfg.R_shaft + 0.004
        parts = [shaft_sdf]
        hub_z = cfg.rotor_half_length - 0.002
        spoke_pitch = 2.0 * np.pi / self.n_hub_spokes
        for sign in (+1, -1):
            z_h = cz + sign * (hub_z - 0.5 * self.hub_spoke_width)
            axial_h = np.abs(Z - z_h) - 0.5 * self.hub_spoke_width
            radial_bound = r - (r_rotor_bore + 0.002)  # overlap into rotor iron
            web = np.full(cfg.shape, 1e9, dtype=np.float32)
            for i in range(self.n_hub_spokes):
                centre = i * spoke_pitch
                d_ang = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
                # angular band whose arc-width equals the spoke width
                band = np.abs(d_ang) * r - 0.5 * self.hub_spoke_width
                web = np.minimum(web, np.maximum(np.maximum(band, axial_h), radial_bound))
            parts.append(web.astype(np.float32))

        bearing_half = self.bearing_width * 0.5
        for sign in (+1, -1):
            # Bearing sits fully beyond the rotor end so it never bridges
            # shaft and rotor iron (rotor ends at rotor_half_length), and
            # keeps its assembly gap to the crown hub (press fit, no weld).
            z_b = cz + sign * (cfg.rotor_half_length + cfg.axial_airgap + self.bearing_width + bearing_half)
            axial_b = np.abs(Z - z_b) - bearing_half
            bearing_outer = np.maximum(r - r_bearing, axial_b)
            bearing_hole = np.maximum(r - (r_shaft - 0.0001), axial_b)
            parts.append(np.maximum(bearing_outer, -bearing_hole))

        combined = parts[0]
        for p in parts[1:]:
            combined = np.minimum(combined, p)

        mf.add(SDFVoxelField(sdf=combined.astype(np.float32),
                             spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)
        return mf


@dataclass
class MotorHousing:
    """Petal-style housing cage with mounting ears and cutout windows.

    LEAP 71 ``structural exoskeleton``: instead of a solid cylinder that
    hides the motor interior, the housing is a ring of petal/blade
    segments with large angular windows between them.  The windows expose
    the copper windings and stator teeth, while the blades provide
    structural stiffness and cooling-surface area.  Mounting ears with
    bolt holes make it a bolted assembly.
    """

    cfg: MotorConfig3D
    housing_radius: float = 0.055
    housing_thickness: float = 0.003
    n_blades: int = 8
    blade_fraction: float = 0.45
    n_mounting_ears: int = 4
    ear_size: float = 0.007

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField, smooth_boolean_subtract

        cfg = self.cfg
        cx, cy, cz = cfg.center
        hz = cfg.stator_half_length + 0.001
        r_in = self.housing_radius
        r_out = r_in + self.housing_thickness
        blade_pitch = 2.0 * np.pi / self.n_blades
        blade_span = blade_pitch * self.blade_fraction

        X, Y, Z, r, theta = _angles_of(cfg)
        axial_ring = np.abs(Z - cz) - hz
        radial_outer = r - r_out
        radial_inner = r_in - r
        ring_sdf = np.maximum(radial_outer, np.maximum(radial_inner, axial_ring))

        for i in range(self.n_blades):
            centre = i * blade_pitch + blade_pitch * 0.5
            d_theta = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
            angular_d = np.abs(d_theta) - blade_span * 0.5
            window_sdf = np.maximum(
                np.maximum(r - (r_out + 0.002), (r_in - 0.002) - r),
                np.maximum(angular_d, np.abs(Z - cz) - (hz + 0.002))
            ).astype(np.float32)
            ring_sdf = np.maximum(ring_sdf, -window_sdf).astype(np.float32)

        mf.add(SDFVoxelField(sdf=ring_sdf, spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)

        ring_half = 0.0018
        for sign in (+1, -1):
            z_rim = cz + sign * (hz + ring_half)
            axial_rim = np.abs(Z - z_rim) - ring_half
            rim_sdf = np.maximum(r - r_out, np.maximum(r_in - r, axial_rim))
            for i in range(self.n_blades):
                centre = i * blade_pitch + blade_pitch * 0.5
                d_theta = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
                angular_d = np.abs(d_theta) - blade_span * 0.5
                window_sdf = np.maximum(
                    np.maximum(r - (r_out + 0.002), (r_in - 0.002) - r),
                    np.maximum(angular_d, axial_rim)
                ).astype(np.float32)
                rim_sdf = np.maximum(rim_sdf, -window_sdf).astype(np.float32)
            mf.add(SDFVoxelField(sdf=rim_sdf.astype(np.float32),
                                 spacing=cfg.spacing, origin=cfg.origin),
                   "iron", priority=True)

        ear_pitch = 2.0 * np.pi / self.n_mounting_ears
        for i in range(self.n_mounting_ears):
            angle = i * ear_pitch + ear_pitch * 0.5
            ex = float(cx + r_out * np.cos(angle))
            ey = float(cy + r_out * np.sin(angle))
            ear = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                                cylinder_z((ex, ey), self.ear_size * 0.5, hz + 0.002))
            bolt = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                                 cylinder_z((ex, ey), self.ear_size * 0.2, hz + 0.002))
            ear = boolean_subtract(ear, bolt)
            mf.add(ear, "iron", priority=True)
        return mf


@dataclass
class FunctionalVoids:
    """Protected functional voids subtracted from ALL materials last.

    The radial air gap between the rotor's outermost solid (sleeve/magnet)
    and the stator inner surface must never be bridged.  This final pass
    subtracts that gap annulus from every material, ensuring later
    additions cannot fill it.  This is the LEAP 71 ``functional void
    first`` principle enforced as a guaranteed final pass.

    The void starts at the sleeve outer surface, NOT at R_rotor_outer —
    otherwise the PM and sleeve would be deleted along with the gap.
    """

    cfg: MotorConfig3D
    rotor_solid_outer: float = 0.0

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        cz = cfg.center[2]
        hz = cfg.stator_half_length

        _X, _Y, Z, r, _theta = _angles_of(cfg)

        r_eff = self.rotor_solid_outer
        if r_eff <= 0.0:
            r_eff = min(cfg.R_sleeve_outer + 0.0001, cfg.R_stator_inner - 0.0005)

        radial_gap = np.maximum(r - cfg.R_stator_inner, (r_eff + 0.0001) - r)
        axial_limit = np.abs(Z - cz) - hz
        gap_sdf = np.maximum(radial_gap, axial_limit)

        mf.add(SDFVoxelField(sdf=gap_sdf.astype(np.float32),
                             spacing=cfg.spacing, origin=cfg.origin),
               "air", priority=True)
        return mf


@dataclass
class StructuralContinuity:
    """Final pass: enforce the structural connectivity graph.

    LEAP 71 doctrine: connectivity belongs in the GROWTH rules (features
    grow from anchors with deliberate overlap), and this pass is the
    safety net that makes the invariant hold regardless of what came
    before -- every iron/PM component must trace back to the shaft
    (rotor side) or the housing ring (stator side).  Floating islands
    are deleted, exactly like disconnected metal would be rejected in
    manufacture.  Copper is untouched: its connectivity belongs to the
    ELECTRICAL graph (phase networks), not this one.
    """

    cfg: MotorConfig3D

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.connectivity import prune_floating_islands

        return prune_floating_islands(mf, self.cfg)


# ---------------------------------------------------------------------------
# P5: field-grown stator cell (swept multi-band copper, arched walls)
# ---------------------------------------------------------------------------


def _band_centerline(
    r_k: float,
    amp_k: float,
    side_angle: float,
    zc: float,
    n_arch: int = 10,
) -> np.ndarray:
    """Centreline polyline for one swept copper band around a tooth.

    A closed loop in the ``(theta, z)`` plane at constant radius ``r_k``:
    two axial sides at ``theta = +/- side_angle`` (the slot sides) joined
    by raised-cosine arches above and below the stack.  The cosine gives
    a smooth (C1) tangent at the junctions — no kink, no stress
    concentration, and a self-supporting overhang angle everywhere.

    ``amp_k`` grows with ``r_k`` so outer bands arch higher than inner
    ones: the stacked bands form a dome whose apex is the outermost band.
    """
    ca, sa = np.cos(side_angle), np.sin(side_angle)
    thetas = np.linspace(-side_angle, side_angle, n_arch)
    cos_profile = (1.0 + np.cos(np.pi * thetas / side_angle)) * 0.5

    pts = [
        [r_k * ca, -r_k * sa, -zc],   # side A bottom
        [r_k * ca, -r_k * sa,  zc],   # side A top
    ]
    for th, c in zip(thetas, cos_profile):
        pts.append([r_k * np.cos(th), r_k * np.sin(th), zc + amp_k * c])

    pts.append([r_k * ca,  r_k * sa,  zc])    # side B top
    pts.append([r_k * ca,  r_k * sa, -zc])    # side B bottom
    for th, c in zip(reversed(thetas), reversed(cos_profile)):
        pts.append([r_k * np.cos(th), r_k * np.sin(th), -(zc + amp_k * c)])

    return np.array(pts, dtype=np.float64)


@dataclass
class StatorCell:
    """One electromagnetic-thermal-structural cell of a printed stator.

    Replaces the monolithic CSG stator (PrintedStatorCore + Winding +
    Insulation + Exoskeleton + Cooling) with a single field-grown cell
    that is then replicated polarly by :class:`StatorCellArray`.

    Each cell carries:
      - iron tooth (prism, the magnetic pole piece)
      - local yoke arc (one cell pitch of back-iron)
      - 6-8 swept copper bands (raised-cosine arched end-turns, dome-
        stacked with outer bands rising higher than inner)
      - interface-only insulation (thin cladding on tooth flanks + end
        caps — NOT a sock; copper surface stays EXPOSED)
      - in-band coolant channels (offset inside each copper band, coolant
        at the Joule-heat source)
    """

    cfg: MotorConfig3D
    tooth_index: int = 0
    n_bands: int = 7
    band_radius: float = 0.0007
    channel_wall: float = 0.0004
    arch_base: float = 0.002
    arch_slope: float = 1.0
    clad_thickness: float = 0.0003
    tooth_tip_dome: float = 0.0015

    def _band_radii(self, cfg: MotorConfig3D) -> tuple[np.ndarray, np.ndarray]:
        r_wi = cfg.R_winding_inner
        r_wo = cfg.R_winding_outer - 0.0005
        r_k = np.linspace(r_wi + 0.0003, r_wo - 0.0003, self.n_bands)
        amp = self.arch_base + (r_k - r_wi) * self.arch_slope
        return r_k, amp

    def _theta0(self) -> float:
        from organic_motor.construct.winding_netlist import printed_netlist
        netlist = printed_netlist(self.cfg)
        pitch = 2.0 * np.pi / netlist.n_slots
        return self.tooth_index * pitch

    def build_iron(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import (
            PRINTED_TOOTH_HALF, printed_netlist,
        )
        cfg = self.cfg
        angles = _angles_of(cfg)
        _X, _Y, Z, r, theta = angles
        cz = cfg.center[2]
        zh = cfg.stator_half_length
        th0 = self._theta0()
        d_ang = np.mod(theta - th0 + np.pi, 2.0 * np.pi) - np.pi

        # tooth prism with slightly domed tip (self-supporting AM)
        tooth_body = np.maximum(
            np.maximum(r - cfg.R_winding_outer, cfg.R_stator_inner - r),
            np.abs(d_ang) - PRINTED_TOOTH_HALF,
        )
        tooth_dz = np.abs(Z - cz) - (zh + self.tooth_tip_dome)
        tooth = np.maximum(tooth_body, tooth_dz)
        # local yoke arc (one cell pitch)
        yoke = np.maximum(
            np.maximum(r - cfg.R_design, cfg.R_winding_outer - r),
            np.abs(d_ang) - (np.pi / 6.0),
        )
        yoke = np.maximum(yoke, np.abs(Z - cz) - zh)
        iron = np.minimum(tooth, yoke).astype(np.float32)
        mf.add(SDFVoxelField(sdf=iron, spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)
        return mf

    def build_copper(self, mf: MaterialField, grid=None) -> MaterialField:
        from organic_motor.construct.field import polyline_capsule_sdf
        from organic_motor.construct.winding_netlist import (
            PRINTED_FRAME_HALF, printed_netlist,
        )
        cfg = self.cfg
        netlist = printed_netlist(cfg)
        th0 = self._theta0()
        cz = cfg.center[2]
        zc = netlist.coil_zc(cfg)
        r_k, amp = self._band_radii(cfg)
        table = {t: ph for t, ph, _ in netlist.coil_table()}
        phase = table[self.tooth_index]

        copper_sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        phase_sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        for k in range(self.n_bands):
            pts = _band_centerline(r_k[k], amp[k], PRINTED_FRAME_HALF, zc)
            # rotate to tooth position
            rot = th0
            ca, sa = np.cos(rot), np.sin(rot)
            pts[:, 0:2] = pts[:, 0:2] @ np.array([[ca, -sa], [sa, ca]])
            band = polyline_capsule_sdf(cfg.shape, cfg.spacing, cfg.origin,
                                        pts, self.band_radius, grid=grid)
            copper_sdf = np.minimum(copper_sdf, band)
            phase_sdf = np.minimum(phase_sdf, band)

        mf.add(SDFVoxelField(sdf=copper_sdf, spacing=cfg.spacing, origin=cfg.origin),
               "copper", priority=True)

        # accumulate phase SDF for the whole stator (multi-call safe)
        prev = mf.metadata.get("winding_phase_sdf")
        if prev is None:
            mf.metadata["winding_phase_sdf"] = np.full(
                (3,) + cfg.shape, 1e9, dtype=np.float32)
        ps = mf.metadata["winding_phase_sdf"]
        ps[phase] = np.minimum(ps[phase], phase_sdf)
        mf.metadata["winding_phase_sdf"] = ps
        return mf

    def build_insulation(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import (
            PRINTED_TOOTH_HALF, PRINTED_CLAD_HALF, printed_netlist,
        )
        cfg = self.cfg
        angles = _angles_of(cfg)
        _X, _Y, Z, r, theta = angles
        cz = cfg.center[2]
        zh = cfg.stator_half_length
        th0 = self._theta0()
        d_ang = np.mod(theta - th0 + np.pi, 2.0 * np.pi) - np.pi
        t = self.clad_thickness

        # (1) tooth-flank cladding: thin shell between tooth and clad angles
        clad_outer = np.maximum(
            np.maximum(r - cfg.R_winding_outer, cfg.R_stator_inner - r),
            np.abs(d_ang) - PRINTED_CLAD_HALF,
        )
        clad_inner = np.maximum(
            np.maximum(r - cfg.R_winding_outer, (cfg.R_stator_inner + 0.0002) - r),
            np.abs(d_ang) - PRINTED_TOOTH_HALF,
        )
        clad = np.maximum(clad_outer, -clad_inner + 2 * t * 0)  # wall logic
        clad = np.maximum(
            np.maximum(clad, -(clad_inner - t)),
            np.abs(Z - cz) - (zh + 0.0001),
        )
        # simpler: ring between TOOTH_HALF and CLAD_HALF
        flank = np.maximum(
            np.maximum(r - cfg.R_winding_outer, cfg.R_stator_inner - r),
            np.maximum(
                np.abs(d_ang) - PRINTED_CLAD_HALF,
                -(np.abs(d_ang) - PRINTED_TOOTH_HALF),
            ),
        )
        flank = np.maximum(flank, np.abs(Z - cz) - (zh + 0.0006))
        insulator = flank.astype(np.float32)

        # (2) tooth end caps
        for sign in (+1, -1):
            cap = np.maximum(
                np.maximum(r - cfg.R_winding_outer, cfg.R_stator_inner - r),
                np.maximum(
                    np.abs(d_ang) - PRINTED_TOOTH_HALF,
                    np.abs(Z - (cz + sign * (zh + 0.0003))) - 0.0004,
                ),
            )
            insulator = np.minimum(insulator, cap.astype(np.float32))

        # (3) slot-center separator (thin wall at cell boundary)
        pitch = 2.0 * np.pi / printed_netlist(cfg).n_slots
        sep_d = np.abs(np.mod(theta - th0 - 0.5 * pitch + np.pi, 2*np.pi) - np.pi)
        sep = np.maximum(
            np.maximum(r - cfg.R_winding_outer, cfg.R_stator_inner - r),
            np.maximum(sep_d - np.deg2rad(0.5), np.abs(Z - cz) - (zh + 0.002)),
        )
        insulator = np.minimum(insulator, sep.astype(np.float32))

        prev = mf.metadata.get("_insulator_sdf")
        if prev is not None:
            insulator = np.minimum(prev, insulator)
        mf.metadata["_insulator_sdf"] = insulator
        return mf

    def build_coolant(self, mf: MaterialField, grid=None) -> MaterialField:
        from organic_motor.construct.field import polyline_capsule_sdf
        from organic_motor.construct.winding_netlist import (
            PRINTED_FRAME_HALF, printed_netlist,
        )
        cfg = self.cfg
        netlist = printed_netlist(cfg)
        th0 = self._theta0()
        cz = cfg.center[2]
        zc = netlist.coil_zc(cfg)
        r_k, amp = self._band_radii(cfg)
        ch_r = self.band_radius - self.channel_wall
        if ch_r <= 0.0:
            return mf

        coolant = np.full(cfg.shape, 1e9, dtype=np.float32)
        for k in range(self.n_bands):
            pts = _band_centerline(r_k[k], amp[k], PRINTED_FRAME_HALF, zc)
            ca, sa = np.cos(th0), np.sin(th0)
            pts[:, 0:2] = pts[:, 0:2] @ np.array([[ca, -sa], [sa, ca]])
            ch = polyline_capsule_sdf(cfg.shape, cfg.spacing, cfg.origin,
                                      pts, ch_r, grid=grid)
            coolant = np.minimum(coolant, ch)

        prev = mf.metadata.get("_coolant_sdf")
        if prev is not None:
            coolant = np.minimum(prev, coolant)
        mf.metadata["_coolant_sdf"] = coolant
        return mf


@dataclass
class StatorCellArray:
    """Twelve polarly-replicated StatorCells + shared exoskeleton.

    Replaces the five P4 monolithic stator classes with one cell mother
    replicated 12 times around the circumference.  The exoskeleton (bearing
    crown, arched radial walls, base collar, bottom plate) is shared
    structural iron grown between the cells.
    """

    cfg: MotorConfig3D
    n_bands: int = 7
    band_radius: float = 0.0007
    channel_wall: float = 0.0004
    arch_base: float = 0.002
    arch_slope: float = 1.0
    clad_thickness: float = 0.0003
    hub_inner: float = 0.0155
    hub_outer: float = 0.020
    wall_half_angle: float = np.deg2rad(0.6)
    base_z0: float = -0.0415
    base_z1: float = -0.031
    plate_z0: float = -0.0435
    plate_z1: float = -0.0412

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.winding_netlist import printed_netlist
        from organic_motor.construct.field import SDFVoxelField
        cfg = self.cfg
        netlist = printed_netlist(cfg)
        n_slots = netlist.n_slots
        pitch = 2.0 * np.pi / n_slots
        cz = cfg.center[2]
        angles = _angles_of(cfg)
        _X, _Y, Z, r, theta = angles

        # --- iron: teeth + local yoke arcs (per-cell) ---
        for n in range(n_slots):
            cell = StatorCell(cfg, tooth_index=n, n_bands=self.n_bands,
                              band_radius=self.band_radius,
                              channel_wall=self.channel_wall,
                              arch_base=self.arch_base,
                              arch_slope=self.arch_slope,
                              clad_thickness=self.clad_thickness)
            # build iron into a temp mf
            cell.build_iron(mf)
        # The per-cell build_iron already adds to mf via mf.add() — good.

        # --- exoskeleton: bearing crown + arched walls + base + plate ---
        from organic_motor.construct.winding_netlist import (
            PRINTED_FRAME_HALF, PRINTED_CLAD_HALF,
        )
        # bearing crown hub ring
        crown = np.maximum(
            np.maximum(r - self.hub_outer, self.hub_inner - r),
            np.abs(Z - (cz + 0.035)) - 0.002,
        )
        exo = crown
        # arched radial walls at slot centres
        for n in range(n_slots):
            wall_c = (n + 0.5) * pitch
            d_ang = np.mod(theta - wall_c + np.pi, 2*np.pi) - np.pi
            ang_bound = np.abs(d_ang) - self.wall_half_angle
            # z(r) arch: wall descends from crown to yoke
            t = np.clip((r - self.hub_outer) / (cfg.R_design - self.hub_outer), 0, 1)
            z_wall_top = (cz + 0.037) - (0.037 - 0.0305) * np.sqrt(t)
            z_wall_bot = cz + 0.0305
            d_z_wall = np.maximum(Z - z_wall_top, z_wall_bot - Z)
            r_bound = np.maximum(r - cfg.R_design, self.hub_inner - r)
            wall = np.maximum(np.maximum(ang_bound * r, d_z_wall), r_bound)
            exo = np.minimum(exo, wall.astype(np.float32))

        # base collar (carries winding pockets)
        frame_func = _printed_frame_sdf(cfg, margin=0.0008, netlist=netlist, cache=angles)
        shell_base = np.maximum(
            np.maximum(r - cfg.R_design, cfg.R_stator_inner - r),
            np.abs(Z - (cz + (self.base_z0 + self.base_z1) * 0.5)) - (self.base_z1 - self.base_z0) * 0.5,
        )
        base = np.maximum(shell_base, -frame_func)
        exo = np.minimum(exo, base.astype(np.float32))

        # bottom plate
        plate = np.maximum(
            np.maximum(r - cfg.R_design, cfg.R_stator_inner - r),
            np.abs(Z - (cz + (self.plate_z0 + self.plate_z1) * 0.5)) - (self.plate_z1 - self.plate_z0) * 0.5,
        )
        exo = np.minimum(exo, plate.astype(np.float32))

        mf.add(SDFVoxelField(sdf=exo.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)

        # --- copper: 12 cells of swept bands ---
        # Pre-compute meshgrid once for all polyline_capsule_sdf calls
        grid = (_X, _Y, Z)
        for n in range(n_slots):
            cell = StatorCell(cfg, tooth_index=n, n_bands=self.n_bands,
                              band_radius=self.band_radius,
                              channel_wall=self.channel_wall,
                              arch_base=self.arch_base,
                              arch_slope=self.arch_slope,
                              clad_thickness=self.clad_thickness)
            cell.build_copper(mf, grid=grid)
        # merge phase_sdf from mf.metadata
        phase_sdf = mf.metadata.get("winding_phase_sdf")
        copper_sdf = mf.sdfs.get("copper")
        if copper_sdf is not None:
            phase_owner = np.argmin(phase_sdf, axis=0).astype(np.int8)
            phase_owner[copper_sdf.sdf >= 0.0] = -1
            mf.metadata["winding_phase_owner"] = phase_owner
            mf.metadata["winding_netlist"] = netlist
            mf.metadata["winding_style"] = "printed"

        # --- insulation (interface-only, no sock) ---
        for n in range(n_slots):
            cell = StatorCell(cfg, tooth_index=n, n_bands=self.n_bands,
                              band_radius=self.band_radius,
                              channel_wall=self.channel_wall,
                              arch_base=self.arch_base,
                              arch_slope=self.arch_slope,
                              clad_thickness=self.clad_thickness)
            cell.build_insulation(mf)
        ins_sdf = mf.metadata.get("_insulator_sdf")
        if ins_sdf is not None:
            mf.add(SDFVoxelField(sdf=ins_sdf.astype(np.float32),
                                 spacing=cfg.spacing, origin=cfg.origin),
                   "insulator", priority=True)

        # --- coolant (in-band channels) ---
        for n in range(n_slots):
            cell = StatorCell(cfg, tooth_index=n, n_bands=self.n_bands,
                              band_radius=self.band_radius,
                              channel_wall=self.channel_wall,
                              arch_base=self.arch_base,
                              arch_slope=self.arch_slope,
                              clad_thickness=self.clad_thickness)
            cell.build_coolant(mf, grid=grid)
        cool_sdf = mf.metadata.get("_coolant_sdf")
        if cool_sdf is not None:
            mf.add(SDFVoxelField(sdf=cool_sdf.astype(np.float32),
                                 spacing=cfg.spacing, origin=cfg.origin),
                   "coolant", priority=True)

        return mf


# ---------------------------------------------------------------------------
# Whole motor assembly
# ---------------------------------------------------------------------------

@dataclass
class Motor:
    """Assemble a complete inner-rotor surface-PM motor.

    The component list is the agent's design vocabulary: adding, removing or
    re-parameterising a component is the whole of a redesign.  The assembly
    order encodes material priority (later components win overlaps).
    """

    cfg: MotorConfig3D
    components: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.components:
            self.components = [
                RotorCore(self.cfg),
                SurfaceMagnets(self.cfg),
                StatorCore(self.cfg),
                DistributedWinding(self.cfg),
                CoolingJacket(self.cfg),
            ]

    def build(self) -> MaterialField:
        mf = _grid_of(self.cfg)
        for component in self.components:
            mf = component.build(mf)
        return mf

    def magnetization(self) -> np.ndarray | None:
        for component in self.components:
            if hasattr(component, "magnetization") and callable(getattr(component, "magnetization")):
                return component.magnetization()
        return None


def baseline_motor(cfg: MotorConfig3D | None = None) -> Motor:
    """Construct the default baseline surface-PM motor for ``cfg``."""
    return Motor(cfg or MotorConfig3D())


def field_driven_motor(cfg: MotorConfig3D | None = None) -> Motor:
    """A motor built entirely from field-driven computational objects.

    P5 topology: the printed multi-material stator is now grown from
    field-driven CELLS.  Each of the 12 teeth is one electromagnetic/
    thermal/structural cell -- iron tooth, 6-8 swept copper bands with
    raised-cosine arched end-turns (dome-stacked), interface-only
    insulation (no sock -- copper stays exposed), in-band coolant
    channels -- replicated polarly by :class:`StatorCellArray`.  The
    exoskeleton (bearing crown, arched radial walls, base collar) is
    grown between the cells with z(r) arch profiles, not flat z=const.

    The forbidden forms (outer spiral barrel, cylindrical housing, flat
    spoked end caps, distributed lap winding, monolithic copper frame,
    insulator sock) are gone.  The rotor keeps the P2-validated radius
    budget (4mm magnets, 3mm gap, non-magnetic sleeve).
    """
    cfg = cfg or MotorConfig3D()
    return Motor(cfg, components=[
        ShaftAndBearings(cfg),
        RotorCore(cfg),
        FieldDrivenMagnets(cfg),
        RotorSleeve(cfg),
        StatorCellArray(cfg, n_bands=7, channel_wall=0.0007),
        FunctionalVoids(cfg),
        StructuralContinuity(cfg),
    ])
