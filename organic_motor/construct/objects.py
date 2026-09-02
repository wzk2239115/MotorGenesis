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
    clearance: float = 0.0006

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
    ``thickness`` is the radial magnet thickness.  Magnetisation alternates
    radially between poles; :meth:`magnetization` returns the per-voxel
    magnetisation vector the critic needs.
    """

    cfg: MotorConfig3D
    thickness: float = 0.0035
    pole_fraction: float = 0.72

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
        cfg = self.cfg
        hz = cfg.rotor_half_length - 0.0002
        for a0, a1, r0, r1, _sign, _centre in self._pole_specs():
            pole = from_implicit(
                cfg.shape, cfg.spacing, cfg.origin,
                annular_sector((cfg.center[0], cfg.center[1]), r0, r1, a0, a1, hz),
            )
            mf.add(pole, "pm", priority=True)
        return mf

    def magnetization(self) -> np.ndarray:
        """Per-voxel unit magnetisation vector, alternating radially per pole.

        Radial outward for even poles, inward for odd -- the canonical
        surface-PM magnetisation pattern.  ``normalized_magnetization3d``
        multiplies this by ``rho_pm``, so the direction is only used where
        magnet material is actually present.
        """
        cfg = self.cfg
        _X, _Y, _Z, _r, angle = _angles_of(cfg)
        poles = 2 * cfg.pole_pairs
        pitch = 2.0 * np.pi / poles
        pole_index = np.rint(angle / pitch)
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
    max_thickness: float = 0.0045
    pole_fraction: float = 0.72
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
        axial_factor = 0.4 + 0.6 * np.sin(np.pi * z_norm)

        sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        for p in range(poles):
            centre = p * pitch
            ang = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
            in_pole = np.abs(ang) < pitch * self.pole_fraction * 0.5
            b_avg = float(fmag[in_pole].mean()) / fmax if in_pole.any() else 0.0
            base_t = self.min_thickness + (self.max_thickness - self.min_thickness) * b_avg
            r1_z = r0 + base_t * axial_factor
            rm = 0.5 * (r0 + r1_z)
            dr = 0.5 * (r1_z - r0)
            radial = np.abs(r - rm) - dr
            axial = np.abs(Z - cz) - hz
            cyl = np.maximum(radial, axial)
            a0 = centre - pitch * self.pole_fraction * 0.5
            a1 = centre + pitch * self.pole_fraction * 0.5
            c0, s0 = np.cos(a0), np.sin(a0)
            c1, s1 = np.cos(a1), np.sin(a1)
            px = X - cx
            py = Y - cy
            h0 = c0 * py - s0 * px
            h1 = s1 * px - c1 * py
            pole_sdf = np.maximum(cyl, np.maximum(-h0, -h1))
            sdf = np.minimum(sdf, pole_sdf)
        from organic_motor.construct.field import SDFVoxelField

        mf.add(SDFVoxelField(sdf=sdf, spacing=cfg.spacing, origin=cfg.origin), "pm", priority=True)
        return mf

    def magnetization(self) -> np.ndarray:
        return SurfaceMagnets(self.cfg).magnetization()


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
    slot_opening: float = 0.0028
    min_yoke_thickness: float = 0.006
    max_yoke_thickness: float = 0.018
    angular_slices: int = 48
    flux_field: object | None = None  # ScalarField; default = airgap_B

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.fields_motor import airgap_B
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        hz = cfg.stator_half_length
        cx, cy = cfg.center[0], cfg.center[1]
        # The yoke MUST stay connected to the air gap (r = R_stator_inner) so
        # the magnetic circuit never opens; thickness grows outward from there.
        r_inner = cfg.R_stator_inner
        max_available = cfg.R_design - cfg.R_stator_inner
        max_thickness = min(self.max_yoke_thickness, max_available)

        field = self.flux_field
        if field is None:
            field = airgap_B(cfg)
        fmag = np.abs(field.data)
        fmax = max(float(fmag.max()), 1e-9)

        _X, _Y, _Z, r, theta = _angles_of(cfg)
        slice_pitch = 2.0 * np.pi / self.angular_slices
        sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        for s in range(self.angular_slices):
            centre = s * slice_pitch
            mask = np.abs(((theta - centre + np.pi) % (2 * np.pi)) - np.pi) < slice_pitch * 0.5
            b_avg = float(fmag[mask].mean()) / fmax if mask.any() else 0.0
            thickness = self.min_yoke_thickness + (max_thickness - self.min_yoke_thickness) * b_avg
            r_outer = r_inner + thickness
            a0 = centre - slice_pitch * 0.5
            a1 = centre + slice_pitch * 0.5
            slice_sdf = from_implicit(
                cfg.shape, cfg.spacing, cfg.origin,
                annular_sector((cx, cy), r_inner, r_outer, a0, a1, hz),
            )
            sdf = np.minimum(sdf, slice_sdf.sdf)
        yoke = SDFVoxelField(sdf=sdf.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin)

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
    """Real 3D distributed winding with slot conductors and end turns.

    Each coil is a continuous copper loop: two straight axial sections in
    slots on opposite sides of a tooth, connected by curved end turns that
    arc over the stator ends.  Multiple concentric turns give the layered
    coil-pack appearance of a real form-wound or hairpin winding.

    This replaces the continuous copper annulus with readable coil geometry
    that has clear slot conductors, end-winding overhangs and turn-to-turn
    spacing -- the visual signature of a real motor.
    """

    cfg: MotorConfig3D
    n_slots: int = 12
    coil_span: int = 3
    n_layers: int = 2
    conductor_radial: float = 0.004
    end_turn_rise: float = 0.003
    end_turn_gap: float = 0.001

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
        hz = cfg.stator_half_length
        r_wi = cfg.R_winding_inner
        r_wo = cfg.R_winding_outer
        slot_pitch = 2.0 * np.pi / self.n_slots
        cond_half = self.conductor_radial * 0.5

        X, Y, Z, r, theta = _angles_of(cfg)
        copper_sdf = np.full(cfg.shape, 1e9, dtype=np.float32)

        slot_width = slot_pitch * 0.72

        for slot in range(self.n_slots):
            theta_slot = slot * slot_pitch
            d_theta = np.mod(theta - theta_slot + np.pi, 2 * np.pi) - np.pi
            in_slot = np.abs(d_theta) < slot_width * 0.5

            for layer in range(self.n_layers):
                r_mid = r_wi + (layer + 0.5) * (r_wo - r_wi) / self.n_layers
                radial_dist = np.abs(r - r_mid) - cond_half
                angular_dist = r_mid * (np.abs(d_theta) - slot_width * 0.5)
                angular_dist = np.maximum(angular_dist, 0.0)
                seg_dist = np.sqrt(radial_dist ** 2 + angular_dist ** 2)
                axial = np.abs(Z - cz) - hz
                axial_clamped = np.maximum(axial, 0.0)
                dist = np.sqrt(seg_dist ** 2 + axial_clamped ** 2)
                copper_sdf = np.minimum(copper_sdf, dist - cond_half * 0.5)

        for coil in range(self.n_slots):
            slot_a = coil
            slot_b = (coil + self.coil_span) % self.n_slots
            theta_a = slot_a * slot_pitch
            theta_b = slot_b * slot_pitch

            for layer in range(self.n_layers):
                r_mid = r_wi + (layer + 0.5) * (r_wo - r_wi) / self.n_layers

                for sign in (+1, -1):
                    z_end = cz + sign * (hz + self.end_turn_gap + cond_half)
                    mid = 0.5 * (theta_a + theta_b)
                    half_span = 0.5 * self.coil_span * slot_pitch
                    d_mid = np.mod(theta - mid + np.pi, 2 * np.pi) - np.pi
                    in_arc = np.abs(d_mid) < half_span
                    d_a = np.abs(np.mod(theta - theta_a + np.pi, 2 * np.pi) - np.pi)
                    d_b = np.abs(np.mod(theta - theta_b + np.pi, 2 * np.pi) - np.pi)
                    arc_dist = np.where(in_arc, 0.0, np.minimum(d_a, d_b)).astype(np.float32)
                    dist_end = np.sqrt(
                        (r - r_mid) ** 2 + (Z - z_end) ** 2 + (r_mid * arc_dist) ** 2
                    )
                    copper_sdf = np.minimum(copper_sdf, dist_end - cond_half * 0.5)

        mf.add(SDFVoxelField(sdf=copper_sdf, spacing=cfg.spacing, origin=cfg.origin), "copper", priority=True)
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
    """Spine-driven helical coolant voids with field-driven wall thickness.

    LEAP 71 ``functional void first``: the coolant path is defined as a swept
    void along a helical spine, then the channel wall grows around it as a
    shell whose thickness follows the local Joule heat.  This replaces the
    uniform gyroid with a self-organising cooling network that thickens where
    the winding runs hot.

    The wall is ``shell(void, t)`` where ``t = f(heat)`` -- the canonical
    LEAP 71 ``Offset`` generation operation with a field-driven distance.
    """

    cfg: MotorConfig3D
    n_channels: int = 4
    n_turns: float = 3.0
    channel_radius: float = 0.004
    min_wall: float = 0.0015
    max_wall: float = 0.0035
    jacket_radius: float = 0.0  # default: R_design + 0.004
    heat_field: object | None = None

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.fields_motor import joule_heat
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        cz = cfg.center[2]
        hz = cfg.stator_half_length - 0.001
        R = self.jacket_radius or (cfg.R_design + 0.004)

        X, Y, Z, r, theta = _angles_of(cfg)
        z_norm = (Z - cz) / (2.0 * hz)
        theta_helix = 2.0 * np.pi * self.n_turns * z_norm

        d_theta_min = np.full_like(r, 1e9, dtype=np.float32)
        for i in range(self.n_channels):
            offset_i = 2.0 * np.pi * i / self.n_channels
            dt = np.mod(theta - theta_helix - offset_i + np.pi, 2.0 * np.pi) - np.pi
            d_theta_min = np.minimum(d_theta_min, np.abs(dt))

        arc_dist = R * d_theta_min
        radial_dist = r - R
        dist_to_spine = np.sqrt(radial_dist ** 2 + arc_dist ** 2)
        void_sdf = (dist_to_spine - self.channel_radius).astype(np.float32)

        q = self.heat_field
        if q is None:
            q = joule_heat(cfg)
        qmax = max(float(np.abs(q.data).max()), 1e-9)
        qnorm = np.clip(np.abs(q.data) / qmax, 0.0, 1.0)
        wall_t = (self.min_wall + (self.max_wall - self.min_wall) * qnorm).astype(np.float32)

        wall_sdf = np.maximum(void_sdf - wall_t, -void_sdf)

        mf.add(SDFVoxelField(sdf=wall_sdf, spacing=cfg.spacing, origin=cfg.origin), "iron", priority=True)
        mf.add(SDFVoxelField(sdf=void_sdf, spacing=cfg.spacing, origin=cfg.origin), "air", priority=True)
        return mf


@dataclass
class RotorSleeve:
    """Thin retaining sleeve over surface magnets.

    Multi-functional (LEAP 71 ``one material, many jobs``):
    - Contains PM against centrifugal stress at speed
    - Provides a smooth outer surface for the air gap
    - Adds a small structural path for torque transfer
    """

    cfg: MotorConfig3D
    thickness: float = 0.002
    clearance: float = 0.0002

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        hz = cfg.rotor_half_length - 0.0001
        r_inner = cfg.R_rotor_outer + 0.0045 + self.clearance
        r_outer = r_inner + self.thickness
        sleeve = _annulus(cfg, r_inner, r_outer, hz)
        mf.add(sleeve, "iron", priority=True)
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
    """Central shaft, bearing races and end caps.

    Completes the mechanical assembly so the motor reads as a machine, not
    an isolated electromagnetic core.  The shaft extends beyond the stator
    for coupling; two bearing races sit at the ends; thin end caps close
    the housing and provide bearing seats.
    """

    cfg: MotorConfig3D
    shaft_extension: float = 0.012
    bearing_width: float = 0.004
    bearing_radius: float = 0.0
    end_cap_thickness: float = 0.003

    def build(self, mf: MaterialField) -> MaterialField:
        from organic_motor.construct.field import SDFVoxelField

        cfg = self.cfg
        cx, cy, cz = cfg.center
        hz = cfg.stator_half_length
        r_shaft = cfg.R_shaft
        r_bearing = self.bearing_radius or (r_shaft + 0.006)
        r_cap = cfg.R_design + 0.006

        X, Y, Z, r, _theta = _angles_of(cfg)
        shaft_half = hz + self.shaft_extension
        shaft_sdf = np.maximum(r - r_shaft, np.abs(Z - cz) - shaft_half)

        bearing_half = self.bearing_width * 0.5
        cap_half = self.end_cap_thickness * 0.5
        parts = [shaft_sdf]
        for sign in (+1, -1):
            z_b = cz + sign * (hz + bearing_half)
            axial_b = np.abs(Z - z_b) - bearing_half
            bearing_outer = np.maximum(r - r_bearing, axial_b)
            bearing_hole = np.maximum(r - r_shaft, axial_b)
            parts.append(np.maximum(bearing_outer, -bearing_hole))

            z_c = cz + sign * (hz + self.bearing_width + cap_half)
            axial_c = np.abs(Z - z_c) - cap_half
            cap_outer = np.maximum(r - r_cap, axial_c)
            cap_hole = np.maximum(r - (r_bearing + 0.001), axial_c)
            parts.append(np.maximum(cap_outer, -cap_hole))

        combined = parts[0]
        for p in parts[1:]:
            combined = np.minimum(combined, p)

        mf.add(SDFVoxelField(sdf=combined.astype(np.float32),
                             spacing=cfg.spacing, origin=cfg.origin),
               "iron", priority=True)
        return mf


@dataclass
class MotorHousing:
    """Outer cylindrical housing with mounting ears.

    Completes the mechanical envelope: a cylindrical shell outside the
    stator provides structural stiffness and mounting points.  Small
    mounting ears with bolt holes make it a bolted assembly, not a free-
    floating core.
    """

    cfg: MotorConfig3D
    housing_radius: float = 0.058
    housing_thickness: float = 0.003
    n_mounting_ears: int = 4
    ear_size: float = 0.008

    def build(self, mf: MaterialField) -> MaterialField:
        cfg = self.cfg
        cx, cy, cz = cfg.center
        hz = cfg.stator_half_length
        r_in = self.housing_radius
        r_out = r_in + self.housing_thickness
        shell = _annulus(cfg, r_in, r_out, hz + 0.002)
        mf.add(shell, "iron", priority=True)
        pitch = 2.0 * np.pi / self.n_mounting_ears
        for i in range(self.n_mounting_ears):
            angle = i * pitch
            ex = cx + r_out * np.cos(angle)
            ey = cy + r_out * np.sin(angle)
            ear = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                                cylinder_z((float(ex), float(ey)),
                                           self.ear_size * 0.5, hz + 0.002))
            bolt = from_implicit(cfg.shape, cfg.spacing, cfg.origin,
                                 cylinder_z((float(ex), float(ey)),
                                            self.ear_size * 0.18, hz + 0.002))
            ear = boolean_subtract(ear, bolt)
            mf.add(ear, "iron", priority=True)
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

    Every solid's local geometry reads a reduced-physics field pointwise:
    magnet thickness follows the air-gap |B| demand with a barrel axial
    profile, the stator yoke thickens where flux is high, the cooling
    channels are helical voids with walls that thicken where the winding
    runs hot, a rotor sleeve contains the magnets, and segmentation slits
    suppress eddy currents.  Real 3D windings have slot conductors and end
    turns.  Shaft, bearings and housing complete the mechanical assembly.
    """
    cfg = cfg or MotorConfig3D()
    return Motor(cfg, components=[
        ShaftAndBearings(cfg),
        RotorCore(cfg),
        FieldDrivenMagnets(cfg),
        RotorSleeve(cfg),
        FieldDrivenStatorYoke(cfg),
        StatorSegmentation(cfg),
        Winding3D(cfg),
        HelicalCoolingChannels(cfg),
        MotorHousing(cfg),
    ])
