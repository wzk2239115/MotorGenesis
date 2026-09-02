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
        cx, cy = cfg.center[0], cfg.center[1]

        # Sample the physics field at each pole's angular centre (z-averaged)
        # to get a per-pole thickness.  This is the field->geometry step for
        # the magnets: poles in higher-flux regions grow thicker.  The pole
        # body itself reuses the proven annular_sector primitive.
        field = self.thickness_field
        if field is None:
            field = airgap_B(cfg)
        fmag = np.abs(field.data)
        fmax = max(float(fmag.max()), 1e-9)
        # Thickness must stay >= ~1 voxel so the magnet registers at this grid
        # resolution; it may poke slightly past R_stator_inner (the stator,
        # added later with priority, carves the overlap back -- same as the
        # baseline SurfaceMagnets).
        sdf = np.full(cfg.shape, 1e9, dtype=np.float32)
        X, Y, Z, r, theta = _angles_of(cfg)
        for p in range(poles):
            centre = p * pitch
            ang = np.mod(theta - centre + np.pi, 2 * np.pi) - np.pi
            in_pole = np.abs(ang) < pitch * self.pole_fraction * 0.5
            b_avg = float(fmag[in_pole].mean()) / fmax if in_pole.any() else 0.0
            thickness = self.min_thickness + (self.max_thickness - self.min_thickness) * b_avg
            r1 = r0 + thickness
            a0 = centre - pitch * self.pole_fraction * 0.5
            a1 = centre + pitch * self.pole_fraction * 0.5
            pole = from_implicit(
                cfg.shape, cfg.spacing, cfg.origin,
                annular_sector((cx, cy), r0, r1, a0, a1, hz),
            )
            sdf = np.minimum(sdf, pole.sdf)
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
    magnet thickness follows the air-gap |B| demand, the stator yoke
    thickens where flux is high, and the cooling gyroid wall thickens where
    the winding runs hot.  This is the LEAP 71 ``radius = f(position,
    physics)`` pattern applied across all four material ``tissues``.
    """
    cfg = cfg or MotorConfig3D()
    return Motor(cfg, components=[
        RotorCore(cfg),
        FieldDrivenMagnets(cfg),
        FieldDrivenStatorYoke(cfg),
        DistributedWinding(cfg),
        FieldDrivenCoolingJacket(cfg),
    ])
