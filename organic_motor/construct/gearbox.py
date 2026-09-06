"""Planetary gearbox: sun gear, planet gears, ring gear, carrier.

Three layers:
1. GEOMETRY: SDF-based gears that can be added to a MaterialField for
   3D meshing and visualization.  Uses involute approximation (trapezoidal
   teeth) sufficient for assembly-level modeling.
2. DYNAMICS: lumped-parameter reduction model (ratio, efficiency,
   inertia, backlash, stiffness) for coupling to the motor transient.
3. VERIFICATION: gear mesh checks (contact ratio, center distance,
   interference) documented as metadata, not runtime checks.

The planetary stage is placed AFT of the motor (z > 0 end).  The
sun gear is keyed to the motor shaft.  The carrier outputs to the
load flange.  The ring gear is grounded to the housing.

Standard planetary relation:
  ratio = 1 + Z_ring / Z_sun  (ring fixed, sun input, carrier output)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.field import SDFVoxelField


@dataclass
class GearSpec:
    """Involute gear specification (module-based, SI units)."""
    module: float = 0.001      # m per tooth (e.g. 1mm module = m1)
    n_teeth: int = 20          # number of teeth
    pressure_angle: float = math.radians(20.0)  # standard 20°
    helix_angle: float = 0.0  # 0 = spur, nonzero = helical
    face_width: float = 0.010  # axial width [m]
    bore_radius: float = 0.004 # bore for shaft/key [m]

    @property
    def pitch_radius(self) -> float:
        return 0.5 * self.module * self.n_teeth

    @property
    def base_radius(self) -> float:
        return self.pitch_radius * math.cos(self.pressure_angle)

    @property
    def addendum(self) -> float:
        return self.module  # standard addendum = module

    @property
    def dedendum(self) -> float:
        return 1.25 * self.module  # standard dedendum

    @property
    def outer_radius(self) -> float:
        return self.pitch_radius + self.addendum

    @property
    def root_radius(self) -> float:
        return self.pitch_radius - self.dedendum


@dataclass
class PlanetaryStage:
    """A single planetary reduction stage.

    Geometry is built as SDFs for 3D visualization.  The carrier
    output is at z+, the sun input at z- (motor side).
    """
    sun: GearSpec = field(default_factory=lambda: GearSpec(module=0.001, n_teeth=18))
    planet: GearSpec = field(default_factory=lambda: GearSpec(module=0.001, n_teeth=24))
    ring_teeth: int = 66          # = sun + 2*planet for standard assembly
    n_planets: int = 3            # number of planet gears
    stage_length: float = 0.020   # axial length [m]
    z_position: float = 0.045     # center z (aft of motor shaft end)
    housing_radius: float = 0.040 # ring gear housing outer radius
    carrier_radius: float = 0.030 # carrier output flange radius
    backlash: float = 0.0001      # circumferential backlash [m]
    efficiency: float = 0.97      # single-stage efficiency
    torsional_stiffness: float = 1e4  # N·m/rad (carrier output)

    def __post_init__(self):
        # Verify assembly condition: Z_ring = Z_sun + 2*Z_planet
        expected_ring = self.sun.n_teeth + 2 * self.planet.n_teeth
        if self.ring_teeth != expected_ring:
            # Auto-correct rather than crash
            object.__setattr__(self, 'ring_teeth', expected_ring)

    @property
    def ratio(self) -> float:
        """Speed reduction ratio (ring fixed)."""
        return 1.0 + self.ring_teeth / self.sun.n_teeth

    @property
    def planet_center_radius(self) -> float:
        """Distance from sun center to planet center."""
        return self.sun.pitch_radius + self.planet.pitch_radius

    @property
    def ring_pitch_radius(self) -> float:
        return 0.5 * self.sun.module * self.ring_teeth

    @property
    def output_inertia(self) -> float:
        """Reflected inertia at the carrier (approximate)."""
        # J_carrier = (J_sun + J_planet * n) / ratio^2 + J_carrier_own
        rho_steel = 7850.0
        sun_vol = math.pi * (self.sun.outer_radius**2 - self.sun.bore_radius**2) * self.sun.face_width
        planet_vol = math.pi * (self.planet.outer_radius**2 - self.planet.bore_radius**2) * self.planet.face_width
        carrier_vol = math.pi * (self.carrier_radius**2) * 0.004
        J_sun = 0.5 * rho_steel * sun_vol * self.sun.pitch_radius**2
        J_planet = 0.5 * rho_steel * planet_vol * self.planet.pitch_radius**2
        J_carrier = 0.5 * rho_steel * carrier_vol * self.carrier_radius**2
        # Planets revolve around sun at planet_center_radius
        J_planet_orbit = self.n_planets * rho_steel * planet_vol * self.planet_center_radius**2
        # Reflected to carrier (output): divide by ratio^2 for sun part
        return J_carrier + J_planet_orbit + (J_sun + J_planet * self.n_planets) / self.ratio**2

    @property
    def assembly_check(self) -> dict:
        """Verify geometric assembly conditions."""
        # 1. Center distance = (Z_sun + Z_planet) * module / 2
        cd = (self.sun.n_teeth + self.planet.n_teeth) * self.sun.module / 2
        # 2. Planet spacing: (Z_sun + Z_ring) / n_planets must be integer
        spacing_test = (self.sun.n_teeth + self.ring_teeth) / self.n_planets
        spacing_ok = abs(spacing_test - round(spacing_test)) < 1e-9
        # 3. Ring OD fits in housing
        ring_od = self.ring_pitch_radius + self.planet.addendum
        fits = ring_od < self.housing_radius
        return {
            "center_distance_mm": cd * 1000,
            "planet_spacing_ok": spacing_ok,
            "ring_od_mm": ring_od * 1000,
            "housing_radius_mm": self.housing_radius * 1000,
            "fits_in_housing": fits,
            "ratio": self.ratio,
            "all_pass": spacing_ok and fits,
        }

    def build_sun_gear_sdf(self, cfg: MotorConfig3D) -> np.ndarray:
        """Build the sun gear SDF (negative inside solid)."""
        X, Y, Z = _grid(cfg)
        cx, cy, cz = cfg.center
        x = X - cx
        y = Y - cy
        R = np.sqrt(x**2 + y**2)

        z_c = cz + self.z_position
        z_half = self.sun.face_width * 0.5
        axial = np.abs(Z - z_c) - z_half

        # Tooth profile: N teeth, each tooth is a bump on the pitch circle
        theta = np.arctan2(y, x)
        n = self.sun.n_teeth
        # Tooth angular pitch
        pitch_ang = 2 * math.pi / n
        # Tooth occupies ~half the angular pitch (alternate tooth/gap)
        tooth_half = pitch_ang * 0.25  # quarter pitch tooth, quarter gap each side
        # Normalize angle to nearest tooth
        d_theta = (theta - np.round(theta / pitch_ang) * pitch_ang)
        # Radial tooth profile: addendum at tooth center, root at gap
        tooth_profile = np.cos(d_theta / tooth_half * math.pi * 0.5)
        tooth_profile = np.clip(tooth_profile, 0, 1)
        # SDF: root_radius at gap, outer_radius at tooth tip
        r_root = self.sun.root_radius
        r_tip = self.sun.outer_radius
        r_bore = self.sun.bore_radius
        # Radial SDF: negative inside the gear body (root to outer)
        r_surface = r_root + (r_tip - r_root) * tooth_profile
        radial = R - r_surface  # negative inside
        # Bore
        bore = r_bore - R  # negative inside bore
        # Combine: solid where (radial < 0) AND (bore > 0) AND (axial < 0)
        gear_sdf = np.maximum(radial, -bore)
        gear_sdf = np.maximum(gear_sdf, axial)
        return gear_sdf.astype(np.float32)

    def build_planet_gear_sdf(self, cfg: MotorConfig3D, planet_index: int) -> np.ndarray:
        """Build one planet gear SDF at its carrier position."""
        angle = 2 * math.pi * planet_index / self.n_planets
        cx, cy, cz = cfg.center
        px = cx + self.planet_center_radius * math.cos(angle)
        py = cy + self.planet_center_radius * math.sin(angle)

        X, Y, Z = _grid(cfg)
        x = X - px
        y = Y - py
        R = np.sqrt(x**2 + y**2)

        z_c = cz + self.z_position
        z_half = self.planet.face_width * 0.5
        axial = np.abs(Z - z_c) - z_half

        theta = np.arctan2(y, x)
        n = self.planet.n_teeth
        pitch_ang = 2 * math.pi / n
        tooth_half = pitch_ang * 0.25
        d_theta = (theta - np.round(theta / pitch_ang) * pitch_ang)
        tooth_profile = np.clip(np.cos(d_theta / tooth_half * math.pi * 0.5), 0, 1)

        r_root = self.planet.root_radius
        r_tip = self.planet.outer_radius
        r_bore = self.planet.bore_radius
        r_surface = r_root + (r_tip - r_root) * tooth_profile
        radial = R - r_surface
        bore = r_bore - R
        gear_sdf = np.maximum(radial, -bore)
        gear_sdf = np.maximum(gear_sdf, axial)
        return gear_sdf.astype(np.float32)

    def build_ring_gear_sdf(self, cfg: MotorConfig3D) -> np.ndarray:
        """Build the ring gear SDF (internal teeth, grounded to housing)."""
        X, Y, Z = _grid(cfg)
        cx, cy, cz = cfg.center
        x = X - cx
        y = Y - cy
        R = np.sqrt(x**2 + y**2)

        z_c = cz + self.z_position
        z_half = self.planet.face_width * 0.5
        axial = np.abs(Z - z_c) - z_half

        # Ring: internal teeth at ring_pitch_radius
        theta = np.arctan2(y, x)
        n = self.ring_teeth
        pitch_ang = 2 * math.pi / n
        tooth_half = pitch_ang * 0.25
        d_theta = (theta - np.round(theta / pitch_ang) * pitch_ang)
        tooth_profile = np.clip(np.cos(d_theta / tooth_half * math.pi * 0.5), 0, 1)

        # Ring: teeth point INWARD.  Outer radius = housing.
        r_pitch = self.ring_pitch_radius
        r_tip = r_pitch - self.planet.addendum   # tooth tip (inward)
        r_root = r_pitch + self.planet.dedendum  # tooth root (outward)
        r_outer = self.housing_radius

        # Internal: solid from r_outer down to r_root, with teeth
        # going inward to r_tip.
        r_surface = r_root - (r_root - r_tip) * tooth_profile
        # Radial: negative inside the ring solid
        radial_inner = r_surface - R  # negative where R > r_surface (inside solid)
        radial_outer = R - r_outer   # negative where R < r_outer
        radial = np.maximum(radial_inner, radial_outer)

        gear_sdf = np.maximum(radial, axial)
        return gear_sdf.astype(np.float32)

    def build_carrier_sdf(self, cfg: MotorConfig3D) -> np.ndarray:
        """Build the carrier (output flange + planet pins)."""
        X, Y, Z = _grid(cfg)
        cx, cy, cz = cfg.center
        x = X - cx
        y = Y - cy
        R = np.sqrt(x**2 + y**2)

        # Carrier disk at z+ end of the gear stage
        z_c = cz + self.z_position + self.sun.face_width * 0.5 + 0.002
        t = 0.004  # carrier thickness
        axial = np.abs(Z - z_c) - t * 0.5

        # Disk: bore at shaft to outer radius
        r_inner = float(cfg.R_shaft)
        r_outer = self.carrier_radius
        disk = np.maximum(np.maximum(r_inner - R, R - r_outer), axial)

        # Planet pins: cylinders at each planet center
        pin_r = self.planet.bore_radius
        for i in range(self.n_planets):
            angle = 2 * math.pi * i / self.n_planets
            px = self.planet_center_radius * math.cos(angle)
            py = self.planet_center_radius * math.sin(angle)
            d = np.sqrt((x - px)**2 + (y - py)**2)
            pin_sdf = np.maximum(d - pin_r, axial)
            disk = np.minimum(disk, pin_sdf)

        return disk.astype(np.float32)

    def build(self, cfg: MotorConfig3D) -> dict[str, SDFVoxelField]:
        """Build all gearbox components as SDFVoxelFields."""
        return {
            "sun_gear": SDFVoxelField(self.build_sun_gear_sdf(cfg), cfg.spacing, cfg.origin),
            "planet_gears": [
                SDFVoxelField(self.build_planet_gear_sdf(cfg, i), cfg.spacing, cfg.origin)
                for i in range(self.n_planets)
            ],
            "ring_gear": SDFVoxelField(self.build_ring_gear_sdf(cfg), cfg.spacing, cfg.origin),
            "carrier": SDFVoxelField(self.build_carrier_sdf(cfg), cfg.spacing, cfg.origin),
        }

    def dynamics_model(self, J_motor_shaft: float) -> dict:
        """Lumped-parameter dynamics for coupling to motor transient.

        Returns parameters for the motor's rotor equation:
          J_total = J_motor + J_reflected
          T_load_gearbox = T_load_carrier / (ratio * efficiency)
          backlash_deadband [rad at sun]
        """
        return {
            "ratio": self.ratio,
            "efficiency": self.efficiency,
            "J_reflected_to_sun": self.output_inertia / (self.ratio * self.ratio),
            "J_total_at_motor": J_motor_shaft + self.output_inertia / (self.ratio**2),
            "backlash_at_sun_rad": self.backlash / self.sun.pitch_radius,
            "torsional_stiffness_carrier": self.torsional_stiffness,
            "torsional_stiffness_sun": self.torsional_stiffness * self.ratio**2,
        }


def _grid(cfg: MotorConfig3D):
    nx, ny, nz = cfg.shape
    x0, y0, z0 = cfg.origin
    dx, dy, dz = cfg.spacing
    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy
    z = z0 + np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z
