"""Parameterized assembly features for a real motor package.

End caps with bearing seats, locating registers, bolt patterns,
mounting flange, and wire exit port — all as SDF geometry that can
be added to a MaterialField.

Design conventions:
  - End cap: disk spanning R_shaft → R_flange_outer, thickness t_cap
  - Bearing seat: precision bore at bearing OD, depth = bearing_width
  - Locating register (止口): annular lip mating with housing bore
  - Bolt circle: N bolts on diameter D_bolt, clearance holes
  - Wire exit: rectangular slot at bottom for cable routing
  - Mounting flange: optional foot with bolt holes for chassis mount
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.field import SDFVoxelField, polyline_capsule_sdf


@dataclass
class EndCapGenerator:
    """End cap with bearing bore, bolt holes, and locating register.

    Generates both front (z+) and rear (z-) caps in one call.
    The cap is iron (structural). Bolt holes and bearing bore are
    voids (subtracted from the cap solid).
    """
    cap_thickness: float = 0.004       # axial thickness [m]
    cap_gap: float = 0.001             # gap from stator end to cap inner face
    bearing_od: float = 0.028          # bearing outer diameter [m]
    bearing_width: float = 0.004       # bearing seat depth [m]
    register_depth: float = 0.002      # locating register lip depth [m]
    register_clearance: float = 0.0003 # register radial clearance [m]
    n_bolts: int = 6                   # bolts per cap
    bolt_circle_radius: float = 0.042  # bolt circle radius [m]
    bolt_hole_radius: float = 0.0022   # clearance hole radius [m]
    flange_outer_radius: float = 0.054 # cap outer radius (flange) [m]
    wire_exit_width: float = 0.012     # wire slot tangential width [m]
    wire_exit_height: float = 0.005    # wire slot radial height [m]
    wire_exit_angle: float = -np.pi/2  # wire exit angular position [rad]
    include_wire_exit: bool = True

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        if self.cap_thickness <= 0 or not 0 < self.bearing_width <= self.cap_thickness:
            raise ValueError("轴承座深度必须大于零且不超过端盖厚度")
        if not 2*cfg.R_shaft < self.bearing_od < 2*self.flange_outer_radius:
            raise ValueError("bearing_od 是直径，必须大于轴径且小于端盖外径")
        if self.n_bolts < 0 or self.bolt_hole_radius <= 0:
            raise ValueError("孔数不能为负，孔半径必须大于零")
        if self.bolt_circle_radius + self.bolt_hole_radius >= self.flange_outer_radius:
            raise ValueError("螺栓孔越出端盖边界")
        X, Y, Z = _grid_arrays(cfg)
        cx, cy, cz = cfg.center
        Z = Z - cz
        x = X - cx
        y = Y - cy
        R = np.sqrt(x**2 + y**2)

        z_half = cfg.stator_half_length
        t = self.cap_thickness
        gap = self.cap_gap

        sdf_caps = np.full(cfg.shape, 1e6, dtype=np.float32)

        for sign in (+1, -1):
            z_inner = sign * (z_half + gap)
            z_outer = sign * (z_half + gap + t)

            # --- Cap disk: R_shaft to flange_outer ---
            r_inner = float(cfg.R_shaft)
            r_outer = self.flange_outer_radius
            disk_radial = np.maximum(r_inner - R, R - r_outer)
            if sign > 0:
                disk_axial = np.maximum(z_inner - Z, Z - z_outer)
            else:
                disk_axial = np.maximum(z_outer - Z, Z - z_inner)
            cap_sdf = np.maximum(disk_radial, disk_axial)

            # --- Bearing seat bore: subtract cylinder at bearing OD ---
            bearing_z_center = sign * (z_half + gap + t - self.bearing_width * 0.5)
            bearing_z_inner = bearing_z_center - sign * self.bearing_width * 0.5
            bearing_z_outer = bearing_z_center + sign * self.bearing_width * 0.5
            if sign > 0:
                bearing_axial = np.maximum(bearing_z_inner - Z, Z - bearing_z_outer)
            else:
                bearing_axial = np.maximum(bearing_z_outer - Z, Z - bearing_z_inner)
            bearing_bore = np.maximum(R - self.bearing_od * 0.5, bearing_axial)
            # Subtract bearing bore from cap (make it void)
            cap_sdf = np.maximum(cap_sdf, -bearing_bore)

            # --- Locating register: annular lip on inner face ---
            reg_r_in = float(cfg.R_design) - self.register_clearance - 0.002
            reg_r_out = float(cfg.R_design) - self.register_clearance
            reg_radial = np.maximum(reg_r_in - R, R - reg_r_out)
            if sign > 0:
                reg_axial = np.maximum(z_inner - self.register_depth - Z, Z - z_inner)
            else:
                reg_axial = np.maximum(z_inner - Z, Z - (z_inner + self.register_depth))
            # Register is ADDITIONAL material (lip protrudes inward)
            reg_sdf = np.maximum(reg_radial, reg_axial)
            cap_sdf = np.minimum(cap_sdf, reg_sdf)

            # --- Bolt holes: subtract N cylinders ---
            for i in range(self.n_bolts):
                angle = 2.0 * np.pi * i / self.n_bolts + (np.pi / self.n_bolts)
                bx = self.bolt_circle_radius * np.cos(angle)
                by = self.bolt_circle_radius * np.sin(angle)
                # Distance from bolt center in XY
                d_xy = np.sqrt((x - bx)**2 + (y - by)**2)
                bolt_radial = d_xy - self.bolt_hole_radius
                if sign > 0:
                    bolt_axial = np.maximum(z_inner - Z, Z - z_outer)
                else:
                    bolt_axial = np.maximum(z_outer - Z, Z - z_inner)
                bolt_sdf = np.maximum(bolt_radial, bolt_axial)
                cap_sdf = np.maximum(cap_sdf, -bolt_sdf)

            # --- Wire exit slot (only on rear cap, sign=-1) ---
            if self.include_wire_exit and sign < 0:
                wa = self.wire_exit_angle
                # Angular distance from wire exit angle
                d_theta = np.abs(np.arctan2(
                    y * np.cos(wa) - x * np.sin(wa),
                    x * np.cos(wa) + y * np.sin(wa)))
                half_w = self.wire_exit_width / (2 * self.bolt_circle_radius)
                # Slot: angular wedge × radial band × through cap thickness
                slot_angular = d_theta - half_w
                r_slot_in = float(cfg.R_winding_outer)
                r_slot_out = r_slot_in + self.wire_exit_height
                slot_radial = np.maximum(r_slot_in - R, R - r_slot_out)
                slot_axial = np.maximum(z_outer - Z, Z - z_inner)
                slot_sdf = np.maximum(np.maximum(slot_angular * self.bolt_circle_radius,
                                                  slot_radial), slot_axial)
                cap_sdf = np.maximum(cap_sdf, -slot_sdf)

            sdf_caps = np.minimum(sdf_caps, cap_sdf)

        return SDFVoxelField(
            sdf_caps.astype(np.float32),
            cfg.spacing, cfg.origin,
        )


@dataclass
class MountingFlangeGenerator:
    """Foot-mounting flange with bolt holes for chassis attachment.

    A rectangular flange at the bottom of the housing with
    bolt holes for mounting the motor to a chassis or baseplate.
    """
    flange_width: float = 0.080      # tangential width [m]
    flange_depth: float = 0.012      # radial protrusion [m]
    flange_thickness: float = 0.005  # axial thickness [m]
    n_mount_holes: int = 4
    mount_hole_radius: float = 0.0025
    mount_offset_z: float = 0.0      # axial center of flange (0 = mid-motor)

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        X, Y, Z = _grid_arrays(cfg)
        cx, cy, cz = cfg.center
        x = X - cx
        y = Y - cy
        R = np.sqrt(x**2 + y**2)

        r_base = self.flange_outer_radius if hasattr(self, 'flange_outer_radius') else float(cfg.R_design) + 0.002
        r_tip = r_base + self.flange_depth

        # Flange is at the bottom (theta = -pi/2, i.e. y < 0)
        # Rectangular: |x| < width/2, R between r_base and r_tip, |z - offset| < t/2
        z_c = cz + self.mount_offset_z
        flange_radial = np.maximum(r_base - R, R - r_tip)
        flange_width_sdf = np.abs(x) - self.flange_width * 0.5
        flange_axial = np.abs(Z - z_c) - self.flange_thickness * 0.5
        # Only where y < 0 (bottom)
        flange_y = y  # negative at bottom
        flange_sdf = np.maximum(np.maximum(flange_radial, flange_width_sdf),
                                np.maximum(flange_axial, flange_y))

        # Subtract bolt holes
        hole_spacing = self.flange_width * 0.7
        for i in range(self.n_mount_holes):
            hx = -hole_spacing/2 + i * (hole_spacing / (self.n_mount_holes - 1))
            hy = -(r_base + r_tip) / 2
            d_xy = np.sqrt((x - hx)**2 + (y - hy)**2)
            hole_sdf = np.maximum(d_xy - self.mount_hole_radius, flange_axial)
            flange_sdf = np.maximum(flange_sdf, -hole_sdf)

        return SDFVoxelField(
            flange_sdf.astype(np.float32),
            cfg.spacing, cfg.origin,
        )


def _grid_arrays(cfg: MotorConfig3D):
    nx, ny, nz = cfg.shape
    x0, y0, z0 = cfg.origin
    dx, dy, dz = cfg.spacing
    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy
    z = z0 + np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z
