"""Parametric honeycomb and helix geometry generators.

These are explicit morphology-preference generators, not physics-derived.
They provide:
  - Honeycomb/strut lattice for support structures (varying density)
  - Helical cooling channels (varying pitch, turns, cross-section)
  - Branching manifold (for coolant distribution)

Each generator outputs SDF fields that compose with the existing
motor geometry via boolean operations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.field import SDFVoxelField


@dataclass
class HoneycombGenerator:
    """Conformal honeycomb/cellular structure for support regions.

    Parameters:
      cell_size: hexagon flat-to-flat distance (m)
      wall_thickness: strut wall thickness (m)
      r_inner, r_outer: radial bounds
      z_bottom, z_top: axial bounds
      density_gradient: optional (r,) array scaling wall thickness
    """

    cell_size: float = 0.004
    wall_thickness: float = 0.0008
    r_inner: float = 0.044
    r_outer: float = 0.046
    z_bottom: float = -0.045
    z_top: float = 0.045
    density_gradient: np.ndarray | None = None

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        nx, ny, nz = cfg.shape
        dx, dy, dz = cfg.spacing
        ox, oy, oz = cfg.origin

        X = ox + dx * np.arange(nx, dtype=np.float32)[:, None, None]
        Y = oy + dy * np.arange(ny, dtype=np.float32)[None, :, None]
        Z = oz + dz * np.arange(nz, dtype=np.float32)[None, None, :]

        R = np.sqrt((X - cfg.center[0])**2 + (Y - cfg.center[1])**2)

        # Hexagonal strut lattice via three sets of parallel plates
        # Offset each set by half period for proper hex pattern
        theta_cart = np.arctan2(Y - cfg.center[1], X - cfg.center[0])
        # Use Cartesian coordinates for the lattice, not polar
        x = X - cfg.center[0]
        y = Y - cfg.center[1]

        a = self.cell_size
        x = X - cfg.center[0]
        y = Y - cfg.center[1]

        # Three sets of parallel plates forming hexagonal struts
        # Distance to nearest plate in each set (negative = inside wall)
        wt = self.wall_thickness * 0.5
        p1 = x
        p2 = 0.5 * x + 0.8660254 * y
        p3 = 0.5 * x - 0.8660254 * y
        d1 = np.abs(np.mod(p1 + a*0.5, a) - a*0.5) - wt
        d2 = np.abs(np.mod(p2 + a*0.5, a) - a*0.5) - wt
        d3 = np.abs(np.mod(p3 + a*0.5, a) - a*0.5) - wt
        # Union of three wall sets (minimum SDF = closest wall)
        d_hex = np.minimum(np.minimum(d1, d2), d3)

        # Clip to radial/axial bounds
        sdf_r = np.maximum(self.r_inner - R, R - self.r_outer)
        sdf_z = np.maximum(self.z_bottom - Z, Z - self.z_top)
        sdf = np.maximum(d_hex, np.maximum(sdf_r, sdf_z))

        return SDFVoxelField(
            sdf.astype(np.float32),
            cfg.spacing, cfg.origin,
        )


@dataclass
class HelicalChannelGenerator:
    """Helical cooling channel for cylindrical surfaces.

    Parameters:
      radius: channel centerline radius (m)
      pitch: axial advance per revolution (m)
      n_turns: number of turns
      channel_radius: cross-section radius (m)
      z_start: starting axial position (m)
      handedness: +1 = right-handed, -1 = left-handed
    """

    radius: float = 0.045
    pitch: float = 0.015
    n_turns: float = 3.0
    channel_radius: float = 0.0015
    z_start: float = -0.020
    handedness: int = 1

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        nx, ny, nz = cfg.shape
        dx, dy, dz = cfg.spacing
        ox, oy, oz = cfg.origin

        X = ox + dx * np.arange(nx, dtype=np.float32)[:, None, None]
        Y = oy + dy * np.arange(ny, dtype=np.float32)[None, :, None]
        Z = oz + dz * np.arange(nz, dtype=np.float32)[None, None, :]

        cx, cy, cz = cfg.center
        dx_ = X - cx
        dy_ = Y - cy
        dz_ = Z - cz

        R = np.sqrt(dx_**2 + dy_**2)
        theta = np.arctan2(dy_, dx_)

        # Axial position along the helix
        z_total = self.n_turns * self.pitch
        z_norm = np.clip((dz_ - self.z_start) / max(z_total, 1e-6), 0, 1)

        # Expected theta at this z
        theta_helix = self.handedness * 2 * np.pi * (dz_ - self.z_start) / max(self.pitch, 1e-6)

        # Angular distance to helix centerline
        d_theta = np.mod(theta - theta_helix + np.pi, 2 * np.pi) - np.pi
        # Distance from helix centerline
        d_helix = np.sqrt((R - self.radius)**2 + (d_theta * self.radius)**2)

        # Channel SDF (negative inside channel)
        sdf = d_helix - self.channel_radius

        # Clip to active z range
        sdf = np.maximum(sdf, np.maximum(self.z_start - dz_, dz_ - self.z_start - z_total))

        return SDFVoxelField(
            sdf.astype(np.float32),
            cfg.spacing, cfg.origin,
        )


@dataclass
class BranchingManifold:
    """Branching coolant distribution structure.

    A simple Y-junction manifold that splits one inlet into two outlets.
    Parameters:
      inlet_r, inlet_z: inlet position
      branch_angle: angle between branches (rad)
      channel_radius: cross-section radius (m)
      length: total path length (m)
    """

    inlet_r: float = 0.044
    inlet_z: float = -0.040
    branch_angle: float = 0.3
    channel_radius: float = 0.002
    length: float = 0.060

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        nx, ny, nz = cfg.shape
        dx, dy, dz = cfg.spacing
        ox, oy, oz = cfg.origin

        X = ox + dx * np.arange(nx, dtype=np.float32)[:, None, None]
        Y = oy + dy * np.arange(ny, dtype=np.float32)[None, :, None]
        Z = oz + dz * np.arange(nz, dtype=np.float32)[None, None, :]

        cx, cy, cz = cfg.center
        R = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # Inlet: straight channel along z
        d_inlet = np.sqrt((R - self.inlet_r)**2 + np.maximum(0, self.inlet_z - (Z - cz))**2)
        sdf = d_inlet - self.channel_radius

        # Two branches at +/- branch_angle from vertical
        for sign in [+1, -1]:
            # Branch goes from (inlet_r, inlet_z) outward at angle
            t_max = self.length / 2
            # Parametric: r(t) = inlet_r + t*sin(angle), z(t) = inlet_z + t*cos(angle)
            sin_a = np.sin(sign * self.branch_angle)
            cos_a = np.cos(sign * self.branch_angle)
            # Distance from point to line segment
            dr = R - (cz + 0) - self.inlet_r  # simplified
            d_branch = np.sqrt(
                (R - self.inlet_r - sin_a * np.clip(
                    ((Z - cz) - self.inlet_z) / max(cos_a, 1e-6), 0, t_max
                ))**2 + 0  # approximate
            )
            sdf = np.minimum(sdf, d_branch - self.channel_radius)

        return SDFVoxelField(
            sdf.astype(np.float32),
            cfg.spacing, cfg.origin,
        )


def honeycomb_support(cfg: MotorConfig3D, **kwargs) -> SDFVoxelField:
    """Convenience: honeycomb between stator outer and housing inner."""
    defaults = dict(
        r_inner=cfg.R_stator_outer + 0.001,
        r_outer=cfg.R_housing_inner - 0.001,
        z_bottom=-cfg.stator_half_length,
        z_top=cfg.stator_half_length,
    )
    defaults.update(kwargs)
    return HoneycombGenerator(**defaults).build(cfg)


def helical_cooling(cfg: MotorConfig3D, **kwargs) -> SDFVoxelField:
    """Convenience: helical channel on stator outer surface."""
    defaults = dict(
        radius=cfg.R_stator_outer + 0.002,
        z_start=-cfg.stator_half_length + 0.003,
        n_turns=4.0,
        pitch=2 * cfg.stator_half_length / 4.0,
    )
    defaults.update(kwargs)
    return HelicalChannelGenerator(**defaults).build(cfg)
