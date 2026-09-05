"""Parametric honeycomb and helix geometry generators.

Honeycomb: true hexagonal cells via proper hex grid SDF.
Helix: 3D centerline swept with capsule (reuses verified polyline_capsule_sdf).
Y-manifold: explicit 3D inlet→fork→outlets with finite segments.

Each generator outputs SDF fields that compose with the motor geometry.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.field import SDFVoxelField, polyline_capsule_sdf


def _grid_arrays(cfg: MotorConfig3D):
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    X = ox + dx * np.arange(nx, dtype=np.float32)[:, None, None]
    Y = oy + dy * np.arange(ny, dtype=np.float32)[None, :, None]
    Z = oz + dz * np.arange(nz, dtype=np.float32)[None, None, :]
    return X, Y, Z


# ============================================================
# Honeycomb — true hexagonal cells
# ============================================================

@dataclass
class HoneycombGenerator:
    """True hexagonal honeycomb lattice with walls between cells.

    Parameters:
      cell_size: flat-to-flat distance of each hex cell (m)
      wall_thickness: strut wall thickness (m)
      r_inner, r_outer: radial bounds (cylindrical clipping)
      z_bottom, z_top: axial bounds
      density_gradient: optional array scaling wall_thickness by radius
    """

    cell_size: float = 0.006
    wall_thickness: float = 0.001
    r_inner: float = 0.044
    r_outer: float = 0.046
    z_bottom: float = -0.045
    z_top: float = 0.045
    density_gradient: np.ndarray | None = None

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        X, Y, Z = _grid_arrays(cfg)
        cx, cy, cz = cfg.center
        x = X - cx
        y = Y - cy
        R = np.sqrt(x**2 + y**2)

        D = self.cell_size  # flat-to-flat
        ir = D * 0.5        # inradius (center to edge midpoint)
        # Hex grid: rows offset by D*sqrt(3)/4, row spacing = D*3/4
        # Using the "offset coordinate" system for pointy-top hexagons
        h = D * np.sqrt(3) / 2   # cell width (flat-to-flat horizontal for pointy-top)
        v = D * 3 / 4            # vertical spacing between rows

        # Convert to axial-ish coordinates
        row = y / v
        row_int = np.round(row)
        x_offset = (np.mod(row_int, 2) - 0.5) * h if False else np.where(
            np.mod(row_int, 2) > 0, h * 0.5, 0.0
        )
        col = (x - x_offset) / h
        col_int = np.round(col)

        # Snap to nearest hex center
        cx_hex = col_int * h + x_offset
        cy_hex = row_int * v

        # Offset from hex center (NOT abs — need sign for edge normals)
        dx = x - cx_hex
        dy = y - cy_hex

        # Hexagon SDF: max of 3 edge-normal distances (6-fold symmetry)
        # Edge normals at 0°, 60°, 120° — abs gives 6 edges
        d1 = np.abs(dx) - ir
        d2 = np.abs(0.5 * dx + 0.8660254 * dy) - ir
        d3 = np.abs(-0.5 * dx + 0.8660254 * dy) - ir
        hex_sdf = np.maximum(np.maximum(d1, d2), d3)

        # Walls are where hex_sdf is in [-wall_t/2, +wall_t/2]
        wt = self.wall_thickness
        if self.density_gradient is not None:
            r_norm = np.clip(
                (R - self.r_inner) / max(self.r_outer - self.r_inner, 1e-6),
                0, 1,
            )
            r_idx = np.clip(
                (r_norm * (len(self.density_gradient) - 1)).astype(int),
                0, len(self.density_gradient) - 1,
            )
            wt = self.wall_thickness * self.density_gradient[r_idx]

        # Material = wall region: |hex_sdf| < wt/2
        # SDF negative inside material:
        wall_sdf = np.abs(hex_sdf) - wt * 0.5

        # Clip to radial/axial bounds
        sdf_r = np.maximum(self.r_inner - R, R - self.r_outer)
        sdf_z = np.maximum(self.z_bottom - Z, Z - self.z_top)
        sdf = np.maximum(wall_sdf, np.maximum(sdf_r, sdf_z))

        return SDFVoxelField(
            sdf.astype(np.float32),
            cfg.spacing, cfg.origin,
        )


# ============================================================
# Helix — 3D centerline with capsule sweep
# ============================================================

@dataclass
class HelicalChannelGenerator:
    """Helical channel via 3D centerline + capsule sweep.

    Uses the verified polyline_capsule_sdf for correct cross-section.

    Parameters:
      radius: channel centerline radius (m)
      pitch: axial advance per revolution (m)
      n_turns: number of turns
      channel_radius: cross-section radius (m)
      z_start: starting axial position (m)
      handedness: +1 = right-handed, -1 = left-handed
      n_segments: centerline discretization resolution
    """

    radius: float = 0.045
    pitch: float = 0.015
    n_turns: float = 3.0
    channel_radius: float = 0.0015
    z_start: float = -0.020
    handedness: int = 1
    n_segments: int = 128

    def _centerline(self, cfg: MotorConfig3D) -> np.ndarray:
        cx, cy, cz = cfg.center
        t_max = 2 * np.pi * self.n_turns
        t = np.linspace(0, t_max, self.n_segments + 1)
        pts = np.column_stack([
            cx + self.radius * np.cos(self.handedness * t),
            cy + self.radius * np.sin(self.handedness * t),
            cz + self.z_start + self.pitch * t / (2 * np.pi),
        ])
        return pts

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        pts = self._centerline(cfg)
        sdf_arr = polyline_capsule_sdf(
            cfg.shape, cfg.spacing, cfg.origin,
            pts, self.channel_radius,
        )
        return SDFVoxelField(sdf_arr, cfg.spacing, cfg.origin)

    def centerline_length(self) -> float:
        """Total centerline length (m)."""
        t_max = 2 * np.pi * self.n_turns
        # Arc length of helix: integral of sqrt(R^2 + (pitch/2pi)^2) dt
        ds_dt = np.sqrt(self.radius**2 + (self.pitch / (2 * np.pi))**2)
        return ds_dt * t_max

    def bounding_box(self, cfg: MotorConfig3D) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) of the centerline."""
        pts = self._centerline(cfg)
        return pts.min(axis=0), pts.max(axis=0)


# ============================================================
# Y-manifold — explicit 3D inlet→fork→outlets
# ============================================================

@dataclass
class BranchingManifold:
    """Y-shaped branching coolant distribution.

    One inlet pipe → fork point → two outlet pipes at ±branch_angle.

    Parameters:
      inlet_pos: (x, y, z) of inlet start
      fork_pos: (x, y, z) of fork point
      outlet_offset: (dx, dy, dz) from fork to each outlet end
      branch_angle: half-angle between branches (rad)
      channel_radius: cross-section radius (m)
      n_segments: discretization
    """

    inlet_pos: tuple = (0.0, 0.0, -0.040)
    fork_pos: tuple = (0.045, 0.0, -0.010)
    outlet_offset: tuple = (0.0, 0.020, 0.030)
    branch_angle: float = 0.3
    channel_radius: float = 0.002
    n_segments: int = 32

    def _segments(self) -> list[np.ndarray]:
        """Return list of centerline polylines: inlet, branch1, branch2."""
        inlet = np.array(self.inlet_pos)
        fork = np.array(self.fork_pos)
        outlet_end = fork + np.array(self.outlet_offset)

        # Inlet: straight from inlet to fork
        pts_in = np.linspace(inlet, fork, self.n_segments + 1)

        # Branch 1: from fork, rotated by +branch_angle around z
        direction = np.array(self.outlet_offset)
        ca, sa = np.cos(self.branch_angle), np.sin(self.branch_angle)
        dir1 = np.array([
            direction[0] * ca - direction[1] * sa,
            direction[0] * sa + direction[1] * ca,
            direction[2],
        ])
        end1 = fork + dir1
        pts_b1 = np.linspace(fork, end1, self.n_segments + 1)

        # Branch 2: rotated by -branch_angle
        dir2 = np.array([
            direction[0] * ca + direction[1] * sa,
            -direction[0] * sa + direction[1] * ca,
            direction[2],
        ])
        end2 = fork + dir2
        pts_b2 = np.linspace(fork, end2, self.n_segments + 1)

        return [pts_in, pts_b1, pts_b2]

    def build(self, cfg: MotorConfig3D) -> SDFVoxelField:
        segments = self._segments()
        sdf = np.full(cfg.shape, 1e6, dtype=np.float32)
        for pts in segments:
            seg_arr = polyline_capsule_sdf(
                cfg.shape, cfg.spacing, cfg.origin,
                pts, self.channel_radius,
            )
            sdf = np.minimum(sdf, seg_arr)
        return SDFVoxelField(sdf, cfg.spacing, cfg.origin)

    def inlet(self) -> np.ndarray:
        return np.array(self.inlet_pos)

    def outlets(self) -> list[np.ndarray]:
        fork = np.array(self.fork_pos)
        direction = np.array(self.outlet_offset)
        ca, sa = np.cos(self.branch_angle), np.sin(self.branch_angle)
        dir1 = np.array([
            direction[0] * ca - direction[1] * sa,
            direction[0] * sa + direction[1] * ca,
            direction[2],
        ])
        dir2 = np.array([
            direction[0] * ca + direction[1] * sa,
            -direction[0] * sa + direction[1] * ca,
            direction[2],
        ])
        return [fork + dir1, fork + dir2]


# ============================================================
# Convenience entry points — using actual config fields
# ============================================================

def honeycomb_support(cfg: MotorConfig3D, **kwargs) -> SDFVoxelField:
    """Honeycomb between stator winding outer and design outer radius."""
    defaults = dict(
        r_inner=cfg.R_winding_outer + 0.001,
        r_outer=cfg.R_design - 0.001,
        z_bottom=-cfg.stator_half_length,
        z_top=cfg.stator_half_length,
    )
    defaults.update(kwargs)
    return HoneycombGenerator(**defaults).build(cfg)


def helical_cooling(cfg: MotorConfig3D, **kwargs) -> SDFVoxelField:
    """Helical channel on stator outer surface."""
    defaults = dict(
        radius=cfg.R_winding_outer + 0.002,
        z_start=-cfg.stator_half_length + 0.003,
        n_turns=4.0,
        pitch=2 * cfg.stator_half_length / 4.0,
    )
    defaults.update(kwargs)
    return HelicalChannelGenerator(**defaults).build(cfg)
