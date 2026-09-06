"""Tests for planetary gearbox: geometry, ratio, assembly, dynamics."""

import pytest
import math
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.gearbox import GearSpec, PlanetaryStage


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig3D(shape=(56, 56, 36))


class TestGearSpec:
    def test_pitch_radius(self):
        g = GearSpec(module=0.001, n_teeth=20)
        assert g.pitch_radius == pytest.approx(0.010)

    def test_outer_radius(self):
        g = GearSpec(module=0.001, n_teeth=20)
        assert g.outer_radius == pytest.approx(0.011)

    def test_root_radius(self):
        g = GearSpec(module=0.001, n_teeth=20)
        assert g.root_radius == pytest.approx(0.00875)


class TestPlanetaryStage:
    def test_ratio(self):
        stage = PlanetaryStage(
            sun=GearSpec(module=0.001, n_teeth=18),
            planet=GearSpec(module=0.001, n_teeth=24),
        )
        # ratio = 1 + Z_ring/Z_sun = 1 + 66/18 = 4.667
        assert stage.ratio == pytest.approx(1 + 66/18, rel=1e-4)

    def test_assembly_condition_auto_correct(self):
        """ring_teeth auto-corrected to sun + 2*planet."""
        stage = PlanetaryStage(
            sun=GearSpec(n_teeth=18),
            planet=GearSpec(n_teeth=24),
            ring_teeth=999,  # wrong
        )
        assert stage.ring_teeth == 18 + 2 * 24  # 66

    def test_planet_spacing(self):
        """(Z_sun + Z_ring) / n_planets must be integer."""
        stage = PlanetaryStage(
            sun=GearSpec(n_teeth=18),
            planet=GearSpec(n_teeth=24),
            n_planets=3,
        )
        check = stage.assembly_check
        assert check["planet_spacing_ok"], (
            f"planet spacing failed: {(18 + 66) / 3}"
        )

    def test_center_distance(self):
        stage = PlanetaryStage(
            sun=GearSpec(module=0.001, n_teeth=18),
            planet=GearSpec(module=0.001, n_teeth=24),
        )
        cd = stage.assembly_check["center_distance_mm"]
        assert cd == pytest.approx((18 + 24) * 1.0 / 2, rel=1e-3)

    def test_fits_in_housing(self):
        stage = PlanetaryStage(
            sun=GearSpec(module=0.001, n_teeth=18),
            planet=GearSpec(module=0.001, n_teeth=24),
            housing_radius=0.040,
        )
        check = stage.assembly_check
        assert check["fits_in_housing"], (
            f"ring OD {check['ring_od_mm']:.1f}mm > housing {check['housing_radius_mm']:.1f}mm"
        )

    def test_sun_gear_sdf(self, cfg):
        stage = PlanetaryStage()
        sdf = stage.build_sun_gear_sdf(cfg)
        assert sdf.shape == cfg.shape
        assert int((sdf < 0).sum()) > 0, "sun gear has zero solid voxels"

    def test_planet_gear_sdf(self, cfg):
        stage = PlanetaryStage()
        sdf = stage.build_planet_gear_sdf(cfg, 0)
        assert int((sdf < 0).sum()) > 0, "planet gear has zero solid voxels"

    def test_ring_gear_sdf(self, cfg):
        stage = PlanetaryStage()
        sdf = stage.build_ring_gear_sdf(cfg)
        assert int((sdf < 0).sum()) > 0, "ring gear has zero solid voxels"

    def test_carrier_sdf(self, cfg):
        stage = PlanetaryStage()
        sdf = stage.build_carrier_sdf(cfg)
        assert int((sdf < 0).sum()) > 0, "carrier has zero solid voxels"

    def test_build_all(self, cfg):
        stage = PlanetaryStage()
        parts = stage.build(cfg)
        assert "sun_gear" in parts
        assert "planet_gears" in parts
        assert len(parts["planet_gears"]) == stage.n_planets
        assert "ring_gear" in parts
        assert "carrier" in parts

    def test_dynamics_model(self):
        stage = PlanetaryStage()
        dyn = stage.dynamics_model(J_motor_shaft=2e-4)
        assert dyn["ratio"] > 1
        assert dyn["J_reflected_to_sun"] > 0
        assert dyn["J_total_at_motor"] > 2e-4  # motor + reflected
        assert dyn["backlash_at_sun_rad"] > 0
        assert dyn["torsional_stiffness_sun"] > dyn["torsional_stiffness_carrier"]

    def test_sun_gear_has_teeth(self, cfg):
        """Sun gear SDF has more boundary voxels than a plain cylinder."""
        stage = PlanetaryStage()
        sdf = stage.build_sun_gear_sdf(cfg)
        cx, cy, cz = cfg.center
        dx, dy, dz = cfg.spacing
        x0, y0, z0 = cfg.origin
        z_c = cz + stage.z_position
        iz = int(round((z_c - z0) / dz))
        iz = max(0, min(cfg.shape[2] - 1, iz))
        # Count surface voxels (|sdf| < dx) at pitch radius
        r_pitch = stage.sun.pitch_radius
        nx, ny = cfg.shape[0], cfg.shape[1]
        n_surface = 0
        for i in range(nx):
            for j in range(ny):
                px = x0 + i * dx
                py = y0 + j * dy
                r = math.sqrt((px - cx)**2 + (py - cy)**2)
                if abs(r - r_pitch) < dx:
                    if abs(sdf[i, j, iz]) < dx:
                        n_surface += 1
        assert n_surface > 5, f"too few tooth surface voxels: {n_surface}"
