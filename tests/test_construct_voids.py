"""Verify functional voids: housing windows exist, PM survives, air gap is clean."""

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import field_driven_motor


def _build_field_driven():
    cfg = MotorConfig3D(
        shape=(56, 56, 36),
        excitation_mode="terminal",
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=3,
        maxwell_maxiter=10,
        thermal_maxiter=20,
        electric_maxiter=20,
    )
    motor = field_driven_motor(cfg)
    mf = motor.build()
    return cfg, mf


def test_housing_has_air_windows():
    """Housing window cutouts must produce air voxels at the housing radius."""
    cfg, mf = _build_field_driven()
    dens = mf.to_densities()
    air = dens.get("air", np.zeros(cfg.shape, dtype=np.float32))
    iron = dens["iron"]

    r_out = 0.055 + 0.003  # housing outer
    dx = cfg.spacing[0]
    cx, cy = cfg.center[0], cfg.center[1]
    nx, ny = cfg.shape[0], cfg.shape[1]

    X = cfg.origin[0] + dx * np.arange(nx)
    Y = cfg.origin[1] + dx * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    housing_mask = (R > 0.053) & (R < r_out + 0.002)

    cz = cfg.shape[2] // 2
    air_at_housing = air[:, :, cz][housing_mask]
    assert air_at_housing.max() > 0.3, (
        f"Housing windows not open: max air density at housing radius is {air_at_housing.max():.3f}"
    )


def test_pm_survives_functional_voids():
    """FunctionalVoids must not delete the PM material."""
    cfg, mf = _build_field_driven()
    dens = mf.to_densities()
    pm = dens["pm"]

    pm_max = float(pm.max())
    pm_count = int((pm > 0.5).sum())
    assert pm_max > 0.5, f"PM density max is {pm_max:.3f} — FunctionalVoids deleted the magnets"
    assert pm_count > 50, f"Only {pm_count} PM voxels above 0.5 — magnets nearly gone"


def test_air_gap_has_no_iron_bridge():
    """The radial air gap must not be bridged by iron."""
    cfg, mf = _build_field_driven()
    dens = mf.to_densities()
    iron = dens["iron"]

    dx = cfg.spacing[0]
    cx, cy = cfg.center[0], cfg.center[1]
    nx, ny = cfg.shape[0], cfg.shape[1]

    X = cfg.origin[0] + dx * np.arange(nx)
    Y = cfg.origin[1] + dx * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)

    gap_mask = (R > 0.0295) & (R < 0.030)
    cz = cfg.shape[2] // 2
    iron_in_gap = iron[:, :, cz][gap_mask]

    assert iron_in_gap.max() < 0.5, (
        f"Iron bridges the air gap: max iron density in gap is {iron_in_gap.max():.3f}"
    )


def test_end_cap_has_spoke_openings():
    """End caps must have angular windows (spoke pattern), not be solid discs."""
    cfg, mf = _build_field_driven()
    dens = mf.to_densities()
    iron = dens["iron"]
    air = dens.get("air", np.zeros(cfg.shape, dtype=np.float32))

    cz_center = cfg.center[2]
    hz = cfg.stator_half_length
    z_end_cap = int(round((cz_center + hz + cfg.spacing[2] * 0.5 - cfg.origin[2]) / cfg.spacing[2]))
    z_end_cap = min(z_end_cap, cfg.shape[2] - 1)

    dx = cfg.spacing[0]
    cx, cy = cfg.center[0], cfg.center[1]
    X = cfg.origin[0] + dx * np.arange(cfg.shape[0])
    Y = cfg.origin[1] + dx * np.arange(cfg.shape[1])
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    cap_mask = (R > 0.02) & (R < 0.05)

    iron_slice = iron[:, :, z_end_cap][cap_mask]
    air_slice = air[:, :, z_end_cap][cap_mask]

    open_fraction = float((air_slice > 0.3).sum()) / max(float(cap_mask.sum()), 1.0)
    assert open_fraction > 0.1, (
        f"End cap is nearly solid: only {open_fraction:.1%} air at end cap radius"
    )
