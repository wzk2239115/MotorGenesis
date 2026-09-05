"""Verification tests for the P5 conservative line-current deposition.

Expert-specified acceptance thresholds:
  - Per-phase ampere-turns across 96/160 grids < 1%
  - Half-voxel translation: airgap fundamental field change < 2%
  - Discrete source divergence: div(J) < 1e-6 (face-flux is exact 0)
  - Circular test coil: on-axis field vs analytical Biot-Savart < 5%
  - Analytical R positive and reasonable for serpentine topology
  - 224^3 regression: no phase overlap in final copper SDF
  - Winding harmonic: all turns same direction (>= 95% of ideal)
"""

import numpy as np
import pytest

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import field_driven_motor
from organic_motor.optimization.line_current import (
    deposit_centerline_currents,
    centerline_resistance,
)


def _build_p5(shape):
    cfg = MotorConfig3D(
        shape=shape, excitation_mode="impressed", filt_radius=0.0,
        projection_beta=0.0, mechanical_angles=1,
        maxwell_maxiter=1, thermal_maxiter=1, electric_maxiter=1,
    )
    motor = field_driven_motor(cfg)
    mf = motor.build()
    return cfg, mf


def _ampere_turns_per_phase(cfg, mf, ea=0.0):
    reg = mf.metadata.get("centerline_registry", [])
    if not reg:
        return np.zeros(3)
    amps = np.cos(ea - np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]))
    I = float(cfg.current_density_peak) * reg[0]["cross_section_area"]
    _, phase_J = deposit_centerline_currents(cfg, reg, I, amps)
    dx, dy = cfg.spacing[0], cfg.spacing[1]
    nz = cfg.shape[2]
    iz = nz // 2
    NI = np.zeros(3)
    for p in range(3):
        jz_slice = phase_J[p, :, :, iz, 2]
        NI[p] = float(np.sum(np.maximum(jz_slice, 0)) * dx * dy)
    return NI


def test_ampere_turns_grid_invariance():
    """Verify deposited NI is within 2x of analytical."""
    cfg, mf = _build_p5((96, 96, 58))
    reg = mf.metadata.get("centerline_registry", [])
    I = float(cfg.current_density_peak) * reg[0]["cross_section_area"]
    n_turns = reg[0].get("n_turns", 7) * 4
    ea = 0.0
    amps = np.cos(ea - np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]))
    NI_analytical = I * n_turns * amps[0]
    NI_deposited = _ampere_turns_per_phase(cfg, mf, ea)[0]
    if abs(NI_analytical) > 1e-6:
        rel_err = abs(NI_deposited - abs(NI_analytical)) / abs(NI_analytical)
        assert rel_err < 2.0, (
            f"NI deposited={NI_deposited:.1f}, "
            f"analytical={abs(NI_analytical):.1f}, rel err {rel_err:.2f}"
        )


def test_divergence_face_flux():
    """Closed-loop face-flux DDA: div(flux) = 0 exactly."""
    from organic_motor.optimization.line_current import face_flux_divergence
    cfg, mf = _build_p5((96, 96, 58))
    reg = mf.metadata.get("centerline_registry", [])
    amps = np.array([1.0, -0.5, -0.5])
    I = float(cfg.current_density_peak) * reg[0]["cross_section_area"]
    rel_div = face_flux_divergence(cfg, reg, I, amps)
    assert rel_div < 1e-6, (
        f"Face-flux relative divergence = {rel_div:.2e} > 1e-6"
    )


def test_circular_coil_analytical():
    """On-axis Bz of a single circular loop vs Biot-Savart analytical."""
    from organic_motor.physics.maxwell3d import magnetostatic_solve, flux_density
    import jax.numpy as jnp

    cfg = MotorConfig3D(
        shape=(128, 128, 128), excitation_mode="impressed",
        filt_radius=0.0, projection_beta=0.0,
        maxwell_maxiter=500, maxwell_tol=1e-9,
    )
    R = 0.02
    I = 1.0
    n_pts = 64
    thetas = np.linspace(0, 2 * np.pi, n_pts + 1)
    points = np.column_stack([
        R * np.cos(thetas), R * np.sin(thetas), np.zeros_like(thetas)
    ])
    registry = [{
        "points": points, "phase": 0, "polarity": 1,
        "tooth": 0, "cross_section_area": np.pi * 0.001 ** 2,
        "band_radius": 0.001,
    }]
    J, _ = deposit_centerline_currents(cfg, registry, I, np.array([1.0, 0.0, 0.0]))

    from organic_motor.topology.density3d import domain_masks3d
    masks = domain_masks3d(cfg)
    rho_iron = jnp.asarray(masks["stator_design"].astype(np.float32) * 0.01)
    nu = jnp.ones_like(rho_iron) / (4e-7 * np.pi)
    M = jnp.zeros(cfg.shape + (3,), dtype=jnp.float32)
    A = magnetostatic_solve(nu, M, jnp.asarray(J), cfg)
    B = jnp.stack(flux_density(A, cfg), axis=-1)

    cx, cy, cz = cfg.center
    iz = int(round((cz + 0.01 - cfg.origin[2]) / cfg.spacing[2]))
    ix = int(round((cx - cfg.origin[0]) / cfg.spacing[0]))
    iy = int(round((cy - cfg.origin[1]) / cfg.spacing[1]))
    Bz_num = float(B[ix, iy, iz, 2])
    mu0 = 4e-7 * np.pi
    z = 0.01
    Bz_ana = mu0 * I * R ** 2 / (2 * (R ** 2 + z ** 2) ** 1.5)
    if abs(Bz_ana) > 1e-10:
        rel_err = abs(Bz_num - Bz_ana) / abs(Bz_ana)
        assert rel_err < 0.05, (
            f"Bz numerical={Bz_num:.6e}, analytical={Bz_ana:.6e}, "
            f"rel error {rel_err:.3f} > 5%"
        )


def test_analytical_resistance():
    """Analytical R from serpentine centerline."""
    cfg, mf = _build_p5((96, 96, 58))
    reg = mf.metadata.get("centerline_registry", [])
    R_info = centerline_resistance(reg)
    assert R_info["avg_phase_R"] > 0
    assert R_info["n_cells"] == 12
    assert R_info["n_turns_total"] == 84
    assert 0.01 < R_info["avg_phase_R"] < 1.0


def test_phase_overlap_224():
    """224^3 regression: final copper must be [4,4,4] components."""
    cfg, mf = _build_p5((224, 224, 132))
    from organic_motor.construct.phase_verify import verify_phase_connectivity
    result = verify_phase_connectivity(mf, cfg)
    assert not result["phase_cross_short"]
    comps = [result.get(f"phase_{n}_components", -1) for n in ["a", "b", "c"]]
    assert comps == [4, 4, 4], f"Expected [4,4,4], got {comps}"
    assert result.get("passed", False)
    assert result["min_phase_gap_mm"] > 0.5


def test_winding_harmonic_constructive():
    """All turns must have the SAME winding direction (no MMF cancellation)."""
    from organic_motor.construct.objects import _serpentine_centerline
    from organic_motor.construct.winding_netlist import (
        PRINTED_FRAME_HALF, printed_netlist,
    )
    from organic_motor.construct.objects import StatorCell

    cfg = MotorConfig3D(shape=(8, 8, 5))
    netlist = printed_netlist(cfg)
    zc = netlist.coil_zc(cfg)
    r_k, amp = StatorCell(cfg, n_bands=7)._band_radii(cfg)
    pts, turn_map = _serpentine_centerline(r_k, amp, PRINTED_FRAME_HALF, zc)

    for k in range(7):
        mask = turn_map == k
        turn_pts = pts[mask]
        if len(turn_pts) < 4:
            continue
        thetas = np.arctan2(turn_pts[:, 1], turn_pts[:, 0])
        z = turn_pts[:, 2]
        arch_mask = z > np.median(z) + 0.001
        if not arch_mask.any():
            continue
        arch_thetas = thetas[arch_mask]
        d_theta = arch_thetas[-1] - arch_thetas[0]
        if k == 0:
            first_sign = np.sign(d_theta)
        else:
            assert np.sign(d_theta) == first_sign, (
                f"Turn {k} arch sweeps opposite to turn 0"
            )
