"""Reduced motor physics fields that drive geometry growth.

LEAP 71 keeps its physics-informed growth in closed-source Noyron; these are
the open equivalent for an inner-rotor surface-PM machine.  Each field is a
cheap analytical/reduced model (NOT the full FEM) -- the same role LEAP 71's
"15-minute" reduced models play -- so the growth loop runs fast and the full
``forward3d`` solver stays the validator, not the in-loop driver.

Fields are returned as :class:`ScalarField` / :class:`VectorField` on the cfg
grid, sampleable at any point -- exactly the contract the modulated implicits
and field-driven objects consume.

  * ``airgap_B(theta, z)``  -- air-gap flux density from a magnetic-equivalent
    circuit; peaks under magnet centres, falls in the interpole gap.
  * ``current_density(theta, z)`` -- stator slot current density from the
    three-phase belt distribution at the design operating point.
  * ``joule_heat(theta, z)``  -- I^2 R volumetric heat from the winding.
  * ``centrifugal_stress(r, z)`` -- rotor hoop stress at the design speed.
"""

from __future__ import annotations

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.field import ScalarField, VectorField


def _polar_grid(cfg: MotorConfig3D):
    cx, cy, _ = cfg.center
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    x = ox + dx * np.arange(nx, dtype=np.float32)
    y = oy + dy * np.arange(ny, dtype=np.float32)
    z = oz + dz * np.arange(nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    theta = np.arctan2(Y - cy, X - cx)
    return X, Y, Z, r, theta


def airgap_B(cfg: MotorConfig3D, *, load_angle: float = np.pi / 4) -> ScalarField:
    """Air-gap flux density magnitude from a magnetic-equivalent-circuit proxy.

    A surface-PM machine's gap field is dominated by the magnet's square-wave
    MMF filtered by the gap permeance.  We model it as the radial projection of
    the magnet remanence onto the fundamental, modulated by a load angle and
    softened by the gap/carter factor.  This is a reduced model -- the full
    Maxwell solve validates it later.
    """
    X, Y, Z, r, theta = _polar_grid(cfg)
    poles = 2 * cfg.pole_pairs
    electrical = poles * theta / 2.0
    # Magnet-shaped gap field: near-trapezoid in the electrical angle.
    magnet_wave = np.sign(np.cos(electrical)) * np.clip(
        np.abs(np.cos(electrical)) ** 0.4, 0.0, 1.0
    )
    gap = cfg.R_stator_inner - cfg.R_rotor_outer
    carter = 1.0 / (1.0 + gap / (2.0 * max(cfg.R_rotor_outer, F1 := 1e-9)))
    # Field falls off away from the gap midpoint; concentrate in the gap band.
    gap_mid = 0.5 * (cfg.R_stator_inner + cfg.R_rotor_outer)
    radial_window = np.exp(-((r - gap_mid) ** 2) / (2 * (gap * 1.5) ** 2 + 1e-18))
    B = cfg.B_r * carter * magnet_wave * np.cos(load_angle) * radial_window
    return ScalarField(data=B.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin)


def current_density(cfg: MotorConfig3D, *, phase: float = 0.0) -> ScalarField:
    """Stator winding current density magnitude from the three-phase belts.

    The belts follow ``cos(p*theta - phase_shift)``; the magnitude is the
    peak current density gated by the winding annulus.  This is the field a
    field-driven copper object samples to thicken conductors where current is
    high.
    """
    X, Y, Z, r, theta = _polar_grid(cfg)
    electrical = cfg.pole_pairs * theta
    # Three-phase envelope: the per-phase belt magnitude at this position.
    envelope = np.abs(np.cos(electrical - phase))
    in_winding = (
        (r >= cfg.R_winding_inner) & (r <= cfg.R_winding_outer)
    ).astype(np.float32)
    J = cfg.current_density_peak * envelope * in_winding
    return ScalarField(data=J.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin)


def joule_heat(cfg: MotorConfig3D, *, phase: float = 0.0) -> ScalarField:
    """Volumetric Joule heating ``J^2 / sigma`` [W/m^3] in the winding."""
    J = current_density(cfg, phase=phase)
    q = (J.data ** 2) / cfg.sigma_copper
    return ScalarField(data=q.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin)


def centrifugal_stress(cfg: MotorConfig3D) -> ScalarField:
    """Rotor hoop stress ``sigma = rho * omega^2 * r^2`` [Pa] at design speed.

    Drives rotor back-iron / magnet-retaining-ring thickness: stress grows
    quadratically with radius, so the outer rotor rim needs the most material.
    """
    X, Y, Z, r, theta = _polar_grid(cfg)
    omega = 2.0 * np.pi * cfg.speed_rpm / 60.0
    # Use iron density inside the rotor region, zero outside.
    in_rotor = (r <= cfg.R_rotor_outer).astype(np.float32)
    stress = cfg.rho_iron_kg * omega ** 2 * r ** 2 * in_rotor
    return ScalarField(data=stress.astype(np.float32), spacing=cfg.spacing, origin=cfg.origin)


def magnetization_field(cfg: MotorConfig3D) -> VectorField:
    """Radial alternating magnetisation direction (3-vector field).

    The PM magnetisation: radial outward for even poles, inward for odd.  This
    is the vector field a field-driven magnet object reads to orient each pole.
    """
    X, Y, Z, r, theta = _polar_grid(cfg)
    poles = 2 * cfg.pole_pairs
    pitch = 2.0 * np.pi / poles
    sign = np.where((np.rint(theta / pitch).astype(int) % 2) == 0, 1.0, -1.0)
    mx = (sign * np.cos(theta)).astype(np.float32)
    my = (sign * np.sin(theta)).astype(np.float32)
    mz = np.zeros_like(mx)
    return VectorField(
        data=np.stack([mx, my, mz], axis=-1),
        spacing=cfg.spacing,
        origin=cfg.origin,
    )


def all_fields(cfg: MotorConfig3D) -> dict:
    """Convenience: build the full reduced-physics field set for a motor."""
    return {
        "B": airgap_B(cfg),
        "J": current_density(cfg),
        "joule_heat": joule_heat(cfg),
        "stress": centrifugal_stress(cfg),
        "magnetization": magnetization_field(cfg),
    }
