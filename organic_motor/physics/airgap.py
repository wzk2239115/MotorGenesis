"""Speed-dependent air effects: gap convection, end-face convection, windage.

D2 of the simulator audit. Established/derivable correlations only:

Air gap (concentric cylinders, inner rotating):
  - Laminar Couette (exact analytic): h = k_air / delta and
    friction coefficient c_f = 2/Re_delta (from tau = mu*omega*r/delta).
  - Above the Taylor transition the laminar result is enhanced; the
    classic Taylor-vortex scaling is Nu ~ Ta^(1/4) (Taylor 1935 gives the
    exponent; the prefactor varies across sources).  Prefactor 0.2 is an
    ENGINEERING ESTIMATE flagged in the notes.

Rotor end faces (disks):
  - Daily & Nece (1960) moment coefficients, canonical values:
    C_M = 3.87/Re_r^0.5 (laminar, Re_r < 3e5), 0.146/Re_r^0.2 (turbulent).
  - End-face h via Chilton-Colburn Reynolds analogy from the SAME C_M
    (transparent, derivable; flagged analogy-based).

Windage power:
  - Cylindrical side: P = pi*c_f*rho*omega^3*r^4*L with the c_f above.
  - Two end disks: T = C_M*(pi/2)*rho*omega^2*r^5, P = T*omega each.

All functions report their regime and applicability so callers can
surface honesty flags (audit: no unflagged extrapolation).
"""

from __future__ import annotations

import math
from typing import NamedTuple

# Air at ~40 degC, 1 atm (approximate; consistent with flow1d water style)
RHO_AIR = 1.127      # kg/m3
MU_AIR = 1.91e-5     # Pa s
NU_AIR = MU_AIR / RHO_AIR  # m2/s
K_AIR = 0.0276       # W/m/K
PR_AIR = 0.70

RE_DISK_TURB = 3.0e5     # Daily-Nece laminar/turbulent disk transition
TA_CRITICAL = 1700.0     # Taylor-vortex onset (Ta = Re_delta^2 * delta/r)


class GapConvectionResult(NamedTuple):
    h_W_m2K: float
    nusselt: float
    reynolds_gap: float
    taylor: float
    regime: str
    notes: str


def air_gap_convection(
    omega_rad_s: float,
    r_rotor_m: float,
    gap_m: float,
) -> GapConvectionResult:
    """Convection coefficient in the annular air gap.

    Laminar branch is the exact conduction limit h = k/delta; above the
    Taylor transition the Nu ~ Ta^(1/4) enhancement is applied with a
    flagged engineering prefactor.
    """
    if omega_rad_s <= 0 or gap_m <= 0 or r_rotor_m <= 0:
        return GapConvectionResult(
            h_W_m2K=K_AIR / max(gap_m, 1e-9) if gap_m > 0 else 0.0,
            nusselt=1.0, reynolds_gap=0.0, taylor=0.0,
            regime="static", notes="conduction only",
        )
    nu = MU_AIR / RHO_AIR
    re_gap = omega_rad_s * r_rotor_m * gap_m / nu
    ta = re_gap ** 2 * (gap_m / r_rotor_m)

    # Continuous formulation: Nu = max(1, 0.2*Ta^0.25).  With this
    # (estimated) prefactor the enhancement crosses 1 at Ta=(1/0.2)^4=625,
    # below the classic vortex onset Ta~1700; the offset reflects the
    # prefactor uncertainty (0.1-0.4 across sources; exponent theoretical,
    # Taylor 1935).
    nu_num = max(1.0, 0.2 * ta ** 0.25)
    h = nu_num * K_AIR / gap_m
    if nu_num > 1.0:
        return GapConvectionResult(
            h_W_m2K=h, nusselt=nu_num, reynolds_gap=re_gap, taylor=ta,
            regime="taylor_vortex",
            notes=("Nu=max(1, 0.2*Ta^0.25) — exponent theoretical "
                   "(Taylor 1935), prefactor engineering estimate"),
        )
    return GapConvectionResult(
        h_W_m2K=h, nusselt=1.0, reynolds_gap=re_gap, taylor=ta,
        regime="laminar_couette",
        notes="exact conduction limit h=k/delta",
    )


def _disk_moment_coefficient(re_disk: float) -> float:
    """Daily & Nece (1960) single-side smooth disk moment coefficient."""
    if re_disk < RE_DISK_TURB:
        return 3.87 / math.sqrt(max(re_disk, 1.0))
    return 0.146 * re_disk ** -0.2


class EndFaceResult(NamedTuple):
    h_W_m2K: float
    moment_coefficient: float
    reynolds_disk: float
    regime: str
    notes: str


def end_face_convection(
    omega_rad_s: float,
    r_disk_m: float,
) -> EndFaceResult:
    """End-face convection via Chilton-Colburn analogy from Daily-Nece C_M.

    St = C_M/2 evaluated at the rms disk radius r*sqrt(2/3); flagged
    analogy-based.
    """
    if omega_rad_s <= 0 or r_disk_m <= 0:
        return EndFaceResult(0.0, 0.0, 0.0, "static", "no rotation")
    nu = MU_AIR / RHO_AIR
    r_ref = r_disk_m * math.sqrt(2.0 / 3.0)  # rms radius of a uniform disk
    re_disk = omega_rad_s * r_disk_m ** 2 / nu
    c_m = _disk_moment_coefficient(re_disk)
    # Chilton-Colburn: St = C_f/2 with C_M approximated as the mean skin
    # friction coefficient; h = St * rho * u * cp, u = omega*r_ref.
    cp = 1005.0  # J/kg/K air
    u = omega_rad_s * r_ref
    st = c_m / 2.0 * prandtl_correction()
    h = st * RHO_AIR * u * cp
    regime = "laminar_disk" if re_disk < RE_DISK_TURB else "turbulent_disk"
    return EndFaceResult(
        h_W_m2K=h, moment_coefficient=c_m, reynolds_disk=re_disk,
        regime=regime,
        notes="Daily-Nece C_M + Chilton-Colburn analogy (estimate)",
    )


def prandtl_correction(j_factor: float = 2.0 / 3.0) -> float:
    """Colburn j-factor Pr^(2/3) correction for the analogy."""
    return PR_AIR ** j_factor


class WindageResult(NamedTuple):
    torque_Nm: float
    power_W: float
    side_torque_Nm: float
    disk_torque_Nm: float
    regime: str
    notes: str


def windage_torque_formula(xp, omega, r_rotor, length, gap,
                           n_end_disks: int = 2):
    """SINGLE-SOURCE windage torque, usable with numpy OR jax.numpy.

    The speed dependence is kept INSIDE the coefficients:
      side, laminar (Ta < 1700):  T = 2*pi*mu*omega*r^3*L/delta
          (exact laminar Couette, LINEAR in omega)
      side, turbulent:            T = pi*0.08*rho*r^4*L*(r*delta/nu)^-0.25
                                      * omega*|omega|^0.75
      disk, laminar (Re < 3e5):   T = (pi/2)*3.87*rho*r^5*(r^2/nu)^-0.5
                                      * omega*|omega|^0.5   (each)
      disk, turbulent:            T = (pi/2)*0.146*rho*r^5*(r^2/nu)^-0.2
                                      * omega*|omega|^0.8   (each)

    All branches are ODD in omega (works for reversed rotation) and
    strictly dissipative (T*omega >= 0).  Returns (total, side, disk).
    """
    gap = xp.maximum(gap, 1e-9)
    w = xp.abs(omega)
    re_gap = w * r_rotor * gap / NU_AIR
    ta = re_gap ** 2 * (gap / r_rotor)
    re_disk = w * r_rotor ** 2 / NU_AIR

    k1 = 2.0 * math.pi * MU_AIR * r_rotor ** 3 * length / gap
    k2 = (math.pi * 0.08 * RHO_AIR * r_rotor ** 4 * length
          * (r_rotor * gap / NU_AIR) ** -0.25)
    side = xp.where(ta < TA_CRITICAL, k1 * omega, k2 * omega * w ** 0.75)

    d1 = n_end_disks * 0.5 * math.pi * 3.87 * RHO_AIR * r_rotor ** 5 \
        * (r_rotor ** 2 / NU_AIR) ** -0.5
    d2 = n_end_disks * 0.5 * math.pi * 0.146 * RHO_AIR * r_rotor ** 5 \
        * (r_rotor ** 2 / NU_AIR) ** -0.2
    disk = xp.where(re_disk < RE_DISK_TURB,
                    d1 * omega * w ** 0.5, d2 * omega * w ** 0.8)
    return side + disk, side, disk


def rotor_windage(
    omega_rad_s: float,
    r_rotor_m: float,
    length_m: float,
    gap_m: float,
    n_end_disks: int = 2,
) -> WindageResult:
    """Windage torque/power of a smooth cylindrical rotor (numpy path).

    Thin wrapper over :func:`windage_torque_formula` — the SAME math the
    JAX transient uses, so the two cannot drift apart.  Side: laminar
    exact / turbulent engineering estimate; disks Daily & Nece (1960).
    """
    if omega_rad_s == 0.0 or r_rotor_m <= 0:
        return WindageResult(0.0, 0.0, 0.0, 0.0, "static", "no rotation")
    import numpy as np
    total, side, disk = windage_torque_formula(
        np, omega_rad_s, r_rotor_m, length_m, gap_m, n_end_disks)
    total, side, disk = (float(total), float(side), float(disk))

    nu = NU_AIR
    re_gap = abs(omega_rad_s) * r_rotor_m * max(gap_m, 1e-9) / nu
    ta = re_gap ** 2 * (max(gap_m, 1e-9) / r_rotor_m)
    re_disk = abs(omega_rad_s) * r_rotor_m ** 2 / nu
    if ta < TA_CRITICAL:
        side_regime, side_note = "laminar_couette_exact", \
            "c_f = 2/Re_delta (exact laminar Couette)"
    else:
        side_regime, side_note = "turbulent_estimate", \
            "c_f = 0.08*Re^-0.25 (Blasius-analog estimate)"
    if re_disk < RE_DISK_TURB:
        disk_regime = "laminar_disk_daily_nece"
    else:
        disk_regime = "turbulent_disk_daily_nece"
    return WindageResult(
        torque_Nm=total, power_W=total * omega_rad_s,
        side_torque_Nm=side, disk_torque_Nm=disk,
        regime=f"{side_regime}+{disk_regime}",
        notes=f"{side_note}; disks Daily-Nece (verified constants); "
              "turbulent side branch flagged engineering estimate",
    )


def windage_scan(
    rpm_list, r_rotor_m, length_m, gap_m,
) -> list[dict]:
    """Fixed-geometry speed sweep (audit D2: fixed heat load, swept speed)."""
    out = []
    for rpm in rpm_list:
        omega = rpm * 2.0 * math.pi / 60.0
        w = rotor_windage(omega, r_rotor_m, length_m, gap_m)
        g = air_gap_convection(omega, r_rotor_m, gap_m)
        out.append({
            "rpm": rpm,
            "omega_rad_s": omega,
            "windage_W": w.power_W,
            "windage_torque_Nm": w.torque_Nm,
            "h_gap_W_m2K": g.h_W_m2K,
            "gap_regime": g.regime,
        })
    return out
