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


def rotor_windage(
    omega_rad_s: float,
    r_rotor_m: float,
    length_m: float,
    gap_m: float,
    n_end_disks: int = 2,
) -> WindageResult:
    """Windage torque/power of a smooth cylindrical rotor.

    Side: exact laminar Couette c_f = 2/Re_delta below the Taylor
    transition; turbulent branch c_f = 0.08*Re_delta^-0.25 (engineering
    estimate, Blasius-analog).  End faces: Daily & Nece C_M.
    """
    if omega_rad_s <= 0 or r_rotor_m <= 0:
        return WindageResult(0.0, 0.0, 0.0, 0.0, "static", "no rotation")
    nu = MU_AIR / RHO_AIR

    # --- cylindrical side through the gap ---
    re_gap = omega_rad_s * r_rotor_m * max(gap_m, 1e-9) / nu
    ta = re_gap ** 2 * (max(gap_m, 1e-9) / r_rotor_m)
    if ta < TA_CRITICAL:
        # Exact laminar Couette: tau = mu*omega*r/delta gives
        # T = pi*c_f*rho*omega^2*r^4*L with c_f = 2/Re_delta.
        c_f = 2.0 / max(re_gap, 1e-9)
        side_regime = "laminar_couette_exact"
        side_note = "c_f = 2/Re_delta (exact laminar Couette)"
    else:
        c_f = 0.08 * re_gap ** -0.25
        side_regime = "turbulent_estimate"
        side_note = "c_f = 0.08*Re^-0.25 (Blasius-analog estimate)"
    side_torque = math.pi * c_f * RHO_AIR * omega_rad_s ** 2 * r_rotor_m ** 4 * length_m

    # --- end disks (Daily & Nece) ---
    re_disk = omega_rad_s * r_rotor_m ** 2 / nu
    c_m = _disk_moment_coefficient(re_disk)
    disk_torque = (
        c_m * 0.5 * math.pi * RHO_AIR * omega_rad_s ** 2 * r_rotor_m ** 5
    )
    total_disk = n_end_disks * disk_torque

    total = side_torque + total_disk
    return WindageResult(
        torque_Nm=total, power_W=total * omega_rad_s,
        side_torque_Nm=side_torque, disk_torque_Nm=total_disk,
        regime=f"{side_regime}+disk",
        notes=f"{side_note}; disks Daily-Nece",
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
