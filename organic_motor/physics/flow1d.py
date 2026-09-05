"""Minimal 1D thermal-flow network for cooling channel evaluation.

Evaluates straight vs helical channels at same pump power.
Computes: pressure drop, flow rate, temperature rise, heat removal.

Physics:
  - Darcy-Weisbach: Δp = f * (L/D) * ρ*v²/2
  - Laminar friction: f = 64/Re (Re < 2300)
  - Blasius turbulent: f = 0.316*Re^-0.25 (Re >= 2300)
  - Helical correction (Ito): f_helix = f_straight * [1 + 0.033*(De^0.5)^2]
    where Dean number De = Re*sqrt(D/(2*R_helix))
  - Energy balance: Q = m_dot * cp * ΔT
  - Convection: Nu = 3.66 (laminar) or Nu = 0.023*Re^0.8*Pr^0.4 (Dittus-Boelter)

All correlations have explicit applicability ranges stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple


# Physical properties (water at 40°C, sourced from NIST)
RHO_WATER = 992.0       # kg/m³
MU_WATER = 0.000653     # Pa·s
CP_WATER = 4179.0       # J/(kg·K)
K_WATER = 0.631         # W/(m·K)
PR_WATER = MU_WATER * CP_WATER / K_WATER  # ~4.3


class FlowResult(NamedTuple):
    """Result of a 1D flow network evaluation."""
    channel_type: str
    length_m: float
    diameter_m: float
    reynolds: float
    friction_factor: float
    velocity_ms: float
    flow_rate_kg_s: float
    pressure_drop_Pa: float
    pump_power_W: float
    heat_removed_W: float
    temp_rise_K: float
    outlet_temp_C: float
    nusselt: float
    h_conv_W_m2K: float
    applicable: bool
    notes: str


def friction_factor(re: float, dean: float = 0.0) -> float:
    """Darcy friction factor with optional helical correction.

    Applicable: Re > 0. Dean > 0 only for helical pipes.
    Ito correlation for helical: f_h = f * (1 + 0.033 * De^0.5)
    Valid for De < 1000 (laminar) — extrapolated for transitional.
    """
    if re < 1:
        return 64.0
    if re < 2300:
        f = 64.0 / re
        if dean > 0:
            de_sqrt = math.sqrt(dean)
            f *= 1.0 + 0.033 * de_sqrt  # Ito, valid De < ~1000
        return f
    else:
        f = 0.316 / re**0.25  # Blasius, valid 4000 < Re < 1e5
        if dean > 0:
            de_sqrt = math.sqrt(dean)
            f *= 1.0 + 0.033 * de_sqrt
        return f


def nusselt_number(re: float, pr: float, dean: float = 0.0) -> float:
    """Nusselt number for internal flow.

    Laminar (Re < 2300): Nu = 3.66 (fully developed, constant wall T)
    Turbulent (Re >= 10000): Nu = 0.023*Re^0.8*Pr^0.4 (Dittus-Boelter)
    Transitional: linear interpolation (not validated — flagged)
    Helical: Nu_helix = Nu * (1 + 3.6*(De/Re)^0.5) (Schmidt)

    Applicability:
      - Laminar: Re < 2300, fully developed
      - Turbulent: 10000 <= Re <= 1.2e5, 0.6 <= Pr <= 120
      - Transitional (2300 < Re < 10000): NOT validated, flagged
    """
    if re < 2300:
        nu = 3.66
        if dean > 0:
            nu *= 1.0 + 3.6 * (dean / max(re, 1))**0.5  # Schmidt
        return nu
    elif re >= 10000:
        nu = 0.023 * re**0.8 * pr**0.4  # Dittus-Boelter
        if dean > 0:
            nu *= 1.0 + 3.6 * (dean / max(re, 1))**0.5
        return nu
    else:
        # Transitional — linear interpolation (NOT validated)
        nu_lam = 3.66
        nu_turb = 0.023 * 10000**0.8 * pr**0.4
        frac = (re - 2300) / (10000 - 2300)
        nu = nu_lam + frac * (nu_turb - nu_lam)
        if dean > 0:
            nu *= 1.0 + 3.6 * (dean / max(re, 1))**0.5
        return nu


def evaluate_channel(
    channel_type: str,
    length_m: float,
    diameter_m: float,
    pump_power_W: float,
    inlet_temp_C: float = 40.0,
    heat_load_W: float = 10.0,
    helix_radius_m: float = 0.0,
) -> FlowResult:
    """Evaluate one channel at fixed pump power.

    Solves: given pump power P, find flow rate Q such that
    P = Δp * Q / η (η=1 for this simplified model).

    Δp = f * (L/D) * ρ * v² / 2, where v = Q / (ρ * A)
    """
    A = math.pi * diameter_m**2 / 4
    eta_pump = 1.0  # simplified

    # Iterative solve: v depends on Re, Re depends on v
    # Start with guess
    v = 1.0  # m/s
    for _ in range(50):
        re = RHO_WATER * v * diameter_m / MU_WATER

        # Dean number for helical
        dean = 0.0
        if channel_type == "helical" and helix_radius_m > 0:
            dean = re * math.sqrt(diameter_m / (2 * helix_radius_m))

        f = friction_factor(re, dean)

        # Pressure drop
        dp = f * (length_m / diameter_m) * RHO_WATER * v**2 / 2

        # Flow rate
        m_dot = RHO_WATER * A * v

        # Pump power = dp * Q / eta
        Q_vol = A * v
        P_calc = dp * Q_vol / eta_pump

        if P_calc < 1e-15:
            break

        # Adjust v to match pump power
        # P = f * (L/D) * rho * A * v^3 / 2
        # v = (2*P / (f * L/D * rho * A))^(1/3)
        v_new = (2 * pump_power_W / (f * length_m / diameter_m * RHO_WATER * A)) ** (1/3)
        if abs(v_new - v) / max(v, 1e-6) < 1e-4:
            v = v_new
            break
        v = v_new

    # Final values
    re = RHO_WATER * v * diameter_m / MU_WATER
    dean = 0.0
    if channel_type == "helical" and helix_radius_m > 0:
        dean = re * math.sqrt(diameter_m / (2 * helix_radius_m))
    f = friction_factor(re, dean)
    dp = f * (length_m / diameter_m) * RHO_WATER * v**2 / 2
    m_dot = RHO_WATER * A * v
    Q_vol = A * v

    # Heat transfer
    nu = nusselt_number(re, PR_WATER, dean)
    h_conv = nu * K_WATER / diameter_m

    # Temperature rise
    if m_dot > 1e-10:
        dT = heat_load_W / (m_dot * CP_WATER)
    else:
        dT = 999.0

    outlet_temp = inlet_temp_C + dT

    # Applicability check
    applicable = True
    notes = []
    if 2300 < re < 10000:
        applicable = False
        notes.append("transitional flow — correlations not validated")
    if re > 1e5:
        applicable = False
        notes.append("Re > 1e5 — Blasius not valid")
    if channel_type == "helical" and dean > 1000:
        notes.append(f"De={dean:.0f} > 1000 — Ito may not apply")
    if not notes:
        notes.append("all correlations within applicability range")

    return FlowResult(
        channel_type=channel_type,
        length_m=length_m,
        diameter_m=diameter_m,
        reynolds=re,
        friction_factor=f,
        velocity_ms=v,
        flow_rate_kg_s=m_dot,
        pressure_drop_Pa=dp,
        pump_power_W=pump_power_W,
        heat_removed_W=heat_load_W,
        temp_rise_K=dT,
        outlet_temp_C=outlet_temp,
        nusselt=nu,
        h_conv_W_m2K=h_conv,
        applicable=applicable,
        notes="; ".join(notes),
    )


def compare_channels(
    pump_power_W: float = 0.5,
    heat_load_W: float = 10.0,
    diameter_m: float = 0.003,
) -> list[FlowResult]:
    """Compare straight vs helical at same pump power and diameter."""
    results = []

    # Straight channel: length = stator axial extent
    L_straight = 0.080  # 80mm
    results.append(evaluate_channel(
        "straight", L_straight, diameter_m, pump_power_W,
        heat_load_W=heat_load_W,
    ))

    # Helical channel: longer path, but smaller cross-section effect
    # 3 turns at R=0.045, pitch=0.015
    R_helix = 0.045
    pitch = 0.015
    n_turns = 3.0
    L_helix = math.sqrt(R_helix**2 + (pitch / (2 * math.pi))**2) * 2 * math.pi * n_turns
    results.append(evaluate_channel(
        "helical", L_helix, diameter_m, pump_power_W,
        heat_load_W=heat_load_W,
        helix_radius_m=R_helix,
    ))

    return results
