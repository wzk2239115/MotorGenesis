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
    wall_temp_C: float
    fluid_avg_temp_C: float
    nusselt: float
    h_conv_W_m2K: float
    surface_area_m2: float
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

    # Wall-fluid heat exchange model:
    # Q_actual = h * A_surface * (T_wall - T_fluid_avg)
    # T_fluid_avg = (T_inlet + T_outlet) / 2
    # Energy balance: Q = m_dot * cp * (T_outlet - T_inlet)
    # Coupled: Q = h * pi*D*L * (T_wall - T_fluid_avg) = m_dot * cp * dT
    # Solving for T_wall:
    #   Q = m_dot * cp * dT  (energy into fluid)
    #   Q = h * pi*D*L * (T_wall - (T_in + T_out)/2)
    #   dT = T_out - T_in
    #   T_out = T_in + Q/(m_dot*cp)
    #   T_fluid_avg = T_in + Q/(2*m_dot*cp)
    #   T_wall = T_fluid_avg + Q/(h*pi*D*L)

    surface_area = math.pi * diameter_m * length_m

    if m_dot > 1e-10 and h_conv > 0 and surface_area > 0:
        # Actual heat transfer: solve coupled system
        # Assume heat_load is the heat available at the wall
        Q_actual = heat_load_W
        dT = Q_actual / (m_dot * CP_WATER)
        T_fluid_avg = inlet_temp_C + dT / 2
        T_wall = T_fluid_avg + Q_actual / (h_conv * surface_area)
        heat_transferred = Q_actual
    else:
        dT = 999.0
        T_wall = 999.0
        T_fluid_avg = inlet_temp_C
        heat_transferred = 0.0

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
        applicable = False
        notes.append(f"De={dean:.0f} > 1000 — Ito correlation not valid")
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
        heat_removed_W=heat_transferred,
        temp_rise_K=dT,
        outlet_temp_C=outlet_temp,
        wall_temp_C=T_wall,
        fluid_avg_temp_C=T_fluid_avg,
        nusselt=nu,
        h_conv_W_m2K=h_conv,
        surface_area_m2=surface_area,
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
    L_helix = math.sqrt(R_helix**2 + (pitch / (2 * np.pi))**2) * 2 * np.pi * n_turns
    results.append(evaluate_channel(
        "helical", L_helix, diameter_m, pump_power_W,
        heat_load_W=heat_load_W,
        helix_radius_m=R_helix,
    ))

    return results


# ============================================================
# D1: node/branch flow network (pressure solver + heat mixing)
# ============================================================

import numpy as np


@dataclass
class Branch:
    """One pipe segment between two nodes."""
    name: str
    from_node: str
    to_node: str
    length_m: float
    diameter_m: float
    minor_loss_K: float = 0.0
    heat_load_W: float = 0.0
    helix_radius_m: float | None = None
    channel_type: str = "straight"


@dataclass
class PumpCurve:
    """Quadratic pump curve dp(Q) = dp_max * (1 - (Q/Q_max)^2)."""
    dp_max_Pa: float
    Q_max_m3s: float

    def dp(self, Q: float) -> float:
        q = max(min(Q / self.Q_max_m3s, 1.0), -1.0)
        return self.dp_max_Pa * (1.0 - q * q)


class NetworkSolution(NamedTuple):
    converged: bool
    iterations: int
    node_pressures_Pa: dict
    branch_flows_kg_s: dict      # signed: + means from_node -> to_node
    branch_results: dict         # name -> FlowResult-style dict
    inlet_flow_kg_s: float
    outlet_temps_C: dict
    heat_removed_W: float
    applicable: bool
    notes: str


def _branch_resistance(br: Branch, m_dot: float) -> tuple[float, float, float, bool, str]:
    """Resistance R (dp = R*m*|m|), plus (Re, f, applicable, note)."""
    A = math.pi * br.diameter_m ** 2 / 4.0
    re = abs(m_dot) * br.diameter_m / (A * MU_WATER) if A > 0 else 0.0
    dean = 0.0
    if br.channel_type == "helical" and br.helix_radius_m:
        dean = re * math.sqrt(br.diameter_m / (2.0 * br.helix_radius_m))
    f = friction_factor(re, dean)
    R = (f * br.length_m / br.diameter_m + br.minor_loss_K) / (2.0 * RHO_WATER * A * A)
    ok = True
    notes = []
    if 2300 < re < 10000:
        ok = False
        notes.append(f"{br.name}: transitional Re={re:.0f}")
    if re > 1e5:
        ok = False
        notes.append(f"{br.name}: Re>1e5 Blasius invalid")
    if br.channel_type == "helical" and dean > 1000:
        ok = False
        notes.append(f"{br.name}: De={dean:.0f} Ito invalid")
    return R, re, f, ok, "; ".join(notes)


def solve_network(
    branches: list[Branch],
    boundary_pressure_Pa: dict,
    inlet_temp_C: float = 40.0,
    pump: PumpCurve | None = None,
    tol: float = 1e-7,
    max_iter: int = 60,
) -> NetworkSolution:
    """Solve nodal pressures (Newton) with outer friction iteration.

    Boundary nodes have fixed pressures; the pump (if given) adds its
    head to the branch connected from a node named 'source'.
    Mass is conserved exactly at every internal node; temperatures are
    propagated by flow-direction traversal with mixing.
    """
    node_set = set()
    for b in branches:
        node_set.update((b.from_node, b.to_node))
    for n in boundary_pressure_Pa:
        if n not in node_set:
            raise ValueError(f"boundary node {n!r} not on any branch")
    internal = sorted(node_set - set(boundary_pressure_Pa))
    index = {n: i for i, n in enumerate(internal)}
    n_int = len(internal)

    p = {n: boundary_pressure_Pa.get(n, 1.0e5) for n in node_set}
    R = {}
    # initialize branch resistances once, then fixed-point on friction
    for b in branches:
        R[b.name], *_ = _branch_resistance(b, 0.05)
    info = {}
    converged = False
    it_total = 0
    flows = {b.name: 0.0 for b in branches}
    prev_total_Q = 0.0  # pump curve operating point [m3/s]

    for outer in range(20):
        for it in range(max_iter):
            it_total += 1
            g = np.zeros(n_int)
            J = np.zeros((n_int, n_int))
            flows = {}
            for b in branches:
                dp = p[b.from_node] - p[b.to_node]
                if pump is not None and b.from_node == "source":
                    dp += pump.dp(prev_total_Q)
                Rb = R[b.name]
                if abs(dp) < 1e-9:
                    m = 0.0
                    dmdp = 1.0 / (2.0 * math.sqrt(Rb * 1e-9))
                else:
                    s = 1.0 if dp > 0 else -1.0
                    m = s * math.sqrt(abs(dp) / Rb)
                    dmdp = 1.0 / (2.0 * math.sqrt(Rb * abs(dp)))
                flows[b.name] = m
                i_f, i_t = index.get(b.from_node), index.get(b.to_node)
                if i_f is not None:
                    g[i_f] += m
                    J[i_f, i_f] += dmdp
                    if i_t is not None:
                        J[i_f, i_t] -= dmdp
                if i_t is not None:
                    g[i_t] -= m
                    J[i_t, i_t] += dmdp
                    if i_f is not None:
                        J[i_t, i_f] -= dmdp
            flows["__prev_total__"] = sum(
                flows[b.name] for b in branches if b.from_node == "source"
            )
            prev_total_Q = abs(flows["__prev_total__"]) / RHO_WATER
            if n_int == 0 or np.max(np.abs(g)) < tol:
                break
            try:
                delta = np.linalg.solve(J, -g)
            except np.linalg.LinAlgError:
                break
            step = np.linalg.norm(delta)
            for n, i in index.items():
                p[n] = max(p[n] + delta[i], 1.0)  # keep physical
        inner_ok = n_int == 0 or np.max(np.abs(g)) < tol
        # outer friction update with converged flows
        changed = False
        for b in branches:
            R_new, re_, f_, ok_, note_ = _branch_resistance(b, flows[b.name])
            info[b.name] = (re_, f_, ok_, note_)
            if abs(R_new - R[b.name]) / max(R[b.name], 1e-12) > 1e-3:
                changed = True
            R[b.name] = R_new
        converged = inner_ok and not changed
        if converged:
            break

    # --- temperature propagation (BFS from boundaries with known T) ---
    T = {n: None for n in node_set}
    for n, pv in boundary_pressure_Pa.items():
        if n.endswith("+") or n == "inlet" or n == "source":
            T[n] = inlet_temp_C
    T.setdefault("source", inlet_temp_C)
    if T.get("inlet") is None:
        T["inlet"] = inlet_temp_C
    for _ in range(len(node_set) + 2):
        for b in branches:
            m = flows[b.name]
            if m > 1e-12:
                t_in = T[b.from_node]
                if t_in is not None:
                    dT = b.heat_load_W / (m * CP_WATER)
                    t_out = t_in + dT
                    if T[b.to_node] is None:
                        T[b.to_node] = t_out
                    else:
                        # mixing handled after full pass below
                        pass
        # mixing pass: recompute mixed temperatures from upstream
        for n in node_set:
            if n in boundary_pressure_Pa and T.get(n) is not None:
                continue
            m_in = 0.0
            h_sum = 0.0
            for b in branches:
                if b.to_node == n and flows[b.name] > 1e-12 and T[b.from_node] is not None:
                    m_b = flows[b.name]
                    m_in += m_b
                    h_sum += m_b * (
                        T[b.from_node]
                        + b.heat_load_W / (m_b * CP_WATER)
                    )
            if m_in > 1e-12:
                T[n] = h_sum / m_in

    # per-branch thermal detail
    branch_results = {}
    all_ok = True
    notes = []
    for b in branches:
        re_, f_, ok_, note_ = info.get(b.name, (0, 0, True, ""))
        all_ok = all_ok and ok_
        if note_:
            notes.append(note_)
        m = abs(flows[b.name])
        A = math.pi * b.diameter_m ** 2 / 4.0
        v = m / (RHO_WATER * A) if A > 0 else 0.0
        dean = 0.0
        if b.channel_type == "helical" and b.helix_radius_m:
            dean = re_ * math.sqrt(b.diameter_m / (2.0 * b.helix_radius_m))
        nu = nusselt_number(re_, PR_WATER, dean)
        h_conv = nu * K_WATER / b.diameter_m
        surf = math.pi * b.diameter_m * b.length_m
        dT = b.heat_load_W / (m * CP_WATER) if m > 1e-12 else float("inf")
        t_avg = ((T.get(b.from_node) or inlet_temp_C)
                 + (T.get(b.to_node) or inlet_temp_C)) / 2.0
        t_wall = t_avg + (b.heat_load_W / (h_conv * surf) if surf > 0 else 0.0)
        branch_results[b.name] = {
            "flow_kg_s": flows[b.name], "reynolds": re_,
            "friction_factor": f_, "velocity_ms": v,
            "h_conv_W_m2K": h_conv, "dT_K": dT,
            "wall_temp_C": t_wall, "applicable": ok_,
        }

    inlet_flow = sum(
        flows[b.name] for b in branches if b.from_node in ("source", "inlet")
    )
    outlet_T = {}
    heat_out = 0.0
    for b in branches:
        if b.to_node in boundary_pressure_Pa and b.to_node not in ("source", "inlet"):
            m = flows[b.name]
            if m > 0 and T.get(b.to_node) is not None:
                outlet_T[b.to_node] = T[b.to_node]
                heat_out += m * CP_WATER * (T[b.to_node] - inlet_temp_C)

    if not notes:
        notes.append("all branches within applicability range")
    return NetworkSolution(
        converged=converged, iterations=it_total,
        node_pressures_Pa=p, branch_flows_kg_s={
            b.name: flows[b.name] for b in branches
        },
        branch_results=branch_results,
        inlet_flow_kg_s=inlet_flow,
        outlet_temps_C=outlet_T,
        heat_removed_W=heat_out,
        applicable=all_ok,
        notes="; ".join(notes),
    )


def network_from_manifold(manifold) -> list[Branch]:
    """Map a BranchingManifold geometry to network branches (D1)."""
    segs = manifold._segments()
    L_in = float(np.sum(np.linalg.norm(np.diff(segs[0], axis=0), axis=1)))
    L_b1 = float(np.sum(np.linalg.norm(np.diff(segs[1], axis=0), axis=1)))
    L_b2 = float(np.sum(np.linalg.norm(np.diff(segs[2], axis=0), axis=1)))
    D = 2.0 * manifold.channel_radius
    return [
        Branch("inlet", "source", "fork", L_in, D, minor_loss_K=0.3),
        Branch("branch1", "fork", "out1", L_b1, D, minor_loss_K=1.0),
        Branch("branch2", "fork", "out2", L_b2, D, minor_loss_K=1.0),
    ]


def network_from_helix(helix) -> list[Branch]:
    """Map a HelicalChannelGenerator to a single network branch."""
    D = 2.0 * helix.channel_radius
    return [
        Branch("helix", "source", "out", helix.centerline_length(), D,
               helix_radius_m=helix.radius, channel_type="helical"),
    ]
