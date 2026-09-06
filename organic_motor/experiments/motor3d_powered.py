"""Powered native-3D motor experiment.

This is a quasi-static field-map transient: several native three-dimensional
magnetostatic/conduction solutions are sampled over rotor angle, then
periodically interpolated while the lumped three-phase circuit and rotor
dynamics advance.  It is deliberately *not* a full time-domain eddy-current
solver.  Every spatial field remains an ``(Nx, Ny, Nz[, 3])`` volume.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.experiments.precision_study3d import reference_design3d
from organic_motor.geometry.domain3d import domain_masks3d
from organic_motor.geometry.grid3d import meshgrid3d
from organic_motor.optimization.objective3d import ForwardResult3D, forward3d
from organic_motor.physics.mechanics3d import (
    air_gap_collision_penalty,
    solve_linear_elasticity,
)
from organic_motor.physics.transient3d import (
    RotorState,
    ThreePhaseState,
    advance_rotor,
    advance_three_phase_rl,
    advance_voxel_temperature,
    load_torque,
    sinusoidal_back_emf,
    transient_iron_loss,
    transient_joule_loss,
)


@dataclass(frozen=True)
class Powered3DSettings:
    """Reduced-order drive and structural settings in SI units."""

    steps: int = 100
    dt: float = 1.0e-5
    phase_voltage_peak: float = 24.0
    phase_resistance: float = 0.4
    phase_inductance: float = 2.0e-3
    flux_linkage: float = 0.03
    current_limit: float | None = None  # A per phase; None = unlimited
    commutation_offset: float = 0.0
    # ^ phase-current angle relative to the rotor at which the mean torque
    #   peaks for THIS winding convention (maps T_p ~ cos(theta_e - alpha_p)
    #   and currents cos(theta_e + comm + s_p): the mean torque is
    #   proportional to cos(comm), so comm = 0 is the q-axis drive and
    #   +/-90 deg is the zero-torque (pure d-axis) point.
    load_torque: float = 0.0
    load_viscous: float = 1.0e-4
    rotor_inertia: float = 2.0e-4
    mechanical_maxiter: int = 100
    mechanical_tol: float = 1.0e-5
    cooling_coefficient: float = 2.0e4
    eddy_loss_coefficient: float = 1.0e-4
    # --- B3: average-inverter current controller (dq PI) ---
    control_mode: str = "open_loop"  # "open_loop" | "current_control"
    i_q_ref_A: float | None = None    # q-axis current reference [A]
    i_d_ref_A: float = 0.0            # d-axis current reference [A]
    current_bw_Hz: float = 500.0      # PI bandwidth [Hz] for auto gains
    voltage_limit_V: float | None = None  # inverter phase limit [V];
    #   default: phase_voltage_peak (ideal three-phase inverter)
    # --- B6: physical power disconnection ---
    power_off_at_s: float | None = None  # voltage zero after this time [s]
    # --- B5: copper resistance temperature feedback ---
    resistance_temp_coeff: float = 0.00393  # copper alpha [1/K]
    resistance_ref_temp_C: float = 20.0
    # --- D3: PM temperature feedback (NdFeB remanence coefficient) ---
    pm_temp_coeff: float = -0.0012  # dB/dT ~ -0.12 %/K
    pm_temp_ref_C: float = 20.0
    # --- over-EMF protection: open the contactor after this many
    # consecutive steps of total-voltage saturation (0 = disabled) ---
    overemf_trip_steps: int = 500
    # --- D7: coupled thermal boundaries (audit item 7) ---
    # "flat": legacy single sink to ambient via cooling_coefficient.
    # "coupled": air-gap surface <-> TRACKED gap-air node <-> end faces
    # <-> ambient; channel wall <-> TRACKED coolant node with through
    # flow m_dot*cp (JAX Darcy fixed point, refreshed every
    # flow_update_steps).  Solid-side heat integrals and fluid-side
    # enthalpy are both returned for the energy audit.
    thermal_coupling: str = "flat"
    flow_update_steps: int = 200
    pump_dp_Pa: float = 5.0e4
    channel_length_m: float = 0.85
    channel_diameter_m: float = 0.003
    gap_air_heat_capacity_J_K: float = 1.5  # small trapped-air node
    coolant_inlet_temp_C: float = 40.0
    # --- D2: aerodynamic rotor load (windage) ---
    include_windage: bool = False
    windage_rotor_radius_m: float | None = None  # default R_sleeve_outer
    windage_rotor_length_m: float | None = None  # default 2*rotor_half_length
    windage_gap_m: float | None = None           # default air gap


def load_design3d(
    cfg: MotorConfig3D, path: str | Path | None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, str]:
    """Load optimization variables or sample the analytic 3-D reference."""
    if path is None:
        logits, rotor_logits, magnetization = reference_design3d(cfg)
        return logits, rotor_logits, magnetization, "reference_design3d"
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = ("logits", "rotor_logits", "magnetization_raw")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{source} is missing {', '.join(missing)}")
        logits = jnp.asarray(data["logits"])
        rotor_logits = jnp.asarray(data["rotor_logits"])
        magnetization = jnp.asarray(data["magnetization_raw"])
    expected = (4,) + cfg.shape
    if logits.shape != expected:
        raise ValueError(
            f"design shape {logits.shape} does not match configured {expected}"
        )
    if rotor_logits.shape != cfg.shape or magnetization.shape != (3,) + cfg.shape:
        raise ValueError("rotor ownership or magnetization has incompatible shape")
    return logits, rotor_logits, magnetization, str(source)


def periodic_interpolate(
    samples: np.ndarray, angles: Sequence[float], query: float, period: float
) -> np.ndarray:
    """Linearly interpolate scalar or volumetric samples on a periodic angle map."""
    values = np.asarray(samples)
    theta = np.mod(np.asarray(angles, dtype=float), period)
    order = np.argsort(theta)
    theta = theta[order]
    values = values[order]
    q = float(np.mod(query, period))
    upper = int(np.searchsorted(theta, q, side="right"))
    lower = (upper - 1) % len(theta)
    upper %= len(theta)
    a0 = theta[lower]
    a1 = theta[upper]
    if upper == 0:
        a1 += period
    if q < a0:
        q += period
    weight = 0.0 if a1 == a0 else (q - a0) / (a1 - a0)
    return (1.0 - weight) * values[lower] + weight * values[upper]


def material_fields3d(
    result: ForwardResult3D, cfg: MotorConfig3D
) -> dict[str, np.ndarray]:
    """Construct physical elastic, expansion, mass and heat-capacity volumes."""
    air = np.asarray(result.rho_air)
    iron = np.asarray(result.rho_iron)
    copper = np.asarray(result.rho_copper)
    pm = np.asarray(result.rho_pm)
    shaft = np.asarray(domain_masks3d(cfg)["shaft"], dtype=float)
    # The topology scaffold reserves the shaft outside the four phase fields.
    iron = np.maximum(iron, shaft)
    air = np.clip(1.0 - iron - copper - pm, 0.0, 1.0)
    fractions = np.stack((air, iron, copper, pm))

    def mix(values: Sequence[float]) -> np.ndarray:
        return np.tensordot(np.asarray(values), fractions, axes=(0, 0))

    return {
        "young_modulus": mix((1.0e6, 200.0e9, 110.0e9, 160.0e9)),
        "poisson_ratio": mix((0.25, 0.29, 0.34, 0.24)),
        "thermal_expansion": mix((0.0, 12.0e-6, 16.5e-6, 6.0e-6)),
        "mass_density": mix(
            (1.2, cfg.rho_iron_kg, cfg.rho_copper_kg, cfg.rho_pm_kg)
        ),
        "thermal_conductivity": mix(
            (
                cfg.thermal_k_air,
                cfg.thermal_k_iron,
                cfg.thermal_k_copper,
                cfg.thermal_k_pm,
            )
        ),
        "volumetric_heat_capacity": mix(
            (1.2e3, 3.6e6, 3.45e6, 3.4e6)
        ),
        "fractions": fractions,
    }


def _collision_diagnostics(
    displacement: np.ndarray, cfg: MotorConfig3D
) -> dict[str, float | bool]:
    x, y, z = (np.asarray(value) for value in meshgrid3d(cfg))
    cx, cy, cz = cfg.center
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    masks = {key: np.asarray(value) for key, value in domain_masks3d(cfg).items()}
    rotor = masks["rotor_design"]
    band = 1.5 * max(cfg.dx, cfg.dy)
    side = rotor & (radius >= cfg.R_rotor_outer - band)
    radial = np.stack(
        (
            (x - cx) / np.maximum(radius, 1e-12),
            (y - cy) / np.maximum(radius, 1e-12),
            np.zeros_like(radius),
        ),
        axis=-1,
    )
    radial_closure = np.sum(displacement * radial, axis=-1)
    radial_gap = cfg.R_stator_inner - cfg.R_rotor_outer

    end = rotor & (
        np.abs(np.abs(z - cz) - cfg.rotor_half_length) <= 1.5 * cfg.dz
    )
    axial_closure = displacement[..., 2] * np.sign(z - cz)
    side_min = (
        float(np.min(radial_gap - radial_closure[side]))
        if np.any(side)
        else radial_gap
    )
    end_min = (
        float(np.min(cfg.axial_airgap - axial_closure[end]))
        if np.any(end)
        else cfg.axial_airgap
    )
    contact = side | end
    signed_gap = np.where(side, radial_gap, cfg.axial_airgap)
    normal = np.where(
        side[..., None],
        -radial,
        np.stack(
            (np.zeros_like(z), np.zeros_like(z), -np.sign(z - cz)), axis=-1
        ),
    )
    penalty = air_gap_collision_penalty(
        displacement,
        signed_gap,
        normal,
        penalty_stiffness=1.0,
        spacing=cfg.spacing,
        contact_mask=contact,
    )
    minimum = min(side_min, end_min)
    return {
        "minimum_gap_m": minimum,
        "minimum_radial_gap_m": side_min,
        "minimum_axial_gap_m": end_min,
        "collision": bool(minimum <= 0.0),
        "collision_penalty": float(penalty),
    }


def _printed_series_coils(cfg: MotorConfig3D) -> int:
    """Coils of one phase wired in series in the printed stator."""
    n_slots = 12
    return max(1, n_slots // 3)


def _nominal_phase_current(result: ForwardResult3D, cfg: MotorConfig3D) -> float:
    phase_j = np.asarray(result.phase_current_density)
    z_index = cfg.Nz // 2
    currents = np.sum(np.maximum(phase_j[:, :, :, z_index, 2], 0.0), axis=(1, 2))
    currents *= cfg.dx * cfg.dy
    return max(float(np.mean(np.abs(currents))), 1.0e-6)


def _plot_outputs(data: dict[str, np.ndarray], path: Path) -> None:
    mid = data["temperature_final"].shape[2] // 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(data["map_angles_rad"], data["torque_map_Nm"], "o-")
    axes[0, 0].set(xlabel="mechanical angle [rad]", ylabel="torque [N m]")
    axes[0, 1].plot(data["time_s"], data["angular_velocity_rad_s"])
    axes[0, 1].set(xlabel="time [s]", ylabel="speed [rad/s]")
    axes[1, 0].plot(data["time_s"], data["currents_A"])
    axes[1, 0].set(xlabel="time [s]", ylabel="phase current [A]")
    image = axes[1, 1].imshow(
        data["temperature_final"][:, :, mid].T, origin="lower", aspect="equal"
    )
    axes[1, 1].set_title("final mid-plane temperature [C]")
    fig.colorbar(image, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _single_phase_current(result, phase: int, cfg: MotorConfig3D) -> float:
    """Phase current [A] of a CONSTANT-amplitude unit solve.

    Integrates the phase's positive-side Jz over the mid-plane -- NOT the
    three-phase mean (which divides by three when the other two phases are
    deliberately zeroed) and NOT angle-dependent (the amplitude is fixed).
    """
    phase_j = np.asarray(result.phase_current_density)
    z_index = cfg.shape[2] // 2
    jz_pos = np.maximum(phase_j[phase, :, :, z_index, 2], 0.0)
    return max(float(np.sum(jz_pos)) * cfg.dx * cfg.dy, 1.0e-6)


def compute_powered_maps(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization_raw: jnp.ndarray,
    angles: Sequence[float],
    settings: Powered3DSettings,
    *,
    phase_solver=None,
    base_belts=None,
    include_mechanics: bool = True,
    keep_volumes: bool = True,
    phases: Sequence[int] = (0, 1, 2),
    progress=None,
    include_cross_terms: bool = False,
    n_turns_override: int | None = None,
    filter_harmonics: bool = False,
    centerline_registry: list | None = None,
) -> dict:
    """Full torque decomposition T0/T1/T2 via sign- and zero-current solves.

    ``progress(done, total, detail)`` (optional) is called before every
    solve so callers can surface observability/cancellation.

    Maxwell stress is quadratic in B, so torque at phase current ``a`` mixes
    T(a) = T0 + a*T1 + a^2*T2 (cogging + linear PMxI coupling + current-self
    terms) and CANNOT be scaled linearly by the transient's actual currents.
    The decomposition (per angle, all solves at CONSTANT amplitude):

        zero-current solve         -> T0(theta)   PM-only cogging
        per phase p, +1 and -1     -> T1_p  = (T+ - T-)/2   linear PM x I
                                    -> T2_pp = (T+ + T-)/2 - T0   self I^2

    The excitation is NOT modulated by cos(p*theta): the transient applies
    the real currents itself (modulating twice manufactures a
    2x-electrical-frequency artefact).  The transient rotor equation uses
    T0 + sum_p T1_p*i_p + sum_p T2_pp*i_p^2; phase-to-phase cross terms
    T2_pq*i_p*i_q are NOT solved for (they need pair solves) and are a
    documented limitation of this model.

    ``phase_solver`` overrides the default ``forward3d`` (e.g. realized
    constructed fields); ``base_belts`` supplies the winding's own belts.
    ``keep_volumes=False`` drops the volumetric maps (transient cannot run)
    -- used by the mesh-convergence check, which only needs the scalars.
    ``phases`` restricts which phase maps are solved (the convergence
    check needs only phase A's T1 plus T0).
    """
    from organic_motor.optimization.objective3d import _phase_belts, forward3d

    if phase_solver is None:
        def phase_solver(belts, angle, amplitudes):
            return forward3d(
                cfg, logits, rotor_logits, magnetization_raw, [angle],
                cfg.sm_temp_final, phase_belts_override=belts,
                phase_amplitudes=amplitudes,
            )

    full = jnp.asarray(base_belts if base_belts is not None else _phase_belts(cfg))
    zero = jnp.zeros_like(full[0])
    singles = [
        jnp.stack([full[p] if q == p else zero for q in range(3)])
        for p in range(3)
    ]
    zero_belts = jnp.zeros_like(full)
    one = jnp.asarray([1.0, 0.0, 0.0])
    plus_amp = {p: jnp.roll(one, p) for p in range(3)}
    minus_amp = {p: -plus_amp[p] for p in range(3)}
    zero_amp = jnp.zeros((3,))

    na = len(angles)
    t_lin = np.zeros((3, na), dtype=np.float64)
    t_static = np.zeros((3, na), dtype=np.float64)
    t0_map = np.zeros(na, dtype=np.float64)
    j_maps_ph = (
        np.zeros((3, na) + cfg.shape + (3,), dtype=np.float32)
        if keep_volumes else None
    )
    b_map = np.zeros((na,) + cfg.shape + (3,), dtype=np.float32) if keep_volumes else None
    temperature_map = (
        np.zeros((na,) + cfg.shape, dtype=np.float32) if keep_volumes else None
    )
    nominal = np.zeros(3, dtype=np.float64)
    last_result = None

    total_solves = na * (1 + 2 * len(phases))
    done = 0

    for i, angle in enumerate(angles):
        if progress:
            progress(done, total_solves, f"cogging θ{i + 1}/{na}")
        r_zero = phase_solver(zero_belts, float(angle), zero_amp)
        t0_map[i] = float(r_zero.torques[0])
        done += 1
        print(f"    [maps] zero-I angle {i + 1}/{na} T0={t0_map[i]:+.4f}", flush=True)

    for p in phases:
        currents = []
        for i, angle in enumerate(angles):
            if progress:
                progress(done, total_solves,
                         f"phase {['A', 'B', 'C'][p]} θ{i + 1}/{na} ±1×nominal")
            r_plus = phase_solver(singles[p], float(angle), plus_amp[p])
            done += 1
            if progress:
                progress(done, total_solves,
                         f"phase {['A', 'B', 'C'][p]} θ{i + 1}/{na} −1×nominal")
            r_minus = phase_solver(singles[p], float(angle), minus_amp[p])
            done += 1
            t_plus = float(r_plus.torques[0])
            t_minus = float(r_minus.torques[0])
            t_lin[p, i] = 0.5 * (t_plus - t_minus)
            t_static[p, i] = 0.5 * (t_plus + t_minus)
            if keep_volumes:
                if p == 0:
                    b_map[i] = np.asarray(r_plus.flux_density)
                    temperature_map[i] = np.asarray(r_plus.temperature)
                j_maps_ph[p, i] = np.asarray(r_plus.phase_current_density)[p]
            currents.append(_single_phase_current(r_plus, p, cfg))
            last_result = r_plus
            print(f"    [maps] phase {p} angle {i + 1}/{na} "
                  f"T+={t_plus:+.4f} T-={t_minus:+.4f} "
                  f"lin={t_lin[p, i]:+.4f} quad={(t_static[p, i] - t0_map[i]):+.4f}",
                  flush=True)
        nominal[p] = float(np.mean(currents))
        # The printed winding's four coil loops are wired in SERIES (N = 4
        # turns): the terminal current equals the per-coil loop current,
        # while the positive-side integral sums ALL FOUR loops at the unit
        # map.  Without this division the map's "nominal current" is ~4x
        # the terminal current and the transient sees a 4x-deflated torque
        # per ampere (measured: motor crawled at 0.06 rad/s from standstill).
        if getattr(cfg, "winding_style", "printed") == "printed":
            n_series = _printed_series_coils(cfg)
            # P5: multiply by turns per cell — n_turns_override (from the
            # centerline registry, typically 7) replaces the cfg attribute
            # which defaults to 1.  Without this the nominal current is 7x
            # too large, i_norm 7x too small, and the torque per terminal
            # amp is 7x deflated.
            n_turns_per_cell = (n_turns_override if n_turns_override is not None
                                else max(1, getattr(cfg, "_n_turns_per_cell", 1)))
            n_series *= max(1, n_turns_per_cell)
            nominal[p] /= max(1, n_series)

    t2_diag = t_static - t0_map[None, :]  # (3, na) self I^2 coefficients

    # Optional cross terms (audit item 6): pair solves at (+1, +1)
    # separate the neglected T2_pq.  T_pair = T0 + T1_p + T1_q + T2_pp +
    # T2_qq + T2_pq, so T2_pq = T_pair - (everything already known).
    t2_cross = None
    if include_cross_terms:
        t2_cross = np.zeros((3, na), dtype=np.float64)  # rows: (0,1),(0,2),(1,2)
        pair_index = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
        for (p, q) in ((0, 1), (0, 2), (1, 2)):
            if p not in phases or q not in phases:
                continue
            for i, angle in enumerate(angles):
                if progress:
                    pi = pair_index[(p, q)]
                    progress(done, total_solves,
                             f"pair {['A', 'B', 'C'][p]}{['A', 'B', 'C'][q]} "
                             f"θ{i + 1}/{na} (+1,+1)")
                belts_pq = jnp.stack([
                    full[r] if r in (p, q) else zero for r in range(3)
                ])
                amp_pq = plus_amp[p] + plus_amp[q]
                r_pair = phase_solver(belts_pq, float(angle), amp_pq)
                done += 1
                t_pair = float(r_pair.torques[0])
                t2_cross[pair_index[(p, q)], i] = (
                    t_pair - t0_map[i] - t_lin[p, i] - t_lin[q, i]
                    - t2_diag[p, i] - t2_diag[q, i]
                )
        total_solves = done  # extend for reporting

    period = 2.0 * np.pi / cfg.pole_pairs
    map_angles = np.mod(np.asarray(angles, dtype=float), period)

    # --- Derive psi from the torque map (audit item 6: emf/torque
    # consistency).  The FEA ∮A·dl flux linkage includes END-TURN
    # LEAKAGE that doesn't produce air-gap torque; the map-derived psi
    # captures only the torque-producing (mutual) component and is
    # CONSISTENT with the maps by construction.  Using this psi for the
    # back-EMF makes ∫emf*i == ∫T*omega (energy balance).
    elec_angles = cfg.pole_pairs * map_angles
    psi_map_amps = []
    psi_map_phases = []
    for p in range(3):
        a = 2.0 / na * np.sum(t_lin[p] * np.cos(elec_angles))
        b = 2.0 / na * np.sum(t_lin[p] * np.sin(elec_angles))
        psi_map_amps.append(float(np.sqrt(a * a + b * b)))
        psi_map_phases.append(float(np.arctan2(b, a)))
    i_nom_mean = float(np.mean(np.abs(nominal)))
    # psi = K / (p * i_nom) — derived from matching the map-based torque
    # sum_p T1_p * i_p/i_nom to the standard PMSM torque 1.5*p*psi*Iq.
    # The 1.5 factor cancels because the 3-phase sum_p cos^2 = 3/2 on
    # BOTH sides of the equation.  The previous formula had an erroneous
    # extra 1.5 in the denominator, making psi 1.5x too small.
    psi_from_map = float(np.mean(psi_map_amps)) / max(
        cfg.pole_pairs * i_nom_mean, 1e-12)

    # --- Save RAW maps (before filtering) for diagnostics ---
    maps_raw_t1 = t_lin.copy()
    maps_raw_t0 = t0_map.copy()

    # --- Harmonic filter: keep only the FUNDAMENTAL (1st electrical
    # harmonic) of T0 and T1, zeroing all higher harmonics.  The
    # reduced-order model's back-EMF is purely sinusoidal (1st
    # harmonic); without this filter the map's slot/discretisation
    # harmonics contribute to torque but have NO corresponding emf,
    # breaking the energy balance by 15-25% on a 96^3 grid.
    # The filter makes ∫emf*i == ∫T*omega (audit item 6).
    if filter_harmonics and na >= 4:
        elec_ang = cfg.pole_pairs * map_angles
        cos_e = np.cos(elec_ang)
        sin_e = np.sin(elec_ang)
        for p in range(3):
            a = 2.0 / na * np.sum(t_lin[p] * cos_e)
            b = 2.0 / na * np.sum(t_lin[p] * sin_e)
            t_lin[p] = a * cos_e + b * sin_e  # fundamental only
        a0 = 2.0 / na * np.sum(t0_map * cos_e)
        b0 = 2.0 / na * np.sum(t0_map * sin_e)
        t0_map = a0 * cos_e + b0 * sin_e
        # T2 (reluctance) kept as-is: it's a 2nd-harmonic effect that
        # doesn't couple to the emf model anyway; its magnitude is ~0.

    materials = material_fields3d(last_result, cfg)
    masks = domain_masks3d(cfg)
    mechanics = None
    if include_mechanics:
        coordinates = jnp.stack(meshgrid3d(cfg), axis=-1)
        rotor_mass = materials["mass_density"] * np.asarray(masks["rotor_design"])
        mechanics = solve_linear_elasticity(
            materials["young_modulus"],
            materials["poisson_ratio"],
            np.asarray(masks["fixed_shaft"]) | np.asarray(masks["boundary"]),
            spacing=cfg.spacing,
            thermal_expansion=materials["thermal_expansion"],
            temperature_change=np.mean(temperature_map, axis=0) - cfg.ambient_temperature,
            density=rotor_mass,
            coordinates=coordinates,
            angular_velocity=cfg.speed_rpm * 2.0 * np.pi / 60.0,
            rotation_center=jnp.asarray(cfg.center),
            maxiter=settings.mechanical_maxiter,
            tol=settings.mechanical_tol,
        )
    maps = {
        "map_angles": map_angles,
        "period": period,
        "torques_ph": t_lin,
        "torque_static": t_static,
        "torque_cogging": t0_map,       # T0: zero-current PM-only torque
        "torque_i2_diag": t2_diag,      # T2_pp: per-phase self I^2 torque
        "torque_i2_cross": t2_cross,    # T2_pq: pair cross terms (or None)
        "psi_from_map": psi_from_map,    # torque-consistent flux linkage [Wb]
        "psi_map_per_phase": psi_map_amps,  # per-phase T1 fundamental [Nm]
        "psi_map_phases": psi_map_phases,   # per-phase T1 phase [rad]
        "maps_raw_t1": maps_raw_t1,         # pre-filter T1 maps (3, na)
        "maps_raw_t0": maps_raw_t0,         # pre-filter T0 maps (na,)
        "filter_harmonics": filter_harmonics,  # whether filtering was applied
        "j_maps_ph": jnp.asarray(j_maps_ph) if keep_volumes else None,
        "b_map": jnp.asarray(b_map) if keep_volumes else None,
        "temperature_map": temperature_map,
        "temperature_init": jnp.asarray(temperature_map.mean(axis=0)) if keep_volumes
        else jnp.zeros(cfg.shape),
        "materials": materials,
        "masks": masks,
        "mechanics": mechanics,
        "nominal_current": jnp.asarray(nominal),
        "_settings_for_scan": settings,
    }

    # --- Analytical copper-loss heat map (grid-independent).
    # The transient solver deposits copper heat as I²ρL/A along the
    # centerline, NOT as |J_grid|²/σ which is grid-dependent and
    # underestimates by ~7x on a 96³ grid (tent kernel footprint vs
    # wire cross-section).  q_joule_ph[p] is the volumetric heat
    # density [W/m³] when phase p alone carries I_per_turn; the
    # transient scales it by i_norm[p]² (loss ∝ I²).
    if centerline_registry is not None and keep_volumes:
        from organic_motor.optimization.line_current import _deposit_joule_heat
        I_per_turn = float(cfg.current_density_peak
                           * centerline_registry[0]["cross_section_area"])
        q_joule_ph = np.zeros((3,) + cfg.shape, dtype=np.float32)
        for p in range(3):
            amp = np.zeros(3)
            amp[p] = 1.0
            q_joule_ph[p] = _deposit_joule_heat(
                cfg, centerline_registry, I_per_turn, amp)
        maps["q_joule_ph"] = jnp.asarray(q_joule_ph)
    return maps


def _interp_uniform(arr: jnp.ndarray, q: jnp.ndarray, period: float) -> jnp.ndarray:
    """JIT-friendly periodic interpolation on a uniform angle grid."""
    na = arr.shape[0]
    step = period / na
    x = jnp.mod(q, period) / step
    i0 = jnp.floor(x).astype(jnp.int32)
    w = x - i0.astype(x.dtype)
    i0m = jnp.mod(i0, na)
    i1m = jnp.mod(i0 + 1, na)
    return (1.0 - w) * arr[i0m] + w * arr[i1m]


def _make_transient_scan(maps: dict, settings: Powered3DSettings, cfg: MotorConfig3D):
    """Build a jitted full-transient scan; initial_angle is a traced arg.

    The 4'000-step loop runs as ONE jax.lax.scan on the GPU: field-map
    interpolation, RL circuit, rotor dynamics and the voxel temperature
    advance all fuse into a single compiled kernel, eliminating the
    per-step Python dispatch that dominated the old loop.

    Control modes:
      ``open_loop``: sinusoidal voltage at fixed amplitude (historical).
      ``current_control``: average-inverter model — dq Park transform of
      measured currents, PI regulators (bandwidth-tuned, conditional
      anti-windup integration, magnitude voltage saturation), inverse
      Park to phase voltages.  This replaces clip-based fake control.

    Physics feedback:
      - Copper phase resistance scales with mean winding temperature
        (R = R_ref * (1 + alpha*(T_cu - T_ref))).
      - Optional physical power disconnection at ``power_off_at_s``.
      - Explicit-Euler thermal stability is verified at build time (D4).

    Energy accounting (B4): per-step electrical input power sum(v*i) and
    converted mechanical power em_torque*omega are returned so callers
    can close the energy balance in post-processing.
    """
    p = cfg.pole_pairs
    period = maps["period"]
    torques_ph = jnp.asarray(maps["torques_ph"], dtype=jnp.float32)   # (3, na)
    torques_t0 = jnp.asarray(maps["torque_cogging"], dtype=jnp.float32)  # (na,)
    torques_t2 = jnp.asarray(maps["torque_i2_diag"], dtype=jnp.float32)  # (3, na)
    _cross_raw = maps.get("torque_i2_cross")
    torques_t2_cross = (
        jnp.asarray(_cross_raw, dtype=jnp.float32)
        if _cross_raw is not None else None
    )  # (3, na): pairs (0,1),(0,2),(1,2)
    j_maps_ph = jnp.asarray(maps["j_maps_ph"], dtype=jnp.float32)     # (3, na, X,Y,Z,3)
    b_map = jnp.asarray(maps["b_map"], dtype=jnp.float32)             # (na, X,Y,Z,3)
    temperature_init = jnp.asarray(maps["temperature_init"], dtype=jnp.float32)
    materials = maps["materials"]
    masks = maps["masks"]
    nominal_current = jnp.asarray(maps["nominal_current"], dtype=jnp.float32)  # (3,)
    phase_shifts = jnp.asarray((0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0), dtype=jnp.float32)
    copper_fraction = jnp.asarray(materials["fractions"][2], dtype=jnp.float32)
    iron_fraction = jnp.asarray(materials["fractions"][1], dtype=jnp.float32)
    sigma = jnp.asarray(cfg.sigma_copper * jnp.maximum(copper_fraction, 1.0e-6), dtype=jnp.float32)
    # Analytical copper heat map (grid-independent I²ρL/A deposited along
    # centerline).  When present, the transient uses this instead of
    # |J_grid|²/σ which is grid-dependent and underestimates by ~7x.
    q_joule_ph = jnp.asarray(maps.get("q_joule_ph")) if maps.get("q_joule_ph") is not None else None
    k_thermal = jnp.asarray(materials["thermal_conductivity"], dtype=jnp.float32)
    c_vol = jnp.asarray(materials["volumetric_heat_capacity"], dtype=jnp.float32)
    cooling_mask = jnp.asarray(masks["boundary"], dtype=jnp.float32)
    dt = settings.dt
    steps = int(settings.steps)
    V = settings.phase_voltage_peak
    R = settings.phase_resistance
    L = max(settings.phase_inductance, 1.0e-9)
    psi = settings.flux_linkage
    i_lim = settings.current_limit
    comm = settings.commutation_offset
    J = settings.rotor_inertia
    load_const = settings.load_torque
    load_visc = settings.load_viscous
    cell_volume = cfg.cell_volume
    k_hyst = cfg.iron_loss_coeff / cfg.iron_loss_B_ref**2
    k_eddy = settings.eddy_loss_coefficient
    ambient = cfg.ambient_temperature
    cool = settings.cooling_coefficient
    spacing = cfg.spacing

    # --- D7: coupled thermal boundary geometry (audit item 7) ---
    thermal_coupling = settings.thermal_coupling
    if thermal_coupling not in ("flat", "coupled"):
        raise ValueError(f"unknown thermal_coupling {thermal_coupling!r}")
    if thermal_coupling == "coupled":
        # Air-gap surface mask: solid voxels within 2 cells of the gap.
        _X, _Y, _Z = meshgrid3d(cfg)
        _cx, _cy, _cz = cfg.center
        _r = np.sqrt((_X - _cx) ** 2 + (_Y - _cy) ** 2)
        r_in = float(getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer))
        r_out = float(cfg.R_stator_inner)
        band = 2.0 * float(min(spacing))
        gap_mask_np = (
            (np.asarray(materials["fractions"][1]) > 0.1)
            & (_r > r_in - band) & (_r < r_out + band)
        ).astype(np.float32)
        # Channel wall mask: from the artifact coolant density if the
        # caller injected it (maps["masks"]["coolant"]), else empty.
        cool_np = np.asarray(masks.get("coolant", 0.0), dtype=np.float32)
        if cool_np.shape != cfg.shape:
            cool_np = np.zeros(cfg.shape, dtype=np.float32)
        dil = cool_np.copy()
        for ax in range(3):
            dil = np.maximum(dil, np.roll(cool_np, 1, ax))
            dil = np.maximum(dil, np.roll(cool_np, -1, ax))
        channel_mask_np = (
            (dil > 0.5) & (cool_np < 0.5)
            & (np.asarray(materials["fractions"][1]) > 0.1)
        ).astype(np.float32)
        gap_mask = jnp.asarray(gap_mask_np)
        channel_mask = jnp.asarray(channel_mask_np)
        gap_cell_count = max(float(gap_mask_np.sum()), 1.0)
        ch_cell_count = max(float(channel_mask_np.sum()), 1.0)
        r_gap = 0.5 * (r_in + r_out)
        gap_width = max(r_out - r_in, 1e-4)
        C_gap = float(settings.gap_air_heat_capacity_J_K)
        T_cool_in = float(settings.coolant_inlet_temp_C)
        flow_every = max(1, int(settings.flow_update_steps))
        L_ch = float(settings.channel_length_m)
        D_ch = float(settings.channel_diameter_m)
        pump_dp = float(settings.pump_dp_Pa)
        # water props (flow1d)
        rho_w, mu_w, cp_w, k_w = 992.0, 6.53e-4, 4179.0, 0.631
        A_ch = math.pi * D_ch ** 2 / 4.0
        pr_w = mu_w * cp_w / k_w

        def _h_gap_jnp(w):
            # Same math as physics.airgap.air_gap_convection (Nu =
            # max(1, 0.2*Ta^0.25), Ta = Re^2 * delta/r), inline for JAX.
            nu_air = 1.91e-5 / 1.127
            re = jnp.abs(w) * r_gap * gap_width / nu_air
            ta = re ** 2 * (gap_width / r_gap)
            nu_num = jnp.maximum(1.0, 0.2 * ta ** 0.25)
            return nu_num * 0.0276 / gap_width

        def _m_dot_jnp():
            # Darcy-Weisbach fixed point on the coolant branch (JAX,
            # traceable): dp = (f*L/D) rho v^2/2, f = Blasius/64-Re.
            v = jnp.asarray(0.5)
            for _ in range(8):
                re = rho_w * v * D_ch / mu_w
                f = jnp.where(re < 2300.0, 64.0 / jnp.maximum(re, 1.0),
                              0.316 * jnp.maximum(re, 1.0) ** -0.25)
                v = jnp.sqrt(2.0 * pump_dp * D_ch / (f * L_ch * rho_w))
            return rho_w * A_ch * v

        def _h_channel_jnp(m_dot):
            re = jnp.abs(m_dot) * D_ch / (A_ch * mu_w)
            nu = jnp.where(re < 2300.0, 3.66,
                           0.023 * jnp.maximum(re, 1.0) ** 0.8 * pr_w ** 0.4)
            return nu * k_w / D_ch

    # --- D4: explicit-Euler thermal stability limit ---
    # 3-D diffusion: dt < dx^2 / (2*ndim*alpha_max), alpha = k / (rho*cp).
    k_np = np.asarray(materials["thermal_conductivity"], dtype=np.float64)
    c_np = np.asarray(materials["volumetric_heat_capacity"], dtype=np.float64)
    alpha_max = float(np.max(k_np / np.maximum(c_np, 1.0e-9)))
    dx_min = float(min(spacing))
    dt_stable = dx_min ** 2 / (6.0 * max(alpha_max, 1e-12))
    if dt > dt_stable:
        raise ValueError(
            f"thermal explicit-Euler unstable: dt={dt:.2e}s exceeds "
            f"dt_stable={dt_stable:.2e}s (dx={dx_min:.2e}m, alpha_max={alpha_max:.2e}m2/s); "
            "reduce dt or coarsen the grid"
        )

    # --- B3: PI gains (bandwidth tuning, pole-zero cancellation ki/kp=R/L) ---
    control_mode = settings.control_mode
    if control_mode not in ("open_loop", "current_control"):
        raise ValueError(f"unknown control_mode {control_mode!r}")
    if control_mode == "current_control" and settings.commutation_offset != 0.0:
        raise ValueError(
            "current_control requires commutation_offset=0 (regulators act "
            "in the rotor dq frame; use i_q_ref sign for direction)"
        )
    w_bw = 2.0 * np.pi * settings.current_bw_Hz
    kp_ctrl = float(L * w_bw)
    ki_ctrl = float(R * w_bw)
    i_q_ref = float(settings.i_q_ref_A if settings.i_q_ref_A is not None else 0.0)
    i_d_ref = float(settings.i_d_ref_A)
    v_max = float(settings.voltage_limit_V if settings.voltage_limit_V is not None else V)
    power_off_at = settings.power_off_at_s
    alpha_R = settings.resistance_temp_coeff
    R_ref_temp = settings.resistance_ref_temp_C
    cu_weight = jnp.maximum(copper_fraction, 0.0)
    cu_total = jnp.maximum(jnp.sum(cu_weight), 1e-12)

    # --- D3: PM temperature feedback coefficients ---
    k_pm = settings.pm_temp_coeff
    pm_ref_temp = settings.pm_temp_ref_C
    pm_weight = jnp.maximum(jnp.asarray(materials["fractions"][3],
                                        dtype=jnp.float32), 0.0)
    pm_total = jnp.maximum(jnp.sum(pm_weight), 1e-12)

    # --- D2: windage via the SHARED formula (same math as
    # physics.airgap.rotor_windage — single source, no drift) ---
    include_windage = settings.include_windage
    # --- over-EMF trip: sustained total-voltage saturation opens the
    # contactor (real inverter protection; also bounds the known
    # reduced-order artifact where phase-lagged currents let the
    # angle-only torque maps extract unbounded energy above the bus) ---
    trip_steps = int(settings.overemf_trip_steps)
    trip_enabled = trip_steps > 0
    if include_windage:
        from organic_motor.physics.airgap import windage_torque_formula
        r_w = float(settings.windage_rotor_radius_m
                    if settings.windage_rotor_radius_m is not None
                    else getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer))
        l_w = float(settings.windage_rotor_length_m
                    if settings.windage_rotor_length_m is not None
                    else 2.0 * cfg.rotor_half_length)
        g_w = float(settings.windage_gap_m
                    if settings.windage_gap_m is not None
                    else (cfg.R_stator_inner
                          - getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer)))

        def windage_torque(w):
            total, _side, _disk = windage_torque_formula(
                jnp, w, r_w, l_w, g_w)
            return total
    else:
        def windage_torque(w):
            return jnp.asarray(0.0)

    two_thirds = 2.0 / 3.0

    def step(carry, idx):
        (angle, omega, currents, temperature, prev_b, v_d_int, v_q_int,
         trip_cnt, t_gap, t_cool, m_dot, h_ch, q_fl_acc) = carry
        t = idx * dt
        elec = p * angle + comm
        # Physical power disconnection (B6): voltage zero after cutoff.
        if power_off_at is None:
            powered = 1.0
        else:
            powered = (t < power_off_at).astype(jnp.float32)
        # Over-EMF latch: once tripped the contactor stays open.
        if trip_enabled:
            powered = powered * (trip_cnt < float(trip_steps))

        # --- winding resistance temperature feedback (B5) ---
        T_cu = jnp.sum(temperature * cu_weight) / cu_total
        R_t = R * (1.0 + alpha_R * (T_cu - R_ref_temp))

        # --- PM temperature feedback (D3): remanence scales psi and the
        # PM torque terms; reluctance terms unchanged (first-order model,
        # maps computed at reference temperature) ---
        T_pm = jnp.sum(temperature * pm_weight) / pm_total
        psi_t = psi * (1.0 + k_pm * (T_pm - pm_ref_temp))
        psi_scale = psi_t / max(psi, 1e-12)

        # Back-EMF is known analytically: use it as controller feedforward.
        back_emf = sinusoidal_back_emf(angle, omega, p, psi_t)

        # --- voltage source: PI current controller or open loop ---
        if control_mode == "current_control":
            # Axis convention (matches the FEA torque maps): the
            # torque-producing current component is the electrical-cos
            # axis, T1_p ~ cos(elec + s_p), so the "q" regulator drives
            # i_cos and the "d" regulator nulls the sin axis.
            cos_e = jnp.cos(elec + phase_shifts)
            sin_e = jnp.sin(elec + phase_shifts)
            # Axis projections (identity: (2/3)*sum cos^2 = (2/3)*sum sin^2
            # = 1, cross = 0): a voltage v_q*cos_e + v_d*sin_e drives
            # currents with cos-component v_q/R and sin-component v_d/R,
            # so the measurements use the SAME signs as the synthesis.
            i_q = two_thirds * jnp.sum(currents * cos_e)    # torque axis
            i_d = two_thirds * jnp.sum(currents * sin_e)    # reactive axis
            err_q = (i_q_ref - i_q) * powered
            err_d = (i_d_ref - i_d) * powered
            v_q_pi = kp_ctrl * err_q + v_q_int
            v_d_pi = kp_ctrl * err_d + v_d_int
            # Back-EMF feedforward lives on the q axis (cos component
            # p*omega*psi).  The INVERTER limit applies to the TOTAL
            # voltage (PI + feedforward): an average-model inverter
            # cannot synthesise more than the bus, so saturate the sum.
            emf_q = p * omega * psi_t
            v_q_tot = v_q_pi + emf_q
            v_d_tot = v_d_pi
            v_mag = jnp.sqrt(v_q_tot ** 2 + v_d_tot ** 2)
            scale = jnp.minimum(1.0, v_max / jnp.maximum(v_mag, 1e-12))
            v_q = v_q_tot * scale
            v_d = v_d_tot * scale
            # Conditional-integration anti-windup: freeze the integrator
            # when the TOTAL voltage saturates AND the PI error would
            # push further into saturation.
            wind_q = (v_mag <= v_max) | (err_q * v_q_pi <= 0.0)
            wind_d = (v_mag <= v_max) | (err_d * v_d_pi <= 0.0)
            v_q_int_new = powered * (v_q_int + ki_ctrl * err_q * dt * wind_q)
            v_d_int_new = powered * (v_d_int + ki_ctrl * err_d * dt * wind_d)
            # Phase voltages whose cos/sin components are v_q/v_d.
            voltage = powered * (v_q * cos_e + v_d * sin_e)
            # Over-EMF latch: monotone counter, never resets — once the
            # contactor opens it stays open for the whole run.
            if trip_enabled:
                saturated = (v_mag > v_max).astype(jnp.float32)
                trip_cnt_new = trip_cnt + saturated
            else:
                trip_cnt_new = trip_cnt
        else:
            voltage = V * powered * jnp.cos(elec + phase_shifts)
            v_q_int_new = v_q_int
            v_d_int_new = v_d_int
            trip_cnt_new = trip_cnt

        circuit = advance_three_phase_rl(
            ThreePhaseState(currents), voltage, back_emf, R_t, L, dt
        )
        # Power-off = open contactor: force phase currents to zero (the
        # spinning machine would otherwise generate into a short).
        currents = powered * circuit.currents
        if i_lim is not None:
            # Physical fault guard (not the control law): trip-level clamp.
            clamped = jnp.clip(currents, -i_lim, i_lim)
            currents = clamped - jnp.mean(clamped)
        # Per-phase current excitation: the commutation angle enters the
        # electromechanics through the ACTUAL phase currents against the
        # per-phase torque maps, not through a projection wave that is
        # collinear with the voltage (and therefore offset-invariant at
        # standstill).
        i_norm = currents / nominal_current
        torque_vec = jnp.stack([
            _interp_uniform(torques_ph[q], angle, period) for q in range(3)
        ])
        t2_vec = jnp.stack([
            _interp_uniform(torques_t2[q], angle, period) for q in range(3)
        ])
        # Full map-based torque decomposition (NOT just the linear term):
        #   em = T0(theta)                        PM-only cogging
        #      + sum_p T1_p(theta) * i_p          linear PM x current
        #      + sum_p T2_pp(theta) * i_p^2       current-self (reluctance)
        #      + sum_pq T2_pq(theta) * i_p * i_q  cross terms (when solved)
        # T0 and T1 scale with the PM remanence factor psi_scale (D3);
        # T2 terms do not.
        em_torque = (
            psi_scale * (
                _interp_uniform(torques_t0, angle, period)
                + jnp.sum(torque_vec * i_norm)
            )
            + jnp.sum(t2_vec * i_norm ** 2)
        )
        if torques_t2_cross is not None:
            in_ = i_norm
            cross = (
                _interp_uniform(torques_t2_cross[0], angle, period) * in_[0] * in_[1]
                + _interp_uniform(torques_t2_cross[1], angle, period) * in_[0] * in_[2]
                + _interp_uniform(torques_t2_cross[2], angle, period) * in_[1] * in_[2]
            )
            em_torque = em_torque + cross
        mech_load = load_torque(omega, constant=load_const, viscous=load_visc)
        aero = windage_torque(omega)           # aerodynamic braking (D2)
        load = mech_load + aero
        rotor = advance_rotor(
            RotorState(angle, omega), em_torque, load, J, dt
        )
        angle, omega = rotor.angle, rotor.angular_velocity
        mapped_j = jnp.sum(jnp.stack([
            _interp_uniform(j_maps_ph[q], angle, period) for q in range(3)
        ]) * i_norm[:, None, None, None, None], axis=0)
        mapped_b = _interp_uniform(b_map, angle, period)
        # Copper joule loss: use the ANALYTICAL heat map (I²ρL/A, grid-
        # independent) when available; fall back to grid J²/σ otherwise.
        if q_joule_ph is not None:
            q_joule = jnp.sum(
                jnp.stack([q_joule_ph[q] * i_norm[q] ** 2 for q in range(3)]),
                axis=0,
            )
        else:
            q_joule = transient_joule_loss(mapped_j, sigma, active_mask=copper_fraction)
        db_dt = (mapped_b - prev_b) / dt
        frequency = p * omega / (2.0 * jnp.pi)
        q_iron = transient_iron_loss(
            mapped_b, db_dt, k_hyst, k_eddy, frequency, iron_mask=iron_fraction
        )
        if thermal_coupling == "coupled":
            # Boundary-specific sinks (audit 7): the gap surface talks to
            # the TRACKED gap-air node (not straight to ambient), and the
            # channel wall talks to the TRACKED coolant node.  h is a
            # SURFACE coefficient; per-voxel exchange area ~ dx^2 and the
            # volumetric sink coefficient is h*dx^2/cell_volume = h/dx.
            h_gap_t = _h_gap_jnp(omega)
            refresh = (idx % flow_every) == 0
            m_dot_new = jnp.where(refresh, _m_dot_jnp(), m_dot)
            h_ch_new = jnp.where(refresh, _h_channel_jnp(m_dot_new), h_ch)
            dx_face = float(min(spacing)) ** 2
            segments = [
                (gap_mask, h_gap_t / dx_min, t_gap),
                (channel_mask, h_ch_new / dx_min, t_cool),
            ]
            q_gap_sink = jnp.sum(h_gap_t * gap_mask * (temperature - t_gap)) \
                * dx_face
            q_ch_sink = jnp.sum(h_ch_new * channel_mask
                                * (temperature - t_cool)) * dx_face
            temperature = jnp.asarray(
                advance_voxel_temperature(
                    temperature, q_joule + q_iron, k_thermal, c_vol,
                    spacing, dt, ambient_temperature=None,
                    cooling_segments=segments,
                ),
                dtype=jnp.float32,
            )
            # Gap-air node: heated by the solid, cooled through the end
            # faces to ambient (h_end from the Daily-Nece analogy, omega-
            # dependent, inline of physics.airgap.end_face_convection).
            nu_air_k = 1.91e-5 / 1.127
            re_d = jnp.abs(omega) * r_gap ** 2 / nu_air_k
            c_m = jnp.where(re_d < 3.0e5, 3.87 / jnp.sqrt(jnp.maximum(re_d, 1.0)),
                            0.146 * jnp.maximum(re_d, 1.0) ** -0.2)
            h_end = (c_m / 2.0 * 0.70 ** (2.0 / 3.0)) * 1.127 * (
                jnp.abs(omega) * r_gap * math.sqrt(2.0 / 3.0)) * 1005.0
            A_end = 2.0 * math.pi * r_gap ** 2
            t_gap_new = t_gap + dt * (
                q_gap_sink - h_end * A_end * (t_gap - ambient)
            ) / C_gap
            # Coolant node: finite-volume through-flow energy balance.
            # m_fluid*cp * dT/dt = q_solid + m_dot*cp*(T_in - T)
            # Implicit Euler for the outflow term (unconditionally stable):
            #   T_new = (T + dt*(q + m_dot*cp*T_in)/(m_fluid*cp))
            #          / (1 + dt*m_dot/m_fluid)
            # This replaces the quasi-steady T_out = T_in + q/(m_dot*cp)
            # which created an algebraic loop that oscillated explosively
            # in explicit time stepping.
            m_fluid = rho_w * A_ch * L_ch  # coolant mass in channel [kg]
            cp_w_val = 4179.0
            tau = dt * m_dot_new / jnp.maximum(m_fluid, 1e-9)
            t_cool_new = (
                t_cool + dt * (q_ch_sink + m_dot_new * cp_w_val * T_cool_in)
                / jnp.maximum(m_fluid * cp_w_val, 1e-9)
            ) / jnp.maximum(1.0 + tau, 1.0)
            # Diagnostic: track total energy transferred to coolant.
            q_fl_acc_new = q_fl_acc + q_ch_sink * dt
        else:
            m_dot_new = m_dot
            h_ch_new = h_ch
            q_gap_sink = jnp.asarray(0.0)
            q_ch_sink = jnp.asarray(0.0)
            q_fl_acc_new = q_fl_acc
            t_gap_new = t_gap
            t_cool_new = t_cool
            temperature = jnp.asarray(
                advance_voxel_temperature(
                    temperature, q_joule + q_iron, k_thermal, c_vol, spacing, dt,
                    ambient_temperature=ambient, cooling_coefficient=cool,
                    cooling_mask=cooling_mask,
                ),
                dtype=jnp.float32,
            )
        # Work-accounting convention: the semi-implicit rotor update is
        # omega_{k+1} = omega_k + (em - load)*dt/J, whose discrete work
        # identity is d(KE) = (em - load) * omega_{k+1} * dt for the LINEAR
        # integrator in exact arithmetic; float32 and the angle-dependent
        # maps make it approximate — the energy audit quantifies the
        # residual rather than assuming it is zero.
        outs = (
            angle, omega, currents, em_torque,
            jnp.sum(q_joule) * cell_volume, jnp.sum(q_iron) * cell_volume,
            jnp.max(temperature),
            jnp.sum(voltage * currents),   # port electrical input [W] (B4)
            em_torque * omega,             # converted mechanical [W]
            jnp.sum(currents ** 2) * R_t,  # circuit copper loss w/ R(T) [W]
            load * omega,                  # load + windage dissipation [W]
            aero * omega,                  # windage part only [W]
            q_gap_sink + q_ch_sink,        # solid heat removed [W] (D7)
            q_ch_sink,                     # fluid-side received [W] (D7)
            t_cool_new,                    # coolant outlet temp [degC] (D7)
        )
        return (angle, omega, currents, temperature, mapped_b,
                v_d_int_new, v_q_int_new, trip_cnt_new, t_gap_new,
                t_cool_new, m_dot_new, h_ch_new, q_fl_acc_new), outs

    @jax.jit
    def run(initial_angle: jnp.ndarray):
        init = (
            initial_angle, jnp.asarray(0.0), jnp.zeros(3), temperature_init,
            _interp_uniform(b_map, initial_angle, period),
            jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0),
            jnp.asarray(float(ambient)),      # t_gap
            jnp.asarray(float(settings.coolant_inlet_temp_C)),  # t_cool
            jnp.asarray(0.05), jnp.asarray(50.0), jnp.asarray(0.0),
        )
        final, hist = jax.lax.scan(step, init, jnp.arange(steps), length=steps)
        return final[3], hist

    return run


def run_powered_transient(maps: dict, settings: Powered3DSettings,
                          cfg: MotorConfig3D, initial_angle: float = 0.0) -> dict:
    """Run the (jitted, GPU) transient from ``initial_angle`` on shared maps.

    The jitted scan is CACHED in ``maps``: repeated calls (one per startup
    angle) reuse the single compiled kernel instead of recompiling.
    """
    scan = maps.get("_scan")
    if scan is None:
        scan = _make_transient_scan(maps, settings, cfg)
        maps["_scan"] = scan
    temperature_final, hist = scan(jnp.asarray(float(initial_angle)))
    (angle_h, speed_h, currents_h, torque_h, joule_h, iron_h, maxt_h,
     elec_h, mech_h, copper_h, loadp_h, windp_h, qsolid_h, qfluid_h,
     tcool_h,
     ) = (np.asarray(x) for x in hist)
    steps = int(settings.steps)
    rotor_angle = np.concatenate([[initial_angle], angle_h])
    speed = np.concatenate([[0.0], speed_h])
    currents = np.concatenate([np.zeros((1, 3)), currents_h])
    max_temperature = np.concatenate([[float(np.max(maps["temperature_init"]))], maxt_h])
    return {
        "time_s": np.arange(steps + 1) * settings.dt,
        "rotor_angle_rad": rotor_angle,
        "angular_velocity_rad_s": speed,
        "currents_A": currents,
        "transient_torque_Nm": torque_h,
        "transient_joule_power_W": joule_h,
        "transient_iron_power_W": iron_h,
        "electrical_power_W": elec_h,        # port input sum(v*i) (B4)
        "mechanical_power_W": mech_h,        # converted em_torque*omega
        "copper_power_RL_W": copper_h,       # circuit i^2*R(T) (audit 6)
        "load_power_W": loadp_h,             # load + windage dissipation
        "windage_power_W": windp_h,          # aerodynamic part only
        "solid_cooling_W": qsolid_h,         # heat leaving solid (D7)
        "fluid_received_W": qfluid_h,        # fluid-side gain (D7)
        "coolant_outlet_C": tcool_h,         # coolant outlet temp (D7)
        "max_temperature_C": max_temperature,
        "temperature_final": np.asarray(temperature_final),
    }


def run_powered3d(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization_raw: jnp.ndarray,
    angles: Sequence[float],
    settings: Powered3DSettings,
    *,
    phase_solver=None,
    base_belts=None,
    initial_angle: float = 0.0,
    include_mechanics: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run field maps, structural loading, and map-driven transient dynamics."""
    maps = compute_powered_maps(
        cfg, logits, rotor_logits, magnetization_raw, angles, settings,
        phase_solver=phase_solver, base_belts=base_belts,
        include_mechanics=include_mechanics,
    )
    transient = run_powered_transient(maps, settings, cfg, initial_angle)
    mechanics = maps["mechanics"]
    materials = maps["materials"]
    displacement = (
        np.asarray(mechanics.displacement) if mechanics is not None
        else np.zeros(cfg.shape + (3,))
    )
    collision = _collision_diagnostics(displacement, cfg)
    speed = transient["angular_velocity_rad_s"]
    max_temperature = transient["max_temperature_C"]
    # Balanced-excitation torque map synthesised from the per-phase maps
    # (for reporting; the transient itself uses the per-phase maps).
    elec_map = cfg.pole_pairs * maps["map_angles"] + cfg.electrical_phase_offset
    shifts = np.asarray((0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0))
    torque_map = np.sum(
        np.cos(elec_map[None, :] - shifts[:, None]) * maps["torques_ph"], axis=0
    )
    data = {
        "map_angles_rad": maps["map_angles"],
        "torque_map_Nm": torque_map,
        "torque_map_per_phase_Nm": maps["torques_ph"],
        "flux_density_map_T": np.asarray(maps["b_map"]),
        "temperature_map_C": maps["temperature_map"],
        "young_modulus_Pa": materials["young_modulus"],
        "poisson_ratio": materials["poisson_ratio"],
        "thermal_expansion_1_K": materials["thermal_expansion"],
        "mass_density_kg_m3": materials["mass_density"],
        "displacement_m": displacement,
        "von_mises_Pa": (
            np.asarray(mechanics.von_mises) if mechanics is not None
            else np.zeros(cfg.shape)
        ),
        "initial_angle_rad": initial_angle,
        **transient,
    }
    summary = {
        "model": "quasi-static field-map transient (jit scan, T0/T1/T2 maps)",
        "full_time_domain_eddy_current": False,
        "shape": list(cfg.shape),
        "angle_samples": len(maps["map_angles"]),
        "steps": settings.steps,
        "torque_mean_Nm": float(np.mean(torque_map)),
        "torque_ripple_peak_to_peak_Nm": float(np.ptp(torque_map)),
        "cogging_t0_amplitude_Nm": float(np.max(np.abs(maps["torque_cogging"]))),
        "t2_diag_amplitude_Nm": float(np.max(np.abs(maps["torque_i2_diag"]))),
        "maximum_displacement_m": float(np.linalg.norm(displacement, axis=-1).max()),
        "maximum_von_mises_Pa": float(np.asarray(data["von_mises_Pa"]).max()),
        "mechanical_relative_residual": (
            float(mechanics.relative_residual) if mechanics is not None else 0.0
        ),
        "final_speed_rad_s": float(speed[-1]),
        "final_max_temperature_C": float(max_temperature[-1]),
        **collision,
    }
    return data, summary


def _shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 3 or min(values) < 3:
        raise argparse.ArgumentTypeError("shape must be Nx,Ny,Nz with values >= 3")
    return values  # type: ignore[return-value]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--shape", type=_shape, default=None)
    ap.add_argument("--angles", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1.0e-5)
    ap.add_argument("--out", type=Path, default=Path("powered3d_out"))
    ap.add_argument("--maxwell-iters", type=int, default=120)
    ap.add_argument("--thermal-iters", type=int, default=240)
    ap.add_argument("--electric-iters", type=int, default=120)
    ap.add_argument("--mechanical-iters", type=int, default=100)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    shape = args.shape
    if shape is None and args.design is not None:
        with np.load(args.design, allow_pickle=False) as design:
            shape = tuple(int(value) for value in design["rotor_logits"].shape)
    cfg = MotorConfig3D(
        shape=(12, 12, 8) if shape is None else shape,
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=args.angles,
        maxwell_maxiter=args.maxwell_iters,
        thermal_maxiter=args.thermal_iters,
        electric_maxiter=args.electric_iters,
        n_theta=32,
        torque_n_z=12,
        torque_n_r=12,
    )
    if args.angles < 2:
        raise ValueError("--angles must be at least 2 for a periodic torque map")
    logits, rotor_logits, magnetization, source = load_design3d(cfg, args.design)
    period = 2.0 * np.pi / cfg.pole_pairs
    angles = np.arange(args.angles) * period / args.angles
    settings = Powered3DSettings(
        steps=args.steps,
        dt=args.dt,
        mechanical_maxiter=args.mechanical_iters,
    )
    data, summary = run_powered3d(
        cfg, logits, rotor_logits, magnetization, angles, settings
    )
    summary["design_source"] = source
    summary["settings"] = asdict(settings)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "powered3d.npz", **data)
    (args.out / "powered3d.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_outputs(data, args.out / "powered3d.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
