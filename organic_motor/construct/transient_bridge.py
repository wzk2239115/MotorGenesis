"""Bridge constructed motor checkpoints to the powered transient solver.

The powered transient (motor3d_powered.py) expects logits/rotor_logits/
magnetization_raw, while the construct layer saves rho_iron/copper/pm/air.
This module bridges that gap so an agent-generated motor can be run through
the dynamic startup simulation.

It also computes electrical parameters (R, L, flux linkage) from the actual
geometry instead of using hand-tuned constants, so the dynamic model
reflects the real winding and magnetic circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.winding_netlist import CoilNetlist


@dataclass
class ElectricalParameters:
    """Per-phase electrical parameters extracted from geometry."""

    phase_resistance: float
    phase_inductance: float
    flux_linkage: float
    n_turns_effective: int
    copper_volume_m3: float
    mean_path_length_m: float
    wire_cross_section_m2: float
    source: str = "geometry"


def load_constructed_checkpoint(
    cfg: MotorConfig3D, path: str | Path,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Load a constructed checkpoint as (logits, rotor_logits, magnetization, meta).

    Converts rho_* densities to the logits format the powered transient
    expects.  Since the construct layer already has hard 0/1 densities,
    the logits are large positive/negative values.
    """
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        rho_iron = np.asarray(data["rho_iron"], dtype=np.float32)
        rho_copper = np.asarray(data["rho_copper"], dtype=np.float32)
        rho_pm = np.asarray(data["rho_pm"], dtype=np.float32)
        rho_air = np.asarray(data["rho_air"], dtype=np.float32)
        magnetization = np.asarray(data.get("magnetization", np.zeros((3,) + cfg.shape, dtype=np.float32)))

    logits = np.zeros((4,) + cfg.shape, dtype=np.float32)
    logits[0] = rho_air * 10 - 5
    logits[1] = rho_iron * 10 - 5
    logits[2] = rho_copper * 10 - 5
    logits[3] = rho_pm * 10 - 5

    rotor_logits = np.asarray(rho_iron > 0.3, dtype=np.float32) * 10 - 5

    meta = {}
    for key in data.files:
        if key.startswith("metric__"):
            meta[key[8:]] = float(np.asarray(data[key]).ravel()[0])

    return jnp.asarray(logits), jnp.asarray(rotor_logits), jnp.asarray(magnetization), meta


def extract_fea_flux_linkage(
    mf, cfg: MotorConfig3D, magnetization_raw=None,
) -> float:
    """Extract PM flux linkage via mutual-energy integral over a full
    electrical cycle.

    Method:
      1. Run PM-only Maxwell solves (zero current) at several rotor
         angles spanning one electrical period (π/pole_pairs mech).
      2. At each angle, compute the mutual-energy flux linkage for each
         phase using the closed-loop line integral of A along the
         winding centerlines (now closed loops):

           λ_p(θ) = Σ_tooth polarity × ∮ A_PM(θ) · dl

         The integral is gauge-invariant because each centerline is a
         CLOSED loop (the serpentine includes a terminal return).

      3. Fit the fundamental component of λ_p(θ) vs θ for each phase.
         The amplitude is the PM flux linkage; the phase should be
         120° apart.

    Returns the mean per-phase flux linkage amplitude [Wb].
    """
    import jax.numpy as jnp
    import numpy as np
    from organic_motor.construct.realize import realize
    from organic_motor.optimization.objective3d import forward3d_fields

    reg = mf.metadata.get("centerline_registry") if hasattr(mf, "metadata") else None
    if not reg:
        return 0.0

    n_angles = 6  # sample one electrical period
    elec_period = 2.0 * np.pi / cfg.pole_pairs
    angles = np.linspace(0, elec_period, n_angles, endpoint=False)

    phase_fluxes = {0: [], 1: [], 2: []}
    fields, mag = realize(mf, cfg, magnetization_raw)

    for angle in angles:
        result = forward3d_fields(
            cfg, fields, mag, angles=[angle],
            phase_amplitudes=jnp.zeros(3),
            centerline_registry=reg,
        )
        A_vec = np.asarray(result.vector_potential)

        for entry in reg:
            pts = entry["points"]
            phase = entry["phase"]
            polarity = entry["polarity"]
            flux = 0.0
            for seg in range(len(pts)):
                p1 = pts[seg]
                p2 = pts[(seg + 1) % len(pts)]
                dl = p2 - p1
                mid = 0.5 * (p1 + p2)
                idx = ((mid - cfg.origin) / cfg.spacing).astype(int)
                i = int(np.clip(idx[0], 0, cfg.shape[0] - 1))
                j = int(np.clip(idx[1], 0, cfg.shape[1] - 1))
                k = int(np.clip(idx[2], 0, cfg.shape[2] - 1))
                A_mid = A_vec[i, j, k, :]
                flux += float(np.dot(A_mid, dl))
            phase_fluxes[phase].append(polarity * flux)

    # Fit fundamental: λ_p(θ) = a_p * cos(p*θ + φ_p)
    elec_angles = cfg.pole_pairs * angles
    amplitudes = []
    for p in range(3):
        # Data layout: phase_fluxes[p] = [angle0_tooth0, angle0_tooth1, ...,
        #   angle1_tooth0, ...].  Reshape to (n_angles, n_teeth_per_phase).
        raw = np.array(phase_fluxes[p])
        n_teeth = len(raw) // n_angles
        flux_by_angle = raw.reshape(n_angles, n_teeth).sum(axis=1)  # (n_angles,)
        if np.max(np.abs(flux_by_angle)) < 1e-12:
            amplitudes.append(0.0)
            continue
        # Fit: a*cos(ea) + b*sin(ea) → amplitude = sqrt(a²+b²)
        a = float(np.dot(flux_by_angle, np.cos(elec_angles)) / n_angles * 2)
        b = float(np.dot(flux_by_angle, np.sin(elec_angles)) / n_angles * 2)
        amplitudes.append(np.sqrt(a ** 2 + b ** 2))

    # Return mean amplitude (should be equal for balanced winding)
    return float(np.mean(amplitudes))


def extract_electrical_parameters(
    mf: MaterialField, cfg: MotorConfig3D,
    b_gap_mean: float = 0.0,
    flux_linkage_fea: float | None = None,
) -> ElectricalParameters:
    """Compute per-phase R, L, flux linkage from actual geometry.

    R = L_wire / (sigma * A_wire)
    L = N^2 * mu0 * A_eff / l_eff  (simplified reluctance)
    flux_linkage = N * kw * B_gap * A_pole

    These replace the hand-tuned constants in Powered3DSettings so the
    dynamic model reflects the actual winding and magnetic circuit.
    """
    densities = mf.to_densities()
    copper = densities["copper"]
    iron = densities["iron"]
    pm = densities["pm"]

    netlist = mf.metadata.get("winding_netlist") if hasattr(mf, "metadata") else None
    if not isinstance(netlist, CoilNetlist):
        netlist = CoilNetlist(n_slots=12, pole_pairs=cfg.pole_pairs)

    # P5 centerline registry: use analytical R and actual turn count
    centerline_registry = (
        mf.metadata.get("centerline_registry") if hasattr(mf, "metadata") else None
    )
    if centerline_registry:
        from organic_motor.optimization.line_current import centerline_resistance
        R_info = centerline_resistance(centerline_registry)
        # Serpentine: 12 entries (one per tooth), each with n_turns
        n_bands = centerline_registry[0].get("n_turns", 7)
        n_turns_total = n_bands * (netlist.n_slots // netlist.n_phases)
        phase_resistance = R_info["avg_phase_R"]
        # Analytical mean path from centerline lengths
        total_L = 0.0
        n_total_turns = 0
        for entry in centerline_registry:
            pts = entry["points"]
            for seg in range(len(pts) - 1):
                d = pts[seg + 1] - pts[seg]
                total_L += float(np.sqrt(d @ d))
            n_total_turns += entry.get("n_turns", n_bands)
        mean_path = total_L / max(n_total_turns, 1)
        wire_area = centerline_registry[0]["cross_section_area"]
        copper_vol = total_L * wire_area
    else:
        copper_vol = float(np.sum(copper)) * cfg.cell_volume
        n_turns = netlist.turns_per_coil
        slots_per_phase = netlist.n_slots // netlist.n_phases
        n_turns_total = n_turns * slots_per_phase

        stack_len = cfg.stack_length
        end_turn_arc = netlist.coil_span * netlist.slot_pitch * cfg.R_winding_inner
        mean_path = 2 * stack_len + 2 * end_turn_arc
        L_wire = n_turns_total * mean_path

        wire_area = copper_vol / max(L_wire, 1e-9)
        phase_resistance = L_wire / (cfg.sigma_copper * max(wire_area, 1e-12))

    # Synchronous inductance from air-gap geometry (no μr_iron — the
    # gap dominates, not the iron):
    # L_s = N² * μ₀ * A_pole / (2 * g_eff)
    # g_eff = R_stator_inner - R_sleeve_outer (physical air gap)
    stack_len = cfg.stack_length
    mu0 = cfg.mu0
    g_eff = cfg.R_stator_inner - getattr(cfg, "R_sleeve_outer", cfg.R_rotor_outer)
    A_pole = stack_len * cfg.R_stator_inner * np.pi / max(2 * cfg.pole_pairs, 1)
    phase_inductance = n_turns_total ** 2 * mu0 * A_pole / max(2 * g_eff, 1e-6)

    kw = 0.933  # 12s10p concentrated winding factor
    if flux_linkage_fea is not None:
        flux_linkage = flux_linkage_fea
    elif b_gap_mean > 0:
        flux_linkage = kw * b_gap_mean * A_pole * n_turns_total
    else:
        # No FEA value available; default to 0 (caller must supply FEA)
        flux_linkage = 0.0

    return ElectricalParameters(
        phase_resistance=phase_resistance,
        phase_inductance=phase_inductance,
        flux_linkage=flux_linkage,
        n_turns_effective=n_turns_total,
        copper_volume_m3=copper_vol,
        mean_path_length_m=mean_path,
        wire_cross_section_m2=wire_area,
    )


def make_powered_settings_from_geometry(
    mf: MaterialField, cfg: MotorConfig3D,
    steps: int = 2000,
    dt: float = 1.0e-5,
    voltage: float = 24.0,
    load_torque: float = 0.01,
    load_viscous: float = 1.0e-4,
    rotor_inertia: float = 2.0e-4,
) -> "Powered3DSettings":
    """Build Powered3DSettings with R/L/flux from actual geometry."""
    from organic_motor.experiments.motor3d_powered import Powered3DSettings
    params = extract_electrical_parameters(mf, cfg)
    return Powered3DSettings(
        steps=steps,
        dt=dt,
        phase_voltage_peak=voltage,
        phase_resistance=params.phase_resistance,
        phase_inductance=params.phase_inductance,
        flux_linkage=params.flux_linkage,
        load_torque=load_torque,
        load_viscous=load_viscous,
        rotor_inertia=rotor_inertia,
    )
