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


def extract_electrical_parameters(
    mf: MaterialField, cfg: MotorConfig3D,
    b_gap_mean: float = 1.0,
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

    copper_vol = float(np.sum(copper)) * cfg.cell_volume
    n_turns = netlist.turns_per_coil * netlist.n_layers
    slots_per_phase = netlist.n_slots // netlist.n_phases
    n_turns_total = n_turns * slots_per_phase

    stack_len = cfg.stack_length
    end_turn_arc = netlist.coil_span * netlist.slot_pitch * cfg.R_winding_inner
    mean_path = 2 * stack_len + 2 * end_turn_arc
    L_wire = n_turns_total * mean_path

    wire_area = copper_vol / max(L_wire, 1e-9)
    phase_resistance = L_wire / (cfg.sigma_copper * max(wire_area, 1e-12))

    mu0 = cfg.mu0
    mu_r = cfg.mu_r_iron
    iron_vol = float(np.sum(iron)) * cfg.cell_volume
    l_eff = stack_len
    A_eff = iron_vol / max(l_eff, 1e-9)
    phase_inductance = n_turns_total ** 2 * mu0 * mu_r * A_eff / max(l_eff, 1e-9) * 1e-3

    kw = 0.9
    A_pole = stack_len * cfg.R_stator_inner * np.pi / max(2 * cfg.pole_pairs, 1)
    flux_linkage = kw * b_gap_mean * A_pole * n_turns_total

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
