"""Audit item 6: electromagnetic-circuit-mechanical energy ledger.

Runs reproducible scenarios on the SYNTHETIC map model (deterministic,
no GPU-quad solves) and writes a full energy account per scenario:

  E_elec   = ∫ sum(v·i) dt                  (port input, actual voltages)
  E_copper = ∫ sum(i²·R(T)) dt              (circuit copper, temp-dependent)
  E_conv   = ∫ em_torque·ω_post dt          (converted mechanical)
  ΔE_mag   = ½·L·Σi_end²                    (magnetic storage change)
  ΔKE      = ½·J·ω_end²
  W_load   = ∫ load·ω dt                    (mechanical + aerodynamic)
  W_wind   = ∫ windage·ω dt                 (aerodynamic part)
  E_voxjoule = ∫ voxel q_joule dt           (heat-solver copper input)
  second-order integrator residual = Σ ½J·Δω²

Identities checked:
  circuit:  E_elec ≈ E_copper + E_conv + ΔE_mag
  rotor:    E_conv ≈ ΔKE + W_load + residual
  copper:   E_copper vs E_voxjoule (two-path comparison)
Scenarios: zero-voltage, no-load start, load step, reversal, power-off
coast, dt-halving refinement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "energy_audit"

from tests.test_powered_control import _synthetic_maps, _base_settings  # noqa: E402


def ledger(data, settings):
    """Full energy account from one transient run."""
    dt = settings.dt
    i = data["currents_A"][1:]
    omega = data["angular_velocity_rad_s"]
    d_omega = np.diff(omega)
    e_elec = float(np.sum(data["electrical_power_W"]) * dt)
    e_copper = float(np.sum(data["copper_power_RL_W"]) * dt)
    e_conv = float(np.sum(data["mechanical_power_W"]) * dt)
    e_vox = float(np.sum(data["transient_joule_power_W"]) * dt)
    w_load = float(np.sum(data["load_power_W"]) * dt)
    w_wind = float(np.sum(data["windage_power_W"]) * dt)
    d_mag = 0.5 * settings.phase_inductance * float(np.sum(i[-1] ** 2))
    d_ke = 0.5 * settings.rotor_inertia * float(omega[-1] ** 2)
    resid2 = float(np.sum(0.5 * settings.rotor_inertia * d_omega ** 2))
    denom = max(abs(e_elec), 1e-6)  # below 1 uJ the ratio is 0/0 noise
    circuit_err = (abs(e_copper + e_conv + d_mag - e_elec) / denom
                   if abs(e_elec) > 1e-6 else None)
    rotor_err = abs(d_ke + w_load + resid2 - e_conv) / max(abs(e_conv), 1e-9)
    return {
        "E_elec_J": e_elec, "E_copper_J": e_copper, "E_conv_J": e_conv,
        "dE_mag_J": d_mag, "dKE_J": d_ke, "W_load_J": w_load,
        "W_windage_J": w_wind, "E_voxel_joule_J": e_vox,
        "integrator_residual_J": resid2,
        "circuit_rel_err": circuit_err, "rotor_rel_err": rotor_err,
        "copper_two_path_ratio": (e_vox / e_copper) if e_copper > 1e-9 else None,
        "final_omega_rad_s": float(omega[-1]),
    }


def run_scenario(cfg, name, **ov):
    maps = _synthetic_maps(cfg)
    s = _base_settings(**ov)
    from organic_motor.experiments.motor3d_powered import run_powered_transient
    data = run_powered_transient(maps, s, cfg, 0.0)
    return {"scenario": name, "settings": {
        "voltage": s.phase_voltage_peak, "load": s.load_torque,
        "dt": s.dt, "steps": s.steps, "control": s.control_mode,
    }, **ledger(data, s)}


def main(cfg_shape=(6, 6, 6)):
    from organic_motor.config3d import MotorConfig3D
    cfg = MotorConfig3D(shape=cfg_shape)
    t0 = time.time()
    scenarios = [
        run_scenario(cfg, "zero_voltage", phase_voltage_peak=0.0),
        run_scenario(cfg, "no_load_start", control_mode="current_control",
                     i_q_ref_A=8.0, load_torque=0.0, load_viscous=0.0),
        run_scenario(cfg, "load_step_high", control_mode="current_control",
                     i_q_ref_A=8.0, load_torque=8e-3, load_viscous=2e-3),
        run_scenario(cfg, "reversal", control_mode="current_control",
                     i_q_ref_A=-8.0, load_torque=0.0, load_viscous=1e-3),
        run_scenario(cfg, "power_off_coast", control_mode="current_control",
                     i_q_ref_A=8.0, power_off_at_s=0.05,
                     load_torque=0.0, load_viscous=1e-3),
        run_scenario(cfg, "dt_half", control_mode="current_control",
                     i_q_ref_A=8.0, load_torque=0.0, load_viscous=0.0,
                     steps=10000, dt=1.0e-5),
        run_scenario(cfg, "windage_on", control_mode="current_control",
                     i_q_ref_A=8.0, load_torque=0.0, load_viscous=0.0,
                     include_windage=True),
    ]
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": f"synthetic maps {cfg_shape} (deterministic; the real "
                 "96^3 motor is gated by map-quality checks)",
        "scenarios": scenarios,
        "wall_time_s": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "energy_ledger.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "| 场景 | 电路误差 | 转子误差 | E_elec | E_conv | W_load | 末速 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in scenarios:
        ce = f"{s['circuit_rel_err']*100:.2f}%" \
            if s["circuit_rel_err"] is not None else "n/a (无能量流动)"
        lines.append(
            f"| {s['scenario']} | {ce} | "
            f"{s['rotor_rel_err']*100:.2f}% | {s['E_elec_J']*1e3:.2f} mJ | "
            f"{s['E_conv_J']*1e3:.2f} mJ | {s['W_load_J']*1e3:.2f} mJ | "
            f"{s['final_omega_rad_s']:.1f} rad/s |")
    (OUT / "energy_ledger.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[energy audit] written to {OUT}")
    return report


if __name__ == "__main__":
    main()
