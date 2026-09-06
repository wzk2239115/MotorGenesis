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
import subprocess
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "energy_audit"

from tests.test_powered_control import _synthetic_maps, _base_settings  # noqa: E402


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


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


def run_artifact_audit(artifact_dir: str | Path, steps: int = 4000):
    """Run the energy ledger on the REAL assembly artifact (not synthetic).

    Loads model_meta.json + checkpoint NPZ, computes solver fields and
    powered maps, then runs the same scenarios as the synthetic audit.
    Requires GPU and takes ~2 min for map solves.
    """
    from organic_motor.construct.model_artifact import ModelArtifact
    from organic_motor.config3d import MotorConfig3D
    from organic_motor.construct.startup_validation import _logits_from_densities
    from organic_motor.construct.transient_bridge import (
        extract_electrical_parameters, extract_fea_flux_linkage,
    )
    from organic_motor.experiments.motor3d_powered import (
        Powered3DSettings, compute_powered_maps, run_powered_transient,
    )
    import inspect
    import jax.numpy as jnp

    artifact = ModelArtifact.load(Path(artifact_dir))
    can_run, reasons = artifact.can_energize()
    if not can_run:
        return {"error": "artifact rejected", "reasons": reasons}

    valid_params = set(inspect.signature(MotorConfig3D.__init__).parameters.keys()) - {"self"}
    cfg_kwargs = {"shape": tuple(artifact.shape)}
    for k, v in artifact.config_dict.items():
        if k not in valid_params:
            continue
        if isinstance(v, list) and len(v) == 3:
            cfg_kwargs[k] = tuple(v)
        elif isinstance(v, (int, float, str, bool)):
            cfg_kwargs[k] = v
    cfg = MotorConfig3D(**cfg_kwargs)

    logits = _logits_from_densities(artifact, cfg)
    fields, mag = artifact.solver_fields(cfg)
    registry = artifact.centerline_registry or None

    flux_fea = extract_fea_flux_linkage(
        None, cfg, artifact.magnetization,
        fields=fields, registry=registry)
    electrical = extract_electrical_parameters(
        None, cfg, flux_linkage_fea=flux_fea,
        registry=registry, copper_fraction=artifact.densities.get("rho_copper"))

    n_turns_override = int(registry[0].get("n_turns", 1)) if registry else None

    p_settings = Powered3DSettings(
        phase_voltage_peak=24.0,
        phase_resistance=electrical.phase_resistance,
        phase_inductance=electrical.phase_inductance,
        flux_linkage=electrical.flux_linkage,
        control_mode="current_control",
        i_q_ref_A=8.0,
        load_torque=5e-3,
        steps=steps,
        dt=2.0e-5,
        thermal_coupling="coupled",
    )

    import numpy as np
    n_map_angles = 6
    elec_period = 2.0 * np.pi / cfg.pole_pairs
    angles_map = np.linspace(0, elec_period, n_map_angles, endpoint=False)

    from organic_motor.optimization.objective3d import forward3d_fields

    def phase_solver(single, angle, amplitudes):
        return forward3d_fields(
            cfg, fields, mag, [angle], single,
            phase_amplitudes=amplitudes,
            centerline_registry=registry,
        )

    maps = compute_powered_maps(
        cfg, logits, None, mag,
        angles_map, p_settings,
        phase_solver=phase_solver,
        include_mechanics=False,
        n_turns_override=n_turns_override,
        filter_harmonics=True,
        centerline_registry=registry,
    )

    psi_map = float(maps.get("psi_from_map", 0.0))
    if psi_map > 1e-8:
        from dataclasses import replace as _replace
        p_settings = _replace(p_settings, flux_linkage=psi_map)
        maps.pop("_scan", None)

    maps["temperature_init"] = jnp.full(cfg.shape, float(cfg.ambient_temperature), dtype=jnp.float32)

    # Run scenarios
    scenarios = []
    for name, ov in [
        ("no_load", {"load_torque": 0.0}),
        ("load_step", {"load_torque": 8e-3}),
        ("power_off", {"power_off_at_s": 0.05, "load_torque": 0.0}),
    ]:
        s = _replace(p_settings, **ov) if ov else p_settings
        maps_copy = dict(maps)
        maps_copy.pop("_scan", None)
        data = run_powered_transient(maps_copy, s, cfg, 0.0)
        scenarios.append({"scenario": name, "settings": {
            "voltage": s.phase_voltage_peak, "load": s.load_torque,
            "dt": s.dt, "steps": s.steps, "control": s.control_mode,
            "thermal_coupling": s.thermal_coupling,
        }, **ledger(data, s)})

    return {
        "design_hash": artifact.design_hash,
        "psi_fea_Wb": electrical.flux_linkage,
        "psi_map_Wb": psi_map,
        "n_turns": n_turns_override,
        "scenarios": scenarios,
    }


def main(cfg_shape=(6, 6, 6), run_artifact=False):
    from organic_motor.config3d import MotorConfig3D
    cfg = MotorConfig3D(shape=cfg_shape)
    t0 = time.time()
    git_hash = _git_hash()
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
        "git_commit": git_hash,
        "model": f"synthetic maps {cfg_shape} (deterministic; the real "
                 "96^3 motor is gated by map-quality checks)",
        "scenarios": scenarios,
        "wall_time_s": time.time() - t0,
    }

    # --- Real artifact audit (if requested) ---
    artifact_dir = Path(__file__).parent.parent / "out" / "assembly"
    if run_artifact and (artifact_dir / "model_meta.json").exists():
        print("[energy audit] running artifact audit on assembly...")
        t_art = time.time()
        try:
            art = run_artifact_audit(artifact_dir, steps=4000)
            art["wall_time_s"] = time.time() - t_art
            report["artifact_audit"] = art
            print(f"[energy audit] artifact done in {art['wall_time_s']:.0f}s")
        except Exception as e:
            report["artifact_audit"] = {"error": str(e)}
            print(f"[energy audit] artifact failed: {e}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "energy_ledger.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# Energy Audit (git {git_hash})",
        "",
        "## Synthetic Model",
        "",
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

    if "artifact_audit" in report and "scenarios" in report.get("artifact_audit", {}):
        art = report["artifact_audit"]
        lines.extend([
            "",
            f"## Real Assembly Artifact (hash {art.get('design_hash', '?')})",
            f"- ψ_FEA = {art.get('psi_fea_Wb', 0):.5f} Wb",
            f"- ψ_map = {art.get('psi_map_Wb', 0):.5f} Wb",
            f"- n_turns = {art.get('n_turns', '?')}",
            "",
            "| 场景 | 电路误差 | 转子误差 | E_elec | E_conv | W_load | 末速 |",
            "|---|---|---|---|---|---|---|",
        ])
        for s in art["scenarios"]:
            ce = f"{s['circuit_rel_err']*100:.2f}%" \
                if s["circuit_rel_err"] is not None else "n/a"
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
    import sys
    run_art = "--artifact" in sys.argv
    main(run_artifact=run_art)
