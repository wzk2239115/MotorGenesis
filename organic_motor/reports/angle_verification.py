"""Off-grid angle verification: compare map interpolation vs direct solve.

The torque maps are computed at N equally-spaced angles.  The transient
interpolates between them.  This script verifies the interpolation by
solving at angles NOT in the map grid and comparing:

  T_direct(theta)   vs   T_interp(theta) = T0(theta) + sum_p T1_p(theta) * i_p

Reports:
  - mean torque error
  - peak torque error
  - waveform L2 error
  - torque ripple (peak-to-peak / mean)

Also runs a time-step refinement: dt and dt/2, comparing the final
state (angle, omega) and energy balance.
"""

from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "angle_verification"


def main(artifact_dir: str | None = None, n_map_angles: int = 6):
    from organic_motor.config3d import MotorConfig3D
    from organic_motor.construct.model_artifact import ModelArtifact
    from organic_motor.construct.startup_validation import _logits_from_densities
    from organic_motor.construct.transient_bridge import (
        extract_electrical_parameters, extract_fea_flux_linkage,
    )
    from organic_motor.experiments.motor3d_powered import (
        Powered3DSettings, compute_powered_maps, run_powered_transient,
    )
    from organic_motor.optimization.objective3d import forward3d_fields
    import jax.numpy as jnp

    if artifact_dir is None:
        artifact_dir = Path(__file__).parent.parent / "out" / "assembly"
    artifact_dir = Path(artifact_dir)

    if not (artifact_dir / "model_meta.json").exists():
        print(f"[angle verify] no artifact at {artifact_dir}")
        return

    print("[angle verify] loading artifact...")
    artifact = ModelArtifact.load(artifact_dir)
    import inspect
    valid_params = set(inspect.signature(MotorConfig3D.__init__).parameters.keys()) - {"self"}
    cfg_kwargs = {"shape": tuple(artifact.shape)}
    for k, v in artifact.config_dict.items():
        if k not in valid_params: continue
        if isinstance(v, list) and len(v) == 3: cfg_kwargs[k] = tuple(v)
        elif isinstance(v, (int, float, str, bool)): cfg_kwargs[k] = v
    cfg = MotorConfig3D(**cfg_kwargs)

    logits = _logits_from_densities(artifact, cfg)
    fields, mag = artifact.solver_fields(cfg)
    registry = artifact.centerline_registry or None

    flux_fea = extract_fea_flux_linkage(
        None, cfg, artifact.magnetization, fields=fields, registry=registry)
    electrical = extract_electrical_parameters(
        None, cfg, flux_linkage_fea=flux_fea,
        registry=registry, copper_fraction=artifact.densities.get("rho_copper"))

    n_turns_override = int(registry[0].get("n_turns", 1)) if registry else None
    p_settings = Powered3DSettings(
        phase_voltage_peak=24.0,
        phase_resistance=electrical.phase_resistance,
        phase_inductance=electrical.phase_inductance,
        flux_linkage=electrical.flux_linkage,
        control_mode="current_control", i_q_ref_A=8.0,
        load_torque=5e-3, steps=2000, dt=2e-5,
        thermal_coupling="coupled",
    )

    elec_period = 2.0 * np.pi / cfg.pole_pairs
    angles_map = np.linspace(0, elec_period, n_map_angles, endpoint=False)

    def phase_solver(single, angle, amplitudes):
        return forward3d_fields(
            cfg, fields, mag, [angle], single,
            phase_amplitudes=amplitudes, centerline_registry=registry)

    print(f"[angle verify] computing maps at {n_map_angles} angles...")
    maps = compute_powered_maps(
        cfg, logits, None, mag, angles_map, p_settings,
        phase_solver=phase_solver, include_mechanics=False,
        n_turns_override=n_turns_override, filter_harmonics=True,
        centerline_registry=registry)

    psi_map = float(maps.get("psi_from_map", 0.0))
    if psi_map > 1e-8:
        from dataclasses import replace as _replace
        p_settings = _replace(p_settings, flux_linkage=psi_map)
        maps.pop("_scan", None)

    # --- Off-grid angle verification ---
    # Solve at angles BETWEEN the map angles (off-grid)
    n_off = 5
    angles_off = np.array([
        elec_period * (i + 0.5) / n_map_angles for i in range(n_off)
    ])

    print(f"[angle verify] solving {n_off} off-grid angles...")
    # For each off-grid angle, solve with phase A at +1 amplitude
    from organic_motor.optimization.objective3d import _phase_belts
    full = jnp.asarray(_phase_belts(cfg))
    zero = jnp.zeros_like(full[0])
    single_a = jnp.stack([full[0], zero, zero])
    one = jnp.asarray([1.0, 0.0, 0.0])

    torques_direct = []
    torques_interp = []
    nominal = float(np.mean(np.abs(np.asarray(maps["nominal_current"]))))
    period = float(maps["period"])

    for angle in angles_off:
        r = phase_solver(single_a, float(angle), one)
        torques_direct.append(float(r.torques[0]))
        # Interpolated: T0(angle) + T1_a(angle) * (1.0 / nominal)
        from organic_motor.experiments.motor3d_powered import _interp_uniform
        t0 = _interp_uniform(np.asarray(maps["torque_cogging"]), angle, period)
        t1 = _interp_uniform(np.asarray(maps["torques_ph"][0]), angle, period)
        torques_interp.append(float(t0 + t1 * 1.0))
        print(f"  theta={angle:.4f} direct={torques_direct[-1]:+.5f} "
              f"interp={torques_interp[-1]:+.5f} "
              f"err={abs(torques_direct[-1]-torques_interp[-1]):.5f}")

    t_dir = np.array(torques_direct)
    t_int = np.array(torques_interp)
    err = np.abs(t_dir - t_int)
    mean_err = float(np.mean(err))
    peak_err = float(np.max(err))
    l2_err = float(np.sqrt(np.mean((t_dir - t_int)**2)))
    ripple_direct = float((np.ptp(t_dir) / max(abs(np.mean(t_dir)), 1e-9)))

    print(f"\n[angle verify] mean error: {mean_err:.5f} Nm")
    print(f"[angle verify] peak error: {peak_err:.5f} Nm")
    print(f"[angle verify] L2 error:   {l2_err:.5f} Nm")
    print(f"[angle verify] ripple:     {ripple_direct*100:.1f}%")

    # --- Time-step refinement ---
    print("\n[angle verify] time-step refinement (dt vs dt/2)...")
    maps["temperature_init"] = jnp.full(cfg.shape, float(cfg.ambient_temperature), dtype=jnp.float32)

    from dataclasses import replace as _replace
    s_dt = _replace(p_settings, steps=2000, dt=2e-5)
    s_dt2 = _replace(p_settings, steps=4000, dt=1e-5)

    maps_dt = dict(maps); maps_dt.pop("_scan", None)
    maps_dt2 = dict(maps); maps_dt2.pop("_scan", None)

    d_dt = run_powered_transient(maps_dt, s_dt, cfg, 0.0)
    d_dt2 = run_powered_transient(maps_dt2, s_dt2, cfg, 0.0)

    # Compare final state
    omega_dt = float(d_dt["angular_velocity_rad_s"][-1])
    omega_dt2 = float(d_dt2["angular_velocity_rad_s"][-1])
    angle_dt = float(d_dt["rotor_angle_rad"][-1])
    angle_dt2 = float(d_dt2["rotor_angle_rad"][-1])

    # Energy balance
    def energy_err(d, s):
        dt = s.dt
        e_elec = float(np.sum(d["electrical_power_W"]) * dt)
        e_joule = float(np.sum(d["copper_power_RL_W"]) * dt)
        e_mech = float(np.sum(d["mechanical_power_W"]) * dt)
        i = d["currents_A"][1:]
        d_mag = 0.5 * s.phase_inductance * float(np.sum(i[-1]**2))
        return abs(e_mech + e_joule + d_mag - e_elec) / max(abs(e_elec), 1e-9)

    e_dt = energy_err(d_dt, s_dt)
    e_dt2 = energy_err(d_dt2, s_dt2)

    print(f"  dt:     omega={omega_dt:.4f} angle={angle_dt:.4f} energy_err={e_dt*100:.2f}%")
    print(f"  dt/2:   omega={omega_dt2:.4f} angle={angle_dt2:.4f} energy_err={e_dt2*100:.2f}%")
    print(f"  d_omega={abs(omega_dt-omega_dt2):.6f} d_angle={abs(angle_dt-angle_dt2):.6f}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "design_hash": artifact.design_hash,
        "n_map_angles": n_map_angles,
        "n_off_grid": n_off,
        "off_grid": {
            "mean_error_Nm": mean_err,
            "peak_error_Nm": peak_err,
            "l2_error_Nm": l2_err,
            "ripple_pct": ripple_direct * 100,
            "torques_direct": torques_direct,
            "torques_interp": torques_interp,
        },
        "time_step_refinement": {
            "dt": {"omega": omega_dt, "angle": angle_dt, "energy_err_pct": e_dt * 100},
            "dt_half": {"omega": omega_dt2, "angle": angle_dt2, "energy_err_pct": e_dt2 * 100},
            "omega_diff": abs(omega_dt - omega_dt2),
            "angle_diff": abs(angle_dt - angle_dt2),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "angle_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[angle verify] written to {OUT}")
    return report


if __name__ == "__main__":
    main()
