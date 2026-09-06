"""Verify the artifact fix end-to-end: flux + full maps on the real motor.

Runs the EXACT web-thread sequence (solver_fields -> flux -> 42 map
solves) and prints the map gate numbers.
"""

import time
import numpy as np
import jax.numpy as jnp

from organic_motor.construct.model_artifact import ModelArtifact
from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.startup_validation import _logits_from_densities
from organic_motor.construct.transient_bridge import (
    extract_electrical_parameters, extract_fea_flux_linkage,
)
from organic_motor.optimization.objective3d import forward3d_fields, _phase_belts
from organic_motor.experiments.motor3d_powered import (
    Powered3DSettings, compute_powered_maps,
)

ART = "organic_motor/out/assembly"


def main():
    artifact = ModelArtifact.load(ART)
    import inspect
    valid = set(inspect.signature(MotorConfig3D.__init__).parameters) - {"self"}
    kw = {"shape": tuple(artifact.shape)}
    for k, v in artifact.config_dict.items():
        if k in valid:
            kw[k] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
    cfg = MotorConfig3D(**kw)
    logits = _logits_from_densities(artifact, cfg)
    fields, mag = artifact.solver_fields(cfg)
    registry = artifact.centerline_registry or None

    t0 = time.time()
    print("[verify] FEA flux linkage via solver_fields…", flush=True)
    flux = extract_fea_flux_linkage(None, cfg, artifact.magnetization,
                                    fields=fields, registry=registry)
    print(f"[verify] flux_linkage = {flux:.5f} Wb  ({time.time()-t0:.0f}s)",
          flush=True)
    electrical = extract_electrical_parameters(
        None, cfg, flux_linkage_fea=flux, registry=registry,
        copper_fraction=artifact.densities.get("rho_copper"))
    print(f"[verify] R={electrical.phase_resistance:.4f} "
          f"L={electrical.phase_inductance:.6f}", flush=True)

    p_settings = Powered3DSettings(
        phase_voltage_peak=24.0,
        phase_resistance=electrical.phase_resistance,
        phase_inductance=electrical.phase_inductance,
        flux_linkage=electrical.flux_linkage,
        control_mode="current_control", i_q_ref_A=10.0,
    )
    import sys
    n_turns_override = int(registry[0].get("n_turns", 1)) if registry else None
    print(f"[verify] n_turns_override = {n_turns_override}", flush=True)
    n_map_angles = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    elec_period = 2.0 * np.pi / cfg.pole_pairs
    angles_map = np.linspace(0, elec_period, n_map_angles, endpoint=False)

    def progress(done, total, detail):
        print(f"    [{done:2d}/{total}] {detail}", flush=True)

    def phase_solver(single, angle, amplitudes):
        return forward3d_fields(
            cfg, fields, mag, [angle], single,
            phase_amplitudes=amplitudes, centerline_registry=registry,
        )

    t1 = time.time()
    maps = compute_powered_maps(
        cfg, logits, None, mag, angles_map, p_settings,
        phase_solver=phase_solver, include_mechanics=False,
        progress=progress, n_turns_override=n_turns_override,
    )
    t1_max = float(np.max(np.abs(np.asarray(maps["torques_ph"]))))
    t2_max = float(np.max(np.abs(np.asarray(maps["torque_i2_diag"]))))
    t0_max = float(np.max(np.abs(np.asarray(maps["torque_cogging"]))))
    i_nom = float(np.max(np.abs(np.asarray(maps["nominal_current"]))))
    psi_map = float(maps.get("psi_from_map", 0.0))
    psi_fea = electrical.flux_linkage
    bound = 2.5 * 1.5 * cfg.pole_pairs * psi_fea * i_nom
    print(f"\n[verify] maps done in {time.time()-t1:.0f}s")
    print(f"[verify] T1_max={t1_max:.4f}  T2_max={t2_max:.6f}  "
          f"T0_max={t0_max:.4f}  I_nom={i_nom:.3f} A")
    print(f"[verify] psi_FEA={psi_fea:.6f} Wb  psi_map={psi_map:.6f} Wb")
    print(f"[verify] psi_FEA/psi_map = {psi_fea/max(psi_map,1e-12):.1f}  "
          f"(leakage fraction = {1-psi_map/max(psi_fea,1e-12):.1%})")
    print(f"[verify] physical bound (FEA) = {bound:.3f}")
    verdict = "PASS — below gate" if max(t1_max, t2_max) <= bound else \
        "FAIL — still above gate"
    print(f"[verify] gate: {verdict}")
    for p in range(3):
        t1p = np.asarray(maps["torques_ph"])[p]
        print(f"  phase {p}: T1 range [{t1p.min():+.4f}, {t1p.max():+.4f}] "
              f"mean {t1p.mean():+.4f}")


if __name__ == "__main__":
    main()
