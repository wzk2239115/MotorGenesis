"""Startup validation experiment for a constructed motor.

Level-3 verification: from multiple initial rotor angles, with a
current-limited drive and a load, does the motor accelerate forward
without reversal, overcurrent or overtemperature -- AND is the machine
behind the transient actually valid (winding / cooling / structure /
manufacturing / mesh convergence)?

The transient uses the full T0/T1/T2 torque decomposition (zero-current
cogging + linear PMxI + self-I^2), so a green spin can no longer hide an
unconquerable cogging torque.  Topology verdicts are evaluated on the
DISPLAY-resolution build (where the geometry resolves) and cross-checked
against the physics grid; ``--convergence`` additionally measures
quantitative torque stability between the physics grid and a 96^3
refinement.

Run with::

    MOTORGENESIS_X64=0 python -m organic_motor.experiments.construct_startup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import field_driven_motor
from organic_motor.construct.startup_validation import (
    constructed_design_from_mf,
    validate_startup,
)
from organic_motor.construct.transient_bridge import extract_electrical_parameters


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angles", type=int, default=4, help="startup initial angles")
    ap.add_argument("--map-angles", type=int, default=6, help="torque-map angles")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--dt", type=float, default=2.0e-5)
    ap.add_argument("--voltage", type=float, default=24.0)
    ap.add_argument("--current-limit", type=float, default=50.0)
    ap.add_argument("--load", type=float, default=0.005)
    ap.add_argument("--inertia", type=float, default=2.0e-4)
    ap.add_argument("--jpeak", type=float, default=5.0e6,
                    help="impressed current density peak [A/m^2]")
    ap.add_argument("--comm", type=float, default=3.1415927,
                    help="commutation offset [rad]; mean torque ~ cos(comm), "
                         "0/180 deg select the sign for this convention")
    ap.add_argument("--maxwell-iters", type=int, default=300,
                    help="Maxwell CG iterations (torque converges ~300)")
    ap.add_argument("--torque-radius", type=float, default=0.029,
                    help="Maxwell stress surface radius [m] (must be in the air gap)")
    ap.add_argument("--shape", type=_parse_shape, default=None,
                    help="physics grid Nx,Ny,Nz (default 96,96,58 where the "
                         "magnet+gap spans ~4.8 cells and the T1 ladder-step "
                         "96->112 agrees to +2.7%%; finer grids 128/160 "
                         "currently DIVERGE (helical-skew/grid moire + stress-"
                         "surface localisation -- see P3 notes); use 56,56,36 "
                         "for quick iterations)")
    ap.add_argument("--convergence", action="store_true",
                    help="measure quantitative torque convergence against a "
                         "96x96x58 refinement (adds ~45 min)")
    ap.add_argument("--out", type=Path, default=Path("startup_out"))
    return ap


def _parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3 or min(values) < 3:
        raise argparse.ArgumentTypeError("shape must be Nx,Ny,Nz with values >= 3")
    return values  # type: ignore[return-value]


def _fine_grid_amplitudes(cfg, settings, map_angles, fine_shape):
    """T0 rms + phase-A T1 amplitude at a REFINED grid (convergence ladder).

    The coarse side of the ladder is reused from the main run's maps, so
    only the fine-grid solves are extra.
    """
    from dataclasses import replace

    import jax.numpy as jnp

    from organic_motor.construct.realize import realize
    from organic_motor.optimization.objective3d import forward3d_fields
    from organic_motor.experiments.motor3d_powered import compute_powered_maps

    g = replace(cfg, shape=fine_shape)
    motor = field_driven_motor(g)
    m = motor.build()
    # Magnetization at the refined grid (a coarse-grid array would not
    # match the refined build).
    mag_g = motor.magnetization()
    fields, _mag = realize(m, g)
    netlist = m.metadata.get("winding_netlist")
    belts = None
    if netlist is not None:
        belts = jnp.asarray(netlist.phase_belts_3d(g))

    def phase_solver(single, angle, amplitudes_arg):
        return forward3d_fields(
            g, fields, mag_g, [angle], single, phase_amplitudes=amplitudes_arg,
        )

    maps = compute_powered_maps(
        g, None, None, mag_g, map_angles, settings,
        phase_solver=phase_solver, base_belts=belts,
        include_mechanics=False, keep_volumes=False,
        phases=(0,),  # phase-A T1 + T0 suffice for the convergence gate
    )
    t0 = np.asarray(maps["torque_cogging"], dtype=float)
    t1 = np.asarray(maps["torques_ph"], dtype=float)
    return {
        "t0_rms": float(np.sqrt(np.mean(t0 ** 2))),
        "t1_phase0_amp": float(np.max(np.abs(t1[0]))),
    }


def _patch_torque_convergence(result, conv):
    """Fold the quantitative torque ladder into the mesh-convergence verdict."""
    verdict = result.verdicts["verdicts"]["mesh_convergence"]
    detail = verdict["detail"]
    detail["torque"] = conv
    t1_change = abs(float(conv["t1_amplitude_change_pct"]))
    t0_change = abs(float(conv["t0_rms_change_pct"]))
    detail["torque_stable"] = bool(max(t1_change, t0_change) <= 5.0)
    topology_stable = bool(detail.get("topology_stable", False))
    verdict["passed"] = bool(topology_stable and detail["torque_stable"])
    suite = result.verdicts
    suite["failed"] = [k for k in suite["verdicts"]
                       if suite["verdicts"][k]["passed"] is False]
    suite["evaluated"] = sum(1 for k in suite["verdicts"]
                             if suite["verdicts"][k]["passed"] is not None)
    suite["passed"] = bool(
        not suite["failed"]
        and suite["verdicts"]["electromechanical"]["passed"] is True
        and suite["verdicts"]["winding"]["passed"] is True
        and suite["verdicts"]["structure"]["passed"] is True
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    shape = _parse_shape(args.shape) if args.shape else (96, 96, 58)
    cfg = MotorConfig3D(
        shape=shape,
        excitation_mode="impressed",
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=args.map_angles,
        current_density_peak=args.jpeak,
        R_torque=args.torque_radius,
        maxwell_maxiter=args.maxwell_iters,
        thermal_maxiter=160,
        electric_maxiter=80,
        n_theta=32,
        torque_n_z=16,
        torque_n_r=16,
    )

    print("[startup] constructing field-driven motor at physics resolution...")
    motor = field_driven_motor(cfg)
    mf = motor.build()
    mag = motor.magnetization()

    elec = extract_electrical_parameters(mf, cfg)
    print("[startup] electrical parameters from geometry:")
    print(f"  R  = {elec.phase_resistance:.4g} ohm")
    print(f"  L  = {elec.phase_inductance:.4g} H")
    print(f"  psi = {elec.flux_linkage:.4g} Wb   (N = {elec.n_turns_effective})")
    print(f"  copper volume = {elec.copper_volume_m3*1e6:.2f} cm^3")

    logits, rotor_logits, magnetization = constructed_design_from_mf(mf, cfg, mag)

    # Display-resolution build: topology verdicts gate on THIS grid (where
    # the geometry is actually resolved), with the physics grid reported
    # alongside -- the discrepancy feeds the mesh-convergence verdict.
    from dataclasses import replace

    display_cfg = replace(cfg, shape=(160, 160, 96))
    print(f"[startup] display build {display_cfg.shape} for topology verdicts...")
    display_mf = field_driven_motor(display_cfg).build()

    print(f"[startup] running {args.angles} initial angles, "
          f"{args.steps} steps x {args.dt*1e3:.1f} ms "
          f"(= {args.steps*args.dt*1e3:.0f} ms), I_limit = {args.current_limit} A")
    result = validate_startup(
        cfg, logits, rotor_logits, magnetization, mf,
        electrical=elec,
        n_angles=args.angles,
        n_map_angles=args.map_angles,
        steps=args.steps,
        dt=args.dt,
        voltage=args.voltage,
        current_limit=args.current_limit,
        commutation_offset=args.comm,
        load_torque=args.load,
        rotor_inertia=args.inertia,
        maxwell_maxiter=args.maxwell_iters,
        thermal_maxiter=160,
        electric_maxiter=80,
        display_mf=display_mf,
        display_cfg=display_cfg,
    )

    if args.convergence and result.verdicts is not None:
        from organic_motor.experiments.motor3d_powered import Powered3DSettings

        conv_settings = Powered3DSettings()
        period = 2.0 * np.pi / cfg.pole_pairs
        map_angles = np.arange(args.map_angles) * period / args.map_angles
        # Adjacent ladder step: 96->112 agreed on T1 (+2.7%%) but T0 rms
        # +768%%; 128/160 steps diverge on both (skew/grid moire) and are
        # recorded in the run report as known-unresolved.
        fine_shape = (112, 112, 66)
        print(f"[startup] torque-convergence refinement {fine_shape} "
              f"(phase-A + zero-I solves)...")
        fine = _fine_grid_amplitudes(cfg, conv_settings, map_angles, fine_shape)
        td = result.torque_decomposition
        conv = {
            "physics_shape": list(cfg.shape),
            "fine_shape": list(fine_shape),
            "t0_rms_physics_Nm": td["t0_rms_Nm"],
            "t0_rms_fine_Nm": fine["t0_rms"],
            "t0_rms_change_pct": 100.0 * (fine["t0_rms"] - td["t0_rms_Nm"])
            / max(td["t0_rms_Nm"], 1e-9),
            "t1_amplitude_physics_Nm": td["t1_amplitudes_Nm"][0],
            "t1_amplitude_fine_Nm": fine["t1_phase0_amp"],
            "t1_amplitude_change_pct": 100.0
            * (fine["t1_phase0_amp"] - td["t1_amplitudes_Nm"][0])
            / max(td["t1_amplitudes_Nm"][0], 1e-9),
        }
        print(f"  T1 amplitude change: {conv['t1_amplitude_change_pct']:+.1f}%")
        print(f"  T0 rms change      : {conv['t0_rms_change_pct']:+.1f}%")
        _patch_torque_convergence(result, conv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    (args.out / "startup.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print()
    print(f"[startup] torque decomposition: "
          f"T0 peak {result.torque_decomposition.get('t0_peak_Nm', 0):.4f} Nm "
          f"(harmonic {result.torque_decomposition.get('t0_dominant_harmonic_per_period', 0)}x/period), "
          f"T1 amps {[round(a, 4) for a in result.torque_decomposition.get('t1_amplitudes_Nm', [])]} Nm")
    if result.verdicts is not None:
        from organic_motor.construct.verdicts import format_verdict_table

        print("[startup] six independent verdicts:")
        print(format_verdict_table(result.verdicts))
        # The headline verdict is the SIX-verdict overall: a green spin
        # must not cover broken topology or non-converged discretisation.
        overall = bool(result.verdicts.get("passed", False))
    else:
        overall = result.passed
    verdict = "VALID MOTOR (six-verdict)" if overall else "NOT VALID YET (six-verdict)"
    print(f"[startup] verdict: {verdict}")
    print(f"  transient spins   : {result.passed}")
    print(f"  all angles started : {result.all_started}")
    print(f"  any reversal       : {result.any_reversal}")
    print(f"  min final speed    : {result.min_final_speed_rad_s:.2f} rad/s "
          f"({result.min_final_speed_rad_s*60/(2*np.pi):.0f} rpm)")
    print(f"  peak current       : {result.max_startup_current_A:.1f} A")
    print(f"  max temperature    : {result.max_temperature_C:.1f} C")
    print(f"  report: {args.out / 'startup.json'}")


if __name__ == "__main__":
    main()
