"""Startup validation experiment for a constructed motor.

Level-3 verification: from multiple initial rotor angles, with a
current-limited drive and a load, does the motor accelerate forward
without reversal, overcurrent or overtemperature?

Builds the field-driven motor at physics resolution, extracts the
electrical parameters (R, L, flux linkage) from the actual geometry,
converts the design to powered-transient inputs and runs the multi-angle
startup suite.

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
    ap.add_argument("--torque-radius", type=float, default=0.0287,
                    help="Maxwell stress surface radius [m] (must be in the air gap)")
    ap.add_argument("--out", type=Path, default=Path("startup_out"))
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    cfg = MotorConfig3D(
        shape=(56, 56, 36),
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
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    (args.out / "startup.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print()
    verdict = "CAN SPIN (simulation)" if result.passed else "CANNOT SPIN YET (simulation)"
    print(f"[startup] verdict: {verdict}")
    print(f"  all angles started : {result.all_started}")
    print(f"  any reversal       : {result.any_reversal}")
    print(f"  min final speed    : {result.min_final_speed_rad_s:.2f} rad/s "
          f"({result.min_final_speed_rad_s*60/(2*np.pi):.0f} rpm)")
    print(f"  peak current       : {result.max_startup_current_A:.1f} A")
    print(f"  max temperature    : {result.max_temperature_C:.1f} C")
    print(f"  report: {args.out / 'startup.json'}")


if __name__ == "__main__":
    main()
