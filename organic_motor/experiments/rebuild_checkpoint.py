"""Rebuild the display checkpoint with the current code and score it.

Builds the field-driven motor at physics resolution for the critic score,
then at display resolution for the viewer checkpoint, saving to a
dedicated 'construct' run directory.

Consistency doctrine: the DISPLAY checkpoint (densities, magnetization,
topology metrics) is built entirely at display resolution -- the physics
build exists for the electromagnetic critic score and is reported
separately.  Mixing physics-grid metrics with a display-grid model in one
artifact is exactly the inconsistency the validation verdicts exist to
prevent.
"""

import time

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import field_driven_motor
from organic_motor.construct.critic import score_fields
from organic_motor.construct.export import save_checkpoint
from organic_motor.construct.startup_validation import constructed_design_from_mf  # noqa: F401


def main() -> None:
    cfg = MotorConfig3D(
        shape=(56, 56, 36), excitation_mode="impressed",
        filt_radius=0.0, projection_beta=0.0, mechanical_angles=3,
        maxwell_maxiter=80, thermal_maxiter=160, electric_maxiter=80,
        n_theta=32, torque_n_z=16, torque_n_r=16,
    )
    from dataclasses import replace
    dcfg = replace(cfg, shape=(160, 160, 96))

    print("[rebuild] physics build + score (56^3, impressed)...")
    t0 = time.perf_counter()
    motor = field_driven_motor(cfg)
    mf = motor.build()
    mag = motor.magnetization()
    metrics = score_fields(mf, cfg, mag)
    print(f"[rebuild] scored in {time.perf_counter()-t0:.0f}s: "
          f"obj={metrics.get('obj', float('nan')):.3g} "
          f"torque={metrics.get('torque', 0):.4g} Nm "
          f"copper_components={metrics.get('copper_components')} "
          f"air_gap_bridge={metrics.get('air_gap_iron_bridge')}")

    print("[rebuild] display build (160^3)...")
    t0 = time.perf_counter()
    dmotor = field_driven_motor(dcfg)
    dmf = dmotor.build()
    print(f"[rebuild] display built in {time.perf_counter()-t0:.0f}s")

    # Display-grid topology metrics: same geometry the viewer shows, so the
    # reported topology matches what is on screen.
    from organic_motor.construct.geometry_metrics import compute_geometry_metrics
    from organic_motor.construct.connectivity import connectivity_report
    from organic_motor.construct.phase_verify import verify_phase_connectivity

    display_metrics = {"display_shape": list(dcfg.shape)}
    display_metrics.update(compute_geometry_metrics(dmf, dcfg))
    display_metrics.update(connectivity_report(dmf, dcfg))
    display_metrics.update(verify_phase_connectivity(dmf, dcfg))
    metrics.update(display_metrics)

    out = "organic_motor/out/construct/checkpoints/step_000000.npz"
    # Magnetization at DISPLAY resolution: it is rendered with the display
    # densities and must match that grid.
    save_checkpoint(dmf, dcfg, out, step=0, metrics=metrics,
                    magnetization=dmotor.magnetization())
    print(f"[rebuild] saved {out}")
    print(f"[rebuild] display topology: "
          f"copper_components={display_metrics.get('copper_components')}, "
          f"phase=({display_metrics.get('phase_a_components')},"
          f"{display_metrics.get('phase_b_components')},"
          f"{display_metrics.get('phase_c_components')}), "
          f"structural={display_metrics.get('structural_components')}, "
          f"floating={display_metrics.get('floating_islands')}, "
          f"coolant_through={display_metrics.get('through_flow_networks')}, "
          f"trapped={display_metrics.get('trapped_voids')}")


if __name__ == "__main__":
    main()
