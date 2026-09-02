"""Rebuild the display checkpoint with the current code and score it.

Builds the field-driven motor at physics resolution for the critic score,
then at display resolution for the viewer checkpoint, saving to a
dedicated 'construct' run directory.
"""

import time

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.objects import field_driven_motor
from organic_motor.construct.critic import score_fields
from organic_motor.construct.export import save_checkpoint
from organic_motor.construct.startup_validation import constructed_design_from_mf


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
    dmf = field_driven_motor(dcfg).build()
    print(f"[rebuild] display built in {time.perf_counter()-t0:.0f}s")

    out = "organic_motor/out/construct/checkpoints/step_000000.npz"
    save_checkpoint(dmf, dcfg, out, step=0, metrics=metrics,
                    magnetization=motor.magnetization())
    print(f"[rebuild] saved {out}")


if __name__ == "__main__":
    main()
