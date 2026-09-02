"""Entry point: ``python -m organic_motor.agent``."""

from organic_motor.agent.loop import run_loop


def main() -> None:
    import argparse

    from organic_motor.config3d import MotorConfig3D

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="48,48,32")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--angles", type=int, default=3)
    ap.add_argument("--maxwell-iters", type=int, default=120)
    ap.add_argument("--thermal-iters", type=int, default=240)
    ap.add_argument("--electric-iters", type=int, default=120)
    ap.add_argument(
        "--out",
        default=str(
            __import__("pathlib").Path(__file__).resolve().parent.parent / "out" / "agent"
        ),
    )
    ap.add_argument("--heuristic", action="store_true", help="no-LLM parametric fallback")
    args = ap.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    cfg = MotorConfig3D(
        shape=shape, excitation_mode="terminal", filt_radius=0.0,
        projection_beta=0.0, mechanical_angles=args.angles,
        maxwell_maxiter=args.maxwell_iters, thermal_maxiter=args.thermal_iters,
        electric_maxiter=args.electric_iters, n_theta=32, torque_n_z=16, torque_n_r=16,
    )
    run_loop(cfg, args.out, max_iters=args.iters, use_llm=not args.heuristic)


if __name__ == "__main__":
    main()
