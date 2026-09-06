"""Diagnose the real-motor map blow-up: which (phase, sign, angle) solves
diverge, and does a higher linear-solver budget fix them?

Reproduces the exact web-thread path (artifact -> cfg -> realize ->
forward3d_fields with centerline registry), then re-runs the FAILING
solves with increasing maxwell_maxiter, reporting torque + residual.
"""

import sys
import time
import numpy as np
import jax.numpy as jnp

from organic_motor.construct.model_artifact import ModelArtifact
from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.startup_validation import (
    _logits_from_densities, _mf_from_artifact,
)
from organic_motor.construct.realize import realize
from organic_motor.optimization.objective3d import forward3d_fields, _phase_belts

ART = "organic_motor/out/assembly"
ANGLE_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # 6/6
PHASE = int(sys.argv[2]) if len(sys.argv) > 2 else 1        # phase B


def build():
    artifact = ModelArtifact.load(ART)
    import inspect
    valid = set(inspect.signature(MotorConfig3D.__init__).parameters) - {"self"}
    kw = {"shape": tuple(artifact.shape)}
    for k, v in artifact.config_dict.items():
        if k in valid:
            kw[k] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
    kw["maxwell_maxiter"] = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    cfg = MotorConfig3D(**kw)
    mf = _mf_from_artifact(artifact, cfg)
    fields, mag = realize(mf, cfg, artifact.magnetization)
    reg = artifact.centerline_registry or None
    logits = _logits_from_densities(artifact, cfg)
    return cfg, fields, mag, reg, logits


def solve(cfg, fields, mag, reg, belts, angle, amp):
    t0 = time.time()
    r = forward3d_fields(cfg, fields, mag, [angle], belts,
                         phase_amplitudes=amp, centerline_registry=reg)
    tor = float(np.asarray(r.torques).ravel()[0])
    res = float(np.asarray(r.maxwell_residual).ravel()
                [0] if np.asarray(r.maxwell_residual).ndim else r.maxwell_residual)
    return tor, float(res), time.time() - t0


def main():
    elec_period = 2.0 * np.pi / 5
    n_angles = 6
    angles = np.linspace(0, elec_period, n_angles, endpoint=False)
    angle = float(angles[ANGLE_IDX])
    print(f"[diag] angle idx {ANGLE_IDX+1}/6 = {np.degrees(angle):.1f} deg mech, "
          f"phase {PHASE}, maxiter sweep")

    full = _phase_belts(cfg) if False else None  # built after cfg
    for maxiter in (400, 800, 1600, 3200):
        cfg, fields, mag, reg = build_with_maxiter(maxiter)
        full = _phase_belts(cfg)
        zero = jnp.zeros_like(full[0])
        singles = [jnp.stack([full[p] if q == p else zero for q in range(3)])
                   for p in range(3)]
        one = jnp.asarray([1.0, 0.0, 0.0])
        plus = jnp.roll(one, PHASE)
        minus = -plus
        t_plus, r_plus, dt = solve(cfg, fields, mag, reg, singles[PHASE],
                                   angle, plus)
        t_minus, r_minus, dt2 = solve(cfg, fields, mag, reg, singles[PHASE],
                                      angle, minus)
        t_lin = 0.5 * (t_plus - t_minus)
        print(f"  maxiter={maxiter:5d}: T+={t_plus:+10.3f} (res {r_plus:.2e}) "
              f"T-={t_minus:+10.3f} (res {r_minus:.2e}) "
              f"T1={t_lin:+10.3f}  [{dt:.1f}s/{dt2:.1f}s]", flush=True)


def build_with_maxiter(maxiter):
    artifact = ModelArtifact.load(ART)
    import inspect
    valid = set(inspect.signature(MotorConfig3D.__init__).parameters) - {"self"}
    kw = {"shape": tuple(artifact.shape)}
    for k, v in artifact.config_dict.items():
        if k in valid:
            kw[k] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
    kw["maxwell_maxiter"] = maxiter
    cfg = MotorConfig3D(**kw)
    mf = _mf_from_artifact(artifact, cfg)
    fields, mag = realize(mf, cfg, artifact.magnetization)
    reg = artifact.centerline_registry or None
    return cfg, fields, mag, reg


if __name__ == "__main__":
    main()
