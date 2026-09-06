"""Angle sweep of solver residuals for the real-motor map solves."""

import time
import numpy as np
import jax.numpy as jnp

from organic_motor.construct.model_artifact import ModelArtifact
from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.startup_validation import _mf_from_artifact
from organic_motor.construct.realize import realize
from organic_motor.optimization.objective3d import forward3d_fields, _phase_belts

ART = "organic_motor/out/assembly"


def build(maxiter=800, direct=False, from_densities=False):
    if direct:
        from organic_motor.construct.objects import field_driven_motor
        kw = {"shape": (96, 96, 58), "maxwell_maxiter": maxiter}
        cfg = MotorConfig3D(**kw)
        motor = field_driven_motor(cfg)
        mf = motor.build()
        mag_raw = motor.magnetization()
        if mag_raw is None:
            mag_raw = np.zeros((3,) + cfg.shape, dtype=np.float32)
    elif from_densities:
        # Hypothesis test: build fields DIRECTLY from artifact densities,
        # skipping the fake-SDF (0.5-rho) round trip entirely.
        from organic_motor.topology.density3d import TopologyFields3D
        from organic_motor.geometry.domain3d import domain_masks3d
        artifact = ModelArtifact.load(ART)
        import inspect
        valid = set(inspect.signature(MotorConfig3D.__init__).parameters) - {"self"}
        kw = {"shape": tuple(artifact.shape)}
        for k, v in artifact.config_dict.items():
            if k in valid:
                kw[k] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
        kw["maxwell_maxiter"] = maxiter
        cfg = MotorConfig3D(**kw)
        d = artifact.densities
        rotor = np.asarray(domain_masks3d(cfg)["rotor_design"], np.float32)
        fields = TopologyFields3D(
            rho_air=jnp.asarray(d.get("rho_air", np.zeros(cfg.shape, np.float32))),
            rho_iron=jnp.asarray(d["rho_iron"]),
            rho_copper=jnp.asarray(d.get("rho_copper", np.zeros(cfg.shape, np.float32))),
            rho_pm=jnp.asarray(d["rho_pm"]),
            rotor_ownership=jnp.asarray(rotor),
        )
        # NOTE: artifact densities are the SMOOTHSTEP of the original SDFs
        # (mf.to_densities at save time), so this is the same information
        # the direct path has — just without re-deriving it through a
        # distance-impostor SDF.
        mag = jnp.asarray(artifact.magnetization)
        from organic_motor.construct.startup_validation import _mf_from_artifact
        mf = _mf_from_artifact(artifact, cfg)  # only for the registry
        reg = mf.metadata.get("centerline_registry") or None
        return cfg, fields, mag, reg
    else:
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
        mag_raw = artifact.magnetization
    fields, mag = realize(mf, cfg, mag_raw)
    reg = mf.metadata.get("centerline_registry") or None
    return cfg, fields, mag, reg


def solve(cfg, fields, mag, reg, belts, angle, amp):
    r = forward3d_fields(cfg, fields, mag, [angle], belts,
                         phase_amplitudes=amp, centerline_registry=reg)
    tor = float(np.asarray(r.torques).ravel()[0])
    res = np.asarray(r.maxwell_residual).ravel()
    sdiv = np.asarray(r.source_divergence_residual).ravel()
    return tor, float(res[0]), float(sdiv[0]) if sdiv.size else float(sdiv)


def main():
    import sys
    direct = "--direct" in sys.argv
    from_densities = "--from-densities" in sys.argv
    cfg, fields, mag, reg = build(800, direct=direct, from_densities=from_densities)
    mode = ("DIRECT construction" if direct
            else "FROM ARTIFACT DENSITIES" if from_densities
            else "ARTIFACT round-trip (fake SDF)")
    full = _phase_belts(cfg)
    zero = jnp.zeros_like(full[0])
    singles = [jnp.stack([full[p] if q == p else zero for q in range(3)])
               for p in range(3)]
    one = jnp.asarray([1.0, 0.0, 0.0])

    elec_period = 2.0 * np.pi / 5
    angles = np.linspace(0, elec_period, 6, endpoint=False)
    print(f"[sweep] {mode}, zero-current PM solves, maxiter=800")
    zb = jnp.zeros_like(full)
    for i, a in enumerate(angles):
        t, res, sdiv = solve(cfg, fields, mag, reg, zb, float(a),
                             jnp.zeros(3))
        print(f"  angle {i+1}/6 ({np.degrees(a):5.1f} deg): T0={t:+8.4f}  "
              f"res={res:.3e}", flush=True)
    print("[sweep] phase-1 minus-current solves")
    for i, a in enumerate(angles):
        t, res, sdiv = solve(cfg, fields, mag, reg, singles[1], float(a),
                             -jnp.roll(one, 1))
        print(f"  angle {i+1}/6 ({np.degrees(a):5.1f} deg): T={t:+10.3f}  "
              f"maxwell_res={res:.3e}  src_div={sdiv:.2e}", flush=True)


if __name__ == "__main__":
    main()
