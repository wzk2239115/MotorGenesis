"""Multi-angle startup validation for constructed motors.

Runs the powered transient from multiple initial rotor angles to verify:
  - Self-starting from every angle (no dead points)
  - Consistent forward torque direction (no reversal)
  - Sustained acceleration over multiple electrical cycles
  - No overcurrent or overtemperature under a current-limited drive

This is validation level 3: "can it actually turn?"  Only after this passes
can we say the motor is rotationally viable in simulation.

Drive model (documented limits):
  - Ideal rotor-angle commutation (the controller knows the rotor angle
    perfectly; sensor error / sensorless start are NOT modelled).
  - Per-phase current clamp emulating the inverter current loop.
  - Torque comes from a periodic static field map interpolated over rotor
    angle; not a full time-domain eddy-current solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.optimization.objective3d import forward3d_fields


@dataclass
class StartupResult:
    """Result of a single startup simulation from one initial angle."""

    initial_angle_rad: float
    final_speed_rad_s: float
    min_speed_rad_s: float
    max_speed_rad_s: float
    mean_torque_Nm: float
    max_current_A: float
    max_temperature_C: float
    reversal: bool
    backward_angle_excursion_deg: float
    electrical_cycles: float
    passed: bool


@dataclass
class MultiAngleStartupResult:
    """Result of startup tests from multiple initial angles."""

    results: list[StartupResult] = field(default_factory=list)
    all_started: bool = False
    any_reversal: bool = False
    any_dead_point: bool = False
    min_final_speed_rad_s: float = 0.0
    max_startup_current_A: float = 0.0
    max_temperature_C: float = 0.0
    passed: bool = False
    n_angles: int = 0
    settings: dict = field(default_factory=dict)
    torque_decomposition: dict = field(default_factory=dict)
    verdicts: dict | None = None

    def summary(self) -> dict:
        out = {
            "n_angles": self.n_angles,
            "all_started": self.all_started,
            "any_reversal": self.any_reversal,
            "any_dead_point": self.any_dead_point,
            "min_final_speed_rad_s": self.min_final_speed_rad_s,
            "max_startup_current_A": self.max_startup_current_A,
            "max_temperature_C": self.max_temperature_C,
            "passed": self.passed,
            "settings": self.settings,
            "torque_decomposition": self.torque_decomposition,
            "angles": [
                {
                    "initial_angle_rad": r.initial_angle_rad,
                    "final_speed_rad_s": r.final_speed_rad_s,
                    "min_speed_rad_s": r.min_speed_rad_s,
                    "mean_torque_Nm": r.mean_torque_Nm,
                    "max_current_A": r.max_current_A,
                    "electrical_cycles": r.electrical_cycles,
                    "reversal": r.reversal,
                    "backward_angle_excursion_deg": r.backward_angle_excursion_deg,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }
        if self.verdicts is not None:
            out["verdicts"] = self.verdicts["verdicts"]
            out["verdict_labels"] = self.verdicts.get("labels", {})
            out["verdicts_evaluated"] = self.verdicts.get("evaluated", 0)
            out["verdicts_failed"] = self.verdicts.get("failed", [])
            # The headline verdict is the six-verdict overall, not the
            # bare transient: a green spin cannot cover broken topology.
            out["passed"] = bool(self.verdicts.get("passed", self.passed))
        return out


def run_single_startup(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization: jnp.ndarray,
    initial_angle: float,
    settings,
    angles_map: Sequence[float],
    *,
    fields=None,
    phase_belts_override=None,
    maps=None,
    min_speed_rad_s: float = 1.0,
    max_current_A: float = 200.0,
) -> StartupResult:
    """Run one startup simulation from a given initial rotor angle.

    ``maps`` (from :func:`compute_powered_maps`) lets all startup angles
    share one set of field-map solves -- the maps are identical for every
    initial angle, so solving them per angle wastes a factor of n_angles.
    Without ``maps`` they are solved here (single-run convenience).
    """
    from organic_motor.experiments.motor3d_powered import (
        compute_powered_maps,
        run_powered_transient,
    )

    if maps is None:
        maps_kwargs = {"include_mechanics": False}
        if fields is not None:
            realized = fields

            def phase_solver(single, angle, amplitudes):
                return forward3d_fields(
                    cfg, realized, magnetization, [angle], single,
                    phase_amplitudes=amplitudes,
                )

            maps_kwargs["phase_solver"] = phase_solver
            if phase_belts_override is not None:
                maps_kwargs["base_belts"] = phase_belts_override
        maps = compute_powered_maps(
            cfg, logits, rotor_logits, magnetization,
            angles_map, settings, **maps_kwargs,
        )
    data = run_powered_transient(maps, settings, cfg, initial_angle)
    return _evaluate_startup(cfg, data, initial_angle, settings,
                             min_speed_rad_s, max_current_A)


def _evaluate_startup(cfg, data, initial_angle, settings,
                      min_speed_rad_s, max_current_A) -> StartupResult:
    speed = np.asarray(data["angular_velocity_rad_s"])
    torque = np.asarray(data["transient_torque_Nm"])
    currents = np.asarray(data["currents_A"])
    temp = np.asarray(data["max_temperature_C"])
    rotor_angle = np.asarray(data["rotor_angle_rad"])

    final_speed = float(speed[-1])
    min_speed = float(speed.min())
    max_speed = float(speed.max())
    mean_torque = float(np.mean(torque))
    max_current = float(np.max(np.abs(currents)))
    max_temp = float(temp.max())
    # Reversal = NET BACKWARD ROTATION beyond one slot pitch (5 mech deg),
    # not an instantaneous negative speed sample: cogging and current
    # ripple can jiggle the speed negative at standstill without the
    # machine actually rotating backwards.
    excursion = float(np.min(rotor_angle) - rotor_angle[0])
    excursion_deg = float(np.degrees(excursion))
    reversal = bool(excursion_deg < -5.0)
    total_electrical = float(cfg.pole_pairs * (rotor_angle[-1] - rotor_angle[0]))
    electrical_cycles = abs(total_electrical) / (2.0 * np.pi)

    passed = bool(
        final_speed >= min_speed_rad_s
        and not reversal
        and max_current <= max_current_A
        and max_temp <= cfg.max_temperature
        and electrical_cycles >= 1.0
    )

    return StartupResult(
        initial_angle_rad=initial_angle,
        final_speed_rad_s=final_speed,
        min_speed_rad_s=min_speed,
        max_speed_rad_s=max_speed,
        mean_torque_Nm=mean_torque,
        max_current_A=max_current,
        max_temperature_C=max_temp,
        reversal=reversal,
        backward_angle_excursion_deg=excursion_deg,
        electrical_cycles=electrical_cycles,
        passed=passed,
    )


def validate_startup(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization: jnp.ndarray,
    mf=None,
    *,
    electrical=None,
    n_angles: int = 4,
    n_map_angles: int = 6,
    steps: int = 8000,
    dt: float = 2.0e-5,
    voltage: float = 24.0,
    current_limit: float = 50.0,
    commutation_offset: float = 3.1415927,
    load_torque: float = 0.005,
    load_viscous: float = 1.0e-4,
    rotor_inertia: float = 2.0e-4,
    min_speed_rad_s: float = 1.0,
    max_current_A: float = 0.0,  # 0 = auto: 1.3x clamp (zero-sequence
    # re-centring of clamped currents can overshoot the per-phase limit)
    maxwell_maxiter: int | None = None,
    thermal_maxiter: int | None = None,
    electric_maxiter: int | None = None,
    display_mf=None,
    display_cfg: MotorConfig3D | None = None,
    torque_convergence: dict | None = None,
) -> MultiAngleStartupResult:
    """Run startup validation from multiple initial rotor angles.

    Tests ``n_angles`` evenly spaced initial positions over one electrical
    period (2*pi/pole_pairs mechanical).  A motor passes when it reaches
    forward speed from EVERY angle without reversal, overcurrent or
    overtemperature, and completes at least one full electrical cycle.

    ``electrical`` (or ``mf``, from which it is extracted) supplies R, L and
    flux linkage from the actual geometry; without either, the historical
    hand constants are used.  The ``*_maxiter`` kwargs are accepted for
    convenience and ignored (set them on ``cfg`` instead).
    """
    from organic_motor.experiments.motor3d_powered import Powered3DSettings
    from organic_motor.construct.transient_bridge import (
        ElectricalParameters,
        extract_electrical_parameters,
    )

    if electrical is None and mf is not None:
        electrical = extract_electrical_parameters(mf, cfg)
    if electrical is None:
        electrical = ElectricalParameters(
            phase_resistance=0.4, phase_inductance=2.0e-3,
            flux_linkage=0.03, n_turns_effective=1,
            copper_volume_m3=0.0, mean_path_length_m=0.0,
            wire_cross_section_m2=0.0, source="fallback_constants",
        )
    phase_resistance = electrical.phase_resistance
    phase_inductance = max(electrical.phase_inductance, 1.0e-6)
    flux_linkage = max(electrical.flux_linkage, 1.0e-4)

    period = 2.0 * np.pi / cfg.pole_pairs
    initial_angles = np.linspace(0, period, n_angles, endpoint=False)
    map_angles = np.arange(n_map_angles) * period / n_map_angles
    if max_current_A <= 0.0:
        max_current_A = 1.3 * current_limit

    settings = Powered3DSettings(
        steps=steps,
        dt=dt,
        phase_voltage_peak=voltage,
        phase_resistance=phase_resistance,
        phase_inductance=phase_inductance,
        flux_linkage=flux_linkage,
        current_limit=current_limit,
        commutation_offset=commutation_offset,
        load_torque=load_torque,
        load_viscous=load_viscous,
        rotor_inertia=rotor_inertia,
    )

    result = MultiAngleStartupResult(
        n_angles=n_angles,
        settings={
            "steps": steps, "dt": dt, "voltage": voltage,
            "R_ohm": phase_resistance, "L_H": phase_inductance,
            "flux_linkage": flux_linkage, "current_limit_A": current_limit,
            "load_torque_Nm": load_torque, "rotor_inertia": rotor_inertia,
            "drive_model": "ideal rotor-angle commutation + per-phase current clamp",
        },
    )
    # Realize the constructed geometry ONCE: the transient's field maps must
    # run on the exact constructed fields (assemble3d would delete surface
    # magnets and clip the winding), and the netlist belts keep the map
    # consistent with the actual winding topology.  The maps are solved
    # ONCE and shared by every startup angle (they are angle-periodic and
    # initial-angle independent).
    fields = None
    phase_belts_override = None
    if mf is not None:
        from organic_motor.construct.realize import realize
        fields, _mag_fields = realize(mf, cfg)
        if hasattr(mf, "metadata"):
            netlist = mf.metadata.get("winding_netlist")
            if netlist is not None:
                phase_belts_override = jnp.asarray(netlist.phase_belts_3d(cfg))

    from organic_motor.experiments.motor3d_powered import compute_powered_maps

    # P5 line-current: pass centerline registry so the solver deposits
    # currents from 3-D swept-band polylines instead of coarse rho_copper.
    centerline_registry = None
    if mf is not None and hasattr(mf, "metadata"):
        centerline_registry = mf.metadata.get("centerline_registry")

    def phase_solver(single, angle, amplitudes):
        return forward3d_fields(
            cfg, fields, magnetization, [angle], single, phase_amplitudes=amplitudes,
            centerline_registry=centerline_registry,
        )

    maps = compute_powered_maps(
        cfg, logits, rotor_logits, magnetization,
        map_angles, settings, phase_solver=phase_solver,
        base_belts=phase_belts_override,
        include_mechanics=False,
    )
    for angle in initial_angles:
        sr = run_single_startup(
            cfg, logits, rotor_logits, magnetization,
            float(angle), settings, map_angles,
            fields=fields,
            phase_belts_override=phase_belts_override,
            maps=maps,
            min_speed_rad_s=min_speed_rad_s, max_current_A=max_current_A,
        )
        result.results.append(sr)
        print(
            f"  [startup] theta0={np.degrees(angle):5.1f}deg  "
            f"final={sr.final_speed_rad_s:8.2f} rad/s  "
            f"min={sr.min_speed_rad_s:7.2f}  "
            f"cyc={sr.electrical_cycles:5.2f}  "
            f"I={sr.max_current_A:6.1f}A  T={sr.max_temperature_C:5.1f}C  "
            f"{'PASS' if sr.passed else 'FAIL'}"
        )

    result.all_started = all(r.passed for r in result.results)
    result.any_reversal = any(r.reversal for r in result.results)
    result.any_dead_point = any(r.final_speed_rad_s < min_speed_rad_s for r in result.results)
    result.min_final_speed_rad_s = min(r.final_speed_rad_s for r in result.results)
    result.max_startup_current_A = max(r.max_current_A for r in result.results)
    result.max_temperature_C = max(r.max_temperature_C for r in result.results)
    result.passed = result.all_started and not result.any_reversal and not result.any_dead_point

    # Torque-decomposition diagnostics (T0 cogging is grid-sensitive: its
    # dominant harmonic tells artifact (4x mech from Cartesian C4 symmetry)
    # from true 4-pole cogging (p-th harmonic)).
    t0 = np.asarray(maps["torque_cogging"], dtype=float)
    t1 = np.asarray(maps["torques_ph"], dtype=float)
    t2 = np.asarray(maps["torque_i2_diag"], dtype=float)
    na = t0.shape[0]
    period = maps["period"]
    fft = abs(np.fft.rfft(t0)) * 2.0 / max(na, 1)
    result.torque_decomposition = {
        "t0_peak_Nm": float(np.max(np.abs(t0))),
        "t0_rms_Nm": float(np.sqrt(np.mean(t0 ** 2))),
        "t0_dominant_harmonic_per_period": int(np.argmax(fft[1:]) + 1),
        "t1_amplitudes_Nm": [float(np.max(np.abs(t1[p]))) for p in range(3)],
        "t2_diag_amplitudes_Nm": [float(np.max(np.abs(t2[p]))) for p in range(3)],
        "cross_terms_i_p_i_q": "not solved (documented limitation)",
    }

    from organic_motor.construct.verdicts import evaluate_verdicts

    if mf is not None or display_mf is not None:
        topo_mf = mf if mf is not None else display_mf
        topo_cfg = cfg if mf is not None else display_cfg
        result.verdicts = evaluate_verdicts(
            topo_mf, topo_cfg, result,
            display_mf=display_mf if mf is not None else None,
            display_cfg=display_cfg if mf is not None else None,
            torque_convergence=torque_convergence,
        )
    return result


def constructed_design_from_mf(mf, cfg: MotorConfig3D, mag=None):
    """Convert a built MaterialField (+ magnetization) to powered-transient inputs."""
    from organic_motor.geometry.domain3d import domain_masks3d

    densities = mf.to_densities()
    logits = np.stack([
        densities["air"], densities["iron"], densities["copper"], densities["pm"],
    ]).astype(np.float32) * 10.0 - 5.0
    rotor_logits = (
        np.asarray(domain_masks3d(cfg)["rotor_design"], dtype=np.float32) * 10.0 - 5.0
    )
    if mag is None:
        mag = np.zeros((3,) + cfg.shape, dtype=np.float32)
    return jnp.asarray(logits), jnp.asarray(rotor_logits), jnp.asarray(mag)


def validate_from_checkpoint(
    cfg: MotorConfig3D,
    checkpoint_path: str | Path,
    n_angles: int = 4,
    steps: int = 8000,
) -> MultiAngleStartupResult:
    """Load a constructed checkpoint and run startup validation."""
    from organic_motor.construct.transient_bridge import load_constructed_checkpoint

    logits, rotor_logits, magnetization, _meta = load_constructed_checkpoint(
        cfg, checkpoint_path
    )
    return validate_startup(
        cfg, logits, rotor_logits, magnetization,
        n_angles=n_angles, steps=steps,
    )
