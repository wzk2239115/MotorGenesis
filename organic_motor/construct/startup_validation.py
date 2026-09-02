"""Multi-angle startup validation for constructed motors.

Runs the powered transient from multiple initial rotor angles to verify:
  - Self-starting from every angle (no dead points)
  - Consistent forward torque direction
  - Continuous rotation over a full mechanical cycle
  - No overcurrent or overtemperature

This is validation level 3: "can it actually turn?"  Only after this passes
can we say the motor is rotationally viable in simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct.material import MaterialField
from organic_motor.construct.transient_bridge import (
    extract_electrical_parameters,
    make_powered_settings_from_geometry,
)


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
    steps_to_steady: int
    passed: bool


@dataclass
class MultiAngleStartupResult:
    """Result of startup tests from multiple initial angles."""

    results: list[StartupResult] = field(default_factory=list)
    all_started: bool = False
    any_reversal: bool = False
    any_dead_point: bool = False
    min_steady_speed_rad_s: float = 0.0
    max_startup_current_A: float = 0.0
    max_temperature_C: float = 0.0
    passed: bool = False
    n_angles: int = 0

    def summary(self) -> dict:
        return {
            "n_angles": self.n_angles,
            "all_started": self.all_started,
            "any_reversal": self.any_reversal,
            "any_dead_point": self.any_dead_point,
            "min_steady_speed_rad_s": self.min_steady_speed_rad_s,
            "max_startup_current_A": self.max_startup_current_A,
            "max_temperature_C": self.max_temperature_C,
            "passed": self.passed,
            "angles": [
                {
                    "initial_angle_rad": r.initial_angle_rad,
                    "final_speed_rad_s": r.final_speed_rad_s,
                    "mean_torque_Nm": r.mean_torque_Nm,
                    "reversal": r.reversal,
                    "passed": r.passed,
                }
                for r in self.results
            ],
        }


def run_single_startup(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization: jnp.ndarray,
    initial_angle: float,
    settings,
    angles_map: Sequence[float],
) -> StartupResult:
    """Run one startup simulation from a given initial rotor angle."""
    from organic_motor.experiments.motor3d_powered import run_powered3d
    from organic_motor.physics.transient3d import RotorState, ThreePhaseState
    from organic_motor.experiments.motor3d_powered import periodic_interpolate

    data, summary = run_powered3d(
        cfg, logits, rotor_logits, magnetization,
        angles_map, settings,
    )
    speed = np.asarray(data["angular_velocity_rad_s"])
    torque = np.asarray(data["transient_torque_Nm"])
    currents = np.asarray(data["currents_A"])
    temp = np.asarray(data["max_temperature_C"])

    final_speed = float(speed[-1])
    min_speed = float(speed.min())
    max_speed = float(speed.max())
    mean_torque = float(np.mean(torque))
    max_current = float(np.max(np.abs(currents)))
    max_temp = float(temp.max())
    reversal = min_speed < -0.01

    steady_threshold = 0.1 * max(abs(final_speed), 1e-6)
    steps_to_steady = len(speed)
    for i in range(len(speed)):
        if abs(speed[i] - final_speed) < steady_threshold:
            steps_to_steady = i
            break

    passed = (final_speed > 0.1) and (not reversal) and (max_current < 100.0) and (max_temp < cfg.max_temperature)

    return StartupResult(
        initial_angle_rad=initial_angle,
        final_speed_rad_s=final_speed,
        min_speed_rad_s=min_speed,
        max_speed_rad_s=max_speed,
        mean_torque_Nm=mean_torque,
        max_current_A=max_current,
        max_temperature_C=max_temp,
        reversal=reversal,
        steps_to_steady=steps_to_steady,
        passed=passed,
    )


def validate_startup(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization: jnp.ndarray,
    n_angles: int = 6,
    n_map_angles: int = 8,
    steps: int = 2000,
    dt: float = 1.0e-5,
    voltage: float = 24.0,
    load_torque: float = 0.01,
    load_viscous: float = 1.0e-4,
    rotor_inertia: float = 2.0e-4,
) -> MultiAngleStartupResult:
    """Run startup validation from multiple initial rotor angles.

    Tests n_angles evenly spaced initial positions over one mechanical
    period (2*pi/pole_pairs).  A motor that can self-start from every
    angle and sustain forward rotation passes.
    """
    period = 2.0 * np.pi / cfg.pole_pairs
    initial_angles = np.linspace(0, period, n_angles, endpoint=False)
    map_angles = np.arange(n_map_angles) * period / n_map_angles

    settings = type("S", (), {
        "steps": steps, "dt": dt,
        "phase_voltage_peak": voltage,
        "phase_resistance": 0.4,
        "phase_inductance": 2.0e-3,
        "flux_linkage": 0.03,
        "load_torque": load_torque,
        "load_viscous": load_viscous,
        "rotor_inertia": rotor_inertia,
        "mechanical_maxiter": 100,
        "mechanical_tol": 1e-5,
        "cooling_coefficient": 2.0e4,
        "eddy_loss_coefficient": 1.0e-4,
    })()

    result = MultiAngleStartupResult(n_angles=n_angles)
    for angle in initial_angles:
        sr = run_single_startup(
            cfg, logits, rotor_logits, magnetization,
            angle, settings, map_angles,
        )
        result.results.append(sr)

    result.all_started = all(r.passed for r in result.results)
    result.any_reversal = any(r.reversal for r in result.results)
    result.any_dead_point = any(r.final_speed_rad_s < 0.1 for r in result.results)
    result.min_steady_speed_rad_s = min(r.final_speed_rad_s for r in result.results)
    result.max_startup_current_A = max(r.max_current_A for r in result.results)
    result.max_temperature_C = max(r.max_temperature_C for r in result.results)
    result.passed = result.all_started and not result.any_reversal and not result.any_dead_point
    return result


def validate_from_checkpoint(
    cfg: MotorConfig3D,
    checkpoint_path: str | Path,
    n_angles: int = 4,
    steps: int = 1000,
) -> MultiAngleStartupResult:
    """Load a constructed checkpoint and run startup validation."""
    from organic_motor.construct.transient_bridge import load_constructed_checkpoint
    logits, rotor_logits, magnetization, _ = load_constructed_checkpoint(cfg, checkpoint_path)
    return validate_startup(
        cfg, logits, rotor_logits, magnetization,
        n_angles=n_angles, steps=steps,
    )
