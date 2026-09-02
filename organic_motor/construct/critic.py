"""Score a constructed motor with the differentiable physics solver.

The critic is the existing 3-D Maxwell / conduction / thermal solver, run as a
forward pass (no gradients).  It returns the same metric dictionary the
optimisation loop uses, so a code-generating agent gets a numerical reward
signal: torque, losses, peak temperature, volume fractions, solver residuals.

This closes the agent loop -- construct (LEAP 71-style) -> realise -> critic
(MotorGenesis-style) -> feedback to the agent -- without ever placing discrete
Booleans on the gradient tape.
"""

from __future__ import annotations

import json
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.optimization.objective3d import forward3d_fields, objective3d
from organic_motor.construct.objects import Motor
from organic_motor.construct.realize import realize


def score(
    motor: Motor,
    cfg: MotorConfig3D | None = None,
    *,
    angles: Sequence[float] | jnp.ndarray | None = None,
    bandwidth: float | None = None,
) -> dict:
    """Build, realise and score a motor; return a metrics dictionary."""
    cfg = cfg or motor.cfg
    mf = motor.build()
    magnetization_raw = motor.magnetization()
    fields, mag = realize(mf, cfg, magnetization_raw, bandwidth=bandwidth)
    if angles is None:
        count = int(getattr(cfg, "mechanical_angles", 3))
        angles = jnp.arange(count) * (2.0 * np.pi / (cfg.pole_pairs * count))
    result = forward3d_fields(cfg, fields, mag, angles)
    _obj, comps = objective3d(cfg, result)
    metrics = {key: float(value) for key, value in comps.items()}
    metrics["obj"] = float(_obj)
    metrics["voxels"] = int(np.prod(cfg.shape))
    metrics["materials"] = mf.materials_present()
    return metrics


def score_fields(
    mf,
    cfg: MotorConfig3D,
    magnetization_raw: np.ndarray | None = None,
    *,
    angles: Sequence[float] | jnp.ndarray | None = None,
    bandwidth: float | None = None,
) -> dict:
    """Score an already-built :class:`MaterialField` directly."""
    fields, mag = realize(mf, cfg, magnetization_raw, bandwidth=bandwidth)
    if angles is None:
        count = int(getattr(cfg, "mechanical_angles", 3))
        angles = jnp.arange(count) * (2.0 * np.pi / (cfg.pole_pairs * count))
    result = forward3d_fields(cfg, fields, mag, angles)
    _obj, comps = objective3d(cfg, result)
    metrics = {key: float(value) for key, value in comps.items()}
    metrics["obj"] = float(_obj)
    return metrics
