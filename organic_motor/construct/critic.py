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
from organic_motor.construct.geometry_metrics import compute_geometry_metrics


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
    geometry_mf=None,
    geometry_cfg: MotorConfig3D | None = None,
) -> dict:
    """Score an already-built :class:`MaterialField` directly.

    ``geometry_mf``/``geometry_cfg`` (optional) is a higher-resolution
    rebuild of the same design used for geometric quality metrics:
    winding wires thinner than the physics voxel cannot be resolved at
    physics resolution, so connectivity checks are meaningless there.
    """
    import jax.numpy as jnp
    from organic_motor.construct.winding_netlist import CoilNetlist
    fields, mag = realize(mf, cfg, magnetization_raw, bandwidth=bandwidth)
    phase_belts_override = None
    netlist = mf.metadata.get("winding_netlist") if hasattr(mf, "metadata") else None
    if isinstance(netlist, CoilNetlist):
        belts_np = netlist.phase_belts_3d(cfg)
        phase_belts_override = jnp.asarray(belts_np)
    if angles is None:
        count = int(getattr(cfg, "mechanical_angles", 3))
        angles = jnp.arange(count) * (2.0 * np.pi / (cfg.pole_pairs * count))
    result = forward3d_fields(cfg, fields, mag, angles, phase_belts_override)
    _obj, comps = objective3d(cfg, result)
    metrics = {key: float(value) for key, value in comps.items()}
    metrics["obj"] = float(_obj)
    if geometry_mf is not None and geometry_cfg is not None:
        geom = compute_geometry_metrics(geometry_mf, geometry_cfg)
        from organic_motor.construct.connectivity import connectivity_report
        geom.update(connectivity_report(geometry_mf, geometry_cfg))
    else:
        geom = compute_geometry_metrics(mf, cfg)
        from organic_motor.construct.connectivity import connectivity_report
        geom.update(connectivity_report(mf, cfg))
    metrics.update(geom)
    if netlist is not None:
        metrics["winding_netlist"] = netlist.summary()
    return metrics
