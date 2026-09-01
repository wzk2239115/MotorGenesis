"""Adam, checkpoint/restart, and multi-resolution optimization for 3-D motors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.topology.density3d import assemble3d, random_init3d


Params3D = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


@dataclass
class OptimizationResult3D:
    logits: jnp.ndarray
    rotor_logits: jnp.ndarray
    magnetization_raw: jnp.ndarray
    history: dict[str, list[float]] = field(default_factory=dict)
    config: MotorConfig3D | None = None
    completed_steps: int = 0


def estimate_memory3d(
    shape: tuple[int, int, int],
    angle_count: int = 3,
    dtype_bytes: int = 4,
    with_gradients: bool = True,
) -> dict[str, float | int]:
    """Conservative working-set estimate for planning CPU/GPU runs.

    The estimate accounts for material/vector fields, Krylov work vectors and
    per-angle reverse-mode intermediates.  XLA may allocate differently.
    """
    voxels = int(np.prod(shape))
    scalar_fields = 18
    vector_fields = 9
    forward_arrays = scalar_fields + 3 * vector_fields
    krylov_arrays = 24
    gradient_factor = (2.5 * max(angle_count, 1)) if with_gradients else 0.0
    arrays = forward_arrays + krylov_arrays + gradient_factor * forward_arrays
    bytes_estimate = int(voxels * dtype_bytes * arrays)
    return {
        "voxels": voxels,
        "estimated_bytes": bytes_estimate,
        "estimated_mib": bytes_estimate / 2**20,
        "angle_count": angle_count,
    }


def temperature_schedule3d(step: int, total: int, cfg: MotorConfig3D) -> float:
    t = min(1.0, step / max(1, int(0.7 * total)))
    log_t = np.log(cfg.sm_temp_init) + t * (
        np.log(cfg.sm_temp_final) - np.log(cfg.sm_temp_init)
    )
    return float(np.exp(log_t))


def _resize(field: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    return jax.image.resize(field, shape, method="linear", antialias=True)


def upsample_params3d(params: Params3D, shape: tuple[int, int, int]) -> Params3D:
    """Trilinearly continue all design fields onto a finer native 3-D grid."""
    logits, rotor_logits, magnetization_raw = params
    return (
        _resize(logits, (4,) + shape),
        _resize(rotor_logits, shape),
        _resize(magnetization_raw, (3,) + shape),
    )


def _zeros_like(params: Params3D) -> Params3D:
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def _adam_update(
    params: Params3D,
    grads: Params3D,
    first: Params3D,
    second: Params3D,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[Params3D, Params3D, Params3D]:
    first = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g, first, grads
    )
    second = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * g * g, second, grads
    )
    b1 = 1.0 - beta1**step
    b2 = 1.0 - beta2**step
    params = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * (m / b1) / (jnp.sqrt(v / b2) + epsilon),
        params,
        first,
        second,
    )
    return params, first, second


def _sanitize_and_clip(
    grads: Params3D, max_norm: float
) -> tuple[Params3D, jnp.ndarray, jnp.ndarray]:
    nonfinite = sum(
        jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)
    )
    clean = jax.tree_util.tree_map(
        lambda value: jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
        grads,
    )
    norm = jnp.sqrt(
        sum(jnp.sum(value * value) for value in jax.tree_util.tree_leaves(clean))
    )
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12))
    return jax.tree_util.tree_map(lambda value: scale * value, clean), norm, nonfinite


def _save_checkpoint(
    path: Path,
    params: Params3D,
    first: Params3D,
    second: Params3D,
    history: dict[str, list[float]],
    stage: int,
    stage_step: int,
    global_step: int,
    cfg: MotorConfig3D,
    temperature: float,
    metrics: dict[str, jnp.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = assemble3d(params[0], params[1], cfg, temperature)
    arrays = {
        "logits": np.asarray(params[0]),
        "rotor_logits": np.asarray(params[1]),
        "magnetization_raw": np.asarray(params[2]),
        "m_logits": np.asarray(first[0]),
        "m_rotor": np.asarray(first[1]),
        "m_magnetization": np.asarray(first[2]),
        "v_logits": np.asarray(second[0]),
        "v_rotor": np.asarray(second[1]),
        "v_magnetization": np.asarray(second[2]),
        "stage": np.asarray(stage),
        "stage_step": np.asarray(stage_step),
        "global_step": np.asarray(global_step),
        "step": np.asarray(global_step),
        "history_json": np.asarray(json.dumps(history)),
        "rho_air": np.asarray(fields.rho_air),
        "rho_iron": np.asarray(fields.rho_iron),
        "rho_copper": np.asarray(fields.rho_copper),
        "rho_pm": np.asarray(fields.rho_pm),
        "rotor_ownership": np.asarray(fields.rotor_ownership),
        "spacing": np.asarray(cfg.spacing),
        "origin": np.asarray(cfg.origin),
    }
    arrays.update(
        {f"metric__{name}": np.asarray(value) for name, value in metrics.items()}
    )
    np.savez_compressed(path, **arrays)


def _load_checkpoint(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        params = (
            jnp.asarray(data["logits"]),
            jnp.asarray(data["rotor_logits"]),
            jnp.asarray(data["magnetization_raw"]),
        )
        first = (
            jnp.asarray(data["m_logits"]),
            jnp.asarray(data["m_rotor"]),
            jnp.asarray(data["m_magnetization"]),
        )
        second = (
            jnp.asarray(data["v_logits"]),
            jnp.asarray(data["v_rotor"]),
            jnp.asarray(data["v_magnetization"]),
        )
        history = json.loads(str(data["history_json"]))
        location = (
            int(data["stage"]), int(data["stage_step"]), int(data["global_step"])
        )
    return params, first, second, history, location


def optimize3d(
    cfg: MotorConfig3D,
    loss_factory: Callable[[MotorConfig3D], Callable],
    key: jax.Array,
    levels: Sequence[tuple[int, int, int]] | None = None,
    steps_per_level: int | Sequence[int] | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: str | Path | None = None,
    progress: bool = True,
) -> OptimizationResult3D:
    """Optimize with Adam and optional coarse-to-fine continuation.

    ``loss_factory(stage_cfg)`` must return
    ``loss(logits, rotor_logits, magnetization_raw, temperature)``.  A
    checkpoint includes design fields, both Adam moments, stage position and
    scalar history, so restart preserves Adam rather than silently resetting it.
    """
    level_shapes = list(levels or [cfg.shape])
    if not level_shapes or level_shapes[-1] != cfg.shape:
        level_shapes.append(cfg.shape)
    if steps_per_level is None:
        counts = [cfg.steps] * len(level_shapes)
    elif isinstance(steps_per_level, int):
        counts = [steps_per_level] * len(level_shapes)
    else:
        counts = list(steps_per_level)
        if len(counts) != len(level_shapes):
            raise ValueError("steps_per_level must match levels")

    start_stage = start_step = global_step = 0
    history: dict[str, list[float]] = {}
    if resume is not None:
        params, first, second, history, location = _load_checkpoint(resume)
        start_stage, start_step, global_step = location
    else:
        initial_cfg = replace(cfg, shape=level_shapes[0])
        logits, rotor_logits = random_init3d(initial_cfg, key)
        direction_key = jax.random.fold_in(key, 1)
        magnetization_raw = jax.random.normal(
            direction_key, (3,) + initial_cfg.shape
        )
        params = (logits, rotor_logits, magnetization_raw)
        first = _zeros_like(params)
        second = _zeros_like(params)

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    total_steps = int(sum(counts))
    for stage, (shape, stage_count) in enumerate(zip(level_shapes, counts)):
        if stage < start_stage:
            continue
        stage_cfg = replace(cfg, shape=shape)
        if params[1].shape != shape:
            params = upsample_params3d(params, shape)
            # Continue Adam as well as the design.  Resetting the moments while
            # retaining the global bias-correction step causes an oversized
            # first update on every new resolution.
            first = upsample_params3d(first, shape)
            second = upsample_params3d(second, shape)
            start_step = 0
        loss_fn = loss_factory(stage_cfg)
        local_start = start_step if stage == start_stage else 0

        for local_step in range(local_start, stage_count):
            temperature = temperature_schedule3d(global_step, total_steps, cfg)
            (obj, comps), grads = jax.value_and_grad(
                loss_fn, argnums=(0, 1, 2), has_aux=True
            )(*params, temperature)
            grads, grad_norm, nonfinite_grads = _sanitize_and_clip(
                grads, float(getattr(cfg, "gradient_clip_norm", 10.0))
            )
            global_step += 1
            params, first, second = _adam_update(
                params, grads, first, second, global_step, cfg.lr
            )
            for name, value in comps.items():
                history.setdefault(name, []).append(float(value))
            history.setdefault("gradient_norm", []).append(float(grad_norm))
            history.setdefault("nonfinite_gradients", []).append(
                float(nonfinite_grads)
            )

            if progress and (
                local_step == local_start
                or (local_step + 1) % 10 == 0
                or local_step + 1 == stage_count
            ):
                print(
                    f"stage {stage + 1}/{len(level_shapes)} step "
                    f"{local_step + 1}/{stage_count} obj={float(obj):.4e} "
                    f"tau_z={float(comps['torque']):.4e} "
                    f"Tmax={float(comps['temperature_max_C']):.2f}"
                )

            checkpoint_due = (
                checkpoint_root is not None
                and cfg.checkpoint_every > 0
                and (
                    global_step % cfg.checkpoint_every == 0
                    or local_step + 1 == stage_count
                )
            )
            if checkpoint_due:
                _save_checkpoint(
                    checkpoint_root / f"step_{global_step:06d}.npz",
                    params,
                    first,
                    second,
                    history,
                    stage,
                    local_step + 1,
                    global_step,
                    stage_cfg,
                    temperature,
                    comps,
                )
        start_step = 0

    return OptimizationResult3D(
        params[0], params[1], params[2], history, cfg, global_step
    )


optimize = optimize3d
estimate_memory = estimate_memory3d
