"""Adam optimisation loop with density continuation and logging."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from organic_motor.config import MotorConfig
from organic_motor.topology.density import assemble, random_init


@dataclass
class OptimizationResult:
    z: jnp.ndarray
    theta: jnp.ndarray
    history: dict = field(default_factory=dict)
    config: MotorConfig | None = None


def temperature_schedule(step: int, total: int, cfg: MotorConfig) -> float:
    """Log-linear softmax-temperature continuation T_init -> T_final."""
    t = min(1.0, step / max(1, int(0.7 * total)))
    logT = jnp.log(cfg.sm_temp_init) + t * (jnp.log(cfg.sm_temp_final)
                                            - jnp.log(cfg.sm_temp_init))
    return float(jnp.exp(logT))


def optimize(cfg: MotorConfig, loss_fn, key, snapshot_fn=None,
             plot_every: int | None = 20, progress: bool = True) -> OptimizationResult:
    """Run Adam on ``loss_fn(z, theta, temperature) -> (obj, comps_dict)``.

    ``snapshot_fn(z, theta, temperature)`` (eager) supplies concrete field
    arrays for periodic rendering; the gradient is taken only through
    ``loss_fn``.
    """
    key, = jax.random.split(key, 1)
    z, theta = random_init(cfg, key)

    opt = optax.adam(cfg.lr)
    params = (z, theta)
    opt_state = opt.init(params)

    history: dict[str, list] = {k: [] for k in
                                ["obj", "torque", "|torque|", "ripple",
                                 "torque/mass", "mass_kg_per_m", "vol_pm",
                                 "vol_iron", "tv"]}

    def step(params, opt_state, temperature):
        (obj, comps), grads = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True)(*params, temperature)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, obj, comps

    plotter = _Plotter(cfg) if plot_every else None
    snapshot_fn = snapshot_fn or _snapshot_for(loss_fn)

    for i in range(cfg.steps):
        temperature = temperature_schedule(i, cfg.steps, cfg)
        params, opt_state, obj, comps = step(params, opt_state, temperature)

        for k in history:
            v = comps.get(k, None)
            if v is not None:
                history[k].append(float(v))

        if progress and (i % 25 == 0 or i == cfg.steps - 1):
            print(f"step {i:4d}  obj {float(obj):.6e}  |T| {float(comps['|torque|']):.4f} "
                  f"torque {float(comps['torque']):.4e}  vol_pm {float(comps['vol_pm']):.3f} "
                  f"vol_iron {float(comps['vol_iron']):.3f}  T {temperature:.3f}")

        # penalty (augmented-Lagrangian) schedule: grow the volume-target
        # weights so the soft targets approach hard volume constraints.
        if i > 0 and (i % cfg.pen_growth_every == 0):
            cfg.w_pm *= cfg.pen_growth
            cfg.w_iron *= cfg.pen_growth

        if plotter is not None and (i % plot_every == 0 or i == cfg.steps - 1):
            fr = snapshot_fn(*params, temperature)
            plotter.render(i, fr, history)

    return OptimizationResult(z=params[0], theta=params[1],
                              history=history, config=cfg)


def _snapshot_for(loss_fn):
    """Fallback: if no snapshot is provided, raise a clear error on first use."""
    def _noop(*a, **k):
        raise RuntimeError("no snapshot_fn provided to optimize()")
    return _noop


class _Plotter:
    def __init__(self, cfg: MotorConfig):
        from pathlib import Path
        self.cfg = cfg
        out = cfg.out_dir or (Path(__file__).parent.parent / "out")
        self.out = Path(out)
        self.frames = self.out / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)

    def render(self, step: int, field_data: jnp.ndarray, history: dict) -> None:
        from organic_motor.visualization.field_plot import plot_field_panel
        from organic_motor.visualization.topology_plot import (
            plot_history, plot_magnetization, plot_material)

        cfg = self.cfg
        out = self.out
        rh = field_data if not hasattr(field_data, "rho_iron") else field_data
        rho_iron = rh.rho_iron
        rho_pm = rh.rho_pm

        plot_material(cfg, rho_iron, rho_pm,
                      str(out / "material.png"),
                      title=f"iteration {step}")
        plot_magnetization(cfg, rho_pm, rh.Mx, rh.My,
                           str(out / "magnetization.png"))
        plot_field_panel(cfg, rh.Bx, rh.By, str(out / "field.png"),
                         title=f"iteration {step}")
        plot_history(history, str(out / "history.png"))

        if self.frames is not None:
            plot_material(cfg, rho_iron, rho_pm,
                          str(self.frames / f"topo_{step:05d}.png"))