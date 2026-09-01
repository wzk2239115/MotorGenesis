"""Reduced-order loss, saturation and thermal-model tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.physics.losses import electromagnetic_losses, saturation_penalty
from organic_motor.physics.thermal import steady_temperature


def _phases(cfg):
    shape = (cfg.N, cfg.N)
    air = jnp.ones(shape)
    zero = jnp.zeros(shape)
    return air, zero, zero, zero


def test_zero_loss_stays_at_ambient():
    cfg = MotorConfig(N=20, thermal_maxiter=80)
    phases = _phases(cfg)
    temp = steady_temperature(jnp.zeros((cfg.N, cfg.N)), *phases, cfg)
    assert np.allclose(np.asarray(temp), cfg.ambient_temperature, atol=1e-5)


def test_positive_heat_creates_interior_hotspot():
    cfg = MotorConfig(N=24, thermal_maxiter=120)
    phases = _phases(cfg)
    q = jnp.zeros((cfg.N, cfg.N)).at[cfg.N // 2, cfg.N // 2].set(1e6)
    temp = np.asarray(steady_temperature(q, *phases, cfg))
    assert temp.max() > cfg.ambient_temperature
    assert np.allclose(temp[0, :], cfg.ambient_temperature, atol=1e-4)


def test_copper_loss_scales_with_current_squared():
    cfg = MotorConfig(N=8)
    ones = jnp.ones((8, 8))
    zero = jnp.zeros((8, 8))
    a = electromagnetic_losses(zero, zero, ones, zero, ones, cfg)
    b = electromagnetic_losses(zero, zero, 2 * ones, zero, ones, cfg)
    assert float(jnp.mean(b.copper)) == pytest.approx(
        4.0 * float(jnp.mean(a.copper)), rel=1e-5)


def test_saturation_penalty_activates_above_limit():
    cfg = MotorConfig(N=8)
    iron = jnp.ones((8, 8))
    low = saturation_penalty(jnp.ones((8, 8)), jnp.zeros((8, 8)), iron, cfg)
    high = saturation_penalty(2 * jnp.ones((8, 8)), jnp.zeros((8, 8)), iron, cfg)
    assert float(low) == 0.0
    assert float(high) > 0.0


def test_temperature_gradient_is_finite():
    cfg = MotorConfig(N=12, thermal_maxiter=60)
    phases = _phases(cfg)

    def peak(scale):
        q = jnp.ones((cfg.N, cfg.N)) * scale
        return jnp.max(steady_temperature(q, *phases, cfg))

    grad = jax.grad(peak)(jnp.asarray(1000.0))
    assert bool(jnp.isfinite(grad))
    assert float(grad) > 0.0
