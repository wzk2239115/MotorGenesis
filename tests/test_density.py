"""Material-field / density parameterisation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from organic_motor.config import MotorConfig
from organic_motor.topology.density import (
    MaterialFields,
    assemble,
    material_label,
    random_init,
)


@pytest.fixture(scope="module")
def cfg():
    return MotorConfig(N=48)


def test_random_init_shapes(cfg):
    z, theta = random_init(cfg, jax.random.PRNGKey(0))
    assert z.shape == (4, cfg.N, cfg.N)
    assert theta.shape == (cfg.N, cfg.N)


def test_assemble_sum_and_support(cfg):
    key = jax.random.PRNGKey(1)
    z, theta = random_init(cfg, key)
    m = assemble(z, theta, cfg)
    assert isinstance(m, MaterialFields)

    rho = np.stack([np.asarray(m.rho_air), np.asarray(m.rho_iron),
                    np.asarray(m.rho_copper),
                    np.asarray(m.rho_pm)])
    assert np.allclose(rho.sum(axis=0), 1.0, atol=1e-3)

    # materials confined to the design annulus
    from organic_motor.geometry.sdf import domain_masks
    design = np.asarray(domain_masks(cfg)["design"])
    assert np.all(np.asarray(m.rho_iron)[~design] < 1e-6)
    assert np.all(np.asarray(m.rho_pm)[~design] < 1e-6)


def test_softmax_sums_to_one(cfg):
    z = jax.random.normal(jax.random.PRNGKey(2), (4, cfg.N, cfg.N))
    rho = jax.nn.softmax(z / cfg.sm_temp_init, axis=0)
    assert np.allclose(np.asarray(rho.sum(axis=0)), 1.0, atol=1e-5)


def test_material_label_consistent(cfg):
    z, theta = random_init(cfg, jax.random.PRNGKey(3))
    m = assemble(z, theta, cfg)
    lab = material_label(m.rho_iron, m.rho_pm)
    assert lab.shape == (cfg.N, cfg.N)
    assert set(np.unique(np.asarray(lab)).tolist()) <= {0, 1, 2, 3}


def test_gradient_flows_to_design(cfg):
    z, theta = random_init(cfg, jax.random.PRNGKey(4))

    def f(zarr):
        m = assemble(zarr, theta, cfg)
        return jnp.sum(m.rho_iron * m.rho_pm)

    g = jax.grad(f)(z)
    assert g.shape == z.shape
    assert np.all(np.isfinite(np.asarray(g)))
