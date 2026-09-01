"""Organic Motor / free-topology electromechanical organ research prototype.

Framework: Physics -> Geometry -> Manufacturable Object (inspired by
OpenSpaceArch), with a differentiable electromagnetic topology-optimization
loop built on JAX (inspired by TOFLUX and ARL_Topologies).
"""

import os

import jax

jax.config.update("jax_enable_x64", os.environ.get("MOTORGENESIS_X64", "1") != "0")

__version__ = "0.3.0"
