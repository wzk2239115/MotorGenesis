"""Objective assembly: forward physics + scalar objective & penalties."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import domain_masks, rotate_rotor
from organic_motor.physics.material import mass_density
from organic_motor.physics.maxwell import flux_density, magnetostatic_solve
from organic_motor.physics.torque import lorentz_torque
from organic_motor.topology.density import assemble
from organic_motor.topology.filters import total_variation


@dataclass
class ForwardResult:
    rho_air: jnp.ndarray
    rho_iron: jnp.ndarray
    rho_pm: jnp.ndarray
    nu: jnp.ndarray
    Mx: jnp.ndarray
    My: jnp.ndarray
    az: jnp.ndarray
    Bx: jnp.ndarray
    By: jnp.ndarray
    tau: jnp.ndarray


def _solve_and_torque(nu, Mx, My, cfg):
    az = magnetostatic_solve(nu, Mx, My, jnp.zeros_like(Mx), cfg)
    Bx, By = flux_density(az, cfg)
    tau = lorentz_torque(Bx, By, Mx, My, cfg)
    return az, Bx, By, tau


def static_forward(cfg: MotorConfig, z: jnp.ndarray, theta: jnp.ndarray,
                   temperature: float | None = None) -> ForwardResult:
    """Single static solve at the design position (rotor angle 0)."""
    mat = assemble(z, theta, cfg, temperature)
    az, Bx, By, tau = _solve_and_torque(mat.nu, mat.Mx, mat.My, cfg)
    return ForwardResult(mat.rho_air, mat.rho_iron, mat.rho_pm, mat.nu,
                         mat.Mx, mat.My, az, Bx, By, tau)


def ripple_forward(cfg: MotorConfig, z: jnp.ndarray, theta: jnp.ndarray,
                   K: int, temperature: float | None = None):
    """Torque evaluated over K rotor angles (the rotor region is rotated).

    Returns (tau_vec (K,), ForwardResult at phi=0).  Used to define the mean
    output torque and the torque-ripple penalty.
    """
    mat = assemble(z, theta, cfg, temperature)
    az0, Bx0, By0, tau0 = _solve_and_torque(mat.nu, mat.Mx, mat.My, cfg)
    base = ForwardResult(mat.rho_air, mat.rho_iron, mat.rho_pm, mat.nu,
                         mat.Mx, mat.My, az0, Bx0, By0, tau0)

    def tau_at(k):
        phi = 2.0 * jnp.pi * k / K
        nu_k = rotate_rotor(mat.nu, phi, cfg)
        Mx_k = rotate_rotor(mat.Mx, phi, cfg)
        My_k = rotate_rotor(mat.My, phi, cfg)
        return _solve_and_torque(nu_k, Mx_k, My_k, cfg)[3]

    taus = jnp.stack([tau_at(k) for k in range(K)])
    return taus, base


def soft_abs(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Smooth |x| (sqrt(x^2 + eps)) so the gradient stays finite at 0.

    ``eps`` is kept tiny so that the smoothed magnitude closely tracks the true
    ``|x|`` (a large ``eps`` would floor the torque and spuriously reward mass
    removal while the true torque is still ~0).
    """
    return jnp.sqrt(x * x + eps)


def objective(cfg: MotorConfig, tau_or_taus: jnp.ndarray,
              mat) -> tuple[jnp.ndarray, dict]:
    """Scalar objective from torque(s) and the material fields.

    ``mat`` is a :class:`MaterialFields` (or ForwardResult with a ``nu``/``Mx``
    attribute-compatible interface); we only need ``rho_iron``/``rho_pm``.
    """
    design = domain_masks(cfg)["design"]
    area = design.astype(jnp.float32).sum()

    # Output torque = mean |torque| over rotor positions.  The absolute value is
    # taken per sample rather than on the mean: with no electrical reference the
    # signed torque flips sign between symmetric rotor positions, so
    # abs(mean(signed)) ~ 0 would hide the real output torque.  For the scalar
    # static case this reduces to |tau|.
    tau_signed = jnp.mean(tau_or_taus)
    tau_mags = soft_abs(tau_or_taus)
    tau_abs = jnp.mean(tau_mags)
    if cfg.w_ripple > 0.0:
        # relative torque ripple: std of the magnitudes over the mean magnitude
        # (dimensionless, O(1) for a bad design -> 0 for a perfectly periodic
        # one).  Smoothed so the gradient stays finite as the samples coincide.
        var = jnp.mean((tau_mags - tau_abs) ** 2)
        ripple = jnp.sqrt(var + 1e-12) / (tau_abs + 1e-9)
    else:
        ripple = jnp.zeros_like(tau_abs)

    vol_pm = jnp.sum(mat.rho_pm) / area
    vol_iron = jnp.sum(mat.rho_iron) / area
    mass = jnp.sum(mass_density(mat.rho_iron, mat.rho_pm, cfg)) * (cfg.h ** 2)

    tv_pm = total_variation(mat.rho_pm, cfg)
    tv_iron = total_variation(mat.rho_iron, cfg)
    tv = tv_pm + tv_iron

    # magnetisation direction smoothness (normalised M field)
    mag_smooth = (total_variation(mat.Mx, cfg) + total_variation(mat.My, cfg)
                  ) / cfg.M_sat

    # torque term, maximised -> negative weight.  Raw torque is O(1e2..1e3)
    # N.m/m in SI, so it is normalised by a fixed reference (tau_ref) to O(1);
    # PM/iron volumes are then held near their targets by the (grown) soft
    # volume penalties.  Iron remains beneficial (flux return) so it is kept up
    # to its target rather than being removed as it would be by a torque/mass
    # objective.
    tq = -cfg.w_torque * tau_abs / cfg.tau_ref

    # soft target-volume constraints (retain material, avoid trivial optimum)
    vol_pm_pen = (vol_pm - cfg.V_pm_target) ** 2
    vol_iron_pen = (vol_iron - cfg.V_iron_target) ** 2

    obj = (tq
           + cfg.w_pm * vol_pm_pen
           + cfg.w_iron * vol_iron_pen
           + cfg.w_tv * tv
           + cfg.w_mag_smooth * mag_smooth
           + cfg.w_ripple * ripple)

    comps = {
        "obj": obj,
        "torque": tau_signed,
        "|torque|": tau_abs,
        "ripple": ripple,
        "torque/mass": tau_abs / (mass + cfg.mass_eps),
        "mass_kg_per_m": mass,
        "vol_pm": vol_pm,
        "vol_iron": vol_iron,
        "tv": tv,
        "tau_penalty_term": tq,
        "vol_pm_pen": vol_pm_pen,
        "vol_iron_pen": vol_iron_pen,
    }
    return obj, comps


def make_static_loss(cfg: MotorConfig):
    """Loss factory for the static (single-position) benchmark."""

    def loss(z, theta, temperature):
        fr = static_forward(cfg, z, theta, temperature)
        obj, comps = objective(cfg, fr.tau, fr)
        return obj, comps

    return loss


def make_ripple_loss(cfg: MotorConfig, K: int):
    """Loss factory for the rotating benchmark (mean torque - ripple)."""

    def loss(z, theta, temperature):
        taus, fr = ripple_forward(cfg, z, theta, K, temperature)
        obj, comps = objective(cfg, taus, fr)
        return obj, comps

    return loss


def make_snapshot(cfg: MotorConfig):
    """Eager forward pass used for rendering (returns concrete field arrays)."""
    def snapshot(z, theta, temperature):
        return static_forward(cfg, z, theta, temperature)
    return snapshot