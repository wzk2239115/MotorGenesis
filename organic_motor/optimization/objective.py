"""Objective assembly: forward physics + scalar objective & penalties."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from organic_motor.config import MotorConfig
from organic_motor.geometry.sdf import (domain_masks, rotate_rotor,
                                        rotate_rotor_vector)
from organic_motor.physics.excitation import (synchronous_electrical_angle,
                                               three_phase_current_density)
from organic_motor.physics.losses import electromagnetic_losses, saturation_penalty
from organic_motor.physics.material import mass_density
from organic_motor.physics.maxwell import (flux_density,
                                           magnetostatic_relative_residual,
                                           magnetostatic_solve)
from organic_motor.physics.thermal import (steady_temperature,
                                           thermal_relative_residual)
from organic_motor.physics.torque import lorentz_torque, maxwell_torque
from organic_motor.topology.density import assemble
from organic_motor.topology.filters import total_variation


@dataclass
class ForwardResult:
    rho_air: jnp.ndarray
    rho_iron: jnp.ndarray
    rho_copper: jnp.ndarray
    rho_pm: jnp.ndarray
    nu: jnp.ndarray
    Mx: jnp.ndarray
    My: jnp.ndarray
    az: jnp.ndarray
    Bx: jnp.ndarray
    By: jnp.ndarray
    tau: jnp.ndarray
    Jz: jnp.ndarray
    loss_copper: jnp.ndarray
    loss_iron: jnp.ndarray
    loss_total: jnp.ndarray
    temperature: jnp.ndarray
    saturation: jnp.ndarray
    maxwell_residual: jnp.ndarray
    thermal_residual: jnp.ndarray


def _solve_and_torque(nu, Mx, My, cfg, Jz=None):
    if Jz is None:
        Jz = jnp.zeros_like(Mx)
    az = magnetostatic_solve(nu, Mx, My, Jz, cfg)
    Bx, By = flux_density(az, cfg)
    if cfg.torque_method == "maxwell":
        tau = maxwell_torque(Bx, By, cfg)
    elif cfg.torque_method == "lorentz":
        tau = lorentz_torque(Bx, By, Mx, My, cfg)
    else:
        raise ValueError(f"unknown torque_method: {cfg.torque_method}")
    return az, Bx, By, tau


def _forward_result(mat, az, Bx, By, tau, Jz, cfg):
    losses = electromagnetic_losses(Bx, By, Jz, mat.rho_iron,
                                    mat.rho_copper, cfg)
    temperature = steady_temperature(losses.total, mat.rho_air, mat.rho_iron,
                                     mat.rho_copper, mat.rho_pm, cfg)
    saturation = saturation_penalty(Bx, By, mat.rho_iron, cfg)
    maxwell_residual = magnetostatic_relative_residual(
        mat.nu, mat.Mx, mat.My, Jz, az, cfg)
    thermal_residual = thermal_relative_residual(
        temperature, losses.total, mat.rho_air, mat.rho_iron,
        mat.rho_copper, mat.rho_pm, cfg)
    return ForwardResult(
        rho_air=mat.rho_air, rho_iron=mat.rho_iron,
        rho_copper=mat.rho_copper, rho_pm=mat.rho_pm, nu=mat.nu,
        Mx=mat.Mx, My=mat.My, az=az, Bx=Bx, By=By, tau=tau, Jz=Jz,
        loss_copper=losses.copper, loss_iron=losses.iron,
        loss_total=losses.total, temperature=temperature,
        saturation=saturation, maxwell_residual=maxwell_residual,
        thermal_residual=thermal_residual,
    )


def static_forward(cfg: MotorConfig, z: jnp.ndarray, theta: jnp.ndarray,
                   temperature: float | None = None) -> ForwardResult:
    """Single static solve at the design position (rotor angle 0)."""
    mat = assemble(z, theta, cfg, temperature)
    Jz = three_phase_current_density(0.0, cfg, mat.rho_copper)
    az, Bx, By, tau = _solve_and_torque(mat.nu, mat.Mx, mat.My, cfg, Jz)
    return _forward_result(mat, az, Bx, By, tau, Jz, cfg)


def ripple_forward(cfg: MotorConfig, z: jnp.ndarray, theta: jnp.ndarray,
                   K: int, temperature: float | None = None):
    """Torque evaluated over K rotor angles (the rotor region is rotated).

    Returns (tau_vec (K,), ForwardResult at phi=0).  Used to define the mean
    output torque and the torque-ripple penalty.
    """
    mat = assemble(z, theta, cfg, temperature)
    Jz0 = three_phase_current_density(0.0, cfg, mat.rho_copper)
    az0, Bx0, By0, tau0 = _solve_and_torque(mat.nu, mat.Mx, mat.My, cfg, Jz0)
    base = _forward_result(mat, az0, Bx0, By0, tau0, Jz0, cfg)

    def tau_at(k):
        phi = 2.0 * jnp.pi * k / K
        nu_k = rotate_rotor(mat.nu, phi, cfg)
        Mx_k, My_k = rotate_rotor_vector(mat.Mx, mat.My, phi, cfg)
        elec = synchronous_electrical_angle(phi, cfg)
        copper_k = rotate_rotor(mat.rho_copper, phi, cfg)
        Jz_k = three_phase_current_density(elec, cfg, copper_k)
        return _solve_and_torque(nu_k, Mx_k, My_k, cfg, Jz_k)[3]

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

    # A driven machine must produce torque in one direction over the cycle.
    # Maximising mean(abs(torque)) would reward alternating alignment torque and
    # can produce a magnetic latch rather than a motor.
    tau_signed = jnp.mean(tau_or_taus)
    tau_mags = soft_abs(tau_or_taus)
    tau_abs = soft_abs(tau_signed)
    if cfg.w_ripple > 0.0:
        # relative torque ripple: std of the magnitudes over the mean magnitude
        # (dimensionless, O(1) for a bad design -> 0 for a perfectly periodic
        # one).  Smoothed so the gradient stays finite as the samples coincide.
        var = jnp.mean((tau_or_taus - tau_signed) ** 2)
        ripple = jnp.sqrt(var + 1e-12) / (soft_abs(tau_signed) + 1e-9)
    else:
        ripple = jnp.zeros_like(tau_abs)

    vol_pm = jnp.sum(mat.rho_pm) / area
    vol_iron = jnp.sum(mat.rho_iron) / area
    vol_copper = jnp.sum(mat.rho_copper) / area
    mass = jnp.sum(mass_density(mat.rho_iron, mat.rho_pm, cfg,
                                mat.rho_copper)) * (cfg.h ** 2)
    copper_loss = jnp.sum(mat.loss_copper) * (cfg.h ** 2)
    iron_loss = jnp.sum(mat.loss_iron) * (cfg.h ** 2)
    total_loss = copper_loss + iron_loss
    temp_max = jnp.max(mat.temperature)
    temp_excess = jnp.maximum(
        (temp_max - cfg.max_temperature) / cfg.max_temperature, 0.0)
    omega = cfg.speed_rpm * 2.0 * jnp.pi / 60.0
    mechanical_power = tau_abs * omega
    efficiency = mechanical_power / (mechanical_power + total_loss + 1e-9)

    tv_pm = total_variation(mat.rho_pm, cfg)
    tv_iron = total_variation(mat.rho_iron, cfg)
    tv = tv_pm + tv_iron
    tv = tv + total_variation(mat.rho_copper, cfg)

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
    vol_copper_pen = (vol_copper - cfg.V_copper_target) ** 2

    obj = (tq
           + cfg.w_pm * vol_pm_pen
           + cfg.w_iron * vol_iron_pen
           + cfg.w_copper * vol_copper_pen
           + cfg.w_tv * tv
           + cfg.w_mag_smooth * mag_smooth
           + cfg.w_ripple * ripple
           + cfg.w_loss * total_loss / cfg.loss_ref
           + cfg.w_temperature * temp_excess ** 2
           + cfg.w_saturation * mat.saturation)

    comps = {
        "obj": obj,
        "torque": tau_signed,
        "|torque|": tau_abs,
        "ripple": ripple,
        "torque/mass": tau_abs / (mass + cfg.mass_eps),
        "mass_kg_per_m": mass,
        "copper_loss_W_per_m": copper_loss,
        "iron_loss_W_per_m": iron_loss,
        "loss_W_per_m": total_loss,
        "temperature_max_C": temp_max,
        "efficiency_proxy": efficiency,
        "saturation": mat.saturation,
        "maxwell_residual": mat.maxwell_residual,
        "thermal_residual": mat.thermal_residual,
        "vol_pm": vol_pm,
        "vol_iron": vol_iron,
        "vol_copper": vol_copper,
        "tv": tv,
        "tau_penalty_term": tq,
        "vol_pm_pen": vol_pm_pen,
        "vol_iron_pen": vol_iron_pen,
        "vol_copper_pen": vol_copper_pen,
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
