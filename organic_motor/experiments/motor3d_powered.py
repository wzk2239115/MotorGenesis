"""Powered native-3D motor experiment.

This is a quasi-static field-map transient: several native three-dimensional
magnetostatic/conduction solutions are sampled over rotor angle, then
periodically interpolated while the lumped three-phase circuit and rotor
dynamics advance.  It is deliberately *not* a full time-domain eddy-current
solver.  Every spatial field remains an ``(Nx, Ny, Nz[, 3])`` volume.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.experiments.precision_study3d import reference_design3d
from organic_motor.geometry.domain3d import domain_masks3d
from organic_motor.geometry.grid3d import meshgrid3d
from organic_motor.optimization.objective3d import ForwardResult3D, forward3d
from organic_motor.physics.mechanics3d import (
    air_gap_collision_penalty,
    solve_linear_elasticity,
)
from organic_motor.physics.transient3d import (
    RotorState,
    ThreePhaseState,
    advance_rotor,
    advance_three_phase_rl,
    advance_voxel_temperature,
    load_torque,
    sinusoidal_back_emf,
    transient_iron_loss,
    transient_joule_loss,
)


@dataclass(frozen=True)
class Powered3DSettings:
    """Reduced-order drive and structural settings in SI units."""

    steps: int = 100
    dt: float = 1.0e-5
    phase_voltage_peak: float = 24.0
    phase_resistance: float = 0.4
    phase_inductance: float = 2.0e-3
    flux_linkage: float = 0.03
    load_torque: float = 0.0
    load_viscous: float = 1.0e-4
    rotor_inertia: float = 2.0e-4
    mechanical_maxiter: int = 100
    mechanical_tol: float = 1.0e-5
    cooling_coefficient: float = 2.0e4
    eddy_loss_coefficient: float = 1.0e-4


def load_design3d(
    cfg: MotorConfig3D, path: str | Path | None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, str]:
    """Load optimization variables or sample the analytic 3-D reference."""
    if path is None:
        logits, rotor_logits, magnetization = reference_design3d(cfg)
        return logits, rotor_logits, magnetization, "reference_design3d"
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = ("logits", "rotor_logits", "magnetization_raw")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{source} is missing {', '.join(missing)}")
        logits = jnp.asarray(data["logits"])
        rotor_logits = jnp.asarray(data["rotor_logits"])
        magnetization = jnp.asarray(data["magnetization_raw"])
    expected = (4,) + cfg.shape
    if logits.shape != expected:
        raise ValueError(
            f"design shape {logits.shape} does not match configured {expected}"
        )
    if rotor_logits.shape != cfg.shape or magnetization.shape != (3,) + cfg.shape:
        raise ValueError("rotor ownership or magnetization has incompatible shape")
    return logits, rotor_logits, magnetization, str(source)


def periodic_interpolate(
    samples: np.ndarray, angles: Sequence[float], query: float, period: float
) -> np.ndarray:
    """Linearly interpolate scalar or volumetric samples on a periodic angle map."""
    values = np.asarray(samples)
    theta = np.mod(np.asarray(angles, dtype=float), period)
    order = np.argsort(theta)
    theta = theta[order]
    values = values[order]
    q = float(np.mod(query, period))
    upper = int(np.searchsorted(theta, q, side="right"))
    lower = (upper - 1) % len(theta)
    upper %= len(theta)
    a0 = theta[lower]
    a1 = theta[upper]
    if upper == 0:
        a1 += period
    if q < a0:
        q += period
    weight = 0.0 if a1 == a0 else (q - a0) / (a1 - a0)
    return (1.0 - weight) * values[lower] + weight * values[upper]


def material_fields3d(
    result: ForwardResult3D, cfg: MotorConfig3D
) -> dict[str, np.ndarray]:
    """Construct physical elastic, expansion, mass and heat-capacity volumes."""
    air = np.asarray(result.rho_air)
    iron = np.asarray(result.rho_iron)
    copper = np.asarray(result.rho_copper)
    pm = np.asarray(result.rho_pm)
    shaft = np.asarray(domain_masks3d(cfg)["shaft"], dtype=float)
    # The topology scaffold reserves the shaft outside the four phase fields.
    iron = np.maximum(iron, shaft)
    air = np.clip(1.0 - iron - copper - pm, 0.0, 1.0)
    fractions = np.stack((air, iron, copper, pm))

    def mix(values: Sequence[float]) -> np.ndarray:
        return np.tensordot(np.asarray(values), fractions, axes=(0, 0))

    return {
        "young_modulus": mix((1.0e6, 200.0e9, 110.0e9, 160.0e9)),
        "poisson_ratio": mix((0.25, 0.29, 0.34, 0.24)),
        "thermal_expansion": mix((0.0, 12.0e-6, 16.5e-6, 6.0e-6)),
        "mass_density": mix(
            (1.2, cfg.rho_iron_kg, cfg.rho_copper_kg, cfg.rho_pm_kg)
        ),
        "thermal_conductivity": mix(
            (
                cfg.thermal_k_air,
                cfg.thermal_k_iron,
                cfg.thermal_k_copper,
                cfg.thermal_k_pm,
            )
        ),
        "volumetric_heat_capacity": mix(
            (1.2e3, 3.6e6, 3.45e6, 3.4e6)
        ),
        "fractions": fractions,
    }


def _collision_diagnostics(
    displacement: np.ndarray, cfg: MotorConfig3D
) -> dict[str, float | bool]:
    x, y, z = (np.asarray(value) for value in meshgrid3d(cfg))
    cx, cy, cz = cfg.center
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    masks = {key: np.asarray(value) for key, value in domain_masks3d(cfg).items()}
    rotor = masks["rotor_design"]
    band = 1.5 * max(cfg.dx, cfg.dy)
    side = rotor & (radius >= cfg.R_rotor_outer - band)
    radial = np.stack(
        (
            (x - cx) / np.maximum(radius, 1e-12),
            (y - cy) / np.maximum(radius, 1e-12),
            np.zeros_like(radius),
        ),
        axis=-1,
    )
    radial_closure = np.sum(displacement * radial, axis=-1)
    radial_gap = cfg.R_stator_inner - cfg.R_rotor_outer

    end = rotor & (
        np.abs(np.abs(z - cz) - cfg.rotor_half_length) <= 1.5 * cfg.dz
    )
    axial_closure = displacement[..., 2] * np.sign(z - cz)
    side_min = (
        float(np.min(radial_gap - radial_closure[side]))
        if np.any(side)
        else radial_gap
    )
    end_min = (
        float(np.min(cfg.axial_airgap - axial_closure[end]))
        if np.any(end)
        else cfg.axial_airgap
    )
    contact = side | end
    signed_gap = np.where(side, radial_gap, cfg.axial_airgap)
    normal = np.where(
        side[..., None],
        -radial,
        np.stack(
            (np.zeros_like(z), np.zeros_like(z), -np.sign(z - cz)), axis=-1
        ),
    )
    penalty = air_gap_collision_penalty(
        displacement,
        signed_gap,
        normal,
        penalty_stiffness=1.0,
        spacing=cfg.spacing,
        contact_mask=contact,
    )
    minimum = min(side_min, end_min)
    return {
        "minimum_gap_m": minimum,
        "minimum_radial_gap_m": side_min,
        "minimum_axial_gap_m": end_min,
        "collision": bool(minimum <= 0.0),
        "collision_penalty": float(penalty),
    }


def _nominal_phase_current(result: ForwardResult3D, cfg: MotorConfig3D) -> float:
    phase_j = np.asarray(result.phase_current_density)
    z_index = cfg.Nz // 2
    currents = np.sum(np.maximum(phase_j[:, :, :, z_index, 2], 0.0), axis=(1, 2))
    currents *= cfg.dx * cfg.dy
    return max(float(np.mean(np.abs(currents))), 1.0e-6)


def _plot_outputs(data: dict[str, np.ndarray], path: Path) -> None:
    mid = data["temperature_final"].shape[2] // 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(data["map_angles_rad"], data["torque_map_Nm"], "o-")
    axes[0, 0].set(xlabel="mechanical angle [rad]", ylabel="torque [N m]")
    axes[0, 1].plot(data["time_s"], data["angular_velocity_rad_s"])
    axes[0, 1].set(xlabel="time [s]", ylabel="speed [rad/s]")
    axes[1, 0].plot(data["time_s"], data["currents_A"])
    axes[1, 0].set(xlabel="time [s]", ylabel="phase current [A]")
    image = axes[1, 1].imshow(
        data["temperature_final"][:, :, mid].T, origin="lower", aspect="equal"
    )
    axes[1, 1].set_title("final mid-plane temperature [C]")
    fig.colorbar(image, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_powered3d(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization_raw: jnp.ndarray,
    angles: Sequence[float],
    settings: Powered3DSettings,
    *,
    forward_solver: Callable[..., ForwardResult3D] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run field maps, structural loading, and map-driven transient dynamics."""
    solver = forward3d if forward_solver is None else forward_solver
    map_results = [
        solver(
            cfg,
            logits,
            rotor_logits,
            magnetization_raw,
            [float(angle)],
            cfg.sm_temp_final,
        )
        for angle in angles
    ]
    period = 2.0 * np.pi / cfg.pole_pairs
    map_angles = np.mod(np.asarray(angles, dtype=float), period)
    torques = np.asarray([float(result.torques[0]) for result in map_results])
    b_map = np.stack([np.asarray(result.flux_density) for result in map_results])
    j_map = np.stack([np.asarray(result.current_density) for result in map_results])
    joule_map = np.stack([np.asarray(result.joule_loss) for result in map_results])
    iron_map = np.stack([np.asarray(result.iron_loss) for result in map_results])
    temperature_map = np.stack(
        [np.asarray(result.temperature) for result in map_results]
    )
    base = map_results[0]
    materials = material_fields3d(base, cfg)
    x, y, z = meshgrid3d(cfg)
    coordinates = jnp.stack((x, y, z), axis=-1)
    masks = domain_masks3d(cfg)
    rotor_mass = materials["mass_density"] * np.asarray(masks["rotor_design"])
    mechanics = solve_linear_elasticity(
        materials["young_modulus"],
        materials["poisson_ratio"],
        np.asarray(masks["fixed_shaft"]) | np.asarray(masks["boundary"]),
        spacing=cfg.spacing,
        thermal_expansion=materials["thermal_expansion"],
        temperature_change=np.mean(temperature_map, axis=0)
        - cfg.ambient_temperature,
        density=rotor_mass,
        coordinates=coordinates,
        angular_velocity=cfg.speed_rpm * 2.0 * np.pi / 60.0,
        rotation_center=jnp.asarray(cfg.center),
        maxiter=settings.mechanical_maxiter,
        tol=settings.mechanical_tol,
    )
    displacement = np.asarray(mechanics.displacement)
    collision = _collision_diagnostics(displacement, cfg)

    state_rotor = RotorState(jnp.asarray(0.0), jnp.asarray(0.0))
    state_circuit = ThreePhaseState(jnp.zeros(3))
    temperature = jnp.asarray(temperature_map.mean(axis=0))
    previous_b = jnp.asarray(periodic_interpolate(b_map, map_angles, 0.0, period))
    nominal_current = _nominal_phase_current(base, cfg)
    time = np.arange(settings.steps + 1) * settings.dt
    rotor_angle = np.zeros(settings.steps + 1)
    speed = np.zeros(settings.steps + 1)
    currents = np.zeros((settings.steps + 1, 3))
    torque_history = np.zeros(settings.steps)
    joule_power = np.zeros(settings.steps)
    iron_power = np.zeros(settings.steps)
    max_temperature = np.zeros(settings.steps + 1)
    max_temperature[0] = float(jnp.max(temperature))
    phase_shifts = jnp.asarray((0.0, -2.0 * jnp.pi / 3.0, 2.0 * jnp.pi / 3.0))
    copper_fraction = jnp.asarray(materials["fractions"][2])
    iron_fraction = jnp.asarray(materials["fractions"][1])
    sigma = cfg.sigma_copper * jnp.maximum(copper_fraction, 1.0e-6)

    for index in range(settings.steps):
        electrical_angle = cfg.pole_pairs * state_rotor.angle
        voltage = settings.phase_voltage_peak * jnp.cos(
            electrical_angle + phase_shifts
        )
        back_emf = sinusoidal_back_emf(
            state_rotor.angle,
            state_rotor.angular_velocity,
            cfg.pole_pairs,
            settings.flux_linkage,
        )
        state_circuit = advance_three_phase_rl(
            state_circuit,
            voltage,
            back_emf,
            settings.phase_resistance,
            settings.phase_inductance,
            settings.dt,
        )
        torque_wave = jnp.cos(electrical_angle + phase_shifts)
        current_scale = jnp.sum(state_circuit.currents * torque_wave) / (
            1.5 * nominal_current
        )
        base_torque = periodic_interpolate(
            torques, map_angles, float(state_rotor.angle), period
        )
        electromagnetic_torque = jnp.asarray(base_torque) * current_scale
        applied_load = load_torque(
            state_rotor.angular_velocity,
            constant=settings.load_torque,
            viscous=settings.load_viscous,
        )
        state_rotor = advance_rotor(
            state_rotor,
            electromagnetic_torque,
            applied_load,
            settings.rotor_inertia,
            settings.dt,
        )

        mapped_j = jnp.asarray(
            periodic_interpolate(
                j_map, map_angles, float(state_rotor.angle), period
            )
        )
        mapped_b = jnp.asarray(
            periodic_interpolate(
                b_map, map_angles, float(state_rotor.angle), period
            )
        )
        q_joule = transient_joule_loss(
            mapped_j * current_scale, sigma, active_mask=copper_fraction
        )
        db_dt = (mapped_b - previous_b) / settings.dt
        frequency = (
            cfg.pole_pairs * state_rotor.angular_velocity / (2.0 * jnp.pi)
        )
        q_iron = transient_iron_loss(
            mapped_b,
            db_dt,
            cfg.iron_loss_coeff / cfg.iron_loss_B_ref**2,
            settings.eddy_loss_coefficient,
            frequency,
            iron_mask=iron_fraction,
        )
        previous_b = mapped_b
        temperature = advance_voxel_temperature(
            temperature,
            q_joule + q_iron,
            materials["thermal_conductivity"],
            materials["volumetric_heat_capacity"],
            cfg.spacing,
            settings.dt,
            ambient_temperature=cfg.ambient_temperature,
            cooling_coefficient=settings.cooling_coefficient,
            cooling_mask=np.asarray(masks["boundary"], dtype=float),
        )
        rotor_angle[index + 1] = float(state_rotor.angle)
        speed[index + 1] = float(state_rotor.angular_velocity)
        currents[index + 1] = np.asarray(state_circuit.currents)
        torque_history[index] = float(electromagnetic_torque)
        joule_power[index] = float(jnp.sum(q_joule) * cfg.cell_volume)
        iron_power[index] = float(jnp.sum(q_iron) * cfg.cell_volume)
        max_temperature[index + 1] = float(jnp.max(temperature))

    data = {
        "map_angles_rad": map_angles,
        "torque_map_Nm": torques,
        "flux_density_map_T": b_map,
        "current_density_map_A_m2": j_map,
        "joule_loss_map_W_m3": joule_map,
        "iron_loss_map_W_m3": iron_map,
        "temperature_map_C": temperature_map,
        "young_modulus_Pa": materials["young_modulus"],
        "poisson_ratio": materials["poisson_ratio"],
        "thermal_expansion_1_K": materials["thermal_expansion"],
        "mass_density_kg_m3": materials["mass_density"],
        "displacement_m": displacement,
        "von_mises_Pa": np.asarray(mechanics.von_mises),
        "time_s": time,
        "rotor_angle_rad": rotor_angle,
        "angular_velocity_rad_s": speed,
        "currents_A": currents,
        "transient_torque_Nm": torque_history,
        "transient_joule_power_W": joule_power,
        "transient_iron_power_W": iron_power,
        "max_temperature_C": max_temperature,
        "temperature_final": np.asarray(temperature),
    }
    summary = {
        "model": "quasi-static field-map transient",
        "full_time_domain_eddy_current": False,
        "shape": list(cfg.shape),
        "angle_samples": len(map_angles),
        "steps": settings.steps,
        "torque_mean_Nm": float(np.mean(torques)),
        "torque_ripple_peak_to_peak_Nm": float(np.ptp(torques)),
        "maximum_displacement_m": float(np.linalg.norm(displacement, axis=-1).max()),
        "maximum_von_mises_Pa": float(np.asarray(mechanics.von_mises).max()),
        "mechanical_relative_residual": float(mechanics.relative_residual),
        "final_speed_rad_s": float(speed[-1]),
        "final_max_temperature_C": float(max_temperature[-1]),
        **collision,
    }
    return data, summary


def _shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 3 or min(values) < 3:
        raise argparse.ArgumentTypeError("shape must be Nx,Ny,Nz with values >= 3")
    return values  # type: ignore[return-value]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--shape", type=_shape, default=None)
    ap.add_argument("--angles", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1.0e-5)
    ap.add_argument("--out", type=Path, default=Path("powered3d_out"))
    ap.add_argument("--maxwell-iters", type=int, default=120)
    ap.add_argument("--thermal-iters", type=int, default=240)
    ap.add_argument("--electric-iters", type=int, default=120)
    ap.add_argument("--mechanical-iters", type=int, default=100)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    shape = args.shape
    if shape is None and args.design is not None:
        with np.load(args.design, allow_pickle=False) as design:
            shape = tuple(int(value) for value in design["rotor_logits"].shape)
    cfg = MotorConfig3D(
        shape=(12, 12, 8) if shape is None else shape,
        filt_radius=0.0,
        projection_beta=0.0,
        mechanical_angles=args.angles,
        maxwell_maxiter=args.maxwell_iters,
        thermal_maxiter=args.thermal_iters,
        electric_maxiter=args.electric_iters,
        n_theta=32,
        torque_n_z=12,
        torque_n_r=12,
    )
    if args.angles < 2:
        raise ValueError("--angles must be at least 2 for a periodic torque map")
    logits, rotor_logits, magnetization, source = load_design3d(cfg, args.design)
    period = 2.0 * np.pi / cfg.pole_pairs
    angles = np.arange(args.angles) * period / args.angles
    settings = Powered3DSettings(
        steps=args.steps,
        dt=args.dt,
        mechanical_maxiter=args.mechanical_iters,
    )
    data, summary = run_powered3d(
        cfg, logits, rotor_logits, magnetization, angles, settings
    )
    summary["design_source"] = source
    summary["settings"] = asdict(settings)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "powered3d.npz", **data)
    (args.out / "powered3d.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_outputs(data, args.out / "powered3d.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
