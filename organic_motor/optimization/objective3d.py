"""Native 3-D electromagnetic, loss, thermal, and topology objective.

The winding model is an impressed three-phase conductor model.  Each phase has
two opposite axial coil sides, fed through the two axial end faces.  The source
is constant in ``z`` and each phase is balanced to zero net axial current, so
``div(J)=0`` and the terminal/current-balance diagnostics are directly
verifiable without pretending that a 2-D solution was extruded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.domain3d import domain_masks3d
from organic_motor.geometry.grid3d import meshgrid3d, rotate_vector_z, rotate_volume_z
from organic_motor.physics.electric3d import (
    _variable_diffusion,
    electric_relative_residual,
    ersatz_conductivity,
    solve_electric,
)
from organic_motor.physics.maxwell3d import (
    flux_density,
    magnetostatic_solve,
    maxwell_relative_residual,
)
from organic_motor.physics.operators3d import divergence3d
from organic_motor.physics.thermal3d import (
    steady_temperature,
    thermal_conductivity,
    thermal_relative_residual,
)
from organic_motor.physics.torque3d import maxwell_torque
from organic_motor.topology.density3d import TopologyFields3D, assemble3d
from organic_motor.topology.filters3d import (
    curvature_penalty3d,
    island_penalty3d,
    total_variation3d,
)


@dataclass
class ForwardResult3D:
    rho_air: jnp.ndarray
    rho_iron: jnp.ndarray
    rho_copper: jnp.ndarray
    rho_pm: jnp.ndarray
    rotor_ownership: jnp.ndarray
    nu: jnp.ndarray
    magnetization: jnp.ndarray
    vector_potential: jnp.ndarray
    flux_density: jnp.ndarray
    current_density: jnp.ndarray
    phase_current_density: jnp.ndarray
    torques: jnp.ndarray
    joule_loss: jnp.ndarray
    iron_loss: jnp.ndarray
    loss_total: jnp.ndarray
    temperature: jnp.ndarray
    maxwell_residual: jnp.ndarray
    thermal_residual: jnp.ndarray
    electric_residual: jnp.ndarray
    source_divergence_residual: jnp.ndarray
    phase_balance_residual: jnp.ndarray

    @property
    def tau(self) -> jnp.ndarray:
        return self.torques


def reluctivity3d(
    rho_iron: jnp.ndarray, rho_pm: jnp.ndarray, cfg: MotorConfig3D
) -> jnp.ndarray:
    """SIMP reluctivity including iron and PM permeability."""
    iron = jnp.clip(rho_iron, 0.0, 1.0) ** cfg.simp_p
    pm = jnp.clip(rho_pm, 0.0, 1.0)
    return cfg.nu_air + (cfg.nu_iron - cfg.nu_air) * iron + (
        cfg.nu_pm - cfg.nu_air
    ) * pm


def normalized_magnetization3d(
    rho_pm: jnp.ndarray, direction_raw: jnp.ndarray, cfg: MotorConfig3D
) -> jnp.ndarray:
    """Return ``M_sat*rho_pm*unit(direction_raw)`` with three free components."""
    expected = (3,) + cfg.shape
    if direction_raw.shape != expected:
        raise ValueError(f"magnetization direction must be {expected}")
    norm = jnp.sqrt(jnp.sum(direction_raw * direction_raw, axis=0) + 1e-8)
    unit = jnp.moveaxis(direction_raw / norm, 0, -1)
    return cfg.M_sat * rho_pm[..., None] * unit


def _phase_belts(cfg: MotorConfig3D, override: jnp.ndarray | None = None) -> jnp.ndarray:
    """Three disjoint phase belts; each contains positive and negative coil sides.

    If ``override`` is provided (from a CoilNetlist), it is used directly.
    Otherwise the analytic cosine assignment is used as a fallback.
    """
    if override is not None:
        return override
    X, Y, _ = meshgrid3d(cfg)
    cx, cy, _ = cfg.center
    radius = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    angle = jnp.arctan2(Y - cy, X - cx)
    phases = jnp.asarray((0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0))
    waves = jnp.cos(cfg.pole_pairs * angle[None, ...] - phases[:, None, None, None])
    owner = jnp.argmax(jnp.abs(waves), axis=0)
    # These are axial conductor columns connected to terminals on both box end
    # faces.  Their conductivity is supplied by the z-averaged winding
    # topology, so they remain a genuine 3-D source while div(J) is exactly
    # zero.  Using domain_masks3d()["winding"] here would incorrectly truncate
    # them before the terminals at the box ends.
    winding = (radius >= cfg.R_winding_inner) & (radius < cfg.R_winding_outer)
    return jnp.stack(
        [jnp.where((owner == phase) & winding, jnp.sign(waves[phase]), 0.0)
         for phase in range(3)],
        axis=0,
    )


def _printed_end_closure_currents(
    phase_jz: jnp.ndarray, cfg: MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """End-turn bridge currents for the PRINTED concentrated winding.

    The printed coil is a physical closed loop: axial current in the two
    side bands, azimuthal current in the copper bridges over both tooth
    ends.  The impressed columns supply the side bands; this routine
    supplies the bridges -- with per-radius flux matching
    ``J_arc(r) = J_col(r) * side_width(r) / end_band`` so the total loop
    current is preserved at every radius, not just globally.

    Coil geometry comes from :class:`PrintedCoilNetlist` (the same single
    source of truth the copper geometry uses): coil n's bridge spans the
    full frame pitch, top bridge current runs +theta * polarity, bottom
    bridge -theta * polarity -- the loop closes exactly.
    """
    nx, ny, nz = cfg.shape
    X, Y, Z = meshgrid3d(cfg)
    cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
    r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    theta = jnp.arctan2(Y - cy, X - cx)
    hz = cfg.stator_half_length

    from organic_motor.construct.winding_netlist import (
        PRINTED_CLAD_HALF,
        PRINTED_END_BAND,
        PRINTED_FRAME_HALF,
        printed_netlist,
    )

    netlist = printed_netlist(cfg)
    pitch = 2.0 * jnp.pi / netlist.n_slots
    n_tooth = jnp.mod(jnp.round(theta / pitch), netlist.n_slots).astype(jnp.int32)
    u = jnp.mod(theta - n_tooth * pitch + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    phase_of = jnp.zeros((netlist.n_slots,), dtype=jnp.int32)
    sign_of = jnp.zeros((netlist.n_slots,), dtype=jnp.float32)
    for tooth, ph, pol in netlist.coil_table():
        phase_of = phase_of.at[tooth].set(ph)
        sign_of = sign_of.at[tooth].set(float(pol))
    phase_grid = phase_of[n_tooth]
    sign_grid = sign_of[n_tooth]

    top_slab = ((Z - cz > hz) & (Z - cz <= hz + PRINTED_END_BAND)).astype(jnp.float32)
    bot_slab = ((cz - Z > hz) & (cz - Z <= hz + PRINTED_END_BAND)).astype(jnp.float32)
    stack_window = (jnp.abs(Z - cz) <= hz).astype(jnp.float32)
    window = (jnp.abs(u) <= PRINTED_FRAME_HALF).astype(jnp.float32)

    side_width = (PRINTED_FRAME_HALF - PRINTED_CLAD_HALF) * r
    gain = side_width / PRINTED_END_BAND
    th_hat_x = -jnp.sin(theta)
    th_hat_y = jnp.cos(theta)

    jx_all = jnp.zeros((3, nx, ny, nz), dtype=phase_jz.dtype)
    jy_all = jnp.zeros((3, nx, ny, nz), dtype=phase_jz.dtype)
    for phase in range(3):
        amp = jnp.maximum(jnp.max(jnp.abs(phase_jz[phase])), 1e-30)
        sel = (phase_grid == phase).astype(phase_jz.dtype)
        # top: +theta * polarity, bottom: -theta * polarity
        mag = amp * sign_grid * sel * gain * (top_slab - bot_slab) * window
        jx_all = jx_all.at[phase].set(jx_all[phase] + th_hat_x * mag)
        jy_all = jy_all.at[phase].set(jy_all[phase] + th_hat_y * mag)
    return jx_all, jy_all, stack_window


def _end_closure_currents(
    phase_jz: jnp.ndarray, cfg: MotorConfig3D, belts: jnp.ndarray,
    n_slots: int = 12, coil_span: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Azimuthal end-turn currents closing each phase's loops IN-DOMAIN.

    The z-uniform belt columns leave the box through the end faces; the
    solver boundary-zeroes the RHS there, which silently deletes part of
    the source and makes the effective source non-solenoidal (measured
    effective div(J_rhs) ~ 0.24 on the old geometry).  This constructs the
    physical return path instead: per coil (slot s <-> slot s+coil_span),
    an arc current in the end slabs at the phase's own radial band carries
    the column current from the + side to the - side at the top and back
    at the bottom -- exactly the real end winding, as a vector source.

    Arc current density is flux-matched to its columns:
    ``J_arc = J_col * (slot-sector arc length) / (end-slab thickness)``,
    so the TOTAL current around each loop is preserved on the voxel grid.
    """
    nx, ny, nz = cfg.shape
    X, Y, Z = meshgrid3d(cfg)
    cx, cy, cz = cfg.center[0], cfg.center[1], cfg.center[2]
    r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    theta = jnp.arctan2(Y - cy, X - cx)
    hz = cfg.stator_half_length
    t_end = max(3.0 * cfg.dz, 0.0035)  # end-slab thickness
    pitch = 2.0 * jnp.pi / n_slots

    top_slab = ((Z - cz > hz) & (Z - cz <= hz + t_end)).astype(jnp.float32)
    bot_slab = ((cz - Z > hz) & (cz - Z <= hz + t_end)).astype(jnp.float32)
    stack_window = (jnp.abs(Z - cz) <= hz).astype(jnp.float32)

    r_mid = 0.5 * (cfg.R_winding_inner + cfg.R_winding_outer)
    arc_gain = (r_mid * pitch) / t_end

    th_hat_x = -jnp.sin(theta)
    th_hat_y = jnp.cos(theta)
    mid_z = nz // 2
    theta_mid = theta[:, :, mid_z]
    jx_all = jnp.zeros((3, nx, ny, nz), dtype=phase_jz.dtype)
    jy_all = jnp.zeros((3, nx, ny, nz), dtype=phase_jz.dtype)
    for phase in range(3):
        any_band3 = jnp.broadcast_to(
            jnp.any(jnp.abs(belts[phase]) > 0, axis=2)[..., None], (nx, ny, nz)
        )
        belt_mid = belts[phase][:, :, mid_z]
        amp = jnp.maximum(jnp.max(jnp.abs(phase_jz[phase])), 1e-30)
        for slot in range(n_slots):
            theta_s = slot * pitch
            partner = (slot + coil_span) % n_slots
            d = jnp.mod(partner * pitch - theta_s + jnp.pi, 2 * jnp.pi) - jnp.pi
            near_s = jnp.abs(
                jnp.mod(theta_mid - theta_s + jnp.pi, 2 * jnp.pi) - jnp.pi
            ) <= 0.5 * pitch
            sign_s = jnp.sign(jnp.sum(belt_mid * near_s))
            centre = theta_s + 0.5 * d
            in_arc = jnp.abs(
                jnp.mod(theta - centre + jnp.pi, 2 * jnp.pi) - jnp.pi
            ) <= 0.5 * jnp.abs(d)
            window = (any_band3 & in_arc).astype(jnp.float32)
            # Top: current from the + side to the - side; bottom: reverse.
            mag = amp * sign_s * jnp.sign(d) * arc_gain
            slab = mag * (top_slab - bot_slab)
            jx_all = jx_all.at[phase].set(
                jx_all[phase] + th_hat_x * window * slab
            )
            jy_all = jy_all.at[phase].set(
                jy_all[phase] + th_hat_y * window * slab
            )
    return jx_all, jy_all, stack_window


def three_phase_impressed_source3d(
    rho_copper: jnp.ndarray, electrical_angle: float, cfg: MotorConfig3D,
    phase_belts_override: jnp.ndarray | None = None,
    phase_amplitudes: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return total and per-phase impressed ``Jz`` fields in A/m².

    Copper is averaged axially and rebroadcast over the FULL box height,
    making every phase current exactly constant along z -- axial current
    columns closed through the box end faces, hence ``div(J) = 0``
    EXACTLY (a solenoidal impressed source; severing the columns at the
    stack ends breaks current continuity and the Coulomb-gauge penalty
    then absorbs most of the armature field).

    ``phase_amplitudes`` (optional (3,) array) overrides the cos(elec-phi)
    excitation with CONSTANT per-phase amplitudes -- required when
    building per-phase torque maps: the map must be the torque at a fixed
    unit current, NOT modulated by cos(p*theta) (the transient multiplies
    the actual currents separately; modulating twice manufactures a
    2x-electrical-frequency artefact).

    The angular restriction to discrete slot sectors (from the netlist
    belts) keeps end-turn copper from injecting axial current at wrong
    angles.  Positive/negative coil sides are normalized to equal
    discrete current via MID-plane sums (the z = 0 grid plane lies
    outside all geometry).
    """
    belts = _phase_belts(cfg, phase_belts_override)
    conductor = jnp.broadcast_to(
        jnp.mean(jnp.clip(rho_copper, 0.0, 1.0), axis=2, keepdims=True),
        cfg.shape,
    )
    mid_z = cfg.shape[2] // 2
    if phase_amplitudes is not None:
        phase_currents = jnp.asarray(phase_amplitudes)
    else:
        phase_currents = jnp.cos(
            electrical_angle
            - jnp.asarray((0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0))
        )
    sources = []
    tiny = 1e-12
    for phase in range(3):
        positive = conductor * (belts[phase] > 0)
        negative = conductor * (belts[phase] < 0)
        pos_sum = jnp.sum(positive[:, :, mid_z])
        neg_sum = jnp.sum(negative[:, :, mid_z])
        usable = (pos_sum > tiny) & (neg_sum > tiny)
        negative_scale = pos_sum / jnp.maximum(neg_sum, tiny)
        basis = jnp.where(usable, positive - negative_scale * negative, 0.0)
        sources.append(cfg.current_density_peak * phase_currents[phase] * basis)
    phase_jz = jnp.stack(sources, axis=0)
    return jnp.sum(phase_jz, axis=0), phase_jz


def phase_terminal_masks3d(
    cfg: MotorConfig3D, phase_belts_override: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Boolean phase terminals on the two ends of the winding region."""
    belts = _phase_belts(cfg, phase_belts_override) != 0.0
    _, _, Z = meshgrid3d(cfg)
    ends = (
        jnp.abs(jnp.abs(Z - cfg.center[2]) - cfg.stator_half_length)
        <= 0.51 * cfg.dz
    )
    return belts & ends[None, ...]


def three_phase_terminal_conduction3d(
    rho_copper: jnp.ndarray, electrical_angle: float, cfg: MotorConfig3D,
    phase_belts_override: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve three terminal-driven conductor fields.

    The three phase belts are electrically independent.  Their opposite coil
    sides receive reversed end potentials, producing balanced axial currents
    while allowing the current magnitude and Joule heat to respond to the
    genuine ``rho_copper(x,y,z)`` field.
    """
    belts = _phase_belts(cfg, phase_belts_override)
    terminals = phase_terminal_masks3d(cfg, phase_belts_override)
    _, _, Z = meshgrid3d(cfg)
    axial_sign = jnp.where(Z >= cfg.center[2], 1.0, -1.0)
    phase_amplitude = jnp.cos(
        electrical_angle
        - jnp.asarray((0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0))
    )
    voltage = cfg.terminal_voltage
    if voltage is None:
        voltage = (
            cfg.current_density_peak
            * (2.0 * cfg.stator_half_length)
            / cfg.sigma_copper
        )

    phase_vectors = []
    phase_losses = []
    residuals = []
    balances = []
    for phase in range(3):
        region = jnp.abs(belts[phase])
        conductivity = ersatz_conductivity(
            jnp.clip(rho_copper * region, 0.0, 1.0), cfg
        )
        terminal_values = (
            0.5
            * voltage
            * phase_amplitude[phase]
            * jnp.sign(belts[phase])
            * axial_sign
        )
        potential, current, loss = solve_electric(
            conductivity, terminals[phase], terminal_values, cfg
        )
        vector = jnp.stack(current, axis=-1)
        phase_vectors.append(vector)
        phase_losses.append(loss)
        residuals.append(
            electric_relative_residual(
                conductivity,
                terminals[phase],
                terminal_values,
                potential,
                cfg,
            )
        )
        terminal_reaction = jnp.where(
            terminals[phase],
            _variable_diffusion(conductivity, potential, cfg),
            0.0,
        )
        balances.append(
            jnp.abs(jnp.sum(terminal_reaction))
            / jnp.maximum(jnp.sum(jnp.abs(terminal_reaction)), 1e-12)
        )
    phase_current = jnp.stack(phase_vectors, axis=0)
    return (
        jnp.sum(phase_current, axis=0),
        phase_current,
        jnp.sum(jnp.stack(phase_losses), axis=0),
        jnp.max(jnp.stack(residuals)),
        jnp.max(jnp.stack(balances)),
    )


def _rotated_materials(
    fields: TopologyFields3D, magnetization: jnp.ndarray, angle: float,
    cfg: MotorConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rotor = domain_masks3d(cfg)["rotor_design"].astype(fields.rho_iron.dtype)
    stator_iron = fields.rho_iron * (1.0 - rotor)
    rotor_iron = rotate_volume_z(fields.rho_iron * rotor, angle, cfg)
    pm = rotate_volume_z(fields.rho_pm, angle, cfg)
    mx, my, mz = rotate_vector_z(
        magnetization[..., 0],
        magnetization[..., 1],
        magnetization[..., 2],
        angle,
        cfg,
    )
    return stator_iron + rotor_iron, pm, jnp.stack((mx, my, mz), axis=-1)


def _source_residuals(
    phase_current: jnp.ndarray, cfg: MotorConfig3D,
    phase_belts_override: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if phase_current.ndim == 4:
        zeros = jnp.zeros_like(phase_current)
        phase_current = jnp.stack((zeros, zeros, phase_current), axis=-1)
    total = jnp.sum(phase_current, axis=0)
    div = divergence3d(total[..., 0], total[..., 1], total[..., 2], cfg)
    h = min(cfg.spacing)
    div_res = h * jnp.linalg.norm(div) / jnp.maximum(
        jnp.linalg.norm(total), 1e-12
    )
    terminals = phase_terminal_masks3d(cfg, phase_belts_override)
    _, _, Z = meshgrid3d(cfg)
    outward = jnp.where(Z >= cfg.center[2], 1.0, -1.0)
    terminal_flux = phase_current[..., 2] * terminals * outward
    phase_net = jnp.sum(terminal_flux, axis=(1, 2, 3))
    phase_abs = jnp.sum(jnp.abs(terminal_flux), axis=(1, 2, 3))
    balance = jnp.max(jnp.abs(phase_net) / jnp.maximum(phase_abs, 1e-12))
    return div_res, balance


def _forward3d_core(
    cfg: MotorConfig3D,
    fields: TopologyFields3D,
    magnetization_raw: jnp.ndarray,
    angles: Sequence[float] | jnp.ndarray | None,
    phase_belts_override: jnp.ndarray | None = None,
    phase_amplitudes: jnp.ndarray | None = None,
    centerline_registry: list | None = None,
) -> ForwardResult3D:
    """Body of :func:`forward3d` once the topology fields are assembled.

    Shared by the optimisation entry point (which soft-maxes logits into
    fields) and the constructive-critic entry point (which receives fields
    built from SDF Booleans).  Either way the physics is identical.

    If ``centerline_registry`` is provided (from a P5 swept-band stator),
    the impressed current is deposited directly from the 3-D centreline
    polylines using a conservative tent kernel — bypassing the coarse-grid
    ``rho_copper`` entirely.  This is the expert-recommended hybrid
    dimension approach: the current path and ampere-turns are exact,
    independent of whether the 0.7 mm tubes are resolved on the physics
    grid.
    """
    magnetization = normalized_magnetization3d(fields.rho_pm, magnetization_raw, cfg)
    if angles is None:
        count = int(getattr(cfg, "mechanical_angles", 3))
        angles = jnp.arange(count) * (2.0 * jnp.pi / (cfg.pole_pairs * count))

    torques = []
    maxwell_residuals = []
    electric_residuals = []
    joule_losses = []
    iron_losses = []
    last = None
    for angle in angles:
        rho_iron, rho_pm, M = _rotated_materials(fields, magnetization, angle, cfg)
        nu = reluctivity3d(rho_iron, rho_pm, cfg)
        electrical_angle = cfg.pole_pairs * angle + cfg.electrical_phase_offset
        if centerline_registry is not None:
            # P5 hybrid dimension: deposit line currents from 3-D centreline
            # polylines directly onto the coarse grid.  The current path
            # follows the actual swept bands, not the planar P4 belts.
            # This takes PRIORITY over terminal mode — the centreline IS
            # the electrical source of truth for printed windings.
            from organic_motor.optimization.line_current import (
                deposit_centerline_currents,
                centerline_resistance,
            )
            import numpy as _np
            if phase_amplitudes is not None:
                amps = _np.asarray(phase_amplitudes)
            else:
                ea = float(electrical_angle)
                offs = _np.array([0.0, 2.0 * _np.pi / 3.0, 4.0 * _np.pi / 3.0])
                amps = _np.cos(ea - offs)
            I_per_turn = float(cfg.current_density_peak) * _np.pi * (
                centerline_registry[0]["band_radius"] ** 2
            )
            J_np, phase_J_np = deposit_centerline_currents(
                cfg, centerline_registry, I_per_turn, amps,
            )
            J = jnp.asarray(J_np)
            phase_current = jnp.asarray(phase_J_np)
            # Analytical copper loss: I²ρL/A deposited along centreline
            # (NOT J²/σ on coarse grid — that is grid-dependent)
            from organic_motor.optimization.line_current import \
                _deposit_joule_heat
            q_cu_np = _deposit_joule_heat(
                cfg, centerline_registry, I_per_turn, amps,
            )
            q_cu = jnp.asarray(q_cu_np)
            electric_residual = jnp.asarray(0.0, dtype=q_cu.dtype)
            _, phase_balance = _source_residuals(phase_current, cfg, phase_belts_override)
        elif cfg.excitation_mode == "terminal":
            jz, phase_jz = three_phase_impressed_source3d(
                fields.rho_copper, electrical_angle, cfg, phase_belts_override,
                phase_amplitudes,
            )
            zeros = jnp.zeros_like(jz)
            if getattr(cfg, "impressed_end_closure", False):
                # Close every phase loop INSIDE the domain.  For the printed
                # concentrated winding the end-turn bridges are PHYSICAL
                # copper (per-radius flux matched, from the netlist's coil
                # table); for the legacy distributed winding the generic
                # slot-to-slot arc closure is used.
                belts = _phase_belts(cfg, phase_belts_override)
                if getattr(cfg, "winding_style", "printed") == "printed":
                    jx, jy, stack_window = _printed_end_closure_currents(
                        phase_jz, cfg,
                    )
                else:
                    jx, jy, stack_window = _end_closure_currents(
                        phase_jz, cfg, belts,
                    )
                jz = jz * stack_window
                phase_jz = phase_jz * stack_window[None]
                J = jnp.stack((jnp.sum(jx, axis=0), jnp.sum(jy, axis=0), jz), axis=-1)
            else:
                J = jnp.stack((zeros, zeros, jz), axis=-1)
            phase_current = jnp.stack(
                (
                    jnp.zeros_like(phase_jz),
                    jnp.zeros_like(phase_jz),
                    phase_jz,
                ),
                axis=-1,
            )
            conductor = jnp.broadcast_to(
                jnp.mean(fields.rho_copper, axis=2, keepdims=True), cfg.shape
            )
            q_cu = jnp.sum(phase_jz * phase_jz, axis=0) / (
                cfg.sigma_copper * (conductor + 1e-6)
            )
            q_cu = jnp.where(conductor > 1e-6, q_cu, 0.0)
            electric_residual = jnp.asarray(0.0, dtype=q_cu.dtype)
            _, phase_balance = _source_residuals(phase_current, cfg, phase_belts_override)
        A = magnetostatic_solve(nu, M, J, cfg)
        B = jnp.stack(flux_density(A, cfg), axis=-1)
        torque = maxwell_torque(B[..., 0], B[..., 1], B[..., 2], cfg)[2]

        b2 = jnp.sum(B * B, axis=-1)
        q_fe = (
            cfg.iron_loss_coeff
            * cfg.electrical_frequency
            * b2
            / (cfg.iron_loss_B_ref**2)
            * rho_iron
        )
        torques.append(torque)
        joule_losses.append(q_cu)
        iron_losses.append(q_fe)
        maxwell_residuals.append(maxwell_relative_residual(nu, M, J, A, cfg))
        electric_residuals.append(electric_residual)
        last = (nu, M, A, B, J, phase_current)

    assert last is not None
    q_cu = jnp.mean(jnp.stack(joule_losses), axis=0)
    q_fe = jnp.mean(jnp.stack(iron_losses), axis=0)
    q_total = q_cu + q_fe
    conductivity = thermal_conductivity(
        fields.rho_air, fields.rho_iron, fields.rho_copper, fields.rho_pm, cfg,
        rho_insulator=getattr(fields, "rho_insulator", None),
    )
    temperature_field = steady_temperature(
        q_total,
        fields.rho_air,
        fields.rho_iron,
        fields.rho_copper,
        fields.rho_pm,
        cfg,
        rho_insulator=getattr(fields, "rho_insulator", None),
        rho_coolant=getattr(fields, "rho_coolant", None),
    )
    thermal_residual = thermal_relative_residual(
        temperature_field, q_total, conductivity, cfg,
        internal_sink_beta=(
            getattr(cfg, "thermal_h_coolant", 3000.0)
            * getattr(cfg, "thermal_channel_s_v", 2000.0)
            * fields.rho_coolant
        ) if getattr(fields, "rho_coolant", None) is not None else None,
        internal_sink_temperature=getattr(cfg, "thermal_coolant_temperature", 40.0),
    )
    if cfg.excitation_mode == "terminal":
        source_div = jnp.max(jnp.stack(electric_residuals))
    else:
        source_div, phase_balance = _source_residuals(last[5], cfg, phase_belts_override)
    return ForwardResult3D(
        fields.rho_air,
        fields.rho_iron,
        fields.rho_copper,
        fields.rho_pm,
        fields.rotor_ownership,
        last[0],
        last[1],
        last[2],
        last[3],
        last[4],
        last[5],
        jnp.stack(torques),
        q_cu,
        q_fe,
        q_total,
        temperature_field,
        jnp.max(jnp.stack(maxwell_residuals)),
        thermal_residual,
        jnp.max(jnp.stack(electric_residuals)),
        source_div,
        phase_balance,
    )


def forward3d(
    cfg: MotorConfig3D,
    logits: jnp.ndarray,
    rotor_logits: jnp.ndarray,
    magnetization_raw: jnp.ndarray,
    angles: Sequence[float] | jnp.ndarray | None = None,
    temperature: float | None = None,
    phase_belts_override: jnp.ndarray | None = None,
    phase_amplitudes: jnp.ndarray | None = None,
) -> ForwardResult3D:
    """Run native 3-D Maxwell solves at multiple mechanical rotor angles.

    Softmax-maps unconstrained ``logits`` into the four-phase topology, then
    runs the shared physics core.  This is the optimisation-time entry point.
    """
    fields = assemble3d(logits, rotor_logits, cfg, temperature)
    return _forward3d_core(cfg, fields, magnetization_raw, angles, phase_belts_override, phase_amplitudes)


def forward3d_fields(
    cfg: MotorConfig3D,
    fields: TopologyFields3D,
    magnetization_raw: jnp.ndarray,
    angles: Sequence[float] | jnp.ndarray | None = None,
    phase_belts_override: jnp.ndarray | None = None,
    phase_amplitudes: jnp.ndarray | None = None,
    centerline_registry: list | None = None,
) -> ForwardResult3D:
    """Critic entry point: score an already-assembled constructed topology.

    The constructive layer builds ``fields`` directly from SDF Booleans; this
    bypasses the softmax/masking of :func:`assemble3d` so the geometry the
    critic solves is exactly what was constructed (cooling jackets, shaft
    bores and other non-design-region solids are preserved).

    If ``phase_belts_override`` is provided (from a CoilNetlist), the solver
    uses the actual winding topology instead of the analytic cosine phase
    belts.  ``phase_amplitudes`` overrides the cos(elec-phi) excitation with
    constant per-phase amplitudes (unit-current per-phase maps).

    If ``centerline_registry`` is provided (P5 swept-band stator), the
    impressed current is deposited from the 3-D centreline polylines
    instead of using the coarse-grid ``rho_copper``.  This is the hybrid
    dimension approach: exact current path, grid-independent ampere-turns.
    """
    return _forward3d_core(
        cfg, fields, magnetization_raw, angles, phase_belts_override,
        phase_amplitudes, centerline_registry,
    )


def _anchor_seeds(cfg: MotorConfig3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X, Y, Z = meshgrid3d(cfg)
    cx, cy, cz = cfg.center
    r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    h = max(cfg.dx, cfg.dy, cfg.dz)
    masks = domain_masks3d(cfg)
    rotor = masks["rotor_design"] & (r <= cfg.R_shaft + 1.5 * h)
    stator = masks["stator_design"] & (r >= cfg.R_design - 1.5 * h)
    winding = masks["winding"]
    z_edge = jnp.abs(jnp.abs(Z - cz) - cfg.stator_half_length) <= 1.5 * cfg.dz
    copper = winding & z_edge
    return rotor.astype(jnp.float32), stator.astype(jnp.float32), copper.astype(jnp.float32)


def objective3d(
    cfg: MotorConfig3D, result: ForwardResult3D
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Combine torque, losses, temperature, volume, TV, curvature and connectivity."""
    design = domain_masks3d(cfg)["design"].astype(result.rho_iron.dtype)
    design_volume = jnp.maximum(jnp.sum(design), 1.0)
    torque_mean = jnp.mean(result.torques)
    torque_variance = jnp.mean((result.torques - torque_mean) ** 2)
    # jnp.std has an undefined derivative at exactly zero variance (notably a
    # one-angle smoke run).  The smooth floor keeps the unused w_ripple=0 path
    # from contaminating every gradient through IEEE 0*NaN.
    torque_ripple = jnp.sqrt(torque_variance + 1e-12) / (
        jnp.abs(torque_mean) + 1e-9
    )
    cell_volume = cfg.cell_volume
    copper_loss = jnp.sum(result.joule_loss) * cell_volume
    iron_loss = jnp.sum(result.iron_loss) * cell_volume
    total_loss = copper_loss + iron_loss
    temp_max = jnp.max(result.temperature)
    temp_excess = jnp.maximum(
        (temp_max - cfg.max_temperature) / cfg.max_temperature, 0.0
    )

    vol_pm = jnp.sum(result.rho_pm) / design_volume
    vol_iron = jnp.sum(result.rho_iron) / design_volume
    vol_copper = jnp.sum(result.rho_copper) / design_volume
    tv = sum(
        total_variation3d(field, cfg)
        for field in (result.rho_iron, result.rho_copper, result.rho_pm)
    )
    curvature = sum(
        curvature_penalty3d(field, cfg)
        for field in (result.rho_iron, result.rho_copper, result.rho_pm)
    )
    rotor_seed, stator_seed, copper_seed = _anchor_seeds(cfg)
    masks = domain_masks3d(cfg)
    rotor_solid = (result.rho_iron + result.rho_pm) * masks["rotor_design"]
    stator_iron = result.rho_iron * masks["stator_design"]
    steps = int(getattr(cfg, "connectivity_steps", 12))
    connectivity = (
        island_penalty3d(rotor_solid, rotor_seed, steps=steps)
        + island_penalty3d(stator_iron, stator_seed, steps=steps)
        + island_penalty3d(result.rho_copper, copper_seed, steps=steps)
    ) / 3.0
    ownership_consistency = jnp.mean(
        rotor_solid * (1.0 - result.rotor_ownership)
    )

    torque_term = -cfg.w_torque * torque_mean / cfg.tau_ref
    obj = (
        torque_term
        + cfg.w_pm * (vol_pm - cfg.V_pm_target) ** 2
        + cfg.w_iron * (vol_iron - cfg.V_iron_target) ** 2
        + cfg.w_copper * (vol_copper - cfg.V_copper_target) ** 2
        + cfg.w_tv * tv
        + getattr(cfg, "w_curvature", 1e-7) * curvature
        + getattr(cfg, "w_connectivity", 0.2) * connectivity
        + getattr(cfg, "w_ownership", 0.1) * ownership_consistency
        + cfg.w_loss * total_loss / cfg.loss_ref
        + cfg.w_temperature * temp_excess**2
        + cfg.w_ripple * torque_ripple
    )
    comps = {
        "obj": obj,
        "torque": torque_mean,
        "|torque|": jnp.abs(torque_mean),
        "torque_ripple": torque_ripple,
        "copper_loss_W": copper_loss,
        "iron_loss_W": iron_loss,
        "loss_W": total_loss,
        "temperature_max_C": temp_max,
        "vol_pm": vol_pm,
        "vol_iron": vol_iron,
        "vol_copper": vol_copper,
        "tv": tv,
        "curvature": curvature,
        "connectivity_penalty": connectivity,
        "ownership_penalty": ownership_consistency,
        "maxwell_residual": result.maxwell_residual,
        "thermal_residual": result.thermal_residual,
        "electric_residual": result.electric_residual,
        "source_divergence_residual": result.source_divergence_residual,
        "phase_balance_residual": result.phase_balance_residual,
    }
    return obj, comps


def make_loss3d(
    cfg: MotorConfig3D, angles: Sequence[float] | jnp.ndarray | None = None
):
    def loss(logits, rotor_logits, magnetization_raw, temperature):
        result = forward3d(
            cfg, logits, rotor_logits, magnetization_raw, angles, temperature
        )
        return objective3d(cfg, result)

    return loss


make_objective3d = make_loss3d
