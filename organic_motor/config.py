"""Central configuration for the free-topology motor benchmark.

Everything (geometry, materials, physics, optimization) is driven off a single
:class:`MotorConfig` dataclass so an experiment is fully reproducible from one
object.  All geometric lengths are in metres; magnetic fields in tesla; implies
SI units throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MotorConfig:
    # --- simulation box (square, [-L, L]^2) ---
    L: float = 0.060            # half width [m]
    N: int = 128                # grid nodes per side

    # --- radial partitions for an inner-rotor machine ---
    R_shaft: float = 0.008
    R_rotor_outer: float = 0.025
    R_stator_inner: float = 0.027
    R_design: float = 0.050
    R_torque: float = 0.026

    # Distributed three-phase winding source.  This phase-1 excitation model
    # represents the axial current sheet in the stator slot band; copper will
    # become an explicit topology phase in the next model tier.
    R_winding_inner: float = 0.030
    R_winding_outer: float = 0.043
    pole_pairs: int = 2
    current_density_peak: float = 5.0e6  # A/m^2
    electrical_phase_offset: float = 0.0
    speed_rpm: float = 3000.0
    stack_length: float = 0.020

    # --- materials (SI) ---
    mu0: float = 1.25663706127e-6    # vacuum permeability
    mu_r_iron: float = 2000.0        # soft-magnetic relative permeability
    mu_r_pm: float = 1.05            # permanent-magnet relative permeability (~air)
    B_r: float = 1.2                 # PM remanence [T]  ->  M_sat = B_r/mu0
    rho_iron_kg: float = 7700.0      # mass density iron [kg/m^3]
    rho_pm_kg: float = 7500.0        # mass density NdFeB [kg/m^3]
    rho_copper_kg: float = 8960.0
    sigma_copper: float = 5.8e7       # S/m at reference temperature
    B_sat_iron: float = 1.65           # T, soft saturation design limit

    # --- reduced-order loss and thermal model ---
    # Iron loss proxy q_fe = k_fe * f_e * (|B| / B_ref)^2 * rho_fe.
    # It is intentionally replaceable by fitted Steinmetz data later.
    iron_loss_coeff: float = 140.0     # W/m^3/Hz at B_ref
    iron_loss_B_ref: float = 1.0
    thermal_k_air: float = 0.026       # W/(m K)
    thermal_k_iron: float = 25.0
    thermal_k_copper: float = 385.0
    thermal_k_pm: float = 8.0
    ambient_temperature: float = 25.0  # degC
    max_temperature: float = 120.0     # degC
    thermal_maxiter: int = 1200
    thermal_tol: float = 1e-8

    # --- topology parameterisation ---
    sm_temp_init: float = 1.0        # softmax temperature (annealed down)
    sm_temp_final: float = 0.25
    simp_p: float = 2.0              # SIMP penalisation exponent for iron
    filt_radius: float = 0.004       # density filter radius [m]
    filt_power: float = 1.0          # Helmholtz filter power (1 = linear)
    projection_beta: float = 0.0     # Heaviside projection sharpness (0 = off)

    # --- physics solver ---
    maxwell_tol: float = 1e-8
    maxwell_maxiter: int = 800
    torque_method: str = "maxwell"       # "maxwell" (air gap) or "lorentz"
    filter_tol: float = 1e-9

    # torque samples for Maxwell stress integration
    n_theta: int = 512

    # --- objective weights ---
    w_torque: float = 1.0
    tau_ref: float = 1000.0         # reference torque [N.m/m] normalising the torque term to O(1)
    w_mass: float = 1.0             # objective is -(w_torque*torque)/tau_ref style; see objective.py
    w_pm: float = 8.0               # PM *target-volume* penalty weight (grown by pen_growth)
    w_iron: float = 3.0             # iron *target-volume* penalty weight (grown by pen_growth)
    w_copper: float = 4.0
    w_tv: float = 0.0005            # total-variation (feature size) penalty
    w_mag_smooth: float = 0.0       # magnetisation direction smoothness (M already Helmholtz-filtered)
    w_ripple: float = 0.0           # torque-ripple penalty (ripple experiment)
    w_loss: float = 0.02
    loss_ref: float = 1000.0          # W/m, objective normalization
    w_temperature: float = 2.0
    w_saturation: float = 0.5
    mass_eps: float = 1e-9

    # target volume fractions (of the design annulus) -- soft constraints that
    # keep material present and avoid the trivial "all air" optimum while still
    # maximising torque per unit mass
    V_pm_target: float = 0.22
    V_iron_target: float = 0.38
    V_copper_target: float = 0.18

    # penalty (augmented-Lagrangian) schedule: volume-target weights are grown
    # geometrically through the run to approach hard volume constraints.
    pen_growth: float = 1.6
    pen_growth_every: int = 40

    # --- optimisation ---
    lr: float = 0.05
    steps: int = 400
    seed: int = 0
    checkpoint_every: int = 20
    generate_growth_report: bool = True
    growth_report_max_frames: int = 12

    # output directory (None -> default under experiments/out)
    out_dir: str | None = None

    @property
    def h(self) -> float:
        """cell size [m]."""
        return 2.0 * self.L / (self.N - 1)

    @property
    def R_gap(self) -> float:
        """Compatibility alias: outer radius of the mechanical air gap."""
        return self.R_stator_inner

    @property
    def R_split(self) -> float:
        """Compatibility alias for the moving rotor boundary."""
        return self.R_rotor_outer

    @property
    def nu_air(self) -> float:
        return 1.0 / self.mu0

    @property
    def nu_iron(self) -> float:
        return 1.0 / (self.mu0 * self.mu_r_iron)

    @property
    def nu_pm(self) -> float:
        return 1.0 / (self.mu0 * self.mu_r_pm)

    @property
    def M_sat(self) -> float:
        """saturation magnetisation of the PM [A/m]; remanence divided by mu0."""
        return self.B_r / self.mu0

    @property
    def electrical_frequency(self) -> float:
        """Fundamental electrical frequency [Hz]."""
        return self.pole_pairs * self.speed_rpm / 60.0


DEFAULT_CONFIG = MotorConfig()
