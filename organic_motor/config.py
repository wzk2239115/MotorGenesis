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

    # --- radial partitions (fixed geometry, the *benchmark* envelope) ---
    #   r <  R_shaft : central shaft (fixed air)
    #   R_shaft..R_gap : air gap (fixed air) -- torque surface lives here
    #   R_gap..R_design : FREE DESIGN DOMAIN (materials are optimised here)
    #   R_design..L : outer air ring up to the Dirichlet boundary
    R_shaft: float = 0.008
    R_gap: float = 0.016
    R_design: float = 0.050

    # rotor/stator split (used by the ripple experiment to rotate the rotor)
    R_split: float = 0.033

    # torque evaluation circle (in the air gap)
    R_torque: float = 0.012

    # --- materials (SI) ---
    mu0: float = 1.25663706127e-6    # vacuum permeability
    mu_r_iron: float = 2000.0        # soft-magnetic relative permeability
    mu_r_pm: float = 1.05            # permanent-magnet relative permeability (~air)
    B_r: float = 1.2                 # PM remanence [T]  ->  M_sat = B_r/mu0
    rho_iron_kg: float = 7700.0      # mass density iron [kg/m^3]
    rho_pm_kg: float = 7500.0        # mass density NdFeB [kg/m^3]

    # --- topology parameterisation ---
    sm_temp_init: float = 1.0        # softmax temperature (annealed down)
    sm_temp_final: float = 0.25
    simp_p: float = 2.0              # SIMP penalisation exponent for iron
    filt_radius: float = 0.004       # density filter radius [m]
    filt_power: float = 1.0          # Helmholtz filter power (1 = linear)
    projection_beta: float = 0.0     # Heaviside projection sharpness (0 = off)

    # --- physics solver ---
    maxwell_tol: float = 1e-8
    maxwell_maxiter: int = 400

    # torque samples for Maxwell stress integration
    n_theta: int = 512

    # --- objective weights ---
    w_torque: float = 1.0
    tau_ref: float = 1000.0         # reference torque [N.m/m] normalising the torque term to O(1)
    w_mass: float = 1.0             # objective is -(w_torque*torque)/tau_ref style; see objective.py
    w_pm: float = 8.0               # PM *target-volume* penalty weight (grown by pen_growth)
    w_iron: float = 3.0             # iron *target-volume* penalty weight (grown by pen_growth)
    w_tv: float = 0.0005            # total-variation (feature size) penalty
    w_mag_smooth: float = 0.0       # magnetisation direction smoothness (M already Helmholtz-filtered)
    w_ripple: float = 0.0           # torque-ripple penalty (ripple experiment)
    mass_eps: float = 1e-9

    # target volume fractions (of the design annulus) -- soft constraints that
    # keep material present and avoid the trivial "all air" optimum while still
    # maximising torque per unit mass
    V_pm_target: float = 0.22
    V_iron_target: float = 0.38

    # penalty (augmented-Lagrangian) schedule: volume-target weights are grown
    # geometrically through the run to approach hard volume constraints.
    pen_growth: float = 1.6
    pen_growth_every: int = 40

    # --- optimisation ---
    lr: float = 0.05
    steps: int = 400
    seed: int = 0

    # output directory (None -> default under experiments/out)
    out_dir: str | None = None

    @property
    def h(self) -> float:
        """cell size [m]."""
        return 2.0 * self.L / (self.N - 1)

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


DEFAULT_CONFIG = MotorConfig()