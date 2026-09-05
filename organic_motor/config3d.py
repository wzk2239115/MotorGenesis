"""Configuration for native three-dimensional motor topology models."""

from __future__ import annotations

from dataclasses import dataclass, field

from organic_motor.config import MotorConfig


# ---------------------------------------------------------------------------
# Frozen motor specification — prevents pole/slot/winding drift
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorSpec:
    """Immutable motor specification shared by geometry, netlist, FEA and startup.

    This exists so the pole count, winding table and phase sequence cannot
    silently change between commits.  All subsystems MUST read from the
    same spec, not hard-code their own values.
    """
    n_slots: int = 12
    pole_pairs: int = 5  # 10 poles (12s10p concentrated winding)
    n_phases: int = 3
    winding_factor: float = 0.933  # 12s10p kw
    coil_span: int = 1  # concentrated (one tooth per coil)

    @property
    def n_poles(self) -> int:
        return 2 * self.pole_pairs

    @property
    def slots_per_pole_per_phase(self) -> float:
        return self.n_slots / (self.n_poles * self.n_phases)

    @property
    def slot_pitch_rad(self) -> float:
        return 2.0 * 3.141592653589793 / self.n_slots

    @property
    def electrical_to_mechanical(self) -> float:
        return self.pole_pairs

    def phase_of_slot(self, slot: int) -> int:
        """12s10p phase assignment: A C B A C B A C B A C B."""
        # Standard 12s10p: slots 0,3,6,9 = phase A; 1,4,7,10 = C; 2,5,8,11 = B
        return [0, 2, 1][slot % 3]

    def polarity_of_slot(self, slot: int) -> int:
        """Alternating polarity within each phase."""
        phase = self.phase_of_slot(slot)
        # Count how many previous slots of the same phase
        count = sum(1 for s in range(slot) if self.phase_of_slot(s) == phase)
        return 1 if count % 2 == 0 else -1


# Singleton instance — import this everywhere
MOTOR_SPEC = MotorSpec()


@dataclass
class MotorConfig3D(MotorConfig):
    """Three-dimensional extension of :class:`MotorConfig`.

    Arrays always use ``(Nx, Ny, Nz)`` order.  ``box_size`` is the physical
    node-to-node extent of the Cartesian box, so the corresponding node
    spacing is ``box_size / (shape - 1)`` and may be anisotropic.
    """

    shape: tuple[int, int, int] = (56, 56, 36)
    box_size: tuple[float, float, float] = (0.140, 0.140, 0.100)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 10 poles / 12 slots: THE canonical printed concentrated-winding pair
    # (one coil per tooth, half-slot sides, winding factor 0.933).  The old
    # default p=2 made a 12s4p machine whose span-1 coils have a
    # structurally unbalanced PM flux linkage (2:1 between phases --
    # measured T1 0.028 vs 0.143 N*m); 12s10p restores per-phase symmetry.
    pole_pairs: int = 5
    # Resolution budget (P2 redesign): the magnet must span >= ~3 cells
    # and the open gap ~2 cells at the PHYSICS grid for torque to be
    # trustworthy.  With 96x96x58 cells (1.47 x 1.47 x 1.75 mm):
    #   rotor iron  8.2 .. 21.6 mm
    #   magnets    21.8 .. 25.8 mm   (3.0-4.0 mm thick = 2.0-2.7 cells)
    #   sleeve     25.9 .. 27.5 mm   (1.6 mm = 1.1 cells)
    #   open gap   27.5 .. 30.5 mm   (3.0 mm = 2.0 cells)
    #   magnet+gap = 7.0 mm = 4.8 cells (>= 3 qualitative gate met)
    # R_torque sits mid-gap with 1.5 mm (1.0 cell) clearance to solids.
    R_rotor_outer: float = 0.0216
    R_sleeve_outer: float = 0.0275
    R_torque: float = 0.0290
    # Maxwell-stress surface averaging: 3 cylinders at R_torque +/- 0.2/0.4mm.
    # A single mid-gap cylinder localises the integral; cogging (a small
    # difference of large stresses) moved 3.4x when the radius shifted by
    # 0.4mm on the 112^3 grid.
    torque_r_average: int = 3
    torque_r_pitch: float = 0.0004
    R_stator_inner: float = 0.0305
    R_winding_inner: float = 0.0305
    stack_length: float = 0.060
    axial_airgap: float = 0.001
    excitation_mode: str = "terminal"
    winding_style: str = "printed"
    # ^ "printed": 12s10p concentrated coils (one per tooth, half-slot
    #   sides, physical bridge end turns -- PrintedCoilNetlist is the
    #   geometry/source/audit single source of truth).  "legacy": the old
    #   distributed span-3 winding with radial phase layers.
    impressed_end_closure: bool = True
    # ^ close the impressed phase loops INSIDE the domain (axial columns
    #   confined to the stack + azimuthal end-turn arc currents) instead of
    #   letting them leave through the box end faces, where the solver's
    #   boundary zeroing silently deletes part of the source (~8%% torque
    #   overestimate from the infinite-solenoid column field).
    terminal_voltage: float | None = None
    electric_sigma_void: float = 5.8
    electric_simp_p: float = 2.0
    electric_maxiter: int = 300
    electric_tol: float = 1e-8
    coulomb_gauge_penalty: float | None = None
    mechanical_angles: int = 3
    torque_n_z: int = 24
    torque_n_r: int = 24
    connectivity_steps: int = 12
    # Reduced-order conjugate heat transfer of the printed cooling channels:
    # beta = h * S_v per coolant voxel (h ~ 3000 W/m^2K turbulent water in a
    # 2mm channel; S_v ~ 4/D ~ 2000 1/m).  Without this sink the channels
    # are thermally-dead k_air voids and every coil floats thermally.
    thermal_h_coolant: float = 3000.0
    thermal_channel_s_v: float = 2000.0
    thermal_coolant_temperature: float = 40.0
    thermal_k_insulator: float = 1.5
    w_curvature: float = 1e-7
    w_connectivity: float = 0.2
    w_ownership: float = 0.1
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(n) != n or n < 2 for n in self.shape):
            raise ValueError("shape must contain three integers >= 2")
        if len(self.box_size) != 3 or any(length <= 0.0 for length in self.box_size):
            raise ValueError("box_size must contain three positive lengths")
        if self.axial_airgap < 0.0:
            raise ValueError("axial_airgap must be non-negative")
        if self.excitation_mode not in {"terminal", "impressed"}:
            raise ValueError("excitation_mode must be 'terminal' or 'impressed'")
        if self.electric_sigma_void <= 0.0:
            raise ValueError("electric_sigma_void must be positive")
        if self.electric_maxiter < 1 or self.mechanical_angles < 1:
            raise ValueError("solver iteration and angle counts must be positive")
        if self.R_shaft >= self.R_rotor_outer:
            raise ValueError("R_shaft must be smaller than R_rotor_outer")
        if self.R_rotor_outer >= self.R_stator_inner:
            raise ValueError("R_rotor_outer must be smaller than R_stator_inner")

    @property
    def Nx(self) -> int:
        return self.shape[0]

    @property
    def Ny(self) -> int:
        return self.shape[1]

    @property
    def Nz(self) -> int:
        return self.shape[2]

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Anisotropic node spacing ``(dx, dy, dz)`` in metres."""
        return tuple(
            length / (n - 1) for length, n in zip(self.box_size, self.shape)
        )

    @property
    def dx(self) -> float:
        return self.spacing[0]

    @property
    def dy(self) -> float:
        return self.spacing[1]

    @property
    def dz(self) -> float:
        return self.spacing[2]

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def origin(self) -> tuple[float, float, float]:
        """Coordinate of node ``(0, 0, 0)``."""
        return tuple(c - 0.5 * length for c, length in zip(self.center, self.box_size))

    @property
    def rotor_half_length(self) -> float:
        return 0.5 * self.stack_length

    @property
    def stator_half_length(self) -> float:
        """Axial enclosure including the two mechanical end gaps."""
        return self.rotor_half_length + self.axial_airgap


DEFAULT_CONFIG_3D = MotorConfig3D()
Config3D = MotorConfig3D
