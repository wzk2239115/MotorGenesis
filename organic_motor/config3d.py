"""Configuration for native three-dimensional motor topology models."""

from __future__ import annotations

from dataclasses import dataclass

from organic_motor.config import MotorConfig


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
