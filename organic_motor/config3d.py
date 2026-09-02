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
    # Maxwell stress surface radius: must lie in the true AIR gap.  The
    # constructed rotor has surface magnets to r=28.4mm and a sleeve to
    # 29.4mm, so the stator-side gap (29.4..30mm) is the only valid
    # integration surface -- the 2-D default (26mm) sits inside the
    # magnetized material, where the simple B-stress integral is invalid.
    R_torque: float = 0.0297
    # A wider coarse-grid gap than the converged 2-D benchmark keeps the
    # mechanical separation representable at the default 48^2 cross-section.
    R_stator_inner: float = 0.030
    stack_length: float = 0.060
    axial_airgap: float = 0.001
    excitation_mode: str = "terminal"
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
