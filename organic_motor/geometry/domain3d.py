"""Native 3-D motor masks including a fixed shaft and closed air gap."""

from __future__ import annotations

import jax.numpy as jnp

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.grid3d import meshgrid3d


def boundary_mask3d(shape: tuple[int, int, int]) -> jnp.ndarray:
    """Boolean mask of all six faces of a 3-D node box."""
    mask = jnp.zeros(shape, dtype=bool)
    mask = mask.at[0, :, :].set(True)
    mask = mask.at[-1, :, :].set(True)
    mask = mask.at[:, 0, :].set(True)
    mask = mask.at[:, -1, :].set(True)
    mask = mask.at[:, :, 0].set(True)
    return mask.at[:, :, -1].set(True)


def domain_masks3d(cfg: MotorConfig3D) -> dict[str, jnp.ndarray]:
    """Build a non-overlapping motor scaffold on the full 3-D grid.

    The mechanical air gap is a closed shell: its cylindrical side joins two
    annular end gaps.  Consequently the rotating design never touches the
    stationary stator at either the radial or axial boundary.  The shaft spans
    the complete box and is separately marked as fixed ownership.
    """
    X, Y, Z = meshgrid3d(cfg)
    cx, cy, cz = cfg.center
    radius = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    abs_z = jnp.abs(Z - cz)

    rotor_axial = abs_z <= cfg.rotor_half_length
    enclosed_axial = abs_z <= cfg.stator_half_length
    shaft = radius < cfg.R_shaft

    rotor_design = (
        (radius >= cfg.R_shaft)
        & (radius < cfg.R_rotor_outer)
        & rotor_axial
    )
    side_airgap = (
        (radius >= cfg.R_rotor_outer)
        & (radius < cfg.R_stator_inner)
        & rotor_axial
    )
    end_airgap = (
        (radius >= cfg.R_shaft)
        & (radius < cfg.R_stator_inner)
        & (abs_z > cfg.rotor_half_length)
        & enclosed_axial
    )
    airgap = side_airgap | end_airgap

    stator_design = (
        (radius >= cfg.R_stator_inner)
        & (radius < cfg.R_design)
        & enclosed_axial
    )
    design = rotor_design | stator_design
    winding = (
        (radius >= cfg.R_winding_inner)
        & (radius < cfg.R_winding_outer)
        & enclosed_axial
    )
    outer = ~(shaft | design | airgap)

    rotor = rotor_design
    stator = stator_design
    fixed = ~rotor
    boundary = boundary_mask3d(cfg.shape)
    return {
        "shaft": shaft,
        "fixed_shaft": shaft,
        "fixed_axis": shaft,
        "rotor_design": rotor_design,
        "stator_design": stator_design,
        "design": design,
        "side_airgap": side_airgap,
        "end_airgap": end_airgap,
        "airgap": airgap,
        "closed_airgap": airgap,
        "rotor": rotor,
        "stator": stator,
        "fixed": fixed,
        "winding": winding,
        "outer": outer,
        "boundary": boundary,
    }


def apply_fixed_shaft(
    field: jnp.ndarray, cfg: MotorConfig3D, value: float | jnp.ndarray = 0.0
) -> jnp.ndarray:
    """Set the non-rotating shaft to a fixed value without in-place mutation."""
    return jnp.where(domain_masks3d(cfg)["fixed_shaft"], value, field)


domain_masks = domain_masks3d
