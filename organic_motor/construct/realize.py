"""Bridge the constructive layer to the differentiable critic.

``realize`` turns a :class:`MaterialField` (disjoint SDFs of iron/copper/PM)
into the :class:`TopologyFields3D` that ``forward3d_fields`` consumes, plus
the magnetisation vector for the PM poles.  No masking is re-applied -- the
critic solves exactly the geometry that was constructed, including any
solids outside the optimisation design region (e.g. the cooling jacket).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.geometry.domain3d import domain_masks3d
from organic_motor.topology.density3d import TopologyFields3D
from organic_motor.construct.material import MaterialField


def realize(
    mf: MaterialField,
    cfg: MotorConfig3D,
    magnetization_raw: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
) -> tuple[TopologyFields3D, jnp.ndarray]:
    """Convert a constructed motor into critic-ready topology fields.

    Returns ``(fields, magnetization_raw)``.  ``fields`` carries the four
    continuous phase densities and a rotor-ownership mask derived from the
    solver's own rotor-design region, so rotation in the critic matches the
    geometry.  ``magnetization_raw`` is the per-voxel magnetisation direction
    (or zeros if none was supplied).

    Materials listed in ``mf.metadata["nonmagnetic_regions"]`` (SDF regions
    such as the retaining sleeve, which must be STRUCTURAL iron but
    NON-MAGNETIC like the real Inconel/carbon-fibre sleeve) are moved from
    ``rho_iron`` to ``rho_air`` before the solve: a mu_r = 2000 sleeve
    shunts the pole flux, and the shunt resolves ever better with grid
    refinement -- which is exactly the monotonic fine-grid torque decline
    the convergence ladder caught.  Structurally and visually they remain
    iron.
    """
    densities = mf.to_densities(bandwidth)
    regions = mf.metadata.get("nonmagnetic_regions", ()) if hasattr(mf, "metadata") else ()
    if regions:
        rho_iron = np.asarray(densities["iron"], dtype=np.float32)
        rho_air = np.asarray(densities["air"], dtype=np.float32)
        for sdf_field in regions:
            nm = np.asarray(sdf_field.to_density(bandwidth), dtype=np.float32)
            moved = np.minimum(nm, rho_iron)
            rho_iron = (rho_iron - moved).astype(np.float32)
            rho_air = (rho_air + moved).astype(np.float32)
        densities = dict(densities)
        densities["iron"] = rho_iron
        densities["air"] = rho_air
    rotor_design = np.asarray(domain_masks3d(cfg)["rotor_design"], dtype=np.float32)
    fields = TopologyFields3D(
        rho_air=jnp.asarray(densities["air"]),
        rho_iron=jnp.asarray(densities["iron"]),
        rho_copper=jnp.asarray(densities["copper"]),
        rho_pm=jnp.asarray(densities["pm"]),
        rotor_ownership=jnp.asarray(rotor_design),
    )
    if magnetization_raw is None:
        mag = np.zeros((3,) + cfg.shape, dtype=np.float32)
    else:
        mag = np.asarray(magnetization_raw, dtype=np.float32)
    if mag.shape != (3,) + cfg.shape:
        raise ValueError(
            f"magnetization_raw must be {(3,) + cfg.shape}, got {mag.shape}"
        )
    return fields, jnp.asarray(mag)
