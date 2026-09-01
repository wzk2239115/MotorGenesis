"""Differentiable native-3D solid mechanics utilities.

The linear solver is matrix free: its stiffness matvec is the Hessian-vector
product of a three-dimensional small-strain elastic energy.  All spatial
quantities use ``(nx, ny, nz, ...)`` arrays; no plane-strain extrusion is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from organic_motor.physics.linear import cg_fixed

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass
class Mechanics3DResult:
    """Fields returned by :func:`solve_linear_elasticity`."""

    displacement: Array
    strain: Array
    stress: Array
    von_mises: Array
    relative_residual: Array

    def tree_flatten(self):
        return (
            self.displacement,
            self.strain,
            self.stress,
            self.von_mises,
            self.relative_residual,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def _check_vector_field(value: Array, name: str) -> None:
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (nx, ny, nz, 3)")


def _spacing(spacing: float | Sequence[float], dtype) -> Array:
    h = jnp.asarray(spacing, dtype=dtype)
    if h.ndim == 0:
        h = jnp.repeat(h[None], 3)
    if h.shape != (3,):
        raise ValueError("spacing must be a scalar or a length-three sequence")
    return h


def _fixed_dofs(fixed_mask: Array, shape: tuple[int, ...]) -> Array:
    fixed = jnp.asarray(fixed_mask, dtype=bool)
    if fixed.shape == shape[:-1]:
        fixed = fixed[..., None]
    return jnp.broadcast_to(fixed, shape)


def lame_parameters(
    young_modulus: Array, poisson_ratio: Array
) -> tuple[Array, Array]:
    """Convert spatial Young's modulus and Poisson ratio to Lamé fields."""
    young_modulus = jnp.asarray(young_modulus)
    poisson_ratio = jnp.asarray(poisson_ratio)
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return lam, mu


def displacement_gradient(
    displacement: Array, spacing: float | Sequence[float] = 1.0
) -> Array:
    """Return ``du_i/dx_j`` on a native 3-D voxel grid.

    Forward differences are used in the interior and backward differences on
    the high face.  This keeps the output collocated with material fields.
    """
    _check_vector_field(displacement, "displacement")
    h = _spacing(spacing, displacement.dtype)
    gradients = []
    for axis in range(3):
        n = displacement.shape[axis]
        if n < 2:
            gradients.append(jnp.zeros_like(displacement))
            continue
        forward = (
            jnp.roll(displacement, -1, axis=axis) - displacement
        ) / h[axis]
        backward = (
            displacement - jnp.roll(displacement, 1, axis=axis)
        ) / h[axis]
        index = jnp.arange(n)
        shape = [1, 1, 1, 1]
        shape[axis] = n
        use_backward = (index == n - 1).reshape(shape)
        gradients.append(jnp.where(use_backward, backward, forward))
    return jnp.stack(gradients, axis=-1)


def small_strain(
    displacement: Array, spacing: float | Sequence[float] = 1.0
) -> Array:
    """Return the symmetric small-strain tensor ``(..., 3, 3)``."""
    grad_u = displacement_gradient(displacement, spacing)
    return 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))


def linear_stress(
    strain: Array,
    young_modulus: Array,
    poisson_ratio: Array,
    thermal_expansion: Array | float = 0.0,
    temperature_change: Array | float = 0.0,
) -> Array:
    """Return isotropic Cauchy stress, including thermal eigenstrain."""
    lam, mu = lame_parameters(young_modulus, poisson_ratio)
    eye = jnp.eye(3, dtype=strain.dtype)
    elastic_strain = strain - (
        jnp.asarray(thermal_expansion) * jnp.asarray(temperature_change)
    )[..., None, None] * eye
    trace = jnp.trace(elastic_strain, axis1=-2, axis2=-1)
    return (
        2.0 * mu[..., None, None] * elastic_strain
        + lam[..., None, None] * trace[..., None, None] * eye
    )


def von_mises_stress(stress: Array) -> Array:
    """Return the full three-dimensional von Mises equivalent stress."""
    eye = jnp.eye(3, dtype=stress.dtype)
    mean = jnp.trace(stress, axis1=-2, axis2=-1) / 3.0
    deviator = stress - mean[..., None, None] * eye
    return jnp.sqrt(
        jnp.maximum(1.5 * jnp.sum(deviator * deviator, axis=(-2, -1)), 0.0)
    )


def elastic_energy(
    displacement: Array,
    young_modulus: Array,
    poisson_ratio: Array,
    spacing: float | Sequence[float] = 1.0,
    thermal_expansion: Array | float = 0.0,
    temperature_change: Array | float = 0.0,
) -> Array:
    """Total small-strain elastic energy on the voxel grid."""
    h = _spacing(spacing, displacement.dtype)
    strain = small_strain(displacement, h)
    stress = linear_stress(
        strain,
        young_modulus,
        poisson_ratio,
        thermal_expansion,
        temperature_change,
    )
    eye = jnp.eye(3, dtype=strain.dtype)
    eigenstrain = (
        jnp.asarray(thermal_expansion) * jnp.asarray(temperature_change)
    )[..., None, None] * eye
    return 0.5 * jnp.prod(h) * jnp.sum(stress * (strain - eigenstrain))


def centrifugal_body_force(
    density: Array,
    coordinates: Array,
    angular_velocity: Array | float,
    axis: Array = jnp.array([0.0, 0.0, 1.0]),
    center: Array = jnp.zeros(3),
) -> Array:
    """Return centrifugal force density ``-rho * omega x (omega x r)``."""
    _check_vector_field(coordinates, "coordinates")
    axis = jnp.asarray(axis, dtype=coordinates.dtype)
    axis = axis / jnp.maximum(
        jnp.linalg.norm(axis), jnp.finfo(coordinates.dtype).tiny
    )
    omega = jnp.asarray(angular_velocity, dtype=coordinates.dtype) * axis
    omega = jnp.broadcast_to(omega, coordinates.shape)
    radius = coordinates - jnp.asarray(center, dtype=coordinates.dtype)
    return -jnp.asarray(density)[..., None] * jnp.cross(
        omega, jnp.cross(omega, radius)
    )


def linear_elasticity_operator(
    young_modulus: Array,
    poisson_ratio: Array,
    fixed_mask: Array,
    spacing: float | Sequence[float] = 1.0,
):
    """Build a symmetric matrix-free stiffness operator with fixed DOFs."""
    young_modulus = jnp.asarray(young_modulus)
    shape = young_modulus.shape + (3,)
    fixed = _fixed_dofs(fixed_mask, shape)
    free = (~fixed).astype(young_modulus.dtype)
    zero = jnp.zeros(shape, dtype=young_modulus.dtype)

    def mechanical_energy(u):
        return elastic_energy(u, young_modulus, poisson_ratio, spacing)

    gradient = jax.grad(mechanical_energy)

    def operator(u):
        projected = free * u
        stiffness_u = jax.jvp(gradient, (zero,), (projected,))[1]
        return free * stiffness_u + (1.0 - free) * u

    return operator


def solve_linear_elasticity(
    young_modulus: Array,
    poisson_ratio: Array,
    fixed_mask: Array,
    spacing: float | Sequence[float] = 1.0,
    body_force: Array | None = None,
    thermal_expansion: Array | float = 0.0,
    temperature_change: Array | float = 0.0,
    density: Array | None = None,
    coordinates: Array | None = None,
    angular_velocity: Array | float = 0.0,
    rotation_axis: Array = jnp.array([0.0, 0.0, 1.0]),
    rotation_center: Array = jnp.zeros(3),
    initial_displacement: Array | None = None,
    maxiter: int = 100,
    tol: float | None = 1e-6,
) -> Mechanics3DResult:
    """Solve heterogeneous 3-D linear elasticity with matrix-free CG.

    ``body_force`` is force per volume.  A centrifugal load is added when
    ``density`` and ``coordinates`` are supplied.  ``fixed_mask`` may be a
    scalar voxel mask or a component-wise ``(..., 3)`` mask.
    """
    young_modulus = jnp.asarray(young_modulus)
    if young_modulus.ndim != 3:
        raise ValueError("young_modulus must have shape (nx, ny, nz)")
    shape = young_modulus.shape + (3,)
    fixed = _fixed_dofs(fixed_mask, shape)
    free = (~fixed).astype(young_modulus.dtype)
    h = _spacing(spacing, young_modulus.dtype)
    zero = jnp.zeros(shape, dtype=young_modulus.dtype)
    force = zero if body_force is None else jnp.asarray(body_force)
    _check_vector_field(force, "body_force")
    force = jnp.broadcast_to(force, shape)
    if (density is None) != (coordinates is None):
        raise ValueError("density and coordinates must be supplied together")
    if density is not None:
        force = force + centrifugal_body_force(
            density,
            coordinates,
            angular_velocity,
            rotation_axis,
            rotation_center,
        )

    operator = linear_elasticity_operator(
        young_modulus, poisson_ratio, fixed, h
    )
    thermal_gradient = jax.grad(
        lambda u: elastic_energy(
            u,
            young_modulus,
            poisson_ratio,
            h,
            thermal_expansion,
            temperature_change,
        )
    )(zero)
    rhs = free * (force * jnp.prod(h) - thermal_gradient)
    x0 = zero if initial_displacement is None else free * initial_displacement

    # A positive scalar stiffness scale is a robust matrix-free preconditioner
    # for strongly heterogeneous topology fields without assembling a matrix.
    lam, mu = lame_parameters(young_modulus, poisson_ratio)
    stiffness_scale = (
        jnp.prod(h)
        * (lam + 2.0 * mu)[..., None]
        * jnp.sum(1.0 / (h * h))
    )
    diagonal = free * jnp.maximum(stiffness_scale, 1e-12) + (1.0 - free)
    displacement = cg_fixed(
        operator,
        rhs,
        x0,
        lambda value: value / diagonal,
        maxiter,
        tol,
    )
    displacement = free * displacement
    strain = small_strain(displacement, h)
    stress = linear_stress(
        strain,
        young_modulus,
        poisson_ratio,
        thermal_expansion,
        temperature_change,
    )
    residual = rhs - operator(displacement)
    tiny = jnp.finfo(displacement.dtype).tiny
    relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
        jnp.linalg.norm(rhs), tiny
    )
    return Mechanics3DResult(
        displacement=displacement,
        strain=strain,
        stress=stress,
        von_mises=von_mises_stress(stress),
        relative_residual=relative_residual,
    )


def neo_hookean_energy_density(
    deformation_gradient: Array,
    young_modulus: Array,
    poisson_ratio: Array,
) -> Array:
    """Compressible Neo-Hookean energy density for future nonlinear solves."""
    lam, mu = lame_parameters(young_modulus, poisson_ratio)
    jacobian = jnp.linalg.det(deformation_gradient)
    safe_jacobian = jnp.maximum(jacobian, 1e-12)
    log_j = jnp.log(safe_jacobian)
    i1 = jnp.sum(
        deformation_gradient * deformation_gradient, axis=(-2, -1)
    )
    return (
        0.5 * mu * (i1 - 3.0)
        - mu * log_j
        + 0.5 * lam * log_j * log_j
    )


def neo_hookean_first_piola(
    deformation_gradient: Array,
    young_modulus: Array,
    poisson_ratio: Array,
) -> Array:
    """First Piola stress matching :func:`neo_hookean_energy_density`."""
    lam, mu = lame_parameters(young_modulus, poisson_ratio)
    jacobian = jnp.linalg.det(deformation_gradient)
    log_j = jnp.log(jnp.maximum(jacobian, 1e-12))
    inverse_transpose = jnp.swapaxes(
        jnp.linalg.inv(deformation_gradient), -1, -2
    )
    return (
        mu[..., None, None] * (deformation_gradient - inverse_transpose)
        + lam[..., None, None]
        * log_j[..., None, None]
        * inverse_transpose
    )


def air_gap_collision_penalty(
    displacement: Array,
    signed_gap: Array,
    surface_normal: Array,
    penalty_stiffness: Array | float,
    spacing: float | Sequence[float] = 1.0,
    contact_mask: Array | float = 1.0,
) -> Array:
    """Quadratic penalty for closure beyond a spatial signed air-gap field.

    ``surface_normal`` points in the direction that opens the gap, hence the
    deformed gap is ``signed_gap + dot(displacement, surface_normal)``.
    """
    _check_vector_field(displacement, "displacement")
    _check_vector_field(surface_normal, "surface_normal")
    h = _spacing(spacing, displacement.dtype)
    deformed_gap = signed_gap + jnp.sum(
        displacement * surface_normal, axis=-1
    )
    penetration = jax.nn.relu(-deformed_gap)
    return (
        0.5
        * jnp.prod(h)
        * jnp.sum(contact_mask * penalty_stiffness * penetration * penetration)
    )
