"""E2: ReferenceSolverAdapter — external solver interface for cross-checks.

The audit requires that claims of engineering accuracy be backed by
comparison against independent solvers (FEMM, Elmer, FEniCS, measured
data).  This module defines the adapter contract and a registry; concrete
adapters are added when the corresponding tool/licence is available in
the execution environment.  Nothing silently fabricates a reference: an
unavailable adapter raises, and benchmarks refuse to run.

Contract: an adapter consumes the SAME geometry/material/boundary
description (a ModelArtifact) and returns a ReferenceResult with field
and integral quantities comparable to our own solver's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ReferenceResult:
    """Comparable outputs of an independent solve."""
    solver_name: str
    solver_version: str
    torque_Nm: dict | None = None          # angle -> torque (electrical setup)
    flux_linkage_Wb: float | None = None
    b_on_axis_T: dict | None = None        # z [m] -> |B| for coil benchmarks
    peak_B_T: float | None = None
    degrees_of_freedom: int | None = None
    residual: float | None = None
    wall_time_s: float | None = None
    notes: str = ""


@runtime_checkable
class ReferenceSolverAdapter(Protocol):
    """Interface every external reference solver must implement."""

    name: str

    def available(self) -> bool:
        """True only when the tool + licence are usable in this env."""
        ...

    def solve_magnetostatic(
        self, artifact, *, angles=None, options=None,
    ) -> ReferenceResult:
        """Solve the artifact geometry magnetostatically."""
        ...


_REGISTRY: dict[str, ReferenceSolverAdapter] = {}


def register_adapter(adapter: ReferenceSolverAdapter) -> None:
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> ReferenceSolverAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"reference solver {name!r} not registered; available: "
            f"{sorted(_REGISTRY) or 'none'}"
        )
    return _REGISTRY[name]


def list_adapters() -> dict[str, bool]:
    """name -> availability, for UI/reporting."""
    return {n: bool(a.available()) for n, a in _REGISTRY.items()}


class UnavailableAdapter:
    """Placeholder for tools not installed — fails loudly, never fakes."""

    def __init__(self, name: str, reason: str):
        self.name = name
        self._reason = reason

    def available(self) -> bool:
        return False

    def solve_magnetostatic(self, *args, **kwargs) -> ReferenceResult:
        raise RuntimeError(
            f"reference solver {self.name!r} unavailable: {self._reason}; "
            "install/configure it before claiming reference-verified accuracy"
        )


# Register the known-but-absent references explicitly so reports can list
# them instead of pretending they exist.
register_adapter(UnavailableAdapter(
    "femm", "FEMM (femm-python) not installed in this environment"))
register_adapter(UnavailableAdapter(
    "elmer", "ElmerFEM mesh/solve not available on PATH"))


# ---------------------------------------------------------------------------
# EXECUTABLE analytic reference: the Biot-Savart coil benchmark.
# This one IS available everywhere (pure numpy) and exercises the full
# adapter contract end-to-end; raw inputs/outputs are persisted by the
# benchmark runner for auditability (audit item 8).
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402


class BiotSavartCoilAdapter:
    """Analytic on-axis field of a circular current loop.

    solve_magnetostatic(coil) with coil = {"radius_m", "current_A",
    "z_samples_m"} returns ReferenceResult.b_on_axis computed from
    B_z = mu0*I*R^2 / (2*(R^2+z^2)^{3/2}) — the independent reference our
    voxel solver is compared against (tests/test_reference_coil.py).
    """

    name = "biot_savart_coil"

    def available(self) -> bool:
        return True

    def solve_magnetostatic(self, coil, *, angles=None, options=None):
        R = float(coil["radius_m"])
        I = float(coil["current_A"])
        zs = np.asarray(coil.get("z_samples_m", [0.003, 0.010, 0.030]))
        mu0 = 4e-7 * np.pi
        bz = {float(z): mu0 * I * R ** 2 / (2.0 * (R ** 2 + z ** 2) ** 1.5)
              for z in zs}
        return ReferenceResult(
            solver_name=self.name,
            solver_version="analytic",
            b_on_axis_T=bz,
            degrees_of_freedom=0,
            residual=0.0,
            notes="closed-form circular loop, on-axis B_z",
        )


register_adapter(BiotSavartCoilAdapter())


def run_reference_benchmark(out_path=None) -> dict:
    """Run the coil benchmark against OUR solver and persist raw I/O.

    Returns the cross-check table; writes JSON (inputs, reference values,
    our values, errors) so the comparison is reproducible (audit item 8:
    接通一个实际可执行的独立参考算例，保存原始输入输出).
    """
    import json
    from pathlib import Path as _P

    R = 0.020
    z_mm_list = (3, 5, 10, 20, 30)
    zs = np.array(z_mm_list, dtype=float) * 1e-3

    adapter = get_adapter("biot_savart_coil")
    ref = adapter.solve_magnetostatic(
        {"radius_m": R, "current_A": 1.0, "z_samples_m": zs.tolist()})

    # Our solver side: reuse the reference-coil battery machinery.
    import jax.numpy as jnp
    from organic_motor.config3d import MotorConfig3D
    from organic_motor.construct.material import MaterialField
    from organic_motor.construct.field import SDFVoxelField
    from organic_motor.construct.realize import realize
    from organic_motor.optimization.objective3d import forward3d_fields

    cfg = MotorConfig3D(
        shape=(64, 64, 64), excitation_mode="impressed",
        filt_radius=0.0, projection_beta=0.0,
        maxwell_maxiter=240, thermal_maxiter=10, electric_maxiter=10,
    )
    dx, dy, dz = cfg.spacing
    ox, oy, oz = cfg.origin
    cx, cy, cz = cfg.center
    I_per_turn = cfg.current_density_peak * 1e-6

    n_pts = 64
    angles = np.linspace(0, 2 * np.pi, n_pts + 1)
    pts = np.column_stack([
        cx + R * np.cos(angles), cy + R * np.sin(angles),
        np.full(n_pts + 1, cz),
    ])
    reg = [{
        "points": pts, "physical_points": pts[:-1],
        "turn_map": np.ones(n_pts + 1, dtype=int),
        "phase": 0, "polarity": 1, "n_turns": 1, "tooth": 0,
        "cross_section_area": 1e-6, "band_radius": 0.001,
        "solver_closure": False,
    }]
    mf = MaterialField(shape=cfg.shape, spacing=cfg.spacing, origin=cfg.origin)
    mf.add(SDFVoxelField(np.full(cfg.shape, 1.0, dtype=np.float32),
                         cfg.spacing, cfg.origin), "air")
    mag = np.zeros((3,) + cfg.shape, dtype=np.float32)
    fields, _ = realize(mf, cfg, mag)
    result = forward3d_fields(cfg, fields, mag, jnp.asarray([0.0]),
                              centerline_registry=reg)
    Bz = np.asarray(result.flux_density)[..., 2]
    ix, iy = cfg.shape[0] // 2, cfg.shape[1] // 2

    checks = []
    for z_m, b_ref in ref.b_on_axis_T.items():
        z_vox = int((cz + z_m - oz) / dz)
        b_ours = float(Bz[ix, iy, z_vox])
        b_ana_scale = I_per_turn  # analytic computed at 1 A
        cc = CrossCheck(
            quantity=f"B_z(z={z_m*1e3:.0f} mm)",
            ours=b_ours, reference=b_ref * I_per_turn, unit="T",
            tolerance=0.25,  # verified envelope: -3% near, -20% far
        )
        cc.evaluate()
        checks.append({
            "quantity": cc.quantity, "ours_T": b_ours,
            "reference_T": b_ref * I_per_turn,
            "rel_error": float(cc.rel_error), "passed": bool(cc.passed),
            "tolerance": float(cc.tolerance),
        })

    record = {
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {"radius_m": R, "current_A": I_per_turn,
                   "z_samples_mm": list(z_mm_list),
                   "grid": list(cfg.shape), "maxiter": 240},
        "solver": "organic_motor forward3d_fields (voxel, impressed)",
        "reference": "analytic Biot-Savart circular loop",
        "checks": checks,
        "verified_envelope": (
            "on-axis B_z within 25% for z in [3, 30] mm at R=20 mm, "
            "64^3 grid, maxiter=240; reversal and 2x-scaling linearity "
            "verified exactly (see tests/test_reference_coil.py). This "
            "verifies the CURRENT DEPOSIT + magnetostatic solve on a "
            "coil — NOT motor-level torque accuracy."
        ),
    }
    if out_path is None:
        out_path = _P(__file__).resolve().parent.parent / "reports" / \
            "reference" / "coil_benchmark.json"
    out_path = _P(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


@dataclass
class CrossCheck:
    """One quantity compared between our solver and a reference."""
    quantity: str
    ours: float
    reference: float
    unit: str = ""
    rel_error: float = field(default=0.0)

    def __post_init__(self):
        denom = abs(self.reference) if self.reference else 1.0
        self.rel_error = abs(self.ours - self.reference) / denom

    passed: bool = False
    tolerance: float = 0.05

    def evaluate(self) -> bool:
        self.passed = self.rel_error <= self.tolerance
        return self.passed
