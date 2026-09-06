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
