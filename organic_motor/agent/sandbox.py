"""Sandboxed execution of agent-generated motor construction code.

The agent emits Python that calls the LEAP 71-style construct API to build a
:class:`MaterialField`.  We execute it in a restricted namespace exposing only
the geometry primitives, implicits, lattices and ``numpy`` -- no file I/O, no
imports, no introspection -- so a misbehaving model cannot escape the design
vocabulary.  This is API-surface sandboxing, not OS-level isolation; for
untrusted models run the loop in a container.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np

from organic_motor.config3d import MotorConfig3D
from organic_motor.construct import (
    MaterialField,
    SDFVoxelField,
    annular_sector,
    boolean_add,
    boolean_intersect,
    boolean_subtract,
    box,
    capsule,
    cylinder,
    cylinder_z,
    empty_field,
    from_implicit,
    gyroid,
    gyroid_sheet,
    offset,
    plane,
    rounded_box,
    sheet_lattice,
    shell,
    sphere,
    strut_lattice,
    torus,
)
from organic_motor.construct.objects import (
    FieldDrivenCoolingJacket,
    FieldDrivenMagnets,
    FieldDrivenStatorYoke,
    FunctionalVoids,
    HelicalCoolingChannels,
    MotorHousing,
    RotorCore,
    RotorSleeve,
    ShaftAndBearings,
    StatorCore,
    StatorSegmentation,
    StructuralContinuity,
    SurfaceMagnets,
    DistributedWinding,
    Winding3D,
)
from organic_motor.construct.fields_motor import (
    airgap_B,
    centrifugal_stress,
    current_density,
    joule_heat,
    magnetization_field,
)
from organic_motor.construct.field import ScalarField, VectorField
from organic_motor.construct.winding_netlist import CoilNetlist, default_netlist
from organic_motor.construct.modulation import (
    ConstField,
    FuncField,
    LineMod,
    SurfaceMod,
    field_sample_grid,
)


_SAFE_IMPORTS = {"numpy", "math", "np"}


def _safe_import(name, *args, **kwargs):
    """An import that admits only numpy/math, mirroring the exposed namespace."""
    if name in _SAFE_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"{name!r} is not available in the agent sandbox; use the exposed primitives")


def make_namespace(cfg: MotorConfig3D) -> dict:
    """The restricted vocabulary an agent builds motors with.

    Everything here is total and deterministic (SDF Booleans never fail), so
    the only failure mode is a geometry that produces no material -- which the
    caller detects and reports back as a critic score of zero.
    """
    return {
        "np": np,
        "math": math,
        "cfg": cfg,
        "MaterialField": MaterialField,
        "SDFVoxelField": SDFVoxelField,
        "from_implicit": from_implicit,
        "empty_field": empty_field,
        "boolean_add": boolean_add,
        "boolean_subtract": boolean_subtract,
        "boolean_intersect": boolean_intersect,
        "offset": offset,
        "shell": shell,
        "sphere": sphere,
        "cylinder": cylinder,
        "cylinder_z": cylinder_z,
        "box": box,
        "rounded_box": rounded_box,
        "torus": torus,
        "plane": plane,
        "capsule": capsule,
        "annular_sector": annular_sector,
        "gyroid": gyroid,
        "gyroid_sheet": gyroid_sheet,
        "strut_lattice": strut_lattice,
        "sheet_lattice": sheet_lattice,
        "ScalarField": ScalarField,
        "VectorField": VectorField,
        "ConstField": ConstField,
        "FuncField": FuncField,
        "LineMod": LineMod,
        "SurfaceMod": SurfaceMod,
        "field_sample_grid": field_sample_grid,
        "airgap_B": airgap_B,
        "current_density": current_density,
        "joule_heat": joule_heat,
        "centrifugal_stress": centrifugal_stress,
        "magnetization_field": magnetization_field,
        "RotorCore": RotorCore,
        "SurfaceMagnets": SurfaceMagnets,
        "StatorCore": StatorCore,
        "DistributedWinding": DistributedWinding,
        "FieldDrivenMagnets": FieldDrivenMagnets,
        "FieldDrivenStatorYoke": FieldDrivenStatorYoke,
        "FieldDrivenCoolingJacket": FieldDrivenCoolingJacket,
        "HelicalCoolingChannels": HelicalCoolingChannels,
        "RotorSleeve": RotorSleeve,
        "ShaftAndBearings": ShaftAndBearings,
        "MotorHousing": MotorHousing,
        "StatorSegmentation": StatorSegmentation,
        "Winding3D": Winding3D,
        "FunctionalVoids": FunctionalVoids,
        "CoilNetlist": CoilNetlist,
        "StructuralContinuity": StructuralContinuity,
    }


@dataclass
class BuildSpec:
    """An executable motor design produced by the agent."""

    code: str
    label: str = "agent"

    def build(self, cfg: MotorConfig3D) -> tuple[MaterialField, np.ndarray | None]:
        namespace = make_namespace(cfg)
        restricted_builtins = {
            "min": min, "max": max, "abs": abs, "round": round,
            "len": len, "range": range, "sum": sum, "True": True, "False": False,
            "None": None, "float": float, "int": int, "bool": bool,
            "zip": zip, "enumerate": enumerate, "print": print,
            "__import__": _safe_import,
        }
        namespace["__builtins__"] = restricted_builtins
        exec(compile(self.code, "<agent>", "exec"), namespace)
        if "build" not in namespace or not callable(namespace["build"]):
            raise ValueError("agent code must define a function build(cfg) -> MaterialField")
        mf = namespace["build"](cfg)
        if not isinstance(mf, MaterialField):
            raise TypeError(f"build() must return MaterialField, got {type(mf).__name__}")
        mag = None
        if "magnetization" in namespace and callable(namespace["magnetization"]):
            mag = np.asarray(namespace["magnetization"](cfg), dtype=np.float32)
        return mf, mag

    def describe(self) -> str:
        return self.code


def execute_agent_code(code: str, cfg: MotorConfig3D) -> tuple[MaterialField, np.ndarray | None, str | None]:
    """Run agent code; return ``(material_field, magnetization, error)``.

    On any exception the error traceback is returned in place of the field, so
    the loop can feed the failure back to the agent as a correction signal
    rather than crashing the run.
    """
    try:
        spec = BuildSpec(code=code)
        mf, mag = spec.build(cfg)
        return mf, mag, None
    except Exception:
        return None, None, traceback.format_exc()
