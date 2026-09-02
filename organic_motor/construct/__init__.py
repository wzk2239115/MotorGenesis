"""LEAP 71-style constructive motor layer.

The construction layer is the *non-differentiable* geometry substrate: an
engineer (or an LLM agent) composes a motor from SDF primitives using a tiny,
``never-fail'' instruction set.  The differentiable physics solver is reused
unchanged as a *critic* that scores the constructed geometry, closing the
agent loop without placing discrete Booleans inside the gradient tape.
"""

from organic_motor.construct.field import (
    SDFVoxelField,
    boolean_add,
    boolean_intersect,
    boolean_subtract,
    empty_field,
    from_implicit,
    offset,
    resample,
    shell,
    smooth_boolean_add,
    smooth_boolean_subtract,
)
from organic_motor.construct.implicit import (
    annular_sector,
    box,
    capsule,
    cylinder,
    cylinder_z,
    gyroid,
    plane,
    rounded_box,
    sphere,
    torus,
)
from organic_motor.construct.lattice import gyroid_sheet, sheet_lattice, strut_lattice
from organic_motor.construct.material import MaterialField
from organic_motor.construct.objects import (
    CoolingJacket,
    DistributedWinding,
    FieldDrivenCoolingJacket,
    FieldDrivenMagnets,
    FieldDrivenStatorYoke,
    HelicalCoolingChannels,
    Motor,
    MotorHousing,
    RotorCore,
    RotorSleeve,
    ShaftAndBearings,
    StatorCore,
    StatorSegmentation,
    SurfaceMagnets,
    Winding3D,
    baseline_motor,
    field_driven_motor,
)
from organic_motor.construct.realize import realize
from organic_motor.construct.critic import score, score_fields

__all__ = [
    "SDFVoxelField",
    "boolean_add",
    "boolean_intersect",
    "boolean_subtract",
    "empty_field",
    "from_implicit",
    "offset",
    "resample",
    "shell",
    "box",
    "capsule",
    "cylinder",
    "cylinder_z",
    "gyroid",
    "plane",
    "rounded_box",
    "sphere",
    "torus",
    "annular_sector",
    "gyroid_sheet",
    "sheet_lattice",
    "strut_lattice",
    "MaterialField",
    "CoolingJacket",
    "DistributedWinding",
    "FieldDrivenCoolingJacket",
    "FieldDrivenMagnets",
    "FieldDrivenStatorYoke",
    "HelicalCoolingChannels",
    "Motor",
    "MotorHousing",
    "RotorCore",
    "RotorSleeve",
    "ShaftAndBearings",
    "StatorCore",
    "StatorSegmentation",
    "SurfaceMagnets",
    "Winding3D",
    "baseline_motor",
    "field_driven_motor",
    "realize",
    "score",
    "score_fields",
]
