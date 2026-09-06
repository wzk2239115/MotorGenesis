"""Versioned ModelArtifact: save/load a complete motor model for simulation.

A ModelArtifact is the single source of truth that connects geometry
construction to energized simulation. It stores everything needed to
reproduce a simulation: material densities, magnetization, winding
centerlines, motion groups, design hash, config, and boundary conditions.

Format: JSON sidecar (metadata) + NPZ (arrays). No pickle — every
field is auditable.

Usage:
    artifact = ModelArtifact.from_motor(motor, mf, cfg)
    artifact.save(path)
    artifact2 = ModelArtifact.load(path)
    assert artifact2.design_hash == artifact.design_hash
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "1.0.0"

ROTOR_COMPONENTS = (
    "ShaftAndBearings", "RotorCore", "FieldDrivenMagnets", "RotorSleeve",
)
STATOR_COMPONENTS = (
    "StatorCellArray", "StatorCore", "DistributedWinding",
    "CoolingJacket", "Exoskeleton",
)


def _cfg_to_dict(cfg) -> dict:
    d = {}
    for attr in sorted(dir(cfg)):
        if attr.startswith("_"):
            continue
        val = getattr(cfg, attr)
        if isinstance(val, (int, float, str, bool, tuple, list)):
            d[attr] = list(val) if isinstance(val, tuple) else val
        elif isinstance(val, np.ndarray):
            d[attr] = val.tolist()
    return d


def _cfg_from_dict(d: dict):
    from organic_motor.config3d import MotorConfig3D
    valid = {}
    for k, v in d.items():
        if k == "shape":
            valid["shape"] = tuple(v)
        elif isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            valid[k] = tuple(v)
        elif isinstance(v, (int, float, str, bool)):
            valid[k] = v
    return MotorConfig3D(**valid)


def _registry_to_arrays(registry: list[dict]) -> dict:
    """Convert centerline registry entries to serializable arrays."""
    if not registry:
        return {"count": 0}
    out = {"count": len(registry)}
    for i, entry in enumerate(registry):
        out[f"cl_{i}_points"] = np.asarray(entry["points"], dtype=np.float32)
        out[f"cl_{i}_physical_points"] = np.asarray(entry.get("physical_points", entry["points"]), dtype=np.float32)
        out[f"cl_{i}_turn_map"] = np.asarray(entry["turn_map"], dtype=np.int32)
        out[f"cl_{i}_meta"] = json.dumps({
            "phase": entry["phase"],
            "polarity": entry["polarity"],
            "n_turns": entry["n_turns"],
            "tooth": entry["tooth"],
            "cross_section_area": entry["cross_section_area"],
            "band_radius": entry["band_radius"],
            "solver_closure": entry.get("solver_closure", True),
        })
    return out


def _arrays_to_registry(npz_data) -> list[dict]:
    count = int(npz_data["count"]) if "count" in npz_data.files else 0
    registry = []
    for i in range(count):
        meta = json.loads(str(npz_data[f"cl_{i}_meta"]))
        entry = {
            "points": np.asarray(npz_data[f"cl_{i}_points"], dtype=np.float32),
            "physical_points": np.asarray(npz_data[f"cl_{i}_physical_points"], dtype=np.float32),
            "turn_map": np.asarray(npz_data[f"cl_{i}_turn_map"], dtype=np.int32),
            **meta,
        }
        registry.append(entry)
    return registry


def _compute_design_hash(rho_dict: dict, magnetization: np.ndarray,
                         registry: list | None = None,
                         config_dict: dict | None = None) -> str:
    """Hash EVERYTHING that affects the solved fields (audit item 4).

    Densities and magnetization alone are NOT enough: the winding
    centerlines (current deposition), the electrical connection they
    encode, and solver-facing config (pole pairs, radii, conductivities,
    excitation) change the solution without changing densities.  All are
    folded in so caches keyed on this hash cannot go stale.
    """
    h = hashlib.sha256()
    for name in sorted(rho_dict):
        arr = rho_dict[name]
        h.update(name.encode())
        h.update(arr.tobytes())
    h.update(b"magnetization")
    h.update(magnetization.tobytes())
    if registry:
        h.update(b"centerlines")
        h.update(str(len(registry)).encode())
        for entry in registry:
            for key in ("phase", "polarity", "n_turns", "tooth",
                        "cross_section_area", "band_radius",
                        "solver_closure"):
                h.update(str(entry.get(key)).encode())
            h.update(np.ascontiguousarray(entry["points"], dtype=np.float32).tobytes())
            tm = entry.get("turn_map")
            if tm is not None:
                h.update(np.ascontiguousarray(tm, dtype=np.int32).tobytes())
    if config_dict:
        h.update(b"config")
        for key in sorted(config_dict):
            h.update(f"{key}={config_dict[key]!r}".encode())
    return h.hexdigest()[:16]


@dataclass
class ModelArtifact:
    """A versioned, self-describing motor model for simulation."""

    densities: dict[str, np.ndarray]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    magnetization: np.ndarray
    centerline_registry: list[dict]
    motion_groups: dict[str, list[str]]
    config_dict: dict
    design_hash: str
    version: str = VERSION
    has_magnetization: bool = True
    has_centerlines: bool = True
    timestamp: str = ""
    rotor_mask: np.ndarray | None = None
    support_mask: np.ndarray | None = None  # honeycomb/support iron
    housing_mask: np.ndarray | None = None  # outer wall iron

    @classmethod
    def from_motor(cls, motor, mf, cfg) -> "ModelArtifact":
        """Build artifact from a Motor + MaterialField + config."""
        vol = mf.to_volume()
        densities = {
            "rho_iron": vol.iron.astype(np.float32),
            "rho_pm": vol.pm.astype(np.float32),
        }
        if vol.copper is not None:
            densities["rho_copper"] = vol.copper.astype(np.float32)
        if vol.air is not None:
            densities["rho_air"] = vol.air.astype(np.float32)
        if vol.coolant is not None:
            densities["rho_coolant"] = vol.coolant.astype(np.float32)
        if vol.insulator is not None:
            densities["rho_insulator"] = vol.insulator.astype(np.float32)

        mag = motor.magnetization()
        if mag is None:
            mag = np.zeros((3,) + cfg.shape, dtype=np.float32)

        registry = mf.metadata.get("centerline_registry", [])

        comp_names = [type(c).__name__ for c in motor.components]
        motion_groups = {
            "rotor": [n for n in comp_names if n in ROTOR_COMPONENTS],
            "stator": [n for n in comp_names if n in STATOR_COMPONENTS],
        }

        design_hash = _compute_design_hash(densities, mag, registry,
                                           _cfg_to_dict(cfg))

        from organic_motor.geometry.domain3d import domain_masks3d
        rotor_mask = np.asarray(
            domain_masks3d(cfg)["rotor_design"], dtype=np.float32
        )

        return cls(
            densities=densities,
            spacing=cfg.spacing,
            origin=cfg.origin,
            shape=cfg.shape,
            magnetization=mag.astype(np.float32),
            centerline_registry=registry,
            motion_groups=motion_groups,
            config_dict=_cfg_to_dict(cfg),
            design_hash=design_hash,
            has_magnetization=mag.max() > 0,
            has_centerlines=len(registry) > 0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            rotor_mask=rotor_mask,
        )

    @classmethod
    def from_material_field(cls, mf, cfg, magnetization=None) -> "ModelArtifact":
        """Build artifact from a MaterialField + config (no Motor object).

        Used when only the constructed geometry is available (e.g. from
        assembly_demo) without the component list.
        """
        vol = mf.to_volume()
        densities = {
            "rho_iron": vol.iron.astype(np.float32),
            "rho_pm": vol.pm.astype(np.float32),
        }
        if vol.copper is not None:
            densities["rho_copper"] = vol.copper.astype(np.float32)
        if vol.air is not None:
            densities["rho_air"] = vol.air.astype(np.float32)
        if vol.coolant is not None:
            densities["rho_coolant"] = vol.coolant.astype(np.float32)
        if vol.insulator is not None:
            densities["rho_insulator"] = vol.insulator.astype(np.float32)

        if magnetization is None:
            magnetization = np.zeros((3,) + cfg.shape, dtype=np.float32)

        registry = mf.metadata.get("centerline_registry", [])

        design_hash = _compute_design_hash(
            densities, magnetization, registry, _cfg_to_dict(cfg))

        from organic_motor.geometry.domain3d import domain_masks3d
        rotor_mask = np.asarray(
            domain_masks3d(cfg)["rotor_design"], dtype=np.float32
        )

        return cls(
            densities=densities,
            spacing=cfg.spacing,
            origin=cfg.origin,
            shape=cfg.shape,
            magnetization=magnetization.astype(np.float32),
            centerline_registry=registry,
            motion_groups={"rotor": [], "stator": []},
            config_dict=_cfg_to_dict(cfg),
            design_hash=design_hash,
            has_magnetization=magnetization.max() > 0,
            has_centerlines=len(registry) > 0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            rotor_mask=rotor_mask,
        )

    def save(self, path: str | Path) -> Path:
        """Save as NPZ + JSON sidecar. Returns the NPZ path."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        ckpt_dir = path / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        npz_path = ckpt_dir / "step_000000.npz"

        arrays = {}
        arrays["spacing"] = np.array(self.spacing, dtype=np.float32)
        arrays["origin"] = np.array(self.origin, dtype=np.float32)
        for name, arr in self.densities.items():
            arrays[name] = arr
        arrays["magnetization"] = self.magnetization
        arrays["step"] = np.array(0, dtype=np.int32)
        arrays["design_hash"] = np.array(self.design_hash)
        arrays["version"] = np.array(self.version)
        arrays["has_magnetization"] = np.array(self.has_magnetization)
        arrays["has_centerlines"] = np.array(self.has_centerlines)
        if self.rotor_mask is not None:
            arrays["rotor_mask"] = self.rotor_mask.astype(np.float32)
        if self.support_mask is not None:
            arrays["support_mask"] = self.support_mask.astype(np.float32)
        if self.housing_mask is not None:
            arrays["housing_mask"] = self.housing_mask.astype(np.float32)

        reg_arrays = _registry_to_arrays(self.centerline_registry)
        for k, v in reg_arrays.items():
            if isinstance(v, str):
                arrays[k] = np.array(v)
            elif isinstance(v, (int, float)):
                arrays[k] = np.array(v)
            else:
                arrays[k] = v

        np.savez(npz_path, **arrays)

        meta = {
            "version": self.version,
            "design_hash": self.design_hash,
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "has_magnetization": self.has_magnetization,
            "has_centerlines": self.has_centerlines,
            "motion_groups": self.motion_groups,
            "config": self.config_dict,
            "timestamp": self.timestamp,
            "centerline_count": len(self.centerline_registry),
            "materials": sorted(self.densities.keys()),
            "has_rotor_mask": self.rotor_mask is not None,
        }
        (path / "model_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8",
        )

        return npz_path

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        """Load from a directory containing model_meta.json + checkpoints/."""
        path = Path(path)
        meta = json.loads((path / "model_meta.json").read_text(encoding="utf-8"))
        npz_path = path / "checkpoints" / "step_000000.npz"
        return cls.load_npz(npz_path, meta)

    @classmethod
    def load_npz(cls, npz_path: str | Path, meta: dict | None = None) -> "ModelArtifact":
        """Load from NPZ file, optionally with pre-loaded metadata."""
        npz_path = Path(npz_path)
        if meta is None:
            meta_path = npz_path.parent.parent / "model_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = {}

        with np.load(npz_path, allow_pickle=False) as data:
            spacing = tuple(float(v) for v in np.asarray(data["spacing"]).ravel())
            origin = tuple(float(v) for v in np.asarray(data["origin"]).ravel())

            densities = {}
            for name in ("rho_iron", "rho_pm", "rho_copper", "rho_air",
                         "rho_coolant", "rho_insulator"):
                if name in data.files:
                    densities[name] = np.asarray(data[name], dtype=np.float32)

            magnetization = np.asarray(
                data["magnetization"] if "magnetization" in data.files
                else np.zeros((3,) + densities["rho_iron"].shape, dtype=np.float32)
            )

            design_hash = str(data["design_hash"]) if "design_hash" in data.files else ""
            version = str(data["version"]) if "version" in data.files else "unknown"
            has_mag = bool(data["has_magnetization"]) if "has_magnetization" in data.files else (magnetization.max() > 0)
            has_cl = bool(data["has_centerlines"]) if "has_centerlines" in data.files else False

            registry = []
            if has_cl or "count" in data.files:
                registry = _arrays_to_registry(data)

            rotor_mask = None
            if "rotor_mask" in data.files:
                rotor_mask = np.asarray(data["rotor_mask"], dtype=np.float32)
            support_mask = None
            if "support_mask" in data.files:
                support_mask = np.asarray(data["support_mask"], dtype=np.float32)
            housing_mask = None
            if "housing_mask" in data.files:
                housing_mask = np.asarray(data["housing_mask"], dtype=np.float32)

        shape = tuple(densities["rho_iron"].shape)

        return cls(
            densities=densities,
            spacing=spacing,
            origin=origin,
            shape=shape,
            magnetization=magnetization,
            centerline_registry=registry,
            motion_groups=meta.get("motion_groups", {"rotor": [], "stator": []}),
            config_dict=meta.get("config", {}),
            design_hash=design_hash,
            version=version,
            has_magnetization=has_mag,
            has_centerlines=has_cl,
            timestamp=meta.get("timestamp", ""),
            rotor_mask=rotor_mask,
            support_mask=support_mask,
            housing_mask=housing_mask,
        )

    def can_energize(self) -> tuple[bool, list[str]]:
        """Check if this artifact has enough data for energized simulation."""
        reasons = []
        if not self.has_magnetization or self.magnetization.max() == 0:
            reasons.append("magnetization is zero — not a PM motor")
        if not self.has_centerlines or len(self.centerline_registry) == 0:
            reasons.append("no centerline registry — cannot deposit winding currents")
        if "rho_copper" not in self.densities:
            reasons.append("no copper density — winding missing")
        if "rho_iron" not in self.densities:
            reasons.append("no iron density — stator/rotor missing")
        return (len(reasons) == 0, reasons)

    def to_volume(self):
        """Convert to VoxelVolume for the existing export pipeline."""
        from organic_motor.geometry.voxel import VoxelVolume
        return VoxelVolume(
            iron=self.densities.get("rho_iron"),
            pm=self.densities.get("rho_pm"),
            spacing=self.spacing,
            origin=self.origin,
            copper=self.densities.get("rho_copper"),
            air=self.densities.get("rho_air"),
            coolant=self.densities.get("rho_coolant"),
            insulator=self.densities.get("rho_insulator"),
        )

    def solver_fields(self, cfg):
        """Build critic-ready TopologyFields3D DIRECTLY from the saved
        densities.

        The artifact densities are the smoothstep of the ORIGINAL
        construction SDFs (mf.to_densities at save time) — the same
        information the direct-construction path feeds the solver.
        Re-deriving an impostor SDF (0.5 - rho) and re-smoothing it is NOT
        equivalent: the impostor is near-binary, and after the rotor
        rotation warp its material fractions wreck the CG conditioning
        (measured: relative residual stalls at 0.3-1.0 at every non-zero
        rotor angle on the 96^3 motor, vs 1.3e-5 via this path and via
        direct construction — reports/diag_angle_sweep.py).

        Returns (fields, magnetization_raw).
        """
        import jax.numpy as jnp
        from organic_motor.topology.density3d import TopologyFields3D
        from organic_motor.geometry.domain3d import domain_masks3d
        import numpy as np

        zeros = np.zeros(self.shape, dtype=np.float32)
        rotor = np.asarray(domain_masks3d(cfg)["rotor_design"],
                           dtype=np.float32)
        d = self.densities
        fields = TopologyFields3D(
            rho_air=jnp.asarray(d.get("rho_air", zeros)),
            rho_iron=jnp.asarray(d["rho_iron"]),
            rho_copper=jnp.asarray(d.get("rho_copper", zeros)),
            rho_pm=jnp.asarray(d["rho_pm"]),
            rotor_ownership=jnp.asarray(rotor),
            rho_insulator=jnp.asarray(d["rho_insulator"])
            if "rho_insulator" in d else None,
            rho_coolant=jnp.asarray(d["rho_coolant"])
            if "rho_coolant" in d else None,
        )
        return fields, jnp.asarray(self.magnetization)
