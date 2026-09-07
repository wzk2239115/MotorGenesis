"""FastAPI application serving smoothed 3-D motor meshes and live checkpoints.

Run with::

    python -m organic_motor.web --out organic_motor/out --port 8000

The server scans one or more run directories (each typically produced by
``motor3d_organic grow``) and, per checkpoint, generates a Taubin-smoothed GLB
on demand and caches it next to the checkpoint.  A Server-Sent Events stream
pushes new checkpoints as they appear while a growth run is in progress, so a
browser tab can watch the motor differentiate live.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from organic_motor.web import builder

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = None
    Field = None


if BaseModel is not None:
    class SimRequest(BaseModel):
        """Typed, range-checked simulation request (audit item 2).

        Defaults apply ONLY to missing fields; legal zero values (0 V,
        0 N·m) are preserved end-to-end.
        """
        voltage: float = Field(24.0, ge=0.0, le=1000.0)
        current_limit: float = Field(50.0, ge=0.0, le=1000.0)
        initial_angle: float = Field(0.0, ge=-4 * 3.14159, le=4 * 3.14159)
        steps: int = Field(4000, ge=1, le=200_000)
        dt: float = Field(2.0e-5, gt=0.0, le=1.0e-2)
        load_torque: float = Field(0.005, ge=0.0, le=100.0)
        load_viscous: float = Field(1.0e-4, ge=0.0, le=10.0)
        rotor_inertia: float = Field(2.0e-4, gt=0.0, le=10.0)
        control_mode: str = Field("open_loop",
                                  pattern="^(open_loop|current_control)$")
        i_q_ref_A: Optional[float] = Field(None, ge=-1000.0, le=1000.0)
        power_off_at_s: Optional[float] = Field(None, ge=0.0)
        include_windage: bool = False
else:  # pragma: no cover
    class SimRequest:  # type: ignore[no-redef]
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

        def model_dump(self):
            return dict(self.__dict__)


def _cache_dir(run_dir: Path) -> Path:
    cache = run_dir / "web_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _glb_cache_path(
    run_dir: Path, step: int, level: float, smoothing: str, iterations: int,
    npz_mtime: float = 0.0, extra: str = "",
) -> Path:
    key = (f"step_{step:06d}_level{level:g}_{smoothing}_{iterations}"
           f"_mt{int(npz_mtime)}_{extra}")
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return _cache_dir(run_dir) / f"{digest}_{key}.glb"


def create_app(out_root: str | Path = "organic_motor/out") -> FastAPI:
    """Build a FastAPI app rooted at ``out_root``.

    ``out_root`` may be a single directory or a colon-separated list; every
    immediate child directory that contains checkpoints or meshes becomes a
    selectable run.
    """
    roots = [Path(p).resolve() for p in str(out_root).split(":") if p]

    app = FastAPI(title="MotorGenesis viewer")
    from organic_motor.web.prototype_api import install
    install(app, roots[0])
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    def _find_run(run_name: str) -> Path:
        for root in roots:
            candidate = root / run_name
            if candidate.is_dir():
                return candidate
        raise HTTPException(status_code=404, detail=f"run {run_name!r} not found")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(Path(__file__).parent / "static" / "index.html"))

    @app.get("/api/runs")
    def get_runs() -> list[dict]:
        runs: list[dict] = []
        for root in roots:
            for entry in builder.list_runs(root):
                runs.append(entry)
        return runs

    @app.get("/api/runs/{run_name}")
    def get_run(run_name: str) -> dict:
        run_dir = _find_run(run_name)
        return builder.run_summary(run_dir)

    @app.get("/api/runs/{run_name}/startup")
    def get_run_startup(run_name: str) -> dict:
        """Multi-angle startup validation verdict, when the run has one."""
        run_dir = _find_run(run_name)
        for name in ("startup.json", "startup_boost/startup.json",
                     "startup_quick/startup.json"):
            candidate = run_dir / name
            if not candidate.is_file():
                candidate = run_dir.parent / name
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="no startup validation for this run")

    @app.get("/api/runs/{run_name}/checkpoint/{step}/glb")
    def get_checkpoint_glb(
        run_name: str,
        step: int,
        level: float = 0.35,
        smoothing: str = "taubin",
        iterations: int = 5,
        view: str = "full",
    ) -> Response:
        if view not in ("full", "stator"):
            raise HTTPException(status_code=400, detail="view must be full or stator")
        run_dir = _find_run(run_name)
        npz = run_dir / "checkpoints" / f"step_{step:06d}.npz"
        if not npz.is_file():
            raise HTTPException(status_code=404, detail="checkpoint not found")
        npz_mtime = npz.stat().st_mtime
        cache = _glb_cache_path(run_dir, step, level, smoothing, iterations, npz_mtime,
                                extra="parts-v1-"+view)
        if not cache.is_file():
            glb = builder.checkpoint_to_glb(
                npz,
                level=level,
                smoothing=smoothing,
                smoothing_iterations=iterations,
                view=view,
            )
            cache.write_bytes(glb)
        return Response(content=cache.read_bytes(), media_type="model/gltf-binary")

    @app.get("/api/runs/{run_name}/checkpoint/{step}/parts")
    def get_parts(run_name: str, step: int):
        from organic_motor.web.parts import part_meshes, manifest
        path = _find_run(run_name) / "checkpoints" / f"step_{step:06d}.npz"
        if not path.is_file():
            raise HTTPException(404, "checkpoint not found")
        return manifest(path, list(part_meshes(path)))

    @app.get("/api/runs/{run_name}/checkpoint/{step}/molds.zip")
    def get_molds_zip(run_name: str, step: int):
        from organic_motor.web.parts import mold_package
        path = _find_run(run_name) / "checkpoints" / f"step_{step:06d}.npz"
        if not path.is_file():
            raise HTTPException(404, "checkpoint not found")
        return Response(mold_package(path), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="stator-casting-fit-test.zip"'})

    @app.get("/api/runs/{run_name}/checkpoint/{step}/parts.zip")
    def get_parts_zip(run_name: str, step: int):
        from organic_motor.web.parts import package
        path = _find_run(run_name) / "checkpoints" / f"step_{step:06d}.npz"
        if not path.is_file():
            raise HTTPException(404, "checkpoint not found")
        return Response(package(path), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="motor-fit-check-parts.zip"'})

    @app.get("/api/runs/{run_name}/checkpoint/{step}/stl")
    def get_checkpoint_stl(
        run_name: str,
        step: int,
        level: float = 0.35,
        smoothing: str = "taubin",
        iterations: int = 5,
    ) -> Response:
        run_dir = _find_run(run_name)
        npz = run_dir / "checkpoints" / f"step_{step:06d}.npz"
        if not npz.is_file():
            raise HTTPException(status_code=404, detail="checkpoint not found")
        npz_mtime = npz.stat().st_mtime
        stl_cache = _cache_dir(run_dir) / f"stl_step_{step:06d}_level{level:g}_{smoothing}_{iterations}_mt{int(npz_mtime)}.stl"
        if not stl_cache.is_file():
            glb_bytes = builder.checkpoint_to_glb(
                npz, level=level, smoothing=smoothing, smoothing_iterations=iterations,
            )
            import io
            import trimesh
            scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
            if not meshes:
                raise HTTPException(status_code=404, detail="no mesh in checkpoint")
            combined = trimesh.util.concatenate(meshes)
            stl_cache.write_bytes(combined.export(file_type="stl"))
        return Response(
            content=stl_cache.read_bytes(),
            media_type="model/stl",
            headers={"Content-Disposition": f"attachment; filename=motor_step{step:06d}.stl"},
        )

    @app.get("/api/runs/{run_name}/mesh/{name}")
    def get_mesh_file(run_name: str, name: str) -> FileResponse:
        run_dir = _find_run(run_name)
        path = run_dir / "meshes" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="mesh not found")
        media = {
            ".glb": "model/gltf-binary",
            ".ply": "application/ply",
            ".stl": "model/stl",
        }.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(str(path), media_type=media, filename=name)

    @app.get("/api/runs/{run_name}/checkpoint/{step}/slice")
    def get_slice(
        run_name: str,
        step: int,
        field: str = "temperature",
        axis: int = 2,
        index: Optional[int] = None,
    ) -> dict:
        run_dir = _find_run(run_name)
        npz = run_dir / "checkpoints" / f"step_{step:06d}.npz"
        if not npz.is_file():
            raise HTTPException(status_code=404, detail="checkpoint not found")
        # Physics fields (temperature/|B|/|J|) only exist in the final forward
        # solve; fall back to it so the slice panel keeps working on every
        # step — but LABEL the source and refuse mismatched grids (C8).
        fallback = run_dir / "final_simulation3d.npz"
        try:
            return builder.field_slice(
                npz,
                field,
                axis=axis,
                index=index,
                fallback_npz=fallback if fallback.is_file() else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_name}/checkpoint/{step}/metrics")
    def get_metrics(run_name: str, step: int) -> dict:
        run_dir = _find_run(run_name)
        npz = run_dir / "checkpoints" / f"step_{step:06d}.npz"
        if not npz.is_file():
            raise HTTPException(status_code=404, detail="checkpoint not found")
        return builder.checkpoint_metrics(npz)

    @app.get("/api/runs/{run_name}/events")
    async def run_events(run_name: str, poll_interval: float = 1.5) -> StreamingResponse:
        """Server-Sent Events stream of new checkpoint steps.

        The endpoint polls the checkpoint directory (no extra deps) and emits a
        ``checkpoint`` event whenever a new ``step_*.npz`` appears.  This is
        what lets a browser watch a live growth run.
        """
        run_dir = _find_run(run_name)
        ckpt_dir = run_dir / "checkpoints"

        async def event_source():
            seen: set[int] = set()
            for info in builder.list_checkpoints(ckpt_dir):
                seen.add(info.step)
            yield f": open run={run_name} seen={len(seen)}\n\n"
            idle = 0.0
            while True:
                await asyncio.sleep(poll_interval)
                current = builder.list_checkpoints(ckpt_dir)
                fresh = [c for c in current if c.step not in seen]
                for info in fresh:
                    seen.add(info.step)
                    payload = {"step": info.step, "path": info.path.name}
                    data = (
                        f"data: {__import__('json').dumps(payload)}\n\n"
                    )
                    yield data
                if not fresh:
                    idle += poll_interval
                else:
                    idle = 0.0
                if idle > 0:
                    yield f": heartbeat idle={idle:.0f}s\n\n"

        return StreamingResponse(
            event_source(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_name}/simulate")
    async def start_simulation(run_name: str, body: Optional[SimRequest] = None) -> dict:
        """Start a powered transient simulation for this run.

        Typed + range-checked request body; the response echoes the
        settings the backend actually adopted (including the voltage
        definition) so the UI never silently changes an operating point.
        """
        import threading
        import uuid

        req = body or SimRequest()
        run_dir = _find_run(run_name)
        sim_id = f"sim_{uuid.uuid4().hex[:8]}"
        settings = req.model_dump()
        sim = {
            "sim_id": sim_id,
            "status": "queued",
            "run_name": run_name,
            "settings_input": settings,
            "adopted": {
                "control_mode": req.control_mode,
                "voltage": req.voltage,
                "voltage_definition": "相电压峰值 (phase voltage peak)",
                "current_limit": req.current_limit,
                "initial_angle": req.initial_angle,
                "load_torque": req.load_torque,
                "steps": req.steps,
                "dt": req.dt,
                "i_q_ref_A": req.i_q_ref_A,
                "power_off_at_s": req.power_off_at_s,
                "include_windage": req.include_windage,
            },
            "progress": {"phase": "queued", "angles_done": 0, "angles_total": None,
                         "elapsed_s": 0.0},
            "started_at": time.time(),
        }
        app.state.simulations[sim_id] = sim
        _persist_sim(run_dir, sim)
        thread = threading.Thread(
            target=_run_simulation_thread,
            args=(sim_id, run_dir, settings, app.state),
            daemon=True,
        )
        thread.start()
        return {"sim_id": sim_id, "status": "queued", "adopted": sim["adopted"]}

    @app.get("/api/simulations/{sim_id}/status")
    def get_simulation_status(sim_id: str) -> dict:
        """Lightweight status (no result payloads)."""
        sim = app.state.simulations.get(sim_id)
        if sim is None:
            raise HTTPException(status_code=404, detail="simulation not found")
        return {k: v for k, v in sim.items() if k != "results"}

    @app.get("/api/simulations/{sim_id}/results")
    def get_simulation_results(sim_id: str) -> dict:
        """Full result payload (only meaningful once status == done)."""
        sim = app.state.simulations.get(sim_id)
        if sim is None:
            raise HTTPException(status_code=404, detail="simulation not found")
        if "results" not in sim:
            raise HTTPException(status_code=404, detail="results not computed")
        return {"sim_id": sim_id, "status": sim["status"], "results": sim["results"]}

    @app.post("/api/simulations/{sim_id}/cancel")
    def cancel_simulation(sim_id: str) -> dict:
        sim = app.state.simulations.get(sim_id)
        if sim is None:
            raise HTTPException(status_code=404, detail="simulation not found")
        if sim["status"] in ("done", "failed", "rejected", "cancelled"):
            return {"sim_id": sim_id, "status": sim["status"]}
        sim["cancel_requested"] = True
        return {"sim_id": sim_id, "status": sim["status"], "cancelling": True}

    app.state.roots = roots
    app.state.simulations = {}
    _restore_sims(app)
    return app


def _persist_sim(run_dir: Path, sim: dict) -> None:
    """Persist a simulation record (status/progress; results only at end)."""
    try:
        d = run_dir / "simulations"
        d.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in sim.items() if k != "results"}
        (d / f"{sim['sim_id']}.json").write_text(
            json.dumps(slim, indent=1, default=str), encoding="utf-8")
        if "results" in sim:
            (d / f"{sim['sim_id']}_results.json").write_text(
                json.dumps(sim["results"], default=str), encoding="utf-8")
    except Exception:
        pass


def _restore_sims(app) -> None:
    """Reload persisted simulations at startup (page refresh recovery)."""
    for root in app.state.roots:
        for f in sorted(root.glob("*/simulations/*.json")):
            if f.name.endswith("_results.json"):
                continue
            try:
                sim = json.loads(f.read_text(encoding="utf-8"))
                sim.setdefault("sim_id", f.stem)
                rf = f.parent / f"{f.stem}_results.json"
                if rf.is_file():
                    sim["results"] = json.loads(rf.read_text(encoding="utf-8"))
                if sim.get("status") in ("queued", "loading", "extracting_flux",
                                         "solving_maps", "running_transient"):
                    sim["status"] = "interrupted"  # server died mid-run
                app.state.simulations[f.stem] = sim
            except Exception:
                continue


_GPU_LOCK = __import__("threading").Lock()
_MAPS_CACHE: dict = {}  # design_hash -> maps dict (audit item 4)


class _Cancelled(Exception):
    pass


def _run_simulation_thread(
    sim_id: str, run_dir: Path, settings: dict, app_state,
):
    """Run a powered transient in a background thread.

    Observable: phase / angles-done / elapsed / detail in sim["progress"].
    Cancellable: sim["cancel_requested"] checked between solves.
    Serialized: one GPU job at a time (others wait with visible status).
    Persisted: every status change lands in run_dir/simulations/.
    """
    import numpy as np
    from organic_motor.construct.model_artifact import ModelArtifact

    sims = app_state.simulations
    sim = sims.get(sim_id)
    if sim is None:
        return
    t0 = time.time()

    def set_status(status, phase=None, **prog):
        sim["status"] = status
        sim["progress"] = {
            "phase": phase or status,
            "elapsed_s": time.time() - t0,
            **prog,
        }
        _persist_sim(run_dir, sim)

    def check_cancel():
        if sim.get("cancel_requested"):
            raise _Cancelled()

    try:
        set_status("loading", "loading model")
        artifact = ModelArtifact.load(run_dir)
        can_run, reasons = artifact.can_energize()
        if not can_run:
            sim["status"] = "rejected"
            sim["error"] = "; ".join(reasons)
            _persist_sim(run_dir, sim)
            return

        from organic_motor.config3d import MotorConfig3D
        import inspect
        valid_params = set(inspect.signature(MotorConfig3D.__init__).parameters.keys()) - {"self"}
        cfg_kwargs = {"shape": tuple(artifact.shape)}
        for k, v in artifact.config_dict.items():
            if k not in valid_params:
                continue
            if isinstance(v, list) and len(v) == 3:
                cfg_kwargs[k] = tuple(v)
            elif isinstance(v, (int, float, str, bool)):
                cfg_kwargs[k] = v
        cfg = MotorConfig3D(**cfg_kwargs)

        from organic_motor.construct.startup_validation import (
            _logits_from_densities,
        )
        import jax.numpy as jnp

        logits = _logits_from_densities(artifact, cfg)
        # CRITICAL: solver fields come DIRECTLY from the saved densities.
        # Re-deriving an impostor SDF (0.5-rho) stalls the Maxwell CG at
        # every rotated angle (residual 0.3-1.0) — see
        # reports/diag_angle_sweep.py.  This path converges (1.3e-5).
        fields, mag = artifact.solver_fields(cfg)
        registry = artifact.centerline_registry or None

        from organic_motor.construct.transient_bridge import (
            extract_electrical_parameters, extract_fea_flux_linkage,
        )

        with _GPU_LOCK:
            check_cancel()
            set_status("extracting_flux", "FEA flux linkage (6 PM solves)")
            flux_fea = extract_fea_flux_linkage(
                None, cfg, artifact.magnetization,
                fields=fields, registry=registry)
        electrical = extract_electrical_parameters(
            None, cfg, flux_linkage_fea=flux_fea,
            registry=registry, copper_fraction=artifact.densities.get("rho_copper"))

        from organic_motor.experiments.motor3d_powered import (
            Powered3DSettings, compute_powered_maps, run_powered_transient,
        )

        voltage = float(settings.get("voltage", 24.0))
        current_limit = float(settings.get("current_limit", 50.0))
        initial_angle = float(settings.get("initial_angle", 0.0))
        steps = int(settings.get("steps", 4000))
        dt = float(settings.get("dt", 2.0e-5))
        load_torque = float(settings.get("load_torque", 0.005))
        load_viscous = float(settings.get("load_viscous", 1.0e-4))
        rotor_inertia = float(settings.get("rotor_inertia", 2.0e-4))
        control_mode = str(settings.get("control_mode", "open_loop"))
        i_q_ref = settings.get("i_q_ref_A")
        i_q_ref = float(i_q_ref) if i_q_ref is not None else None
        power_off_at = settings.get("power_off_at_s")
        power_off_at = float(power_off_at) if power_off_at is not None else None
        include_windage = bool(settings.get("include_windage", False))

        if electrical.flux_linkage < 1e-8:
            sim["status"] = "rejected"
            sim["error"] = "flux_linkage ~0 — FEA extraction failed or winding not connected"
            _persist_sim(run_dir, sim)
            return

        # Initial p_settings — psi from FEA (used only for the map solve;
        # the map solve does NOT depend on psi, only on geometry + unit
        # excitation).  After maps are computed, psi is overridden with
        # the map-consistent value (audit item 6: emf/torque consistency).
        # n_turns from the centerline registry (7 bands per cell) —
        # fixes the nominal-current normalization (was 7x too large).
        n_turns_override = None
        if registry:
            n_turns_override = int(registry[0].get("n_turns", 1))

        p_settings = Powered3DSettings(
            phase_voltage_peak=voltage,
            phase_resistance=electrical.phase_resistance,
            phase_inductance=electrical.phase_inductance,
            flux_linkage=electrical.flux_linkage,
            current_limit=current_limit,
            commutation_offset=0.0,
            control_mode=control_mode,
            i_q_ref_A=i_q_ref,
            power_off_at_s=power_off_at,
            steps=steps,
            dt=dt,
            load_torque=load_torque,
            load_viscous=load_viscous,
            rotor_inertia=rotor_inertia,
            include_windage=include_windage,
            thermal_coupling="coupled",
        )

        n_map_angles = 6
        cache_key = (artifact.design_hash, n_map_angles)
        cached = _MAPS_CACHE.get(cache_key)
        if cached is not None:
            # Shallow copy so the per-run jitted "_scan" (bound to THIS
            # run's settings) never leaks back into the cache.
            maps = dict(cached)
            maps.pop("_scan", None)
            set_status("solving_maps", "reusing cached maps",
                       angles_done=42, angles_total=42,
                       detail=f"cache hit (design {artifact.design_hash}) — "
                              "operating-point-only change, maps identical")
        else:
            set_status("solving_maps", "solving maps", angles_done=0,
                       angles_total=n_map_angles)
            elec_period = 2.0 * np.pi / cfg.pole_pairs
            angles_map = np.linspace(0, elec_period, n_map_angles, endpoint=False)

            from organic_motor.optimization.objective3d import forward3d_fields

            def progress_cb(done, total, detail):
                check_cancel()
                set_status("solving_maps", "solving maps",
                           angles_done=done, angles_total=total,
                           detail=detail)

            def phase_solver(single, angle, amplitudes):
                return forward3d_fields(
                    cfg, fields, mag, [angle], single,
                    phase_amplitudes=amplitudes,
                    centerline_registry=registry,
                )

            with _GPU_LOCK:
                maps = compute_powered_maps(
                    cfg, logits, None, mag,
                    angles_map, p_settings,
                    phase_solver=phase_solver,
                    include_mechanics=False,
                    progress=progress_cb,
                    n_turns_override=n_turns_override,
                    filter_harmonics=True,
                    centerline_registry=registry,
                )
            _MAPS_CACHE[cache_key] = {
                k: v for k, v in maps.items() if k != "_scan"
            }
            if len(_MAPS_CACHE) > 4:  # bounded cache
                _MAPS_CACHE.pop(next(iter(_MAPS_CACHE)))

        maps["temperature_init"] = jnp.full(cfg.shape, float(cfg.ambient_temperature), dtype=jnp.float32)

        # --- psi consistency: override the FEA flux linkage with the
        # MAP-DERIVED value (audit item 6).  The FEA ∮A·dl includes
        # end-turn leakage that doesn't produce torque; the map-derived
        # psi captures only the torque-producing (mutual) component and
        # is consistent with the torque maps BY CONSTRUCTION, so
        # ∫emf*i == ∫T*omega (energy balance).  psi_FEA is retained as
        # a diagnostic (their ratio = 1/mutual_fraction). ---
        psi_fea = electrical.flux_linkage
        psi_map = float(maps.get("psi_from_map", 0.0))
        if psi_map > 1e-8:
            from dataclasses import replace as _replace
            p_settings = _replace(p_settings, flux_linkage=psi_map)
            maps.pop("_scan", None)  # force recompile with new psi
            sim["psi_fea_Wb"] = psi_fea
            sim["psi_map_Wb"] = psi_map
            sim["leakage_fraction"] = 1.0 - psi_map / max(psi_fea, 1e-12)

        # --- map physical-bounds gate: reject unphysical maps rather
        # than letting them produce energy from nothing ---
        t_phys = 1.5 * cfg.pole_pairs * electrical.flux_linkage * \
            float(np.max(np.abs(np.asarray(maps["nominal_current"]))))
        t_bound = 2.5 * t_phys
        t1_max = float(np.max(np.abs(np.asarray(maps["torques_ph"]))))
        t2_max = float(np.max(np.abs(np.asarray(maps["torque_i2_diag"]))))
        t0_max = float(np.max(np.abs(np.asarray(maps["torque_cogging"]))))
        if max(t1_max, t2_max) > t_bound:
            sim["status"] = "rejected"
            sim["error"] = (
                f"torque maps exceed physical bound: T1_max={t1_max:.2f}, "
                f"T2_max={t2_max:.2f} Nm vs bound {t_bound:.2f} "
                f"(1.5*p*psi*I_nom={t_phys:.2f} x2.5) — refine the grid "
                "or increase map angles before trusting this transient"
            )
            _persist_sim(run_dir, sim)
            return

        check_cancel()
        set_status("running_transient", "transient scan (GPU)")
        data = run_powered_transient(maps, p_settings, cfg, initial_angle)

        # --- energy-balance validity check (audit: 失败如实报告) ---
        i_hist = np.asarray(data["currents_A"])[1:]
        e_elec = float(np.sum(data["electrical_power_W"])) * dt
        e_joule = float(np.sum(np.sum(i_hist ** 2, axis=1)
                               * p_settings.phase_resistance)) * dt
        e_mech = float(np.sum(data["mechanical_power_W"])) * dt
        mag_e = 0.5 * p_settings.phase_inductance * float(
            np.sum(i_hist[-1] ** 2))
        imbalance = abs(e_mech + e_joule + mag_e - e_elec) / max(
            abs(e_elec), abs(e_joule), 1e-9)

        sim["status"] = "done"
        sim["results"] = {
            "time_s": data["time_s"].tolist(),
            "rotor_angle_rad": data["rotor_angle_rad"].tolist(),
            "angular_velocity_rad_s": data["angular_velocity_rad_s"].tolist(),
            "currents_A": data["currents_A"].tolist(),
            "torque_Nm": data["transient_torque_Nm"].tolist(),
            "joule_power_W": data["transient_joule_power_W"].tolist(),
            "iron_power_W": data["transient_iron_power_W"].tolist(),
            "electrical_power_W": data["electrical_power_W"].tolist(),
            "mechanical_power_W": data["mechanical_power_W"].tolist(),
            "max_temperature_C": data["max_temperature_C"].tolist(),
            "rpm": (data["angular_velocity_rad_s"] * 60 / (2 * np.pi)).tolist(),
            "energy_elec_J": e_elec,
            "energy_mech_J": e_mech,
            "energy_joule_J": e_joule,
            "energy_imbalance_rel": imbalance,
            "energy_valid": bool(imbalance < 0.2),
            "map_bounds": {"T1_max": t1_max, "T2_max": t2_max,
                           "T0_max": t0_max, "bound": t_bound},
        }
        sim["settings"] = {
            "voltage": voltage,
            "voltage_definition": "相电压峰值 (phase voltage peak)",
            "current_limit": current_limit,
            "initial_angle": initial_angle,
            "steps": steps,
            "dt": dt,
            "load_torque": load_torque,
            "load_viscous": load_viscous,
            "rotor_inertia": rotor_inertia,
            "control_mode": control_mode,
            "i_q_ref_A": i_q_ref,
            "power_off_at_s": power_off_at,
            "include_windage": include_windage,
            "phase_resistance": electrical.phase_resistance,
            "phase_inductance": electrical.phase_inductance,
            "flux_linkage": p_settings.flux_linkage,
            "flux_linkage_source": ("psi_from_map (torque-consistent)"
                                     if psi_map > 1e-8 else "psi_fea"),
            "psi_fea_Wb": psi_fea,
            "leakage_fraction": 1.0 - psi_map / max(psi_fea, 1e-12)
            if psi_fea > 1e-12 else None,
        }
        _persist_sim(run_dir, sim)
    except _Cancelled:
        sim["status"] = "cancelled"
        sim["error"] = "用户取消"
        _persist_sim(run_dir, sim)
    except Exception as e:
        sim["status"] = "failed"
        sim["error"] = str(e)
        import traceback
        sim["traceback"] = traceback.format_exc()
        _persist_sim(run_dir, sim)


app = create_app()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="organic_motor/out",
        help="run output root (colon-separated for multiple roots)",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn

    global app
    app = create_app(args.out)
    uvicorn.run(
        "organic_motor.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
