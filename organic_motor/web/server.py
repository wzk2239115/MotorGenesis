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
import time
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from organic_motor.web import builder


def _cache_dir(run_dir: Path) -> Path:
    cache = run_dir / "web_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _glb_cache_path(
    run_dir: Path, step: int, level: float, smoothing: str, iterations: int
) -> Path:
    key = f"step_{step:06d}_level{level:g}_{smoothing}_{iterations}"
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

    @app.get("/api/runs/{run_name}/checkpoint/{step}/glb")
    def get_checkpoint_glb(
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
        cache = _glb_cache_path(run_dir, step, level, smoothing, iterations)
        if not cache.is_file():
            glb = builder.checkpoint_to_glb(
                npz,
                level=level,
                smoothing=smoothing,
                smoothing_iterations=iterations,
            )
            cache.write_bytes(glb)
        return Response(content=cache.read_bytes(), media_type="model/gltf-binary")

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
        index: int | None = None,
    ) -> dict:
        run_dir = _find_run(run_name)
        npz = run_dir / "checkpoints" / f"step_{step:06d}.npz"
        if not npz.is_file():
            raise HTTPException(status_code=404, detail="checkpoint not found")
        # Physics fields (temperature/|B|/|J|) only exist in the final forward
        # solve; fall back to it so the slice panel keeps working on every step.
        fallback = run_dir / "final_simulation3d.npz"
        return builder.field_slice(
            npz,
            field,
            axis=axis,
            index=index,
            fallback_npz=fallback if fallback.is_file() else None,
        )

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

    app.state.roots = roots
    return app


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
