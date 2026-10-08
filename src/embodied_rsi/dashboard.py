from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .monitor import resources
from .storage import RUN_ID, write_json


class JobRequest(BaseModel):
    action: Literal["cycle", "collect", "train", "evaluate"]
    backend: Literal["compact", "rsi"] = "compact"
    dataset: str | None = None
    checkpoint: str | None = None
    episodes: int = Field(default=30, ge=1, le=300)
    steps: int = Field(default=500, ge=1, le=20000)
    rounds: int = Field(default=2, ge=1, le=5)
    seed_start: int = Field(default=0, ge=0, le=10**8)


class Jobs:
    def __init__(self, store, rsi_python=None):
        self.store, self.rsi_python = store, rsi_python
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, request):
        interpreter = self.rsi_python if request.backend == "rsi" else sys.executable
        if not interpreter or not Path(interpreter).is_file():
            raise ValueError("Configure --rsi-python before launching RSI jobs")
        args = [interpreter, "-m", "embodied_rsi.cli", "--artifacts", str(self.store.root), request.action]
        if request.action == "cycle":
            if request.backend != "compact":
                raise ValueError("Automatic cycle currently uses compact reference backend")
            args += ["--rounds", str(request.rounds), "--episodes", str(request.episodes),
                     "--steps", str(request.steps), "--seed-start", str(request.seed_start)]
        elif request.action == "collect":
            args += ["--episodes", str(request.episodes), "--seed-start", str(request.seed_start)]
        else:
            # Only paths selected from our artifact catalogue, never arbitrary executable arguments.
            name = request.dataset or ""
            if not RUN_ID.fullmatch(name) or not (self.store.root / "datasets" / name / "manifest.json").is_file():
                raise ValueError("Select an existing dataset")
            args += ["--dataset", str(self.store.root / "datasets" / name), "--backend", request.backend]
            if request.action == "train":
                args += ["--steps", str(request.steps)]
                if request.backend == "rsi" and not request.checkpoint:
                    args += ["--checkpoint", "v3.0-2b"]
            else:
                if not request.checkpoint:
                    raise ValueError("Select a checkpoint for evaluation")
                args += ["--episodes", str(request.episodes), "--seed-start", str(request.seed_start)]
            if request.checkpoint:
                path = self.store.run_dir(request.checkpoint) / "checkpoint"
                if not (path / "meta.json").is_file():
                    raise ValueError("Unknown checkpoint")
                backend = json.loads((path / "meta.json").read_text()).get("backend", "rsi-jev")
                if (backend == "compact-numpy") != (request.backend == "compact"):
                    raise ValueError("Checkpoint backend does not match selected backend")
                args += ["--checkpoint", str(path)]
        with self.lock:
            if any(j["process"].poll() is None for j in self.jobs.values()):
                raise ValueError("A job is already running; stop it or wait before launching another")
            job_id = "job-" + uuid.uuid4().hex[:12]
            directory = self.store.root / "jobs" / job_id
            directory.mkdir(parents=True)
            with (directory / "output.log").open("w") as log:
                process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=self.store.root.parent, start_new_session=True)
            metadata = {"id": job_id, "action": request.action, "backend": request.backend,
                        "created": time.time(), "pid": process.pid}
            self.jobs[job_id] = {"process": process, "metadata": metadata, "directory": directory}
            write_json(directory / "job.json", metadata)
            return metadata

    def list(self):
        result = []
        with self.lock:
            for job in self.jobs.values():
                code = job["process"].poll()
                r = {**job["metadata"], "status": "running" if code is None else
                     "completed" if code == 0 else "cancelled" if code < 0 else "failed", "exit_code": code}
                write_json(job["directory"] / "job.json", r)
                result.append(r)
        return result

    def stop(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["process"].poll() is not None:
                raise ValueError("No active job with that id")
            os.killpg(job["process"].pid, signal.SIGINT)


def create_app(store, rsi_python=None):
    jobs = Jobs(store, rsi_python)
    sample = {"ts": None, "data": {}}
    stop = threading.Event()

    def sampling():
        while not stop.is_set():
            sample.update(ts=time.time(), data=resources())
            stop.wait(5)

    @asynccontextmanager
    async def lifespan(_app):
        thread = threading.Thread(target=sampling, daemon=True)
        thread.start()
        yield
        stop.set()
        thread.join(timeout=4)

    app = FastAPI(title="Embodied RSI Dashboard", lifespan=lifespan)
    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Also prevents DNS rebinding and cross-origin launches on a localhost dashboard.
        host = request.url.hostname
        origin = request.headers.get("origin")
        if host not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Use localhost or an SSH tunnel"}, status_code=403)
        if request.method != "GET" and origin and origin != str(request.base_url).rstrip("/"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Cross-origin mutations are blocked"}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "schema_version": 1}

    @app.get("/api/overview")
    def overview():
        datasets = []
        for f in sorted((store.root / "datasets").glob("*/manifest.json")):
            try:
                datasets.append({"id": f.parent.name, **json.loads(f.read_text())})
            except (OSError, ValueError):
                pass
        champion_path = store.root / "champion.json"
        return {"runs": store.runs(), "resources": sample["data"], "resource_sample_at": sample["ts"],
                "datasets": datasets, "jobs": jobs.list(), "rsi_configured": bool(rsi_python),
                "champion": json.loads(champion_path.read_text()) if champion_path.exists() else None}

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = 0):
        if not RUN_ID.fullmatch(run_id) or after < 0:
            raise HTTPException(400, "Invalid event cursor")
        return {"events": store.events(run_id, after)}

    @app.post("/api/jobs", status_code=201)
    def launch(request: JobRequest):
        try:
            return jobs.start(request)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/stop")
    def cancel(job_id: str):
        try:
            jobs.stop(job_id)
            return {"status": "stopping"}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
