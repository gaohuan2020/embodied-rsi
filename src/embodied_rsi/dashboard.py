from __future__ import annotations

import fcntl
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

from .deployment import Runtime, current
from .monitor import resources
from .simulation import Simulation
from .storage import RUN_ID, write_json


class JobRequest(BaseModel):
    action: Literal["cycle", "collect", "train", "evaluate", "task-train", "self-improve", "initialize-rsi"]
    backend: Literal["compact", "rsi"] = "compact"
    dataset: str | None = None
    checkpoint: str | None = None
    episodes: int = Field(default=30, ge=1, le=300)
    explore_episodes: int = Field(default=30, ge=1, le=300)
    steps: int = Field(default=6, ge=1, le=20000)
    rounds: int = Field(default=1, ge=1, le=5)
    seed_start: int = Field(default=0, ge=0, le=10**8)
    replay_only: bool = False


class Control(BaseModel):
    action: Literal["run", "pause", "resume", "step", "stop", "reset", "camera"]
    task: Literal["transfer", "stack", "barrier"] = "transfer"
    seed: int = Field(default=0, ge=0, le=10**8)
    policy: Literal["deployed", "teacher", "rsi-initial", "rsi-selected", "rsi-auto"] = "deployed"
    explore: bool = False
    continuous: bool = False
    max_steps: int = Field(default=20, ge=1, le=100)
    checkpoint: str | None = None
    randomize_scene: bool = True
    random_tasks: bool = True
    auto_train: bool = True
    train_every: int = Field(default=30, ge=1, le=10000)
    min_successes: int = Field(default=5, ge=1, le=1000)
    train_steps: int = Field(default=100, ge=1, le=20000)
    view: Literal["external", "wrist"] = "external"
    azimuth: float = Field(default=135, ge=-360, le=360)
    elevation: float = Field(default=-25, ge=-85, le=-5)
    distance: float = Field(default=1.75, ge=.5, le=3)


class Decision(BaseModel):
    state: dict
    questions: dict
    model: str = "deployed"


class Jobs:
    def __init__(self, store, rsi_python=None):
        self.store, self.rsi_python = store, rsi_python
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, request):
        with (self.store.root / "training.lock").open("a+") as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("已有训练进程运行，继续采集并等待其结束") from None
        interpreter = self.rsi_python if request.backend == "rsi" else sys.executable
        if not interpreter or not Path(interpreter).is_file():
            raise ValueError("Configure --rsi-python before launching RSI jobs")
        args = [interpreter, "-m", "embodied_rsi.cli", "--artifacts", str(self.store.root), request.action]
        if request.action in {"cycle", "task-train", "self-improve"}:
            if request.steps > 1000 and request.action != "self-improve":
                raise ValueError("任务训练每轮最多 1000 次整局更新")
            args += ["--backend", request.backend]
            if request.checkpoint:
                path = self.store.run_dir(request.checkpoint) / "checkpoint"
                if not (path / "meta.json").is_file():
                    raise ValueError("Unknown checkpoint")
                meta = json.loads((path / "meta.json").read_text())
                if (meta.get("backend") == "compact-numpy") != (request.backend == "compact"):
                    raise ValueError("Checkpoint backend does not match selected backend")
                args += ["--checkpoint", str(path)]
        if request.action in {"cycle", "self-improve"}:
            if request.episodes < 30:
                raise ValueError("自动部署评测每任务至少需要 30 个新场景")
            args += ["--rounds", str(request.rounds), "--episodes", str(request.episodes),
                     "--steps", str(request.steps), "--seed-start", str(request.seed_start)]
            if request.action == "self-improve":
                args += ["--explore-episodes", str(request.explore_episodes)]
                if request.replay_only:
                    args += ["--replay-only"]
        elif request.action == "task-train":
            from .workflow import reserve_suite
            suite = reserve_suite(self.store, request.seed_start)
            args += ["--steps", str(request.steps), "--seed-start", str(suite["train"]),
                     "--dev-start", str(suite["dev"])]
        elif request.action == "collect":
            args += ["--episodes", str(request.episodes), "--seed-start", str(request.seed_start)]
        else:
            # Only paths selected from our artifact catalogue, never arbitrary executable arguments.
            name = request.dataset or ""
            if not RUN_ID.fullmatch(name) or not (self.store.root / "datasets" / name / "manifest.json").is_file():
                raise ValueError("Select an existing dataset")
            args += ["--dataset", str(self.store.root / "datasets" / name)]
            if request.action != "initialize-rsi":
                args += ["--backend", request.backend]
            elif request.backend != "rsi":
                raise ValueError("RSI initialization requires the RSI backend")
            if request.action in {"train", "initialize-rsi"}:
                args += ["--steps", str(request.steps)]
                if request.action == "train" and request.backend == "rsi" and not request.checkpoint:
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
    runtime = Runtime(store, rsi_python)
    from .auto_training import AutoTrainer
    automatic = AutoTrainer(store, jobs, JobRequest)
    simulation = Simulation(store, runtime, automatic.observe)
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
        simulation.close()
        runtime.close()

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
    @app.get("/training")
    def index():
        return FileResponse(web / "index.html")

    @app.get("/simulation")
    def simulation_page():
        return FileResponse(web / "simulation.html")

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
        runs = store.runs()
        for run in runs:
            if run["kind"] == "explore" and run["status"] == "running":
                metrics = [e["payload"] for e in store.events(run["id"]) if e["kind"] == "metric"]
                if metrics:
                    last = metrics[-1]
                    run["summary"].update(episodes=last.get("episodes"),
                        successful_episodes=last.get("successful_episodes"),
                        success_rate=last.get("exploration_success_rate"))
        return {"runs": runs, "resources": sample["data"], "resource_sample_at": sample["ts"],
                "datasets": datasets, "jobs": jobs.list(), "rsi_configured": bool(rsi_python),
                "champion": json.loads(champion_path.read_text()) if champion_path.exists() else None,
                "runtime": runtime.status(),
                "rsi_champion": current(store, "rsi"),
                "deployments": [r for r in store.runs() if r["kind"] == "deploy"]}

    @app.get("/api/datasets/{dataset_id}/samples")
    def dataset_samples(dataset_id: str, offset: int = 0, limit: int = 10):
        if not RUN_ID.fullmatch(dataset_id) or offset < 0 or not 1 <= limit <= 50:
            raise HTTPException(400, "Invalid dataset page")
        path = store.root / "datasets" / dataset_id / "train.jsonl"
        if not path.is_file():
            raise HTTPException(404, "Dataset not found")
        rows = []
        with path.open() as file:
            for index, line in enumerate(file):
                if index < offset:
                    continue
                if len(rows) >= limit:
                    break
                rows.append(json.loads(line))
        return {"samples": rows, "offset": offset, "limit": limit}

    @app.get("/api/runs/{run_id}/episodes")
    def episode_samples(run_id: str, outcome: Literal["all", "success", "failure"] = "all",
                        offset: int = 0, limit: int = 10):
        if not RUN_ID.fullmatch(run_id) or offset < 0 or not 1 <= limit <= 50:
            raise HTTPException(400, "Invalid trajectory page")
        path = store.run_dir(run_id) / "episodes.jsonl"
        if not path.is_file():
            raise HTTPException(404, "Trajectory collection not found")
        rows, count = [], 0
        with path.open() as file:
            for line in file:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                if outcome != "all" and bool(row["success"]) != (outcome == "success"):
                    continue
                if count >= offset and len(rows) < limit:
                    rows.append(row)
                count += 1
        return {"episodes": rows, "total": count, "offset": offset}

    @app.get("/api/simulation")
    def simulation_status():
        return simulation.snapshot()

    @app.post("/api/simulation/control")
    def simulation_control(request: Control):
        try:
            checkpoint = None
            if request.policy == "rsi-selected" and request.action in {"run", "step"}:
                if not request.checkpoint or not RUN_ID.fullmatch(request.checkpoint):
                    raise ValueError("Select an RSI checkpoint from the model catalogue")
                checkpoint = store.run_dir(request.checkpoint) / "checkpoint"
                meta = json.loads((checkpoint / "meta.json").read_text())
                if meta.get("backend") != "rsi-jev":
                    raise ValueError("Selected model is not RSI-Jev")
            return simulation.control(request.action, request.task, request.seed, request.policy,
                {"view": request.view, "azimuth": request.azimuth,
                 "elevation": request.elevation, "distance": request.distance}, request.explore,
                 request.continuous, request.max_steps, str(checkpoint) if checkpoint else None,
                 randomize_scene=request.randomize_scene, random_tasks=request.random_tasks,
                 auto_train=request.auto_train, train_every=request.train_every,
                 min_successes=request.min_successes, train_steps=request.train_steps)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/simulation/replay/{index}")
    def replay(index: int):
        try:
            return simulation.replay(index)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/v1/systemone")
    def predict(request: Decision):
        try:
            question = request.questions["action"]
            criteria = question["criteria"]
            if question.get("type") != "choice" or not 2 <= len(criteria) <= 9:
                raise ValueError("Expected an action choice question with 2–9 skills")
            policy, model = runtime.pin()
            try:
                probabilities = policy.predict(request.state, criteria)
            finally:
                runtime.release(model)
            import numpy as np
            if (len(probabilities) != len(criteria) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or abs(probabilities.sum() - 1) > .001):
                raise ValueError("Model produced invalid probabilities")
            return {"model_version": model.get("version", model["sha256"][:12]),
                    "answers": {"action": {"choice": list(criteria)[int(np.argmax(probabilities))],
                    "probabilities": dict(zip(criteria, probabilities.tolist()))}}}
        except (ValueError, KeyError, RuntimeError, TypeError, AttributeError) as exc:
            raise HTTPException(503, str(exc)) from exc

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
