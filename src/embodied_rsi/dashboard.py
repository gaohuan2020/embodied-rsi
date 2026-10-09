from __future__ import annotations

import fcntl
import json
import logging
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

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .deployment import Runtime, current
from .monitor import resources
from .round_metrics import RoundRecorder
from .simulation import Simulation
from .storage import RUN_ID, write_json


class JobRequest(BaseModel):
    action: Literal["cycle", "collect", "train", "evaluate", "task-train", "self-improve", "initialize-rsi", "transfer-study", "grasp-study"]
    backend: Literal["compact", "rsi"] = "compact"
    dataset: str | None = None
    checkpoint: str | None = None
    episodes: int = Field(default=90, ge=1, le=300)
    explore_episodes: int = Field(default=30, ge=1, le=300)
    steps: int = Field(default=6, ge=1, le=20000)
    rounds: int = Field(default=1, ge=1, le=5)
    seed_start: int = Field(default=0, ge=0, le=10**8)
    replay_only: bool = False
    tasks: list[Literal["transfer", "stack", "barrier"]] = Field(default_factory=lambda: ["transfer"], min_length=1)
    randomize_scene: bool = False
    max_steps: int = Field(default=20, ge=1, le=100)
    audit_episodes: int = Field(default=90, ge=30, le=300)
    demos: int = Field(default=100, ge=1, le=1000)
    warmup_steps: int = Field(default=500, ge=1, le=10000)
    dev_episodes: int = Field(default=100, ge=1, le=1000)
    test_episodes: int = Field(default=500, ge=1, le=2000)
    training_seeds: list[int] = Field(default_factory=lambda: [17, 29, 43], min_length=1, max_length=5)


class Control(BaseModel):
    action: Literal["run", "pause", "resume", "step", "stop", "reset", "camera"]
    task: Literal["transfer", "stack", "barrier"] = "transfer"
    seed: int = Field(default=0, ge=0, le=10**8)
    policy: Literal["deployed", "teacher", "rsi-initial", "rsi-selected", "rsi-auto"] = "deployed"
    explore: bool = False
    continuous: bool = False
    max_steps: int = Field(default=20, ge=1, le=100)
    checkpoint: str | None = None
    randomize_scene: bool = False
    random_tasks: bool = False
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


class RestoredJobProcess:
    """Observe a surviving background job after the dashboard restarts."""
    def __init__(self, process, store, metadata):
        self.process, self.store, self.metadata = process, store, metadata
        self.pid = process.pid

    def poll(self):
        try:
            if self.process.is_running() and self.process.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            pass
        kind = {"initialize-rsi": "initialize"}.get(self.metadata["action"], self.metadata["action"])
        with self.store.connect() as db:
            row = db.execute("SELECT status FROM runs WHERE kind=? AND backend=? AND created>=? ORDER BY created LIMIT 1",
                             (kind, self.metadata["backend"], self.metadata["created"] - 1)).fetchone()
        return 0 if row and row["status"] == "completed" else -2 if row and row["status"] == "cancelled" else 1


class Jobs:
    def __init__(self, store, rsi_python=None):
        self.store, self.rsi_python = store, rsi_python
        self.jobs = {}
        self.lock = threading.Lock()
        for path in (store.root / "jobs").glob("*/job.json"):
            try:
                metadata = json.loads(path.read_text())
                if metadata.get("status") != "running":
                    continue
                process = psutil.Process(metadata["pid"])
                args = process.cmdline()
                index = args.index("--artifacts")
                if ("embodied_rsi.cli" not in args or args[index + 1] != str(store.root)
                        or args[index + 2] != metadata["action"]
                        or abs(process.create_time() - metadata["created"]) > 5):
                    continue
                self.jobs[metadata["id"]] = {
                    "process": RestoredJobProcess(process, store, metadata),
                    "metadata": metadata, "directory": path.parent,
                }
            except (OSError, ValueError, KeyError, IndexError, psutil.Error):
                continue

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
        if request.action == "grasp-study":
            if (self.store.root / "grasp-study-latest.json").exists():
                raise ValueError("此目录已有抓取实验，请使用新的实验目录")
            if request.backend != "rsi" or request.tasks != ["transfer"] or request.checkpoint:
                raise ValueError("抓取对照实验使用原始 RSI 和 transfer 任务")
            args += ["--rounds", str(request.rounds), "--steps", str(request.steps),
                     "--explore-episodes", str(request.explore_episodes), "--demos", str(request.demos),
                     "--warmup-steps", str(request.warmup_steps), "--dev-episodes", str(request.dev_episodes),
                     "--test-episodes", str(request.test_episodes), "--max-steps", str(request.max_steps),
                     "--training-seeds", *map(str, request.training_seeds)]
        elif request.action == "transfer-study":
            if request.tasks != ["transfer"] or request.episodes < 90 or request.rounds < 2:
                raise ValueError("搬运测试只接受 transfer、至少 2 轮和 90 个发布场景")
            if request.checkpoint:
                raise ValueError("清空后的测试从原始模型开始，不接受历史检查点")
            args += ["--backend", request.backend, "--rounds", str(request.rounds),
                     "--explore-episodes", str(request.explore_episodes), "--steps", str(request.steps),
                     "--episodes", str(request.episodes), "--audit-episodes", str(request.audit_episodes),
                     "--max-steps", str(request.max_steps)]
        elif request.action in {"cycle", "task-train", "self-improve"}:
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
        if request.action in {"transfer-study", "grasp-study"}:
            pass
        elif request.action in {"cycle", "self-improve"}:
            if request.episodes < 30:
                raise ValueError("自动部署评测每任务至少需要 30 个新场景")
            args += ["--rounds", str(request.rounds), "--episodes", str(request.episodes),
                     "--steps", str(request.steps), "--seed-start", str(request.seed_start)]
            if request.action == "self-improve":
                if request.episodes * len(set(request.tasks)) < 90:
                    raise ValueError("独立发布评测总计至少 90 个新场景；单任务请设置 90")
                args += ["--explore-episodes", str(request.explore_episodes), "--tasks", *request.tasks,
                         "--max-steps", str(request.max_steps)]
                if request.randomize_scene:
                    args += ["--randomize-scene"]
                if request.replay_only:
                    args += ["--replay-only"]
        elif request.action == "task-train":
            from .workflow import reserve_suite
            suite = reserve_suite(self.store, request.seed_start)
            args += ["--steps", str(request.steps), "--seed-start", str(suite["train"]),
                     "--dev-start", str(suite["dev"])]
        elif request.action == "collect":
            args += ["--episodes", str(request.episodes), "--seed-start", str(request.seed_start),
                     "--tasks", *request.tasks]
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
    from .trajectory_replay import TrajectoryReplay
    replay_archive = TrajectoryReplay(store)
    sample = {"ts": None, "data": {}}
    recorder = RoundRecorder(store)
    stop = threading.Event()

    def sampling():
        while not stop.is_set():
            sample.update(ts=time.time(), data=resources())
            try:
                recorder.refresh()
            except Exception:  # Retry reporting without stopping collection or resource sampling.
                logging.getLogger(__name__).exception("Could not refresh round metrics")
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
        replay_archive.close()

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

    @app.get("/grasp")
    def grasp_page():
        return FileResponse(web / "index.html")

    @app.get("/api/grasp/status")
    def grasp_status():
        path = store.root / "grasp-study-latest.json"
        status = json.loads(path.read_text()) if path.exists() else {"status": "idle"}
        observation = store.root / "grasp-observation.json"
        if observation.exists():
            status["live"] = json.loads(observation.read_text())
        status["jobs"] = jobs.list()
        return status

    @app.get("/api/grasp/frame.jpg")
    def grasp_frame():
        path = store.root / "grasp-frame.jpg"
        if not path.exists():
            raise HTTPException(404, "等待物理仿真画面")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"status": "ok", "schema_version": 1}

    @app.get("/api/rounds")
    def round_records():
        path = recorder.directory / "rounds.json"
        if not path.exists():
            recorder.refresh()
        return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/api/rounds/export/{name}")
    def export_round_records(name: Literal["training-rounds", "collection-rounds", "training-loss", "collection-episodes"]):
        path = recorder.directory / f"{name}.csv"
        if not path.exists():
            recorder.refresh()
        return FileResponse(path, media_type="text/csv", filename=path.name,
                            headers={"Cache-Control": "no-store"})

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

    @app.get("/api/runs/{run_id}/parameters")
    def parameters(run_id: str, after: int = -1):
        if not RUN_ID.fullmatch(run_id) or after < -1:
            raise HTTPException(400, "Invalid parameter cursor")
        from .parameter_history import parameter_history
        try:
            return parameter_history(store, run_id, after)
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs/{run_id}/replays")
    def replays(run_id: str, outcome: Literal["all", "success", "failure"] = "all", offset: int = 0, limit: int = 5):
        if not RUN_ID.fullmatch(run_id) or offset < 0 or not 1 <= limit <= 50:
            raise HTTPException(400, "Invalid replay page")
        try:
            return replay_archive.catalogue(run_id, outcome, offset, limit)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/runs/{run_id}/replay/{episode_id}")
    def replay_metadata(run_id: str, episode_id: str):
        if not RUN_ID.fullmatch(run_id) or not RUN_ID.fullmatch(episode_id):
            raise HTTPException(400, "Invalid replay id")
        try:
            return replay_archive.metadata(run_id, episode_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs/{run_id}/replay/{episode_id}/frames/{frame}.jpg")
    def archived_frame(run_id: str, episode_id: str, frame: int, view: Literal["external", "wrist"] = "external"):
        if not RUN_ID.fullmatch(run_id) or not RUN_ID.fullmatch(episode_id) or frame < 0:
            raise HTTPException(400, "Invalid replay frame")
        try:
            return Response(replay_archive.frame(run_id, episode_id, frame, view), media_type="image/jpeg")
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

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
