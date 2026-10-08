"""Interactive physics thread with continuous collection and state-local failure exploration."""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import queue
import threading

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from .exploration import ExplorationMemory, ExploringPolicy
from .sim import episode
from .storage import dumps, write_json


class Cancelled(Exception):
    pass


class Simulation:
    def __init__(self, store, runtime, on_collection=None):
        self.store, self.runtime = store, runtime
        self.on_collection = on_collection
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.cancel, self.closed, self.resume = threading.Event(), threading.Event(), threading.Event()
        self.resume.set()
        self.step_once = False
        self.camera_changed = threading.Event()
        self.renderer, self.world_model, self.initial_policy, self.initial_key = None, None, None, None
        self.memories = {}
        self.session_run, self.session_backend = None, None
        self.collection = {"episodes": 0, "successful_episodes": 0, "failed_episodes": 0, "success_rate": 0.}
        self.camera = {"azimuth": 135, "elevation": -25, "distance": 1.75, "view": "external"}
        self.frames = []
        self.state = {"status": "loading", "task": "transfer", "seed": 0, "policy": "deployed",
                      "version": None, "records": [], "frame": None, "observation": None, "result": None}
        self.thread = threading.Thread(target=self._worker, daemon=True, name="mujoco-simulation")
        self.thread.start()
        self.commands.put(("reset", {}))
        intent = store.root / "live-session.json"
        if intent.exists():
            saved = json.loads(intent.read_text())
            if saved.get("active"):
                config = saved["config"]
                if config.get("_resume_backlog"):
                    config["_auto_backlog"] = config.pop("_resume_backlog")
                self.commands.put(("run", config))

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({**self.state, "replay_frames": len(self.frames), "camera": self.camera,
                                  "collection": self.collection, "collection_run": self.session_run})

    def replay(self, index):
        with self.lock:
            if index < 0 or index >= len(self.frames):
                raise ValueError("Replay frame outside current episode")
            return copy.deepcopy(self.frames[index])

    def control(self, action, task="transfer", seed=0, policy="deployed", camera=None, explore=False,
                continuous=False, max_steps=20, checkpoint=None, randomize_scene=True, random_tasks=True,
                auto_train=True, train_every=30, min_successes=5, train_steps=100):
        with self.lock:
            active = self.state["status"] in {"running", "paused", "loading_model", "starting", "stopping"}
            if active and action == "run":
                raise ValueError("当前任务仍在运行，请先停止或等待完成")
            if action == "camera":
                self.camera.update(camera or {})
                self.camera_changed.set()
                if not active:
                    self.commands.put(("render", {}))
            elif action == "pause":
                self.resume.clear()
                if active:
                    self.state["status"] = "paused"
            elif action in {"resume", "step"} and active:
                self.step_once = action == "step"
                self.state["status"] = "running"
                self.resume.set()
            elif action in {"run", "step"}:
                self.cancel.clear()
                self.resume.set()
                self.step_once = action == "step"
                self.state["status"] = "starting"
                self.collection = {"episodes": 0, "successful_episodes": 0, "failed_episodes": 0, "success_rate": 0.}
                self.state.update(records=[], result=None, version=None)
                self.frames = []
                config = {"task": task, "seed": seed, "policy": policy, "explore": explore,
                          "continuous": continuous, "max_steps": max_steps, "checkpoint": checkpoint,
                          "randomize_scene": randomize_scene, "random_tasks": random_tasks,
                          "auto_train": auto_train, "train_every": train_every,
                          "min_successes": min_successes, "train_steps": train_steps}
                write_json(self.store.root / "live-session.json", {"active": continuous, "config": config})
                self.commands.put(("run", config))
            elif action in {"reset", "stop"}:
                write_json(self.store.root / "live-session.json", {"active": False})
                self.cancel.set()
                self.resume.set()
                if active:
                    self.state["status"] = "stopping"
                self.commands.put((action, {"task": task, "seed": seed, "policy": policy}))
            else:
                raise ValueError("Unsupported simulation control")
        return self.snapshot()

    def _render(self, world, record=True):
        import mujoco
        from PIL import Image
        if self.world_model is not world.model:
            if self.renderer:
                self.renderer.close()
            world.model.vis.global_.offwidth, world.model.vis.global_.offheight = 1024, 640
            world.model.vis.quality.shadowsize = 4096
            world.model.vis.quality.offsamples = 4
            world.model.vis.quality.numslices, world.model.vis.quality.numstacks = 48, 32
            world.model.vis.headlight.ambient[:] = [.22, .24, .27]
            world.model.vis.headlight.diffuse[:] = [.65, .65, .62]
            world.model.vis.headlight.specular[:] = [.32, .32, .32]
            table = world.model.geom("table").id
            world.model.geom_rgba[table] = [.27, .32, .37, 1.]
            self.renderer = mujoco.Renderer(world.model, height=640, width=1024)
            self.world_model = world.model
        with self.lock:
            controls = dict(self.camera)
            self.camera_changed.clear()
        camera = "wrist_camera" if controls["view"] == "wrist" else mujoco.MjvCamera()
        if not isinstance(camera, str):
            camera.lookat[:] = [.25, 0., .28]
            camera.azimuth, camera.elevation, camera.distance = controls["azimuth"], controls["elevation"], controls["distance"]
        self.renderer.update_scene(world.data, camera=camera)
        # Display-only ground: improves spatial reading without adding physics contacts.
        scene = self.renderer.scene
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_PLANE,
                               np.array([3., 3., .1]), np.array([0., 0., -.085]),
                               np.eye(3).ravel(), np.array([.07, .09, .115, 1.]))
            scene.ngeom += 1
        buffer = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=90)
        frame = {"image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
                 "observation": world.observe(), "sim_time": float(world.data.time)}
        with self.lock:
            self.state.update(frame=frame["image"], observation=frame["observation"], render_resolution="1024 × 640 · 4× MSAA")
            if record:
                self.frames.append(frame)
                if len(self.frames) > 1000:
                    self.frames.pop(0)

    def _finish_collection(self, status="completed"):
        if self.session_run:
            self.store.finish(self.session_run, status, {**self.collection, "stopped": True})
            self.session_run = None

    def _worker(self):
        world = None
        try:
            from embodied_jev.physics import RobotWorld

            from .deployment import RPCPolicy, current
            while not self.closed.is_set():
                action, config = self.commands.get()
                if action == "close":
                    break
                if action == "run" and config.get("_continue") and self.cancel.is_set():
                    continue
                if action == "render" and world is not None:
                    self._render(world, record=False)
                    continue
                if action in {"reset", "stop"}:
                    self._finish_collection()
                    if self.initial_policy:
                        self.initial_policy.close()
                        self.initial_policy = None
                    if action == "stop":
                        with self.lock:
                            self.state["status"] = "stopped"
                        continue
                if action == "run" and config.get("random_tasks"):
                    from .scenes import task_for
                    config = {**config, "_base_task": config.get("_base_task", config["task"]),
                              "task": task_for(config["seed"])}
                with self.lock:
                    self.state.update(config, records=[], result=None, error=None, version=None)
                    self.frames = []
                if action == "reset":
                    world = RobotWorld(self.state["task"], self.state["seed"])
                    self._render(world)
                    with self.lock:
                        self.state["status"] = "ready"
                    continue
                if not config.get("_continue"):
                    self._finish_collection()
                    with self.lock:
                        self.collection = {"episodes": 0, "successful_episodes": 0, "failed_episodes": 0, "success_rate": 0.}
                pinned, run, policy, records = None, None, None, []
                try:
                    with self.lock:
                        self.state["status"] = "loading_model"
                    use_rsi = config["policy"] == "rsi-auto" or (config.get("auto_train") and config["policy"] == "rsi-selected")
                    if config["policy"] == "deployed" or (use_rsi and current(self.store, "rsi")):
                        policy, pinned = self.runtime.pin("rsi" if use_rsi else None)
                        version = pinned.get("version", pinned["sha256"][:12])
                        backend = "rsi" if isinstance(policy, RPCPolicy) else "compact"
                    elif config["policy"] in {"rsi-initial", "rsi-selected", "rsi-auto"}:
                        key = config.get("checkpoint") or "v1.0-0.8b"
                        if self.initial_policy and (self.initial_key != key or self.initial_policy.process.poll() is not None):
                            self.initial_policy.close()
                            self.initial_policy = None
                        if not self.initial_policy:
                            self.initial_policy = RPCPolicy(key, self.runtime.rsi_python)
                            self.initial_key = key
                        policy, backend = self.initial_policy, "rsi"
                        version = "RSI-Jev " + (key.split("/")[-2] if "/" in key else key + " / f9248caceb89")
                    else:
                        policy, backend, version = None, "teacher", "rule-baseline"
                    if config.get("continuous") or config.get("explore"):
                        if self.session_run and self.session_backend != backend:
                            self._finish_collection()
                        if not self.session_run:
                            self.session_run = self.store.create("explore", backend, {**config, "model_version": version,
                                "mode": "continuous_simulation" if config.get("continuous") else "interactive_exploration"})
                            self.session_backend = backend
                        run = self.session_run
                    else:
                        run = self.store.create("simulation", backend, {**config, "model_version": version})
                    model_checkpoint = pinned["checkpoint"] if pinned else config.get("checkpoint")
                    memory = None
                    if config.get("explore") and policy is not None:
                        if backend not in self.memories:
                            self.memories[backend] = ExplorationMemory(self.store, backend)
                            self.memories[backend].import_past()
                        memory = self.memories[backend]
                        policy = ExploringPolicy(policy, memory)
                    with self.lock:
                        self.state.update(status="running", version=version,
                            request_protocol="RSI JSON-lines 模型请求" if backend == "rsi" else "本机参考策略",
                            sampling="失败引导探索" if memory else "确定性决策")

                    def on_frame(current_world):
                        nonlocal world
                        world = current_world
                        while not self.resume.wait(.1):
                            if self.cancel.is_set() or self.closed.is_set():
                                raise Cancelled()
                            if self.camera_changed.is_set():
                                self._render(world, record=False)
                        if self.cancel.is_set() or self.closed.is_set():
                            raise Cancelled()
                        self._render(world)
                        if self.cancel.wait(.025):
                            raise Cancelled()

                    def on_step(row, run=run, records=records):
                        records.append(row)
                        with self.lock:
                            self.state["records"].append({k: row[k] for k in
                                ("step", "choice", "probabilities", "criteria", "executed", "rejection", "latency_ms")})
                            if self.step_once:
                                self.step_once = False
                                self.resume.clear()
                                self.state["status"] = "paused"
                        self.store.event(run, "action", {"step": row["step"], "choice": row["choice"], "executed": row["executed"]})

                    result = episode(config["task"], config["seed"], policy, max_steps=config.get("max_steps", 20),
                        label_mode="none", on_frame=on_frame, on_step=on_step,
                        rng=np.random.default_rng(config["seed"]) if config.get("explore") else None,
                        randomize_scene=config.get("randomize_scene", True))
                    result.update(model_version=version, source_policy=backend)
                    if memory:
                        result["exploration_update"] = memory.observe(result)
                    with (self.store.run_dir(run) / "episodes.jsonl").open("a") as output:
                        output.write(dumps(result) + "\n")
                    summary = {k: v for k, v in result.items() if k != "records"}
                    self.store.event(run, "episode", summary)
                    with self.lock:
                        self.collection["episodes"] += 1
                        self.collection["successful_episodes"] += int(result["success"])
                        self.collection["failed_episodes"] += int(not result["success"])
                        self.collection["success_rate"] = self.collection["successful_episodes"] / self.collection["episodes"]
                        self.state.update(status="completed", result=summary)
                    if self.session_run:
                        self.store.event(run, "metric", {"step": self.collection["episodes"],
                            "episodes": self.collection["episodes"], "successful_episodes": self.collection["successful_episodes"],
                            "exploration_success_rate": self.collection["success_rate"]})
                        if self.on_collection:
                            automatic = self.on_collection(config, dict(self.collection), backend, model_checkpoint, run)
                            with self.lock:
                                self.state["automatic_training"] = automatic
                        if not config.get("continuous"):
                            self._finish_collection()
                    else:
                        write_json(self.store.run_dir(run) / "episode.json", result)
                        self.store.finish(run, summary=summary)
                except Cancelled:
                    if run:
                        write_json(self.store.run_dir(run) / "interrupted-episode.json", {"records": records,
                                   "task": config["task"], "seed": config["seed"], "model_version": self.state["version"]})
                        if run != self.session_run:
                            self.store.finish(run, "cancelled", {"model_version": self.state["version"]})
                        else:
                            self._finish_collection("cancelled" if self.closed.is_set() else "completed")
                    with self.lock:
                        self.state["status"] = "cancelled"
                except Exception as exc:  # noqa: BLE001 -- worker failures must reach the UI
                    if run:
                        if run == self.session_run:
                            self._finish_collection("failed")
                        else:
                            self.store.finish(run, "failed", {"error": str(exc)[:1000]})
                    with self.lock:
                        self.state.update(status="failed", error=str(exc)[:1000])
                finally:
                    if pinned:
                        self.runtime.release(pinned)
                if config.get("continuous") and self.state["status"] == "completed" and not self.cancel.is_set():
                    previous = config.get("_auto_backlog", {})
                    next_config = {**config, "seed": config["seed"] + 1, "_continue": True,
                        "_resume_backlog": {"episodes": previous.get("episodes", 0) + self.collection["episodes"],
                            "successful_episodes": previous.get("successful_episodes", 0) + self.collection["successful_episodes"]}}
                    write_json(self.store.root / "live-session.json", {"active": True, "config": next_config})
                    with self.lock:
                        self.state["status"] = "starting"
                    if not self.cancel.wait(.65) and not self.closed.is_set():
                        self.commands.put(("run", next_config))
        except Exception as exc:  # noqa: BLE001 -- rendering setup failures need an actionable error
            with self.lock:
                self.state.update(status="failed", error=str(exc)[:1000])
        finally:
            self._finish_collection("cancelled")
            if self.initial_policy:
                self.initial_policy.close()
            if self.renderer:
                self.renderer.close()

    def close(self):
        self.closed.set()
        self.cancel.set()
        self.resume.set()
        self.commands.put(("close", {}))
        self.thread.join(timeout=5)
