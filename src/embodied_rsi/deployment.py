"""Validated, atomic model deployment; episodes pin a policy version until completion."""
from __future__ import annotations

import fcntl
import json
import selectors
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .compact import CompactPolicy
from .evaluation import fingerprint
from .storage import dumps, write_json


class RPCPolicy:
    """Keep GPU dependencies out of the dashboard's CPU environment."""
    def __init__(self, checkpoint, interpreter=None, revision=None):
        self.lock = threading.Lock()
        if checkpoint == "v1.0-0.8b" and revision is None:
            revision = "f9248caceb89caf2e6c968ea33bf0d6eb7f957b0"
        args = [interpreter or sys.executable, "-m", "embodied_rsi.inference_worker", str(checkpoint)]
        if revision:
            args.append(revision)
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, start_new_session=True)
        try:
            self._read(240)
        except BaseException:
            self.close()
            raise

    def _read(self, timeout=60):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise RuntimeError("Model worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Model worker exited before responding")
        reply = json.loads(line)
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply

    def predict(self, state, criteria):
        with self.lock:
            self.process.stdin.write(dumps({"state": state, "criteria": criteria}) + "\n")
            self.process.stdin.flush()
            return np.asarray(self._read()["probabilities"])

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()


def load_policy(checkpoint, rsi_python=None):
    meta = json.loads((Path(checkpoint) / "meta.json").read_text())
    if meta.get("backend") == "compact-numpy":
        return CompactPolicy.load(checkpoint)
    return RPCPolicy(checkpoint, rsi_python)


def current(store, backend=None):
    path = store.root / (f"champion-{backend}.json" if backend else "champion.json")
    return json.loads(path.read_text()) if path.exists() else None


def deploy(store, report_path, expected_sha=None, rsi_python=None, registry_backend=None):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if not report["gate"]["passed"] or not all(report["gate"]["checks"].values()):
        raise ValueError("Task completion release gates failed")
    checkpoint = Path(report["candidate_checkpoint"])
    sha = report["candidate_sha256"]
    if fingerprint(checkpoint) != sha:
        raise ValueError("Checkpoint changed after evaluation")
    if expected_sha is not None and "champion_sha256" in report and report["champion_sha256"] != expected_sha:
        raise ValueError("Release report was evaluated against a different deployed model")
    run = store.create("deploy", "inference", {"checkpoint": str(checkpoint), "sha256": sha})
    try:
        policy = load_policy(checkpoint, rsi_python)
        try:
            from embodied_jev.physics import RobotWorld

            from .sim import criteria_for, menu_for, state_for
            world = RobotWorld("transfer", 77777777)
            criteria = criteria_for(menu_for(world))
            p = policy.predict(state_for(world, []), criteria)
            if (len(p) != len(criteria) or not np.isfinite(p).all() or (p < 0).any()
                    or abs(p.sum() - 1) > .001):
                raise ValueError("Deployment warmup returned invalid probabilities")
        finally:
            if hasattr(policy, "close"):
                policy.close()
        if fingerprint(checkpoint) != sha:
            raise ValueError("Checkpoint changed during warmup")
        # Serialize writers and reject releases evaluated against a superseded champion.
        with (store.root / "deployment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = current(store, registry_backend)
            if (previous or {}).get("sha256") != expected_sha:
                raise ValueError("Deployed model changed during evaluation; reevaluate candidate")
            model = {"version": run, "checkpoint": str(checkpoint.resolve()), "sha256": sha,
                     "evaluation": str(report_path.resolve()), "report": report,
                     "deployed_at": time.time(), "status": "ready", "scope": "simulation",
                     "previous_version": (previous or {}).get("version"), "warmup": "passed"}
            meta = json.loads((checkpoint / "meta.json").read_text())
            model["backend"] = "compact" if meta.get("backend") == "compact-numpy" else "rsi"
            write_json(store.run_dir(run) / "deployment.json", model)
            write_json(store.root / f"champion-{model['backend']}.json", model)
            write_json(store.root / "champion.json", model)
        store.finish(run, summary=model)
        return model
    except BaseException as exc:
        store.finish(run, "failed", {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise


class Runtime:
    def __init__(self, store, rsi_python=None):
        self.store, self.rsi_python = store, rsi_python
        self.lock = threading.Lock()
        self.loaded = None
        self.loaded_sha = None
        self.error = None
        self.policies = {}
        self.references = {}

    def pin(self, backend=None):
        with self.lock:
            model = current(self.store, backend)
            if model is None:
                raise ValueError("尚无部署模型，请先在训练页运行自动学习闭环")
            if model["sha256"] != self.loaded_sha:
                try:
                    if fingerprint(model["checkpoint"]) != model["sha256"]:
                        raise ValueError("Deployed checkpoint integrity check failed")
                    policy = load_policy(model["checkpoint"], self.rsi_python)
                    self.loaded = (policy, model)
                    self.policies[model["sha256"]] = self.loaded
                    self.loaded_sha = model["sha256"]
                    self.error = None
                except Exception as exc:
                    self.error = str(exc)
                    raise
            sha = self.loaded[1]["sha256"]
            self.references[sha] = self.references.get(sha, 0) + 1
            return self.loaded

    def release(self, model):
        with self.lock:
            sha = model["sha256"]
            self.references[sha] = max(0, self.references.get(sha, 0) - 1)
            for old_sha in list(self.policies):
                if old_sha != self.loaded_sha and self.references.get(old_sha, 0) == 0:
                    policy, _ = self.policies.pop(old_sha)
                    if hasattr(policy, "close"):
                        policy.close()

    def close(self):
        with self.lock:
            for policy, _ in self.policies.values():
                if hasattr(policy, "close"):
                    policy.close()
            self.policies.clear()

    def status(self):
        return {"deployment": current(self.store), "loaded_sha256": self.loaded_sha,
                "error": self.error}
