"""Drive the existing visible simulation -> training -> evaluation -> update APIs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from embodied_rsi.dataset import build_dataset
from embodied_rsi.storage import Store, dumps, write_json

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts"
URL = "http://127.0.0.1:8091"
BASELINE = Path("/home/huan/.cache/huggingface/hub/models--shgao--rsi-jev-v1.0-qwen3.5-0.8b/snapshots/f9248caceb89caf2e6c968ea33bf0d6eb7f957b0")


def api(path, body=None):
    request = urllib.request.Request(URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    store = Store(ROOT)
    protocol = {"data_source": "visible /simulation MuJoCo execution only",
                "bootstrap": {"policy": "teacher", "successful_episodes": 100, "maximum_attempts": 150},
                "training": {"checkpoint": str(BASELINE), "steps": 500, "objective": "supervised_initialization"},
                "release": {"paired_episodes": 90, "seed_start": 100000, "randomize_scene": True},
                "online": {"policy": "rsi-auto", "train_every": 30, "min_successes": 5,
                           "train_steps": 300, "seed_start": 10000000},
                "success_definition": "strict_grasp_v1", "input_contract": "grasp-observation-v1"}
    run = store.create("self-improve", "rsi", {"mode": "visible_simulation_learning", **protocol})
    write_json(store.run_dir(run) / "protocol.json", protocol)
    state = {"run": run, "status": "running", "protocol": protocol, "started": time.time()}
    def stage(name, **values):
        state.update(stage=name, updated=time.time(), **values)
        write_json(ROOT / "simulation-learning-latest.json", state)
        store.event(run, "stage", {"stage": name, **values})
        print(dumps(state), flush=True)
    try:
        stage("exploring", source="visible_simulation_teacher_bootstrap")
        api("/api/simulation/control", {"action": "run", "task": "transfer", "seed": 0,
            "policy": "teacher", "continuous": True, "explore": False, "max_steps": 20,
            "randomize_scene": True, "random_tasks": False, "auto_train": False})
        while True:
            sim = api("/api/simulation")
            collection = sim["collection"]
            if sim["status"] in {"failed", "stopped", "cancelled"}:
                raise RuntimeError(f"Visible collection stopped: {sim.get('error') or sim['status']}")
            if collection["successful_episodes"] >= 100 or collection["episodes"] >= 150:
                collection_run = sim["collection_run"]
                api("/api/simulation/control", {"action": "stop"})
                break
            store.event(run, "heartbeat", {"episodes": collection["episodes"],
                                           "successful_episodes": collection["successful_episodes"]})
            time.sleep(3)
        while api("/api/simulation")["status"] not in {"stopped", "cancelled"}:
            time.sleep(1)
        source = store.run_dir(collection_run) / "episodes.jsonl"
        episodes = [json.loads(line) for line in source.read_text().splitlines() if line]
        good = [e for e in episodes if e["success"]]
        if len(good) < 100:
            raise RuntimeError(f"Only {len(good)} strict successful demonstrations in 150 visible attempts")
        filtered_source = store.run_dir(run) / "simulation-successes.jsonl"
        filtered_source.write_text("".join(dumps(e) + "\n" for e in good))
        dataset = ROOT / "datasets" / "simulation-bootstrap"
        manifest = build_dataset([filtered_source], dataset)
        manifest.update(data_source="visible_simulation", collection_run=collection_run,
                        label_source="teacher", input_contract="grasp-observation-v1",
                        success_definition="strict_grasp_v1")
        write_json(dataset / "manifest.json", manifest)
        stage("success_replay_training", collection=collection_run, dataset=str(dataset),
              successful_episodes=len(good), training_samples=manifest["cases"]["train"],
              label_source="teacher_initialization")
        job = api("/api/jobs", {"action": "initialize-rsi", "backend": "rsi",
                               "dataset": dataset.name, "steps": 500})
        while True:
            jobs = api("/api/overview")["jobs"]
            status = next(j for j in jobs if j["id"] == job["id"])
            if status["status"] != "running":
                if status["status"] != "completed":
                    raise RuntimeError(f"Training {status['status']}; inspect its output.log")
                break
            store.event(run, "heartbeat", {"training_job": job["id"]})
            time.sleep(5)
        training = next(r for r in store.runs() if r["kind"] == "initialize" and r["status"] == "completed")
        checkpoint = store.run_dir(training["id"]) / "checkpoint"
        stage("independent_evaluation", training=training["id"], checkpoint=str(checkpoint))
        args = [sys.executable, "-m", "embodied_rsi.cli", "--artifacts", str(ROOT), "evaluate",
                "--backend", "rsi", "--checkpoint", str(checkpoint), "--champion", str(BASELINE),
                "--tasks", "transfer", "--episodes", "90", "--seed-start", "100000",
                "--dataset", str(dataset), "--max-steps", "20", "--randomize-scene"]
        with (store.run_dir(run) / "evaluation.log").open("w") as log:
            process = subprocess.Popen(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            write_json(store.run_dir(run) / "evaluation-process.json", {"pid": process.pid})
            while process.poll() is None:
                store.event(run, "heartbeat", {"evaluation_pid": process.pid})
                time.sleep(5)
            if process.returncode:
                raise RuntimeError("Paired evaluation failed; inspect evaluation.log")
        evaluation = next(r for r in store.runs() if r["kind"] == "evaluate" and r["status"] == "completed")
        report = evaluation["summary"]
        if report["gate"]["passed"]:
            stage("deploying", success_rate=report["success_rate"], paired_gain=report["paired_gain"])
            args = [sys.executable, "-m", "embodied_rsi.cli", "--artifacts", str(ROOT), "promote",
                    "--report", str(store.run_dir(evaluation["id"]) / "report.json")]
            subprocess.run(args, cwd=REPO, check=True)
            stage("completed", status="deployed", evaluation=evaluation["id"], report=report)
        else:
            stage("completed", status="retained_current_model", evaluation=evaluation["id"], report=report)
        # All subsequent training is triggered only by completed visible model episodes.
        api("/api/simulation/control", {"action": "run", "task": "transfer", "seed": 10000000,
            "policy": "rsi-auto", "continuous": True, "explore": True, "max_steps": 20,
            "randomize_scene": True, "random_tasks": False, "auto_train": True,
            "train_every": 30, "min_successes": 5, "train_steps": 300})
        store.finish(run, summary=state)
    except BaseException as exc:
        state.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        write_json(ROOT / "simulation-learning-latest.json", state)
        store.finish(run, state["status"], {"error": str(exc)})
        raise


if __name__ == "__main__":
    os.environ.setdefault("EMBODIED_RSI_STRICT_GRASP", "1")
    main()
