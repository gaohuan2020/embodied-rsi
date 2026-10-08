import json
import time

import pytest
from fastapi.testclient import TestClient

from embodied_rsi.auto_training import AutoTrainer
from embodied_rsi.dashboard import JobRequest, create_app
from embodied_rsi.scenes import scene_config, task_for
from embodied_rsi.sim import episode
from embodied_rsi.storage import Store


def test_thresholds_wait_for_new_successes_and_do_not_spawn_duplicate_jobs(tmp_path):
    class Jobs:
        def __init__(self):
            self.running = False
            self.requests = []

        def list(self):
            return [{"status": "running"}] if self.running else []

        def start(self, request):
            self.requests.append(request)
            self.running = True
            return {"id": "test-job"}

    store = Store(tmp_path)
    run = store.create("explore", "rsi", {})
    jobs = Jobs()
    auto = AutoTrainer(store, jobs, JobRequest)
    config = {"auto_train": True, "train_every": 10, "min_successes": 3, "train_steps": 100}

    def observed(n, s, backend="rsi"):
        return auto.observe(config, {"episodes": n, "successful_episodes": s}, backend, None, run)

    assert observed(10, 2)["status"] == "collecting"
    assert observed(9, 3)["status"] == "collecting"
    assert observed(10, 3)["status"] == "training"
    request = jobs.requests[0]
    assert request.replay_only and request.backend == "rsi" and request.action == "self-improve"
    assert request.episodes == 30 and request.steps == 100
    assert observed(30, 8)["status"] == "training" and len(jobs.requests) == 1
    jobs.running = False
    assert observed(20, 5)["status"] == "collecting"  # Old successes cannot retrigger training.
    assert observed(20, 6)["status"] == "training" and len(jobs.requests) == 2
    jobs.running = False
    assert observed(100, 100, "teacher")["status"] == "disabled"
    assert len(jobs.requests) == 2


def test_randomized_tasks_and_physical_positions_are_reproducible_and_varied():
    pytest.importorskip("embodied_jev")
    for block in range(4):
        assert {task_for(s) for s in range(block * 3, block * 3 + 3)} == {"transfer", "stack", "barrier"}
    scenes = []
    for seed in range(6):
        task = task_for(seed)
        ep = episode(task, seed, max_steps=1, label_mode="none", randomize_scene=True)
        assert ep["scene_distribution"] == "workspace-random-v1"
        config = scene_config(task, seed)
        assert config == scene_config(task, seed)
        assert ep["scene_config"]["source_xy"] == config["source_xy"]
        assert ep["records"][0]["before"]["destination"][:2] == config["target_xy"]
        scenes.append(ep["scene_hash"])
    assert len(set(scenes)) == 6


def test_continuous_collection_resumes_authorized_intent_after_service_restart(tmp_path):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path)
    request = {"action": "run", "policy": "teacher", "continuous": True, "max_steps": 1,
               "auto_train": False, "random_tasks": True, "randomize_scene": True}
    with TestClient(create_app(store)) as client:
        assert client.post("/api/simulation/control", json=request).status_code == 200
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if client.get("/api/simulation").json()["collection"]["episodes"] >= 1:
                break
            time.sleep(.05)
        else:
            pytest.fail("Initial continuous episode did not finish")
    intent = json.loads((store.root / "live-session.json").read_text())
    assert intent["active"]
    with TestClient(create_app(store)) as client:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            state = client.get("/api/simulation").json()
            if state["collection"]["episodes"] >= 1:
                break
            time.sleep(.05)
        else:
            pytest.fail(f"Collection did not resume: {state}")
        assert state["random_tasks"] and state["randomize_scene"]
        assert client.post("/api/simulation/control", json={"action": "stop"}).status_code == 200
    assert not json.loads((store.root / "live-session.json").read_text())["active"]
