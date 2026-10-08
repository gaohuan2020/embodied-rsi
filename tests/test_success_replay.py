import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_core import sample

from embodied_rsi.dashboard import create_app
from embodied_rsi.exploration import ExplorationMemory, ExploringPolicy
from embodied_rsi.self_improvement import fit_successes, success_dataset
from embodied_rsi.storage import Store, dumps
from embodied_rsi.task_training import CompactLearner


def model_episode(success=True):
    row = sample()["records"][0]
    row.update(choice="grasp", executed=True, gold=[1., 0.])
    return {"success": success, "source_policy": "rsi", "task": "transfer", "seed": 42,
            "group": "transfer:42:0", "episode_id": "model-42", "records": [row]}


def test_replay_keeps_only_completed_model_actions_and_ignores_teacher_gold(tmp_path):
    file = tmp_path / "episodes.jsonl"
    file.write_text(dumps(model_episode()) + "\n" + dumps(model_episode(False)) + "\n")
    output = tmp_path / "success-dataset"
    manifest = success_dataset([file], output)
    assert manifest["episodes_total"] == 2 and manifest["successful_episodes"] == 1
    assert manifest["failed_episodes"] == 1 and manifest["cases"]["train"] == 1
    c = json.loads((output / "train.jsonl").read_text())
    assert c["gold"]["action"] == [0., 1.]  # The executed successful choice, not the teacher label.
    assert c["provenance"]["source_policy"] == "rsi"
    assert c["provenance"]["episode_success"]


def test_teacher_trajectory_is_never_presented_as_model_success(tmp_path):
    ep = model_episode()
    ep["source_policy"] = "teacher"
    file = tmp_path / "episodes.jsonl"
    file.write_text(dumps(ep) + "\n")
    with pytest.raises(ValueError, match="teacher"):
        success_dataset([file], tmp_path / "bad")


def test_zero_successes_cannot_train_and_data_api_shows_failures(tmp_path):
    store = Store(tmp_path / "artifacts")
    run = store.create("explore", "rsi", {})
    file = store.run_dir(run) / "episodes.jsonl"
    file.write_text(dumps(model_episode(False)) + "\n")
    output = store.root / "datasets" / "no-success"
    success_dataset([file], output)
    with pytest.raises(ValueError, match="No successful"):
        fit_successes(store, CompactLearner(), output, steps=1, backend="compact")
    with TestClient(create_app(store)) as client:
        assert client.get("/api/datasets/no-success/samples").json()["samples"] == []
        failed = client.get(f"/api/runs/{run}/episodes?outcome=failure").json()
        assert failed["total"] == 1 and not failed["episodes"][0]["success"]
        assert client.get(f"/api/runs/{run}/episodes?outcome=success").json()["total"] == 0
        assert client.get("/api/datasets/not-present/samples").status_code == 404


def test_failed_grasp_changes_exploration_but_never_mutates_the_model(tmp_path):
    class Model:
        def predict(self, state, criteria):
            return np.array([.5, .5])
    store = Store(tmp_path)
    memory = ExplorationMemory(store)
    model = Model()
    policy = ExploringPolicy(model, memory)
    ep = model_episode(False)
    row = ep["records"][0]
    row["state"]["observation"]["task"] = "transfer"
    row["before"] = row["after"] = row["state"]["observation"]
    before = policy.predict(row["state"], row["criteria"])
    memory.observe(ep)
    assert memory.observe(ep)["already_observed"]
    after = policy.predict(row["state"], row["criteria"])
    assert after[1] < before[1]
    assert np.allclose(model.predict(row["state"], row["criteria"]), [.5, .5])
    assert np.isclose(after.sum(), 1.) and (after > 0).all()
    assert ExplorationMemory(Store(store.root)).stats(row["state"], ["grasp"])[0]["failures"] == 1.


def test_continuous_simulation_restarts_at_budget_and_stops_without_false_success(tmp_path):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path)
    with TestClient(create_app(store)) as client:
        request = {"action": "run", "policy": "teacher", "continuous": True, "max_steps": 1}
        assert client.post("/api/simulation/control", json=request).status_code == 200
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            state = client.get("/api/simulation").json()
            if state["collection"]["episodes"] >= 2 or state["status"] == "failed":
                break
            time.sleep(.05)
        assert state["collection"]["episodes"] >= 2, state.get("error")
        assert state["collection"]["successful_episodes"] == 0
        assert state["seed"] >= 1
        run = state["collection_run"]
        # Stop must remain valid even if the model picker contains a stale checkpoint.
        assert client.post("/api/simulation/control", json={"action": "stop", "policy": "rsi-selected",
                           "checkpoint": "not-a-real-checkpoint"}).status_code == 200
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = client.get("/api/simulation").json()
            if state["status"] == "stopped":
                break
            time.sleep(.05)
        assert state["status"] == "stopped"
        episodes = client.get(f"/api/runs/{run}/episodes").json()["episodes"]
        assert len(episodes) >= 2
        assert all(e["steps"] == 1 and not e["success"] and e["failure"] == "budget_exhausted" for e in episodes)
        time.sleep(.8)
        assert client.get(f"/api/runs/{run}/episodes").json()["total"] == len(episodes)
