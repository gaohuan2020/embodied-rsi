import copy
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_core import paired_rows, sample

from embodied_rsi.compact import CompactPolicy, features
from embodied_rsi.dashboard import create_app
from embodied_rsi.deployment import Runtime, deploy
from embodied_rsi.evaluation import fingerprint, paired_report
from embodied_rsi.storage import Store, write_json
from embodied_rsi.task_training import CompactLearner, task_score, terminal_reward, train_tasks
from embodied_rsi.workflow import evaluate, reserve_suite


def checkpoint(path, bias=0.):
    row = sample()["records"][0]
    p = CompactPolicy(np.full(features(row["state"], row["criteria"]).shape[1], bias))
    p.save(path)
    write_json(path / "meta.json", {"backend": "compact-numpy"})
    return path


def release_report(path, model):
    report = paired_report(paired_rows(), paired_rows(False))
    report.update(candidate_checkpoint=str(model), candidate_sha256=fingerprint(model))
    write_json(path, report)
    return path


def test_rewards_and_gradients_ignore_teacher_gold():
    row = sample()["records"][0]
    row.update(probabilities=[.5, .5], choice="approach")
    trajectory = {"success": True, "records": [row]}
    altered = copy.deepcopy(trajectory)
    altered["records"][0]["gold"] = [0., 1.]
    a, b = CompactLearner(), CompactLearner()
    for learner in (a, b):
        learner.predict(row["state"], row["criteria"])
    a.update([trajectory], [terminal_reward(trajectory)])
    b.update([altered], [terminal_reward(altered)])
    assert np.array_equal(a.policy.weights, b.policy.weights)
    assert a.predict(row["state"], row["criteria"])[0] > .5
    altered["success"] = False
    assert terminal_reward(altered) == 0.
    assert task_score(paired_rows())[0] == 1.


def test_checkpoint_selection_preserves_task_success_even_if_update_regresses(tmp_path, monkeypatch):
    model = checkpoint(tmp_path / "parent")
    row = sample()["records"][0]
    row.update(probabilities=[.5, .5], choice="approach")

    def rollout(task, seed, policy, **kwargs):
        success = kwargs.get("rng") is None and np.linalg.norm(policy.policy.weights) < 1e-8
        return {"task": task, "seed": seed, "success": success, "collisions": 0, "drops": 0,
                "rejections": 0, "records": [row]}

    monkeypatch.setattr("embodied_rsi.task_training.episode", rollout)
    store = Store(tmp_path / "store")
    run = train_tasks(store, model, steps=1, batch_size=1, dev_episodes=1)
    result = json.loads((store.run_dir(run) / "summary.json").read_text())
    assert result["selected_step"] == 0 and result["success_rate"] == 1.
    assert np.all(CompactPolicy.load(store.run_dir(run) / "checkpoint").weights == 0.)


def test_deployment_hot_update_pins_old_episode_and_rejects_failed_gate(tmp_path):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path / "store")
    first = checkpoint(tmp_path / "first")
    initial = deploy(store, release_report(tmp_path / "r1.json", first))
    runtime = Runtime(store)
    old_policy, old_version = runtime.pin()
    second = checkpoint(tmp_path / "second", .01)
    updated = deploy(store, release_report(tmp_path / "r2.json", second), initial["sha256"])
    new_policy, new_version = runtime.pin()
    assert new_version["version"] == updated["version"]
    assert old_version["version"] == initial["version"] and old_policy is not new_policy
    mismatched = json.loads((tmp_path / "r2.json").read_text())
    mismatched["champion_sha256"] = initial["sha256"]
    write_json(tmp_path / "mismatch.json", mismatched)
    with pytest.raises(ValueError, match="different deployed"):
        deploy(store, tmp_path / "mismatch.json", updated["sha256"])
    failed = json.loads((tmp_path / "r2.json").read_text())
    failed["gate"]["passed"] = False
    write_json(tmp_path / "failed.json", failed)
    with pytest.raises(ValueError, match="gates"):
        deploy(store, tmp_path / "failed.json", updated["sha256"])
    assert json.loads((store.root / "champion.json").read_text())["version"] == updated["version"]
    runtime.release(old_version)
    runtime.release(new_version)
    runtime.close()


def test_failed_warmup_and_stale_evaluation_cannot_replace_model(tmp_path, monkeypatch):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path / "store")
    first = checkpoint(tmp_path / "first")
    initial = deploy(store, release_report(tmp_path / "r1.json", first))
    second = checkpoint(tmp_path / "second", .01)
    report = release_report(tmp_path / "r2.json", second)
    with pytest.raises(ValueError, match="changed during evaluation"):
        deploy(store, report, "wrong-champion")

    def fail(*args):
        raise RuntimeError("model cannot load")

    monkeypatch.setattr("embodied_rsi.deployment.load_policy", fail)
    with pytest.raises(RuntimeError, match="cannot load"):
        deploy(store, report, initial["sha256"])
    assert json.loads((store.root / "champion.json").read_text())["sha256"] == initial["sha256"]


def test_scene_reservations_survive_restart_and_evaluation_checks_task_training(tmp_path):
    store = Store(tmp_path / "store")
    a = reserve_suite(store)
    b = reserve_suite(Store(store.root))
    assert b["train"] >= a["release"] + 10000
    model = checkpoint(tmp_path / "model")
    write_json(model / "meta.json", {"train_seed_range": [a["train"], a["train"] + 10000]})
    with pytest.raises(ValueError, match="overlap"):
        evaluate(store, None, model, None, "teacher", ["transfer"], [a["train"]])


def test_simulation_renders_real_physics_accepts_controls_and_serves_deployed_model(tmp_path):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path / "store")
    model = checkpoint(tmp_path / "model")
    deployed = deploy(store, release_report(tmp_path / "report.json", model))
    with TestClient(create_app(store)) as client:
        assert client.get("/simulation").status_code == 200
        assert client.get("/training").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = client.get("/api/simulation").json()
            if state["status"] in {"ready", "failed"}:
                break
            time.sleep(.05)
        assert state["status"] == "ready", state.get("error")
        assert state["frame"].startswith("data:image/jpeg;base64,")
        r = client.post("/api/simulation/control", json={"action": "step", "policy": "teacher"})
        assert r.status_code == 200
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = client.get("/api/simulation").json()
            if state["status"] in {"paused", "failed"}:
                break
            time.sleep(.05)
        assert state["status"] == "paused", state.get("error")
        assert len(state["records"]) == 1 and state["replay_frames"] > 1
        assert client.get("/api/simulation/replay/0").status_code == 200
        row = sample()["records"][0]
        response = client.post("/v1/systemone", json={"state": row["state"],
            "questions": {"action": {"type": "choice", "criteria": row["criteria"]}}})
        assert response.status_code == 200
        assert response.json()["model_version"] == deployed["version"]
        assert client.post("/api/simulation/control", json={"action": "reset"}).status_code == 200
        assert client.post("/api/simulation/control", json={"action": "run"},
                           headers={"Origin": "https://attacker.example"}).status_code == 403
