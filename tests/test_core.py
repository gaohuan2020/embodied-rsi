import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embodied_rsi.compact import CompactPolicy, features
from embodied_rsi.dashboard import create_app
from embodied_rsi.dataset import build_dataset, split_for, validate_dataset
from embodied_rsi.evaluation import fingerprint, paired_report, promote
from embodied_rsi.monitor import Monitor
from embodied_rsi.sim import POLICY_VERSION
from embodied_rsi.storage import Store, dumps, write_json
from embodied_rsi.training import train_compact


def sample(group="transfer:0:0"):
    state = {"observation": {"tcp": [.4, -.1, .22], "object": [.43, -.17, .02],
        "destination": [.43, .18, .026], "held": False, "gripper": "open",
        "support_contact": False, "grasp_secured": False}}
    criteria = {"approach": dumps({"skill": "approach", "target_tcp": [.43, -.17, .16],
                                   "duration_seconds": .8}),
                "grasp": dumps({"skill": "grasp", "target_tcp": [.4, -.1, .22], "duration_seconds": .65})}
    return {"group": group, "task": "transfer", "episode_id": group, "records": [{"step": 0,
            "state": state, "criteria": criteria, "gold": [1., 0.], "label_source": "teacher"}]}


def dataset_fixture(tmp_path):
    eps = []
    found = set()
    for i in range(1000):
        group = f"transfer:{i}:0"
        split = split_for(group)
        if split not in found:
            ep = sample(group)
            ep["records"][0]["state"]["observation"]["tcp"][0] += i * .0001
            eps.append(ep)
            found.add(split)
        if len(found) == 3:
            break
    file = tmp_path / "episodes.jsonl"
    file.write_text("".join(dumps(e) + "\n" for e in eps))
    out = tmp_path / "dataset"
    build_dataset([file], out)
    return out


def test_dataset_grouping_and_checksum(tmp_path):
    dataset = dataset_fixture(tmp_path)
    manifest = validate_dataset(dataset)
    assert all(n == 1 for n in manifest["cases"].values())
    with (dataset / "train.jsonl").open("a") as file:
        file.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(dataset)


def test_identical_decision_cannot_cross_splits(tmp_path):
    a, b = next((i, j) for i in range(20) for j in range(20)
                if split_for(str(i)) != split_for(str(j)))
    file = tmp_path / "episodes.jsonl"
    file.write_text(dumps(sample(str(a))) + "\n" + dumps(sample(str(b))) + "\n")
    with pytest.raises(ValueError, match="across splits"):
        build_dataset([file], tmp_path / "dataset")


def test_compact_probability_tracks_action_when_order_changes():
    r = sample()["records"][0]
    w = np.arange(features(r["state"], r["criteria"]).shape[1]) * .01
    policy = CompactPolicy(w)
    forward = policy.predict(r["state"], r["criteria"])
    reverse = policy.predict(r["state"], dict(reversed(list(r["criteria"].items()))))
    assert np.allclose(forward, reverse[::-1])
    assert math.isclose(float(forward.sum()), 1.)


def test_real_supervised_training_and_checkpoint_reload(tmp_path):
    dataset = dataset_fixture(tmp_path)
    store = Store(tmp_path / "artifacts")
    run = train_compact(store, dataset, steps=30)
    summary = store.runs()[0]["summary"]
    assert summary["final_dev"]["loss"] < summary["initial_dev"]["loss"]
    restored = CompactPolicy.load(store.run_dir(run) / "checkpoint")
    r = sample()["records"][0]
    assert restored.predict(r["state"], r["criteria"])[0] > .9
    assert any(e["kind"] == "metric" for e in store.events(run))


def test_nonfinite_metrics_stop_before_json_or_update(tmp_path):
    store = Store(tmp_path)
    run = store.create("train", "test", {})
    monitor = Monitor(store, run)
    with pytest.raises(FloatingPointError):
        monitor.metric(1, loss=float("nan"))
    assert store.events(run)[-1]["payload"]["severity"] == "critical"


def test_gradient_alert(tmp_path):
    store = Store(tmp_path)
    run = store.create("train", "test", {})
    Monitor(store, run).metric(1, loss=1., grad_norm=101.)
    assert store.events(run)[-2]["payload"]["code"] == "gradient_spike"


def paired_rows(success=True):
    return [{"group": f"{task}:{seed}:0", "task": task, "success": success,
             "policy_version": POLICY_VERSION, "collisions": 0, "drops": 0, "rejections": 0,
             "steps": 8, "latency_p95_ms": 10} for task in ("transfer", "stack", "barrier")
            for seed in range(30)]


def test_pairing_and_risk_gates():
    a, b = paired_rows(), paired_rows(False)
    assert paired_report(a, b)["gate"]["passed"]
    a[0]["collisions"] = 1
    assert not paired_report(a, b)["gate"]["passed"]
    with pytest.raises(ValueError, match="matched"):
        paired_report(a, b[:-1])


def test_promotion_requires_same_evaluated_weights(tmp_path):
    store = Store(tmp_path / "artifacts")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "weights").write_text("original")
    report = {"gate": {"passed": True}, "candidate_checkpoint": str(ckpt),
              "candidate_sha256": fingerprint(ckpt)}
    path = tmp_path / "report.json"
    write_json(path, report)
    (ckpt / "weights").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        promote(store, path)


def test_dashboard_reads_persistent_metrics_and_blocks_rebinding(tmp_path):
    store = Store(tmp_path)
    run = store.create("train", "test", {})
    Monitor(store, run).metric(1, loss=.5)
    with TestClient(create_app(store)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/overview").json()["runs"][0]["id"] == run
        assert client.get(f"/api/runs/{run}/events").json()["events"][-1]["payload"]["loss"] == .5
        assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 403
        assert client.post("/api/jobs", json={"action": "collect"},
                           headers={"Origin": "https://attacker.example"}).status_code == 403
        assert client.post("/api/jobs", json={"action": "train", "dataset": "../../etc"}).status_code == 400


@pytest.mark.parametrize("task", ["transfer", "stack", "barrier"])
def test_real_mujoco_teacher_episode(task):
    pytest.importorskip("embodied_jev")
    from embodied_rsi.sim import episode
    result = episode(task, 0)
    assert result["success"]
    assert result["collisions"] == 0
    for r in result["records"]:
        assert "branch_scores" not in r["state"]
        assert "success" not in r["state"]["observation"]
        assert r["choice"] in r["criteria"]
