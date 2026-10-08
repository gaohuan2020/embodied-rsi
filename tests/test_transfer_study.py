import json

import pytest
from test_core import paired_rows

from embodied_rsi.evaluation import paired_report
from embodied_rsi.self_improvement import success_dataset
from embodied_rsi.storage import Store, dumps, write_json
from embodied_rsi.transfer_study import transfer_study


def test_transfer_only_release_requires_ninety_matched_scenes_and_scoped_coverage():
    candidate, old = [], []
    template = next(r for r in paired_rows() if r["task"] == "transfer")
    for seed in range(90):
        row = {**template, "group": f"transfer:{seed}:0"}
        candidate.append({**row, "success": True})
        old.append({**row, "success": False})
    report = paired_report(candidate, old, expected_tasks=["transfer"])
    assert report["gate"]["passed"] and report["gate"]["thresholds"]["tasks"] == ["transfer"]
    assert not paired_report(candidate[:30], old[:30], expected_tasks=["transfer"])["gate"]["passed"]
    assert not paired_report(candidate, old)["gate"]["passed"]  # Cannot claim three-task coverage.


def test_transfer_dataset_never_imports_other_task_successes(tmp_path):
    from test_success_replay import model_episode
    first = model_episode()
    other = {**model_episode(), "task": "stack", "group": "stack:42:0", "episode_id": "other"}
    source = tmp_path / "episodes.jsonl"
    source.write_text(dumps(first)+"\n"+dumps(other)+"\n")
    manifest = success_dataset([source], tmp_path / "dataset", ["transfer"])
    assert manifest["episodes_total"] == manifest["successful_episodes"] == 1
    assert set(manifest["groups"]) == {"transfer:42:0"}


def test_audit_is_delayed_and_regression_or_plateaus_are_not_hidden(tmp_path, monkeypatch):
    store = Store(tmp_path)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    write_json(baseline / "meta.json", {"backend": "compact-numpy"})
    phase = {"training_complete": False}

    def learning(_store, **kwargs):
        assert kwargs["tasks"] == ["transfer"] and not kwargs["randomize_scene"]
        results = []
        for index, status in enumerate(("deployed", "retained_current_model", "deployed"), 1):
            run = store.create("self-train", "compact", {})
            path = store.run_dir(run) / "checkpoint"
            write_json(path / "meta.json", {"index": index})
            results.append({"round": index, "training": run, "status": status})
        phase["training_complete"] = True
        return results

    def load(path):
        return json.loads((type(baseline)(path) / "meta.json").read_text()).get("index", 0)

    def rollout(task, seed, policy, **kwargs):
        assert phase["training_complete"] and task == "transfer"
        base = next(r for r in paired_rows() if r["task"] == "transfer")
        # Candidate V2 regresses; accepted versions plateau while V2 is rejected.
        return {**base, "group": f"transfer:{seed}:0", "success": seed % 10 < (0, 6, 4, 8)[policy]}

    monkeypatch.setattr("embodied_rsi.transfer_study.self_improve", learning)
    monkeypatch.setattr("embodied_rsi.transfer_study.load_policy", load)
    monkeypatch.setattr("embodied_rsi.transfer_study.episode", rollout)
    report = transfer_study(store, rounds=3, explore_episodes=2, steps=1,
                            audit_episodes=30, backend="compact", checkpoint=baseline)
    assert [p["success_rate"] for p in report["audit"]] == [0., .6, .4, .8]
    assert report["deployed_audit_rates"] == [0., .6, .6, .8]
    assert not report["candidate_strictly_increasing"] and not report["deployed_strictly_increasing"]
    assert report["updated_models"] == 2 and not report["audit_used_for_release"]
    assert report["final_vs_baseline"]["gain_ci95"][0] > 0
    assert store.runs()[0]["kind"] == "self-train" or any(r["kind"] == "study" for r in store.runs())


def test_study_rejects_a_preexisting_deployment(tmp_path):
    store = Store(tmp_path)
    write_json(store.root / "champion-rsi.json", {"version": "old"})
    with pytest.raises(ValueError, match="clean baseline"):
        transfer_study(store)


def test_study_cannot_reuse_historical_training_without_a_deployment(tmp_path):
    store = Store(tmp_path)
    store.create("self-train", "rsi", {})
    with pytest.raises(ValueError, match="clean baseline"):
        transfer_study(store)
