import copy
import json

import pytest
from test_core import paired_rows
from test_success_replay import model_episode

from embodied_rsi.evaluation import paired_report
from embodied_rsi.exploration import ExplorationMemory
from embodied_rsi.self_improvement import fit_successes, self_improve, success_dataset
from embodied_rsi.storage import Store, dumps, write_json
from embodied_rsi.task_training import CompactLearner


def test_historical_collection_survives_dashboard_history_limit(tmp_path, monkeypatch):
    store = Store(tmp_path)
    old = store.create("explore", "compact", {})
    source = store.run_dir(old) / "episodes.jsonl"
    ep = model_episode()
    row = ep["records"][0]
    row["before"] = row["after"] = row["state"]["observation"]
    source.write_text(dumps(ep) + "\n")
    for _ in range(201):
        store.create("simulation", "compact", {})
    assert old not in {r["id"] for r in store.runs()}
    memory = ExplorationMemory(store, "compact")
    memory.import_past()
    assert memory.overview()["trials"] == 1
    seen = []

    def dataset(files, output, tasks):
        seen.extend(files)
        return {"successful_episodes": 0, "cases": {"train": 0}}

    monkeypatch.setattr("embodied_rsi.self_improvement.success_dataset", dataset)
    self_improve(store, rounds=1, backend="compact", replay_only=True, episodes=90, steps=1)
    assert source in seen


def test_repeated_episode_files_do_not_inflate_replay_counts(tmp_path):
    source = tmp_path / "episodes.jsonl"
    source.write_text(dumps(model_episode()) + "\n")
    manifest = success_dataset([source, source], tmp_path / "dataset")
    assert manifest["episodes_total"] == manifest["successful_episodes"] == 1
    assert manifest["cases"]["train"] == 1


def test_development_scenes_cannot_be_replayed_training_scenes(tmp_path):
    ep = model_episode()
    source = tmp_path / "episodes.jsonl"
    source.write_text(dumps(ep) + "\n")
    dataset = tmp_path / "dataset"
    success_dataset([source], dataset)
    with pytest.raises(ValueError, match="Development scenes overlap"):
        fit_successes(Store(tmp_path / "store"), CompactLearner(), dataset, steps=1,
                      dev_start=ep["seed"], dev_episodes=1, backend="compact", tasks=["transfer"])


@pytest.mark.parametrize("field,value", [("scene_hash", "different-scene"),
    ("control_protocol", "different-controller"), ("success_definition", "strict_grasp_v1"),
    ("episode_protocol", {"staged_prompts": True, "strict_grasp": True})])
def test_paired_scores_reject_different_scenes_or_input_contracts(field, value):
    candidate, parent = paired_rows(), paired_rows(False)
    candidate[0][field] = value
    with pytest.raises(ValueError, match=field + " mismatch"):
        paired_report(candidate, parent)


def test_self_improvement_evaluates_the_saved_model_and_keeps_failed_release(tmp_path, monkeypatch):
    store = Store(tmp_path)
    collection = store.create("explore", "compact", {})
    ep = model_episode()
    row = ep["records"][0]
    row["before"] = row["state"]["observation"]
    row["after"] = {**row["before"], "held": True}
    (store.run_dir(collection) / "episodes.jsonl").write_text(dumps(ep) + "\n")
    loaded = object()
    checked = []

    def fit(store, learner, dataset, *args, **kwargs):
        training = store.create("self-train", "compact", {})
        write_json(store.run_dir(training) / "checkpoint" / "meta.json", {"backend": "compact-numpy"})
        return training

    def evaluate(store, candidate, path, old, old_name, tasks, seeds, **kwargs):
        assert candidate is loaded and kwargs["candidate_reloaded"]
        checked.append(str(path))
        evaluation = store.create("evaluate", "mujoco", {})
        write_json(store.run_dir(evaluation) / "report.json", {
            "success_rate": 0., "champion_success_rate": 0., "gate": {"passed": False}})
        return evaluation

    monkeypatch.setattr("embodied_rsi.self_improvement.fit_successes", fit)
    monkeypatch.setattr("embodied_rsi.self_improvement.load_policy", lambda path: loaded)
    monkeypatch.setattr("embodied_rsi.self_improvement.evaluate", evaluate)
    results = self_improve(store, rounds=1, backend="compact", replay_only=True, episodes=90, steps=1)
    assert len(checked) == 1
    assert results[0]["status"] == "retained_current_model"
    assert not (store.root / "champion.json").exists()
    collection = next(r for r in store.runs() if r["config"].get("mode") == "existing_model_replay")
    assert collection["summary"]["success_rate"] is None


def test_compact_reload_keeps_saved_episode_protocol(tmp_path):
    import numpy as np
    from test_core import sample

    from embodied_rsi.compact import CompactPolicy, features
    row = sample()["records"][0]
    policy = CompactPolicy(np.zeros(features(row["state"], row["criteria"]).shape[1]))
    policy.save(tmp_path)
    protocol = {"staged_prompts": True, "strict_grasp": True}
    write_json(tmp_path / "meta.json", {"episode_protocol": protocol})
    assert CompactLearner(tmp_path).episode_protocol == protocol
    assert copy.deepcopy(CompactPolicy.load(tmp_path)).episode_protocol == protocol


def test_missing_legacy_dataset_does_not_silently_change_model_inputs(tmp_path):
    from embodied_rsi.episode_protocol import checkpoint_protocol
    write_json(tmp_path / "meta.json", {"dataset": str(tmp_path / "missing")})
    with pytest.raises(ValueError, match="input protocol is missing"):
        checkpoint_protocol(tmp_path)
    assert json.loads((tmp_path / "meta.json").read_text())["dataset"].endswith("missing")
