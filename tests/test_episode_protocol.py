import json

import numpy as np
import pytest

from embodied_rsi.episode_protocol import (
    checkpoint_protocol,
    episode_protocol,
    replay_protocol,
    rsi_default_protocol,
)
from embodied_rsi.exploration import ExploringPolicy
from embodied_rsi.self_improvement import FrozenParentPolicy
from embodied_rsi.stage_prompts import PROTOCOL, STAGES
from embodied_rsi.storage import dumps, write_json


def staged_case():
    return {"state": dumps({"task_id": "transfer", "control": {"protocol": PROTOCOL,
                            "stage": "acquire", "goal": STAGES["acquire"][0]},
                            "environment": {"required_stable_grasp_s": .16}})}


def test_legacy_checkpoint_recovers_training_inputs_without_environment_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBODIED_RSI_STAGED_PROMPTS", raising=False)
    monkeypatch.delenv("EMBODIED_RSI_STRICT_GRASP", raising=False)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text(dumps(staged_case()) + "\n")
    checkpoint = tmp_path / "model"
    write_json(checkpoint / "meta.json", {"robot_training": {"dataset": str(dataset)}})
    original = (checkpoint / "meta.json").read_bytes()
    assert checkpoint_protocol(checkpoint) == {"staged_prompts": True, "strict_grasp": True}
    assert (checkpoint / "meta.json").read_bytes() == original


def test_saved_protocol_works_without_dataset_and_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIED_RSI_STAGED_PROMPTS", "1")
    monkeypatch.setenv("EMBODIED_RSI_STRICT_GRASP", "1")
    protocol = {"staged_prompts": False, "strict_grasp": False}
    write_json(tmp_path / "meta.json", {"episode_protocol": protocol, "dataset": "/missing"})
    assert checkpoint_protocol(tmp_path) == protocol
    assert checkpoint_protocol(tmp_path, rsi_default_protocol()) == protocol


def test_new_rsi_runs_enable_measured_contract_without_shell_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBODIED_RSI_STAGED_PROMPTS", raising=False)
    monkeypatch.delenv("EMBODIED_RSI_STRICT_GRASP", raising=False)
    assert checkpoint_protocol(tmp_path, rsi_default_protocol()) == {
        "staged_prompts": True, "strict_grasp": True}
    # Explicit legacy experiments and saved checkpoints retain their settings.
    monkeypatch.setenv("EMBODIED_RSI_STAGED_PROMPTS", "0")
    assert not rsi_default_protocol()["staged_prompts"]


def test_replay_rejects_mixed_input_contracts():
    with pytest.raises(ValueError, match="incompatible"):
        replay_protocol([staged_case(), {"state": dumps({"task_id": "transfer"})}])


def test_policy_wrappers_keep_protocol_pinned_when_environment_changes(monkeypatch):
    class Model:
        def __init__(self):
            self.episode_protocol = {"staged_prompts": True, "strict_grasp": True}

    model = Model()
    exploring = ExploringPolicy(model, None)
    parent = FrozenParentPolicy(model, None)
    monkeypatch.setenv("EMBODIED_RSI_STAGED_PROMPTS", "0")
    monkeypatch.setenv("EMBODIED_RSI_STRICT_GRASP", "0")
    assert episode_protocol(exploring) == episode_protocol(parent) == model.episode_protocol
    model.episode_protocol = {"staged_prompts": False, "strict_grasp": False}
    assert episode_protocol(parent)["staged_prompts"]  # Frozen baseline keeps its original contract.


def test_episode_uses_loaded_protocol_without_environment_flags(monkeypatch):
    pytest.importorskip("embodied_jev")
    from embodied_rsi.sim import episode

    monkeypatch.delenv("EMBODIED_RSI_STAGED_PROMPTS", raising=False)
    monkeypatch.delenv("EMBODIED_RSI_STRICT_GRASP", raising=False)

    class Policy:
        def __init__(self):
            self.episode_protocol = {"staged_prompts": True, "strict_grasp": True}

        def predict(self, state, criteria):
            assert state["control"]["protocol"] == PROTOCOL
            assert "required_stable_grasp_s" in state["environment"]
            return np.full(len(criteria), 1 / len(criteria))

    result = episode("transfer", 123, ExploringPolicy(Policy(), FakeMemory()), max_steps=1,
                     label_mode="none", randomize_scene=True)
    assert result["control_protocol"] == PROTOCOL
    assert result["success_definition"] == "strict_grasp_v1"


def test_explicit_physical_world_is_recorded_even_when_policy_defaults_are_legacy(monkeypatch):
    pytest.importorskip("embodied_jev")
    from embodied_rsi.grasp import GraspWorld, grasp_state
    from embodied_rsi.sim import episode
    monkeypatch.delenv("EMBODIED_RSI_STRICT_GRASP", raising=False)
    result = episode("transfer", 123, max_steps=1, world_factory=GraspWorld,
                     state_transform=grasp_state, staged_prompts=True)
    assert result["episode_protocol"] == {"staged_prompts": True, "strict_grasp": True}


class FakeMemory:
    def stats(self, state, keys):
        return [{"trials": 0, "failures": 0} for key in keys]


def test_success_training_saves_and_validates_with_recorded_protocol(tmp_path, monkeypatch):
    from test_success_replay import model_episode

    from embodied_rsi.self_improvement import fit_successes, success_dataset
    from embodied_rsi.storage import Store
    from embodied_rsi.task_training import CompactLearner

    ep = model_episode()
    ep["records"][0]["state"].update(json.loads(staged_case()["state"]))
    source = tmp_path / "episodes.jsonl"
    source.write_text(dumps(ep) + "\n")
    dataset = tmp_path / "dataset"
    success_dataset([source], dataset)
    checked = []

    def rollout(task, seed, policy, **kwargs):
        checked.append(episode_protocol(policy))
        return {"success": False, "collisions": 0, "drops": 0, "rejections": 0}

    monkeypatch.setattr("embodied_rsi.self_improvement.episode", rollout)
    store = Store(tmp_path / "store")
    model = CompactLearner()
    model.predict(ep["records"][0]["state"], ep["records"][0]["criteria"])
    run = fit_successes(store, model, dataset, steps=1, dev_episodes=1, backend="compact", tasks=["transfer"])
    meta = json.loads((store.run_dir(run) / "checkpoint" / "meta.json").read_text())
    expected = {"staged_prompts": True, "strict_grasp": True}
    assert checked == [expected, expected]
    assert meta["episode_protocol"] == expected
    summary = json.loads((store.run_dir(run) / "summary.json").read_text())
    assert summary == next(r["summary"] for r in store.runs() if r["id"] == run)
