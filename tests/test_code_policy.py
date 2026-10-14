import copy
import json

import numpy as np
import pytest

from embodied_rsi.code_policy import CodePolicy, HybridPolicy
from embodied_rsi.code_study import check_holdout, run_code_study, world_factory
from embodied_rsi.self_improvement import success_dataset
from embodied_rsi.sim import PHASES, episode
from embodied_rsi.storage import Store, dumps, write_json


def state(**overrides):
    o = {"tcp": [.38, -.08, .22], "object": [.43, -.17, .02],
         "destination": [.43, .18, .026], "held": False, "gripper": "open",
         "grasp_secured": False, "support_contact": False, **overrides}
    return {"task_id": "transfer", "observation": o}


def choose(policy, observation, keys=PHASES):
    criteria = {key: key for key in keys}
    return list(criteria)[int(np.argmax(policy.predict(observation, criteria)))]


def test_rules_use_measured_effects_and_respect_candidate_order():
    policy = CodePolicy()
    assert choose(policy, state(), tuple(reversed(PHASES))) == "approach"
    assert choose(policy, state(tcp=[.43, -.17, .16])) == "descend"
    assert choose(policy, state(tcp=[.43, -.17, .021])) == "grasp"
    assert choose(policy, state(tcp=[.43, -.17, .021], held=True, gripper="closed",
                                grasp_secured=True)) == "lift"
    assert choose(policy, state(tcp=[.43, -.17, .22], object=[.43, -.17, .219],
                                held=True, gripper="closed", grasp_secured=True)) == "carry"
    assert choose(policy, state(tcp=[.43, .18, .22], object=[.43, .18, .219],
                                held=True, gripper="closed", grasp_secured=True)) == "lower"
    assert choose(policy, state(tcp=[.43, .18, .03], object=[.43, .18, .026],
                                held=True, gripper="closed", grasp_secured=True)) == "release"
    assert choose(policy, state(tcp=[.43, .18, .03], object=[.43, .18, .026],
                                support_contact=True)) == "withdraw"
    assert policy.last_decision["model_calls"] == 0


def test_drop_invalidates_lift_memory_and_recovery_is_bounded():
    policy = CodePolicy(1)
    before = state(held=True, gripper="closed", grasp_secured=True,
                   object=[.43, -.17, .219])["observation"]
    policy.predict({"task_id": "transfer", "observation": before}, dict.fromkeys(PHASES))
    after = state(gripper="closed")["observation"]
    feedback = policy.observe_outcome({"choice": "carry", "before": before, "after": after,
                                       "executed": True, "rejection": None})
    assert feedback["failure_reason"] == "lost_grasp"
    assert not policy.stages.acquired and choose(policy, state(gripper="closed")) == "recover"
    outcome = {"choice": "grasp", "before": after, "after": after, "executed": True, "rejection": None}
    policy.observe_outcome(outcome)
    assert policy.stop_reason == "recovery_budget_exhausted"
    policy.reset()
    assert policy.retry_count == 0 and policy.stop_reason is None and policy.recovery_phase is None


def test_hybrid_masks_illegal_actions_without_changing_model_input():
    class Model:
        def __init__(self):
            self.calls = []

        def predict(self, state, criteria):
            self.calls.append((copy.deepcopy(state), list(criteria)))
            return np.asarray([.9 if key == "release" else .1 if key == "grasp" else 0. for key in criteria])

    model = Model()
    policy = HybridPolicy(model)
    assert choose(policy, state()) == "approach" and not model.calls
    grasp = state(tcp=[.43, -.17, .021])
    assert choose(policy, grasp, tuple(reversed(PHASES))) == "grasp"
    assert model.calls == [(grasp, list(reversed(PHASES)))]
    assert policy.last_decision["model_calls"] == 1
    bad = state(gripper="closed")["observation"]
    policy.observe_outcome({"choice": "grasp", "before": bad, "after": bad,
                            "executed": True, "rejection": None})
    assert choose(policy, state(gripper="closed")) == "recover" and len(model.calls) == 1


def test_missing_skill_and_unsupported_task_fail_explicitly():
    with pytest.raises(ValueError, match="unavailable"):
        choose(CodePolicy(), state(), ("grasp",))
    with pytest.raises(ValueError, match="transfer only"):
        choose(CodePolicy(), {**state(), "task_id": "stack"})


def test_rejected_carry_preserves_grasp_instead_of_opening_fingers():
    policy = CodePolicy()
    choose(policy, state())
    o = state(held=True, gripper="closed", grasp_secured=True,
              object=[.43, -.17, .219])["observation"]
    choose(policy, {"task_id": "transfer", "observation": o})
    assert policy.stages.acquired
    policy.observe_outcome({"choice": "carry", "before": o, "after": o,
                            "executed": False, "rejection": "collision_preview"})
    assert choose(policy, {"task_id": "transfer", "observation": o}) == "lift"
    assert policy.stages.acquired


def test_real_mujoco_recovers_from_fault_without_teacher_or_success_input(monkeypatch):
    pytest.importorskip("embodied_jev")
    from embodied_rsi.grasp import grasp_state

    def forbidden(*args):
        raise AssertionError("Code policy must not request teacher labels")

    monkeypatch.setattr("embodied_rsi.sim.teacher", forbidden)
    class AuditedPolicy(CodePolicy):
        def observe_outcome(self, outcome):
            for o in (outcome["before"], outcome["after"]):
                assert not {"success", "seed", "stable_seconds", "max_lift_m", "audit"} & o.keys()
            return super().observe_outcome(outcome)

    policy = AuditedPolicy()
    for seed in (123, 124):
        result = episode("transfer", seed, policy, label_mode="none", randomize_scene=True,
                         world_factory=world_factory("missed-grasp"), state_transform=grasp_state,
                         staged_prompts=True)
        assert result["success"], result["failure"]
        assert result["stages"]["lift"] and result["stages"]["release"] and not result["stages"]["drop"]
        assert len(result["interventions"]) == 1  # Shadow previews must not consume/duplicate the fault.
        assert policy.retry_count == 1  # Episode-local memory resets when this same instance is reused.
        assert result["source_policy"] == "code"
        assert all(row["gold"] is None for row in result["records"])
        assert any(row["policy_feedback"]["failure_reason"] == "empty_grasp" for row in result["records"])
    failed = episode("transfer", 123, CodePolicy(0), label_mode="none", randomize_scene=True,
                     world_factory=world_factory("missed-grasp"), state_transform=grasp_state,
                     staged_prompts=True)
    assert not failed["success"] and failed["failure"] == "recovery_budget_exhausted"


@pytest.mark.parametrize("source", ["code", "code_no_recovery", "hybrid", "rsi"])
def test_code_and_evaluation_trajectories_cannot_become_model_self_success(tmp_path, source):
    ep = {"source_policy": source, "success": True, "task": "transfer", "data_role": "evaluation_only"}
    file = tmp_path / "episodes.jsonl"
    file.write_text(dumps(ep) + "\n")
    with pytest.raises(ValueError, match="Evaluation-only"):
        success_dataset([file], tmp_path / "dataset")
    if source != "rsi":
        del ep["data_role"]
        file.write_text(dumps(ep) + "\n")
        with pytest.raises(ValueError, match="code/hybrid"):
            success_dataset([file], tmp_path / "dataset2")


def test_paired_study_freezes_protocol_and_separates_fault_results(tmp_path, monkeypatch):
    pytest.importorskip("embodied_jev")
    # Explicit study configuration must override ambient flags.
    monkeypatch.setenv("EMBODIED_RSI_STAGED_PROMPTS", "0")
    store = Store(tmp_path)
    report = run_code_study(store, episodes=2)
    assert not report["deployment_supported"] and not (store.root / "champion.json").exists()
    assert report["protocol"]["staged_prompts"]
    normal, fault = report["policies"]["normal"], report["policies"]["missed-grasp"]
    assert normal["code"]["successes"] == normal["code_no_recovery"]["successes"] == 2
    assert fault["code"]["successful_after_fault"] == 2
    assert fault["code_no_recovery"]["successes"] == 0
    assert fault["code"]["mean_model_calls"] == 0
    assert not report["paired_comparisons"]["missed-grasp"]["code_vs_code_no_recovery"]["gate"]["passed"]
    protocol = json.loads((store.run_dir(report["run"]) / "protocol.json").read_text())
    assert protocol == report["protocol"]
    first_seeds = set(protocol["seeds"])
    second = run_code_study(store, episodes=1, profiles=["normal"])
    assert not first_seeds & set(second["protocol"]["seeds"])


def test_study_rejects_training_seed_leakage_and_code_changes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    write_json(checkpoint / "meta.json", {"train_seed_range": [20020000, 20020002]})
    with pytest.raises(ValueError, match="overlap"):
        run_code_study(Store(tmp_path / "store"), checkpoint, episodes=2)
    with pytest.raises(ValueError, match="overlap"):
        check_holdout(checkpoint, [20020000])
    pytest.importorskip("embodied_jev")
    hashes = iter([{"code": "before"}, {"code": "after"}])
    monkeypatch.setattr("embodied_rsi.code_study.source_hashes", lambda: next(hashes))
    store = Store(tmp_path / "changed")
    with pytest.raises(ValueError, match="changed"):
        run_code_study(store, episodes=1, profiles=["normal"])
    assert store.runs()[0]["status"] == "failed"
    assert not (store.root / "code-study-latest.json").exists()


def test_compact_checkpoint_cannot_be_mislabeled_as_rsi(tmp_path):
    write_json(tmp_path / "meta.json", {"backend": "compact-numpy"})
    with pytest.raises(ValueError, match="RSI-Jev checkpoint"):
        check_holdout(tmp_path, [20000000])
