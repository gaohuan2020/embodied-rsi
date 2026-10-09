import json

import pytest

from embodied_rsi.stage_prompts import STAGES, PickPlaceStages, decision_input


def observation(**values):
    return {"tcp": [.43, -.17, .22], "object": [.43, -.17, .02],
            "destination": [.43, .18, .026], "held": False, "gripper": "open",
            "grasp_secured": False, "support_contact": False, **values}


def test_grasp_and_lift_remain_model_decisions_inside_acquisition():
    machine = PickPlaceStages()
    o = observation(tcp=[.43, -.17, .022], gripper="closed")
    assert machine.update(o)["stage"] == "acquire"
    o.update(held=True, grasp_secured=False)
    assert machine.update(o)["stage"] == "acquire"
    o["grasp_secured"] = True
    assert machine.update(o)["stage"] == "acquire"
    assert {"approach", "descend", "grasp", "lift", "recover"} == set(STAGES["acquire"][1])
    o.update(tcp=[.43, -.17, .1], object=[.43, -.17, .098])
    assert machine.update(o)["stage"] == "transport"


def test_lower_release_withdraw_share_one_stage_and_drop_resets_acquisition():
    machine = PickPlaceStages()
    machine.update(observation())
    o = observation(tcp=[.43, .18, .22], object=[.43, .18, .217],
                    held=True, gripper="closed", grasp_secured=True)
    assert machine.update(o)["stage"] == "place"
    o.update(tcp=[.43, .18, .03], object=[.43, .18, .026])
    assert machine.update(o)["stage"] == "place"
    o.update(held=False, gripper="open", support_contact=True)
    assert machine.update(o)["stage"] == "place"
    assert {"lower", "release", "withdraw"} <= set(STAGES["place"][1])
    # A drop away from the destination cannot inherit a completed placement.
    o.update(object=[.43, -.17, .02], support_contact=False)
    assert machine.update(o)["stage"] == "acquire"


def test_recorded_and_live_states_encode_identical_stage_prompts():
    o = observation(tcp=[.39, -.08, .22])
    state = {"observation": o, "control": PickPlaceStages().update(o)}
    criteria = {"approach": json.dumps({"target_tcp": [.43, -.17, .16],
                "gripper_command": "open", "duration_seconds": .8})}
    live = decision_input(state, criteria, "fallback")
    recorded = decision_input(json.dumps(state), criteria, "fallback")
    assert live == recorded
    assert "Obtain a stable grasp" in live[1]
    assert all(word not in live[0] for word in ["seed", "episode_success", "branch_scores"])


def test_complete_staged_mujoco_episode_uses_contact_based_transitions(monkeypatch):
    pytest.importorskip("embodied_jev")
    from embodied_rsi.sim import episode
    monkeypatch.setenv("EMBODIED_RSI_STAGED_PROMPTS", "1")
    monkeypatch.setenv("EMBODIED_RSI_STRICT_GRASP", "1")
    result = episode("transfer", 123, randomize_scene=True)
    assert result["success"] and result["stages"]["lift"] and result["stages"]["release"]
    assert [r["state"]["control"]["stage"] for r in result["records"]] == [
        "acquire", "acquire", "acquire", "acquire", "transport", "place", "place", "place"]
    assert all(len(r["criteria"]) >= 5 for r in result["records"])


def test_failure_feedback_uses_task_id_and_keeps_stage_contexts_separate(tmp_path):
    from embodied_rsi.exploration import ExplorationMemory, context
    from embodied_rsi.storage import Store
    o = observation(task="把方块放入托盘", gripper="closed")
    state = {"task_id": "transfer", "observation": o, "control": PickPlaceStages().update(o)}
    row = {"choice": "grasp", "state": state, "before": o, "after": o, "executed": True}
    memory = ExplorationMemory(Store(tmp_path))
    memory.observe({"episode_id": "failed-grasp", "task": "transfer", "success": False, "records": [row]})
    assert memory.stats(state, ["grasp"])[0]["failures"] == 1
    placed = {**state, "control": {**state["control"], "stage": "place"}}
    assert context(state) != context(placed)
    assert memory.stats(placed, ["grasp"])[0]["failures"] == 0


def test_premature_release_is_failure_feedback_but_supported_release_is_not(tmp_path):
    from embodied_rsi.exploration import ExplorationMemory
    from embodied_rsi.storage import Store
    memory = ExplorationMemory(Store(tmp_path))
    for index, height in enumerate([.22, .026]):
        before = observation(object=[.43, .18, height], held=True, gripper="closed")
        after = {**before, "held": False, "gripper": "open"}
        state = {"task_id": "transfer", "observation": before}
        row = {"choice": "release", "state": state, "before": before, "after": after, "executed": True}
        memory.observe({"episode_id": str(index), "task": "transfer", "success": index == 1, "records": [row]})
        assert memory.stats(state, ["release"])[0]["failures"] == (1 if index == 0 else 0)
