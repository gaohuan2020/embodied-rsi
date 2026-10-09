import json

import numpy as np
import pytest

from embodied_rsi.parameter_history import ParameterRecorder, parameter_history
from embodied_rsi.sim import episode
from embodied_rsi.storage import Store, dumps
from embodied_rsi.trajectory_replay import TrajectoryReplay


def test_parameter_snapshots_measure_optimizer_changes_and_checkpoint_restore(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.tower = torch.nn.Linear(2, 2)
    model.scorer = torch.nn.Linear(2, 1)
    for p in model.tower.parameters():
        p.requires_grad_(False)
    store = Store(tmp_path)
    run = store.create("self-train", "rsi", {})
    recorder = ParameterRecorder(model, store.run_dir(run))
    before = {n: p.clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.scorer.parameters(), lr=.01)
    model.scorer(model.tower(torch.ones(2))).sum().backward()
    optimizer.step()
    update = recorder.capture(1)
    for tensor in update["tensors"]:
        difference = dict(model.named_parameters())[tensor["name"]] - before[tensor["name"]]
        assert tensor["delta_norm"] == pytest.approx(float(difference.norm().detach()))
    assert update["frozen_changed"] == []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(before[n])
    restored = recorder.capture(0, "selected_checkpoint")
    assert all(t["total_delta_norm"] == 0 for t in restored["tensors"])
    history = parameter_history(store, run, after=0)
    assert [s["index"] for s in history["snapshots"]] == [1, 2]
    assert history["snapshots"][-1]["phase"] == "selected_checkpoint"


def test_archived_poses_match_collection_and_render_both_views_without_a_policy(tmp_path):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path)
    run = store.create("explore", "teacher", {})
    directory = store.run_dir(run)
    original = episode("transfer", 42, max_steps=2, label_mode="none")
    captured = episode("transfer", 42, max_steps=2, label_mode="none", replay_directory=directory / "replays")
    assert original["success"] == captured["success"]
    for a, b in zip(original["records"], captured["records"]):
        assert a["choice"] == b["choice"]
        assert a["after"]["tcp"] == b["after"]["tcp"]
        assert a["after"]["object"] == b["after"]["object"]
    (directory / "episodes.jsonl").write_text(dumps(captured) + "\n")
    replay = TrajectoryReplay(store)
    try:
        metadata = replay.metadata(run, captured["episode_id"])
        assert metadata["source"] == "captured_poses" and metadata["frame_count"] > 2
        first = replay.frame(run, captured["episode_id"], 0)
        last = replay.frame(run, captured["episode_id"], metadata["frame_count"] - 1)
        wrist = replay.frame(run, captured["episode_id"], 0, "wrist")
        assert first.startswith(b"\xff\xd8") and last != first and wrist != first
        assert replay.catalogue(run, "failure")["total"] == 1
        with pytest.raises(ValueError, match="Invalid replay"):
            replay.frame(run, captured["episode_id"], -1)
    finally:
        replay.close()


def test_live_decision_is_published_before_motion_and_does_not_change_the_trajectory(monkeypatch):
    pytest.importorskip("embodied_jev")
    from embodied_rsi import sim

    original = episode("transfer", 42, max_steps=2, label_mode="none")
    events = []
    execute = sim.execute

    def motion(world, candidate, on_frame=None):
        assert events[-1][0] == "selected"
        events.append(("motion", None))
        return execute(world, candidate, on_frame)

    monkeypatch.setattr(sim, "execute", motion)
    observed = episode("transfer", 42, max_steps=2, label_mode="none",
                       on_decision=lambda world, state, criteria, step: events.append(("inference", step)),
                       on_action=lambda row: events.append(("selected", row["choice"])),
                       on_step=lambda row: events.append(("finished", row["choice"])))
    assert [event[0] for event in events] == ["inference", "selected", "motion", "finished"] * 2
    for before, after in zip(original["records"], observed["records"]):
        assert before["choice"] == after["choice"]
        assert before["after"] == after["after"]


def test_legacy_replay_is_verified_and_mismatched_actions_are_rejected(tmp_path):
    pytest.importorskip("embodied_jev")
    store = Store(tmp_path)
    run = store.create("explore", "rsi", {})
    ep = episode("transfer", 13, max_steps=1, label_mode="none")
    path = store.run_dir(run) / "episodes.jsonl"
    path.write_text(dumps(ep) + "\n")
    replay = TrajectoryReplay(store)
    try:
        assert replay.metadata(run, ep["episode_id"])["source"] == "verified_action_reconstruction"
        bad = json.loads(dumps(ep))
        bad["episode_id"] = "bad-trajectory"
        bad["records"][0]["after"]["object"] = np.array([10., 10., 10.]).tolist()
        with path.open("a") as file:
            file.write(dumps(bad) + "\n")
        with pytest.raises(ValueError, match="位置校验"):
            replay.metadata(run, bad["episode_id"])
        assert not (store.run_dir(run) / "replays" / "bad-trajectory.npz").exists()
    finally:
        replay.close()
