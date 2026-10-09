import csv
import json

from embodied_rsi.round_metrics import RoundRecorder, round_history
from embodied_rsi.storage import Store


def test_collection_uses_pinned_version_across_restarts_and_only_completed_rsi_episodes(tmp_path):
    store = Store(tmp_path)
    deploy = store.create("deploy", "inference", {})
    store.finish(deploy, summary={"backend": "rsi", "version": deploy, "checkpoint": "model-1"})
    first = store.create("explore", "rsi", {"continuous": True, "explore": True})
    second = store.create("explore", "rsi", {"continuous": True, "explore": True})
    teacher = store.create("explore", "teacher", {"continuous": True})
    pilot = store.create("explore", "rsi", {"continuous": False})

    def ep(run, key, version, success, source="rsi"):
        store.event(run, "episode", {"episode_id": key, "model_version": version,
                                    "success": success, "source_policy": source, "seed": 1})

    ep(first, "a", "RSI-Jev v1.0-0.8b / revision", False)
    ep(first, "b", deploy, True)  # A hot update within a single session.
    ep(second, "b", deploy, True)  # Duplicate record must not inflate the score.
    ep(second, "c", deploy, False)
    ep(teacher, "teacher", deploy, True, "teacher")
    ep(pilot, "pilot", deploy, True)
    store.event(second, "cancelled", {"records": [], "model_version": deploy})
    groups = round_history(store)["collection"]
    assert len(groups) == 2
    assert groups[0]["model_round"] == 0 and groups[0]["success_rate"] == 0
    deployed = groups[1]
    assert deployed["model_round"] == 1
    assert (deployed["episodes"], deployed["successes"], deployed["failures"]) == (2, 1, 1)
    assert deployed["success_rate"] == .5
    assert [p["success_rate"] for p in deployed["curve"]] == [1., .5]
    assert deployed["sessions"] == [first, second]


def test_rounds_keep_rejected_training_and_partial_evaluation_without_invented_metrics(tmp_path):
    store = Store(tmp_path)
    training = store.create("self-train", "rsi", {"steps": 100})
    checkpoint = str(store.run_dir(training) / "checkpoint")
    for step, loss in [(1, 1.5), (5, 1.2), (100, .8)]:
        store.event(training, "metric", {"step": step, "loss": loss, "training_samples": 64,
                                         "successful_episodes": 5})
    store.event(training, "metric", {"step": 0, "dev_success_rate": .8})
    store.finish(training, summary={"initial_success_rate": .8, "success_rate": .8,
                                   "selected_step": 0, "checkpoint": checkpoint})
    evaluation = store.create("evaluate", "mujoco", {"candidate": checkpoint, "tasks": ["transfer"], "seeds": [2, 3]})
    store.event(evaluation, "pair", {"candidate_success": True, "champion_success": True})
    partial = round_history(store)["training"][0]
    assert partial["evaluated_pairs"] == 1 and partial["total_pairs"] == 2
    assert partial["evaluation_status"] == "running" and partial["gate_passed"] is None
    assert partial["candidate_success_rate"] == 1.
    store.event(evaluation, "pair", {"candidate_success": False, "champion_success": True})
    store.finish(evaluation, summary={"success_rate": .5, "champion_success_rate": 1.,
                                     "paired_gain": -.5, "gate": {"passed": False}})
    report = RoundRecorder(store).refresh()
    row = report["training"][0]
    assert row["loss_first"] == 1.5 and row["loss_last"] == .8
    assert row["logged_updates"] == 3 and row["selected_step"] == 0
    assert row["loss_selected_step"] is None  # There was no optimizer update at step zero.
    assert row["gate_passed"] is False and row["deployment"] is None
    assert row["candidate_success_rate"] == .5 and row["parent_success_rate"] == 1.
    directory = tmp_path / "metrics"
    assert json.loads((directory / "rounds.json").read_text())["training"] == report["training"]
    with (directory / "training-loss.csv").open(encoding="utf-8-sig") as file:
        losses = list(csv.DictReader(file))
    assert [int(r["step"]) for r in losses] == [1, 5, 100]
    with (directory / "training-rounds.csv").open(encoding="utf-8-sig") as file:
        assert next(iter(csv.DictReader(file)))["gate_passed"] == "False"


def test_dashboard_restart_observes_existing_job_without_launching_another(tmp_path, monkeypatch):
    import time

    from embodied_rsi.dashboard import Jobs
    from embodied_rsi.storage import write_json

    store = Store(tmp_path)
    created = time.time()
    metadata = {"id": "job-surviving", "pid": 1234, "action": "self-improve", "backend": "rsi",
                "created": created, "status": "running"}
    write_json(store.root / "jobs" / metadata["id"] / "job.json", metadata)
    run = store.create("self-improve", "rsi", {})

    class Process:
        pid = 1234
        running = True

        def cmdline(self):
            return ["python", "-m", "embodied_rsi.cli", "--artifacts", str(store.root), "self-improve"]

        def create_time(self):
            return created

        def is_running(self):
            return self.running

        def status(self):
            return "running"

    process = Process()
    monkeypatch.setattr("embodied_rsi.dashboard.psutil.Process", lambda pid: process)
    jobs = Jobs(store)
    assert jobs.list()[0]["status"] == "running"
    assert jobs.jobs[metadata["id"]]["process"].pid == 1234
    store.finish(run)
    process.running = False
    assert jobs.list()[0]["status"] == "completed"
