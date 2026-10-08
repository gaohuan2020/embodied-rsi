"""Outcome-filtered RSI self imitation, with task success selecting and releasing models."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .compact import CompactPolicy
from .deployment import current, deploy, load_policy
from .monitor import Monitor
from .sim import INSTRUCTIONS, episode
from .storage import dumps, write_json
from .task_training import TASKS, CompactLearner, task_score
from .workflow import evaluate, reserve_suite


def success_dataset(files, output):
    """Only actually completed model trajectories; never substitute teacher decisions."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Successful trajectory datasets are immutable")
    output.mkdir(parents=True)
    total, successes, cases, groups, filtered = 0, 0, [], {}, 0
    for file in files:
        with Path(file).open() as source:
            for line in source:
                if not line.endswith("\n"):
                    break  # A continuous collector may currently be appending its next episode.
                ep = json.loads(line)
                total += 1
                if not ep["success"]:
                    continue
                if ep.get("source_policy") == "teacher":
                    raise ValueError("Self-improvement cannot relabel teacher trajectories as model successes")
                successes += 1
                groups[ep["group"]] = "train"
                for row in ep["records"]:
                    if not row["executed"]:
                        filtered += 1
                        continue
                    before, after = row.get("before"), row.get("after")
                    if before and after:
                        empty_grasp = row["choice"] == "grasp" and not after["held"]
                        lost_grip = before["held"] and not after["held"] and row["choice"] not in {"release", "withdraw"}
                        unheld_transport = row["choice"] in {"lift", "carry", "lower"} and not before["held"] and not after["held"]
                        if empty_grasp or lost_grip or unheld_transport:
                            filtered += 1
                            continue
                    keys = list(row["criteria"])
                    cases.append({"case_id": f"{ep['episode_id']}:{row['step']}",
                        "source": "successful_model_episode", "state": dumps(row["state"]),
                        "questions": [{"key": "action", "mode": "choice", "instructions": INSTRUCTIONS,
                                       "options": keys, "criteria": row["criteria"]}],
                        "gold": {"action": [float(k == row["choice"]) for k in keys]},
                        "provenance": {"group": ep["group"], "task": ep["task"], "seed": ep["seed"],
                                       "episode_success": True, "episode_id": ep["episode_id"],
                                       "source_policy": ep.get("source_policy"),
                                       "scene_distribution": ep.get("scene_distribution", "upstream-default"),
                                       "scene_config": ep.get("scene_config"),
                                       "choice": row["choice"], "label_source": "completed_episode_return"}})
    for split in ("train", "dev", "test"):
        (output / f"{split}.jsonl").write_text("".join(dumps(c) + "\n" for c in cases) if split == "train" else "")
    manifest = {"schema_version": 1, "objective": "successful_episode_replay",
                "selection": "executed actions from model episodes with terminal success=1",
                "episodes_total": total, "successful_episodes": successes, "failed_episodes": total - successes,
                "success_rate": successes / total if total else 0., "groups": groups,
                "filtered_failure_actions": filtered,
                "cases": {"train": len(cases), "dev": 0, "test": 0},
                "development": "independent complete tasks, not single-step labels",
                "sha256": {s: hashlib.sha256((output / f"{s}.jsonl").read_bytes()).hexdigest()
                           for s in ("train", "dev", "test")}}
    write_json(output / "manifest.json", manifest)
    return manifest


def fit_successes(store, learner, dataset, steps=100, dev_start=1010000, dev_episodes=3,
                  seed_start=1000000, batch_size=16, backend="rsi", initialization=False,
                  randomize_scene=False):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if hashlib.sha256((dataset / "train.jsonl").read_bytes()).hexdigest() != manifest["sha256"]["train"]:
        raise ValueError("Success replay dataset checksum mismatch")
    cases = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines() if line]
    if not cases:
        raise ValueError("No successful model trajectories; train is skipped")
    objective = "supervised_initialization" if initialization else "successful_episode_replay"
    config = {"objective": objective, "steps": steps, "batch_size": batch_size,
              "lr": learner.lr, "randomize_scene": randomize_scene,
              "dataset": str(dataset), "dataset_sha256": manifest["sha256"],
              "train_seed_range": [seed_start, seed_start + 10000],
              "dev_seed_range": [dev_start, dev_start + dev_episodes]}
    run = store.create("initialize" if initialization else "self-train", backend, config)
    directory = store.run_dir(run)
    started = time.perf_counter()
    try:
        with Monitor(store, run) as monitor:
            def validate(step):
                rows = [episode(task, seed, learner, label_mode="none", randomize_scene=randomize_scene) for task in TASKS
                        for seed in range(dev_start, dev_start + dev_episodes)]
                score = task_score(rows)
                monitor.metric(step, dev_success_rate=score[0], dev_risk_count=-score[1])
                return score
            initial = best_score = validate(0)
            best, best_step = learner.snapshot(), 0
            encoded = []
            for index, c in enumerate(cases):
                row = {"state": json.loads(c["state"]), "criteria": c["questions"][0]["criteria"]}
                if not initialization and not c["provenance"].get("episode_success"):
                    raise ValueError("Only successful episode actions may enter self imitation")
                encoded.append((learner.encode_success(row), int(np.argmax(c["gold"]["action"]))))
                if index % 10 == 0:
                    monitor.metric(0, encoded_samples=index + 1, training_samples=len(cases))
            rng = np.random.default_rng(seed_start)
            for step in range(1, steps + 1):
                batch = [encoded[i] for i in rng.integers(len(encoded), size=batch_size)]
                metrics = learner.imitate(batch)
                if step % 5 == 0 or step == 1 or step == steps:
                    monitor.metric(step, **metrics, training_samples=len(cases),
                                   successful_episodes=manifest.get("successful_episodes", 0),
                                   steps_per_second=step / (time.perf_counter() - started))
                if step % 25 == 0 or step == steps:
                    score = validate(step)
                    if score > best_score:
                        best_score, best, best_step = score, learner.snapshot(), step
                    elif score[0] < best_score[0] - .05:
                        monitor.alert("task_completion_regression", "任务完成率下降，保留之前的模型")
            learner.restore(best)
            if hashlib.sha256((dataset / "train.jsonl").read_bytes()).hexdigest() != manifest["sha256"]["train"]:
                raise ValueError("Successful trajectory data changed during training")
            summary = {"objective": objective, "initial_success_rate": initial[0],
                       "success_rate": best_score[0], "selected_step": best_step,
                       "training_samples": len(cases), "successful_episodes": manifest.get("successful_episodes", 0),
                       "checkpoint": str(directory / "checkpoint"),
                       "train_seconds": time.perf_counter() - started}
            learner.save(directory / "checkpoint", {**config, **summary})
        store.finish(run, summary=summary)
        return run
    except BaseException as exc:
        store.finish(run, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise


def self_improve(store, rounds=2, episodes=30, steps=100, checkpoint=None, backend="rsi",
                 revision=None, seed_start=0, explore_episodes=None, replay_only=False):
    """Explore with RSI, replay completed trajectories, select by full task success, deploy if better."""
    if episodes < 30 or episodes > 300 or steps < 1 or steps > 20000:
        raise ValueError("Need 30–300 scenes per task and 1–20000 replay updates")
    if backend == "rsi" and checkpoint is None and revision is None:
        revision = "f9248caceb89caf2e6c968ea33bf0d6eb7f957b0"
    explore_episodes = explore_episodes or episodes
    if not 1 <= explore_episodes <= 300:
        raise ValueError("Exploration budget must be 1–300 scenes per task")
    run = store.create("self-improve", backend, {"objective": "successful_episode_replay", "rounds": rounds,
        "episodes": episodes, "explore_episodes": explore_episodes,
        "steps": steps, "checkpoint": str(checkpoint), "revision": revision,
        "replay_only": replay_only, "scene_distribution": "workspace-random-v1"})
    results, files, collections = [], [], set()
    # Replay actual previous successful model data; failed episodes remain available to inspect.
    for prior in store.runs():
        file = store.run_dir(prior["id"]) / "episodes.jsonl"
        if prior["kind"] == "explore" and prior["backend"] == backend and file.exists():
            files.append(file)
    try:
        with Monitor(store, run):
            for index in range(rounds):
                deployed = current(store, backend)
                active = current(store)
                if not deployed and active:
                    active_meta = json.loads((Path(active["checkpoint"]) / "meta.json").read_text())
                    if (active_meta.get("backend") == "compact-numpy") == (backend == "compact"):
                        deployed = active
                parent = checkpoint
                if not parent and deployed:
                    meta = json.loads((Path(deployed["checkpoint"]) / "meta.json").read_text())
                    if (meta.get("backend") == "compact-numpy") == (backend == "compact"):
                        parent = deployed["checkpoint"]
                reserve_after = seed_start
                if parent and (Path(parent) / "meta.json").exists():
                    reserve_after = max(reserve_after, json.loads((Path(parent) / "meta.json").read_text()).get("reserved_through", 0))
                suite = reserve_suite(store, reserve_after)
                if backend == "rsi":
                    from .task_rsi import RSILearner
                    learner = RSILearner(parent or "v1.0-0.8b", revision=revision)
                else:
                    learner = CompactLearner(parent, .08)
                collection = store.create("explore", backend, {"checkpoint": str(parent or "v1.0-0.8b"),
                    "mode": "existing_model_replay" if replay_only else "sampled_model_actions",
                    "scene_suite": suite, "round": index + 1})
                collections.add(collection)
                store.event(run, "stage", {"stage": "exploring", "round": index + 1, "collection": collection})
                rng = np.random.default_rng(suite["train"])
                from .exploration import ExplorationMemory, ExploringPolicy
                memory = ExplorationMemory(store, backend)
                memory.import_past()
                exploratory = ExploringPolicy(learner, memory)
                complete, count = 0, 0
                with Monitor(store, collection), (store.run_dir(collection) / "episodes.jsonl").open("w") as output:
                    for task in (() if replay_only else TASKS):
                        for seed in range(suite["train"], suite["train"] + explore_episodes):
                            result = episode(task, seed, exploratory, label_mode="none", rng=rng,
                                             randomize_scene=True)
                            result.update(source_policy=backend, checkpoint=str(parent or "v1.0-0.8b"))
                            result["exploration_update"] = memory.observe(result)
                            output.write(dumps(result) + "\n")
                            output.flush()
                            complete += int(result["success"])
                            count += 1
                            store.event(collection, "episode", {k: v for k, v in result.items() if k != "records"})
                            Monitor(store, collection).metric(count, exploration_success_rate=complete / count,
                                                             successful_episodes=complete, episodes=count)
                store.finish(collection, summary={"episodes": count, "successful_episodes": complete,
                                                  "success_rate": complete / count if count else 0., "failed_episodes": count - complete})
                files.append(store.run_dir(collection) / "episodes.jsonl")
                dataset = store.root / "datasets" / f"success-{collection}"
                manifest = success_dataset(files, dataset)
                result = {"round": index + 1, "collection": collection, "dataset": str(dataset),
                          "exploration_success_rate": complete / count if count else None,
                          "successful_episodes": manifest["successful_episodes"], "scene_suite": suite}
                if not manifest["cases"]["train"]:
                    result["status"] = "no_success_data"
                    store.event(run, "alert", {"code": "no_success_trajectories", "severity": "warning",
                        "message": "RSI 探索暂未完成任务，没有成功轨迹可训练；保留全部失败记录，需要继续探索或初始化"})
                else:
                    store.event(run, "stage", {"stage": "success_replay_training", "round": index + 1})
                    training = fit_successes(store, learner, dataset, steps, suite["dev"], seed_start=suite["train"],
                                             backend=backend, randomize_scene=True)
                    path = store.run_dir(training) / "checkpoint"
                    meta = json.loads((path / "meta.json").read_text())
                    meta.update(reserved_through=suite["release"] + 10000, scene_suite=suite)
                    write_json(path / "meta.json", meta)
                    # Compare with the actual initial RSI when no RSI is deployed; model identity stays explicit.
                    from .deployment import RPCPolicy
                    if deployed and (json.loads((Path(deployed["checkpoint"]) / "meta.json").read_text()).get("backend") == "compact-numpy") == (backend == "compact"):
                        old = load_policy(deployed["checkpoint"])
                        old_name = deployed["checkpoint"]
                    elif backend == "rsi":
                        from serve.release import resolve_ckpt
                        old_name = str(resolve_ckpt(parent or "v1.0-0.8b", revision=revision))
                        old = RPCPolicy(old_name)
                    else:
                        old, old_name = CompactPolicy(), "untrained-compact"
                    store.event(run, "stage", {"stage": "independent_evaluation", "round": index + 1})
                    try:
                        evaluation = evaluate(store, learner, path, old, old_name, list(TASKS),
                                              range(suite["release"], suite["release"] + episodes),
                                              dataset=dataset, randomize_scene=True)
                    finally:
                        if hasattr(old, "close"):
                            old.close()
                    report = store.run_dir(evaluation) / "report.json"
                    report_data = json.loads(report.read_text())
                    result.update(training=training, evaluation=evaluation, success_rate=report_data["success_rate"],
                                  parent_success_rate=report_data["champion_success_rate"])
                    if report_data["gate"]["passed"]:
                        store.event(run, "stage", {"stage": "deploying", "round": index + 1})
                        registry = backend if current(store, backend) else None
                        if registry is None and current(store) and not deployed:
                            registry = backend
                        result["deployment"] = deploy(store, report, (deployed or {}).get("sha256"),
                                                      registry_backend=registry)["version"]
                        result["status"] = "deployed"
                        checkpoint = str(path)
                    else:
                        result["status"] = "retained_current_model"
                results.append(result)
                write_json(store.root / "self-improve-latest.json", results)
                store.event(run, "stage", {"stage": "completed", "round": index + 1, "status": result["status"]})
                del learner
        store.finish(run, summary={"rounds": results})
        return results
    except BaseException as exc:
        for child in store.runs():
            if child["id"] in collections and child["status"] == "running":
                store.finish(child["id"], "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                             {"error": str(exc)[:1000]})
        store.finish(run, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000], "rounds": results})
        raise
