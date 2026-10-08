from __future__ import annotations

from pathlib import Path

from .dataset import build_dataset, validate_dataset
from .evaluation import fingerprint, paired_report
from .monitor import Monitor
from .sim import episode
from .storage import dumps, write_json
from .training import train_compact


def collect(store, tasks, seeds, policy=None, policy_name="teacher", max_steps=20,
            label_mode="teacher", intervention=False):
    run_id = store.create("collect", "mujoco", {"tasks": tasks, "seeds": list(seeds),
        "policy": policy_name, "max_steps": max_steps, "label_mode": label_mode, "intervention": intervention})
    successes, count = 0, 0
    try:
        with Monitor(store, run_id), (store.run_dir(run_id) / "episodes.jsonl").open("w") as f:
            for task in tasks:
                for seed in seeds:
                    e = episode(task, seed, policy, max_steps, label_mode, intervention)
                    f.write(dumps(e) + "\n")
                    f.flush()
                    count += 1
                    successes += int(e["success"])
                    store.event(run_id, "episode", {k: v for k, v in e.items() if k != "records"})
        summary = {"episodes": count, "success_rate": successes / count,
                   "episodes_file": str(store.run_dir(run_id) / "episodes.jsonl")}
        store.finish(run_id, summary=summary)
        return run_id
    except BaseException as exc:
        store.finish(run_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise


def evaluate(store, candidate, candidate_path, champion, champion_name, tasks, seeds,
             max_steps=20, dataset=None, intervention=False, randomize_scene=False):
    import json
    seeds = list(seeds)
    metadata_path = Path(candidate_path) / "meta.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    for name in ("train_seed_range", "dev_seed_range"):
        bounds = metadata.get(name)
        if bounds and any(bounds[0] <= seed < bounds[1] for seed in seeds):
            raise ValueError("Evaluation scenes overlap task training/development")
    manifest = validate_dataset(dataset) if dataset else None
    if manifest:
        for task in tasks:
            for seed in seeds:
                group = f"{task}:{seed}:{int(intervention)}"
                if manifest["groups"].get(group) in {"train", "dev"}:
                    raise ValueError(f"Evaluation scene {group} was used for training or development")
    candidate_sha = fingerprint(candidate_path)
    champion_path = Path(champion_name)
    champion_sha = fingerprint(champion_path) if champion_path.is_dir() else None
    run_id = store.create("evaluate", "mujoco", {"candidate": str(candidate_path),
        "champion": champion_name, "tasks": tasks, "seeds": list(seeds), "max_steps": max_steps})
    new, old = [], []
    try:
        with Monitor(store, run_id), (store.run_dir(run_id) / "paired_episodes.jsonl").open("w") as f:
            for task in tasks:
                for seed in seeds:
                    a = episode(task, seed, candidate, max_steps, label_mode="none", intervention=intervention,
                                randomize_scene=randomize_scene)
                    b = episode(task, seed, champion, max_steps, label_mode="none", intervention=intervention,
                                randomize_scene=randomize_scene)
                    new.append(a)
                    old.append(b)
                    f.write(dumps({"candidate": a, "champion": b}) + "\n")
                    f.flush()
                    store.event(run_id, "pair", {"task": task, "seed": seed,
                        "candidate_success": a["success"], "champion_success": b["success"]})
        report = paired_report(new, old, expected_tasks=tasks)
        if fingerprint(candidate_path) != candidate_sha or (
                champion_sha and fingerprint(champion_path) != champion_sha):
            raise ValueError("Checkpoint changed during paired evaluation")
        report.update(candidate_checkpoint=str(Path(candidate_path).resolve()),
                      candidate_sha256=candidate_sha, champion=champion_name, champion_sha256=champion_sha,
                      model_backend=metadata.get("backend"),
                      scene_distribution="workspace-random-v1" if randomize_scene else "upstream-default")
        write_json(store.run_dir(run_id) / "report.json", report)
        store.finish(run_id, summary=report)
        return run_id
    except BaseException as exc:
        store.finish(run_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise


def reserve_suite(store, requested_start=0):
    """Persist independent train/dev/release namespaces, including failed rounds."""
    with store.connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS scene_allocator (id INTEGER PRIMARY KEY, next_seed INTEGER)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT next_seed FROM scene_allocator WHERE id=1").fetchone()
        start = max(1000000, requested_start, row[0] if row else 0)
        db.execute("INSERT OR REPLACE INTO scene_allocator VALUES (1, ?)", (start + 30000,))
    return {"train": start, "dev": start + 10000, "release": start + 20000}


def cycle(store, rounds=1, episodes=30, steps=6, seed_start=0, backend="compact",
          checkpoint=None, revision=None, warmup_steps=500):
    """Episode reward → task validation → independent paired gates → automatic deployment."""
    import json

    from .compact import CompactPolicy
    from .deployment import current, deploy, load_policy
    from .task_training import TASKS, train_tasks
    from .training import train_rsi

    if episodes < 30 or episodes > 300 or steps > 1000:
        raise ValueError("Automatic deployment needs at least 30 fresh scenes per task; max 1000 iterations")
    cycle_id = store.create("cycle", backend, {"objective": "episode_success", "rounds": rounds,
        "episodes_per_task": episodes, "steps": steps, "automatic_deployment": True})
    results = []
    try:
        with Monitor(store, cycle_id):
            for index in range(rounds):
                deployed = current(store)
                expected_sha = deployed["sha256"] if deployed else None
                parent = checkpoint
                if not parent and deployed:
                    meta = json.loads((Path(deployed["checkpoint"]) / "meta.json").read_text())
                    if (meta.get("backend") == "compact-numpy") == (backend == "compact"):
                        parent = deployed["checkpoint"]
                starts = [seed_start]
                for model_path in (parent, (deployed or {}).get("checkpoint")):
                    if model_path and (Path(model_path) / "meta.json").is_file():
                        parent_meta = json.loads((Path(model_path) / "meta.json").read_text())
                        starts.append(parent_meta.get("reserved_through", 0))
                suite = reserve_suite(store, max(starts))
                result = {"round": index + 1, "scene_suite": suite, "objective": "episode_success"}

                def stage(name, round_number=index + 1, **payload):
                    store.event(cycle_id, "stage", {"round": round_number, "stage": name, **payload})

                # Teacher imitation is initialization only, never the release objective.
                if not parent:
                    stage("initialization")
                    collection = collect(store, list(TASKS), range(suite["train"] + 1000,
                        suite["train"] + 1030), policy_name="teacher_initialization")
                    dataset = store.root / "datasets" / f"cycle-{collection}"
                    build_dataset([store.run_dir(collection) / "episodes.jsonl"], dataset)
                    if backend == "compact":
                        warmup = train_compact(store, dataset, steps=warmup_steps)
                    else:
                        warmup = train_rsi(store, dataset, "v3.0-2b", steps=warmup_steps,
                                           revision=revision)
                    parent = store.run_dir(warmup) / "checkpoint"
                    result.update(collection=collection, initialization=warmup, dataset=str(dataset))
                stage("task_training")
                training = train_tasks(store, parent, backend, steps, suite["train"], suite["dev"],
                                       revision=revision)
                candidate_path = store.run_dir(training) / "checkpoint"
                candidate_meta = json.loads((candidate_path / "meta.json").read_text())
                candidate_meta.update(scene_suite=suite, reserved_through=suite["release"] + 10000)
                write_json(candidate_path / "meta.json", candidate_meta)
                stage("independent_evaluation", training=training)
                candidate = load_policy(candidate_path)
                if deployed:
                    old = load_policy(deployed["checkpoint"])
                    old_name = deployed["checkpoint"]
                elif backend == "rsi":
                    from serve.release import resolve_ckpt

                    from .deployment import RPCPolicy
                    old_name = str(resolve_ckpt(checkpoint or "v3.0-2b", revision=revision))
                    old = RPCPolicy(old_name)
                else:
                    old, old_name = CompactPolicy(), "untrained-compact"
                try:
                    evaluation = evaluate(store, candidate, candidate_path, old,
                                          old_name,
                                          list(TASKS), range(suite["release"], suite["release"] + episodes))
                finally:
                    for policy in (candidate, old):
                        if hasattr(policy, "close"):
                            policy.close()
                report_path = store.run_dir(evaluation) / "report.json"
                report = json.loads(report_path.read_text())
                result.update(training=training, checkpoint=str(candidate_path), evaluation=evaluation,
                              success_rate=report["success_rate"], gate=report["gate"])
                if report["gate"]["passed"]:
                    stage("deploying")
                    result["deployment"] = deploy(store, report_path, expected_sha)["version"]
                    result["status"] = "deployed"
                else:
                    result["status"] = "retained_current_model"
                    stage("retained_current_model", checks=report["gate"]["checks"])
                results.append(result)
                write_json(store.root / "cycle-latest.json", results)
                stage("completed", status=result["status"])
        store.finish(cycle_id, summary={"rounds": results})
        return results
    except BaseException as exc:
        store.finish(cycle_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000], "rounds": results})
        raise
