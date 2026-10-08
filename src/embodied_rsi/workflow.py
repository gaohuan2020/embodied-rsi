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
             max_steps=20, dataset=None, intervention=False):
    manifest = validate_dataset(dataset) if dataset else None
    if manifest:
        for task in tasks:
            for seed in seeds:
                group = f"{task}:{seed}:{int(intervention)}"
                if manifest["groups"].get(group) in {"train", "dev"}:
                    raise ValueError(f"Evaluation scene {group} was used for training or development")
    run_id = store.create("evaluate", "mujoco", {"candidate": str(candidate_path),
        "champion": champion_name, "tasks": tasks, "seeds": list(seeds), "max_steps": max_steps})
    new, old = [], []
    try:
        with Monitor(store, run_id), (store.run_dir(run_id) / "paired_episodes.jsonl").open("w") as f:
            for task in tasks:
                for seed in seeds:
                    a = episode(task, seed, candidate, max_steps, intervention=intervention)
                    b = episode(task, seed, champion, max_steps, intervention=intervention)
                    new.append(a)
                    old.append(b)
                    f.write(dumps({"candidate": a, "champion": b}) + "\n")
                    f.flush()
                    store.event(run_id, "pair", {"task": task, "seed": seed,
                        "candidate_success": a["success"], "champion_success": b["success"]})
        report = paired_report(new, old)
        report.update(candidate_checkpoint=str(Path(candidate_path).resolve()),
                      candidate_sha256=fingerprint(candidate_path), champion=champion_name)
        write_json(store.run_dir(run_id) / "report.json", report)
        store.finish(run_id, summary=report)
        return run_id
    except BaseException as exc:
        store.finish(run_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise


def cycle(store, rounds=2, episodes=30, steps=500, seed_start=0):
    """A real CPU/MuJoCo learning cycle. Release evaluation is a separate command."""
    from .compact import CompactPolicy
    parent = None
    results = []
    for index in range(rounds):
        policy = CompactPolicy.load(parent) if parent else None
        run = collect(store, ["transfer", "stack", "barrier"],
                      range(seed_start + index * episodes, seed_start + (index + 1) * episodes),
                      policy=policy, policy_name=str(parent) if parent else "teacher")
        dataset = store.root / "datasets" / f"cycle-{run}"
        # Replay past rounds so new state coverage cannot silently delete old skills.
        files = [store.run_dir(r["collection"]) / "episodes.jsonl" for r in results]
        files.append(store.run_dir(run) / "episodes.jsonl")
        build_dataset(files, dataset)
        training = train_compact(store, dataset, steps=steps, parent=parent)
        parent = store.run_dir(training) / "checkpoint"
        results.append({"collection": run, "training": training, "dataset": str(dataset),
                        "checkpoint": str(parent)})
        write_json(store.root / "cycle-latest.json", results)
    return results
