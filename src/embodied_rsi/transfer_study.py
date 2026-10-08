"""Frozen-protocol transfer-only study; audit is revealed after all training rounds."""
from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path

from .deployment import RPCPolicy, current, load_policy
from .evaluation import fingerprint, paired_report
from .monitor import Monitor
from .self_improvement import self_improve
from .sim import episode
from .storage import dumps, write_json
from .workflow import reserve_suite

REVISION = "f9248caceb89caf2e6c968ea33bf0d6eb7f957b0"


def transfer_study(store, rounds=3, explore_episodes=60, steps=100, release_episodes=90,
                   audit_episodes=90, backend="rsi", checkpoint=None, max_steps=20):
    if not 2 <= rounds <= 5 or not 30 <= audit_episodes <= 300 or release_episodes < 90:
        raise ValueError("Use 2–5 rounds, 30–300 audit episodes and at least 90 release episodes")
    if current(store, backend):
        raise ValueError("A clean baseline is required; reset prior deployments or use a fresh artifact store")
    if any(r["kind"] in {"study", "self-train", "task-train", "train", "initialize", "self-improve", "cycle"}
           for r in store.runs()):
        raise ValueError("A clean baseline is required; reset prior training runs or use a fresh artifact store")
    if backend == "rsi":
        from serve.release import resolve_ckpt
        baseline = str(resolve_ckpt(checkpoint or "v1.0-0.8b", revision=REVISION if not checkpoint else None))
    else:
        if checkpoint is None:
            from embodied_jev.physics import RobotWorld

            from .sim import criteria_for, menu_for, state_for
            from .task_training import CompactLearner
            initial = store.create("initialize", "compact", {"mode": "zero_weight_baseline", "tasks": ["transfer"]})
            checkpoint = store.run_dir(initial) / "checkpoint"
            model = CompactLearner()
            world = RobotWorld("transfer", 999999)
            model.predict(state_for(world, []), criteria_for(menu_for(world)))
            model.save(checkpoint, {"tasks": ["transfer"], "initialization": "zero_weights"})
            store.finish(initial, summary={"checkpoint": str(checkpoint)})
        baseline = str(Path(checkpoint).resolve())
    audit_suite = reserve_suite(store)
    protocol = {"tasks": ["transfer"], "rounds": rounds, "explore_episodes": explore_episodes,
        "steps": steps, "release_episodes": release_episodes, "audit_episodes": audit_episodes,
        "max_steps": max_steps, "scene_distribution": "upstream-default",
        "baseline": baseline, "baseline_sha256": fingerprint(baseline), "revision": REVISION,
        "seeded_source_jitter_m": .025, "random_target": False,
        "audit_seed_start": audit_suite["release"], "audit_revealed": "after_all_training_rounds",
        "labels": "actual successful model actions; no teacher initialization",
        "release": "fresh paired task success, confidence interval and risk gates",
        "hypothesis": "success improves across rounds; no monotonicity is imposed"}
    run = store.create("study", backend, protocol)
    protocol_path = store.run_dir(run) / "protocol.json"
    write_json(protocol_path, protocol)
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    try:
        with Monitor(store, run):
            store.event(run, "stage", {"stage": "learning_rounds", "rounds": rounds})
            results = self_improve(store, rounds=rounds, episodes=release_episodes, steps=steps,
                checkpoint=baseline, backend=backend, explore_episodes=explore_episodes,
                tasks=["transfer"], max_steps=max_steps, dev_episodes=10, randomize_scene=False)
            # Audit cannot influence gradient updates, development selection or release decisions.
            models = [{"label": "V0", "checkpoint": baseline, "status": "baseline"}]
            accepted_checkpoint = baseline
            for result in results:
                candidate = store.run_dir(result["training"]) / "checkpoint" if result.get("training") else None
                models.append({"label": f"V{result['round']}", "checkpoint": str(candidate or accepted_checkpoint),
                               "status": result["status"], "round": result["round"]})
                if candidate and result["status"] == "deployed":
                    accepted_checkpoint = str(candidate)
            audited, cached = [], {}
            store.event(run, "stage", {"stage": "sealed_audit", "episodes": audit_episodes})
            for index, model in enumerate(models):
                sha = fingerprint(model["checkpoint"])
                if sha in cached:
                    rows = cached[sha]
                else:
                    policy = load_policy(model["checkpoint"]) if model["checkpoint"] != baseline or backend == "compact" else RPCPolicy(baseline)
                    try:
                        rows = []
                        for seed in range(audit_suite["release"], audit_suite["release"] + audit_episodes):
                            rows.append(episode("transfer", seed, policy, max_steps=max_steps, label_mode="none"))
                            store.event(run, "audit_progress", {"model": model["label"], "completed": len(rows),
                                "total": audit_episodes, "successful_episodes": sum(r["success"] for r in rows)})
                    finally:
                        if hasattr(policy, "close"):
                            policy.close()
                    if fingerprint(model["checkpoint"]) != sha:
                        raise ValueError("Audit checkpoint changed")
                    cached[sha] = rows
                with (store.run_dir(run) / f"audit-{model['label']}.jsonl").open("w") as file:
                    file.write("".join(dumps(r) + "\n" for r in rows))
                rate = sum(r["success"] for r in rows) / len(rows)
                audited.append({**model, "sha256": sha, "success_rate": rate,
                                "successful_episodes": sum(r["success"] for r in rows), "episodes": len(rows)})
                store.event(run, "metric", {"step": index, "audit_success_rate": rate,
                    "model": model["label"], "status": model["status"]})
            if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha:
                raise ValueError("Frozen experiment protocol changed")
            latest = cached[audited[-1]["sha256"]]
            base = cached[audited[0]["sha256"]]
            comparison = paired_report(latest, base, expected_tasks=["transfer"])
            accepted = [audited[0]]
            for point in audited[1:]:
                accepted.append(point if point["status"] == "deployed" else accepted[-1])
            strict = lambda values: all(b > a for a, b in pairwise(values))
            rates = [p["success_rate"] for p in audited]
            deployed_rates = [p["success_rate"] for p in accepted]
            summary = {"protocol_sha256": protocol_sha, "tasks": ["transfer"], "rounds": results,
                "audit": audited, "deployed_audit_rates": deployed_rates,
                "candidate_strictly_increasing": strict(rates),
                "deployed_strictly_increasing": strict(deployed_rates),
                "final_vs_baseline": comparison, "audit_used_for_release": False,
                "conclusion": "候选完成率逐轮上升" if strict(rates) else "未观察到候选完成率逐轮上升",
                "updated_models": sum(r["status"] == "deployed" for r in results)}
            write_json(store.run_dir(run) / "report.json", summary)
            write_json(store.root / "study-latest.json", {"run": run, **summary})
        store.finish(run, summary=summary)
        return summary
    except BaseException as exc:
        store.finish(run, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": str(exc)[:1000], "protocol_sha256": protocol_sha})
        raise
