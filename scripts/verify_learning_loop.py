"""Run real RSI replay, reloaded paired evaluation, release and runtime checks.

Use a fresh artifact directory. Historical inputs must be actual autonomous
baseline episodes, never release/audit episodes or teacher demonstrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from embodied_rsi.deployment import Runtime, current
from embodied_rsi.self_improvement import self_improve
from embodied_rsi.sim import episode
from embodied_rsi.storage import Store, dumps, write_json
from embodied_rsi.workflow import reserve_suite


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=9)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--audit-episodes", type=int, default=30)
    args = parser.parse_args()
    if args.successes < 1 or not 1 <= args.steps <= 20000 or not 1 <= args.audit_episodes <= 300:
        raise ValueError("Use positive replay/update/audit budgets")
    if args.artifacts.exists():
        raise FileExistsError("Use a new artifact directory for verification")
    os.environ["EMBODIED_RSI_STAGED_PROMPTS"] = "1"
    os.environ["EMBODIED_RSI_STRICT_GRASP"] = "1"
    store = Store(args.artifacts)
    run = store.create("explore", "rsi", {"mode": "historical_baseline_replay",
        "source": str(args.episodes_file.resolve()), "successes_requested": args.successes})
    count = good = 0
    with args.episodes_file.open() as source, (store.run_dir(run) / "episodes.jsonl").open("w") as output:
        for line in source:
            ep = json.loads(line)
            if (ep.get("source_policy") != "rsi" or ep.get("data_role") == "evaluation_only"
                    or not ep.get("model_version", "").startswith("RSI-Jev v1.0-0.8b")):
                raise ValueError("Verification needs autonomous original-model episodes")
            output.write(dumps(ep) + "\n")
            count += 1
            good += int(ep["success"])
            if good >= args.successes:
                break
    if good < args.successes:
        raise ValueError("Not enough autonomous baseline successes")
    store.finish(run, summary={"episodes": count, "successful_episodes": good,
                              "success_rate": good / count})
    print(dumps({"stage": "training", "episodes": count, "successful_episodes": good}), flush=True)
    results = self_improve(store, rounds=1, episodes=90, steps=args.steps, checkpoint=str(args.baseline),
        backend="rsi", replay_only=True, tasks=["transfer"], randomize_scene=True,
        max_steps=30, dev_episodes=10)
    result = results[0]
    training = store.run_dir(result["training"])
    parent_head_sha = sha(args.baseline / "scorer.safetensors")
    selected_head_sha = sha(training / "checkpoint" / "scorer.safetensors")
    assert parent_head_sha != selected_head_sha, "Selected head has no parameter update"
    assert result["status"] == "deployed", "Independent release did not establish an improvement"
    report = json.loads((store.run_dir(result["evaluation"]) / "report.json").read_text())
    assert report["candidate_reloaded"] and report["gate"]["passed"]
    print(dumps({"stage": "deployed", "candidate_success_rate": report["success_rate"],
                 "parent_success_rate": report["champion_success_rate"]}), flush=True)
    # The published model must work after a restart with no process-local flags.
    os.environ.pop("EMBODIED_RSI_STAGED_PROMPTS")
    os.environ.pop("EMBODIED_RSI_STRICT_GRASP")
    runtime = Runtime(store)
    policy, model = runtime.pin("rsi")
    audit = []
    try:
        assert model["version"] == current(store, "rsi")["version"] == result["deployment"]
        suite = reserve_suite(store)
        for seed in range(suite["release"], suite["release"] + args.audit_episodes):
            ep = episode("transfer", seed, policy, max_steps=30, label_mode="none", randomize_scene=True)
            audit.append(ep)
            print(dumps({"stage": "runtime_audit", "completed": len(audit),
                         "successes": sum(e["success"] for e in audit)}), flush=True)
        with (store.root / "runtime-audit.jsonl").open("w") as output:
            output.write("".join(dumps(ep) + "\n" for ep in audit))
    finally:
        runtime.release(model)
        runtime.close()
    summary = {"training": result["training"], "evaluation": result["evaluation"],
        "deployment": result["deployment"], "parent_head_sha256": parent_head_sha,
        "selected_head_sha256": selected_head_sha, "parameters_changed": True,
        "training_summary": next(r["summary"] for r in store.runs(limit=None) if r["id"] == result["training"]),
        "release_report": report, "runtime_audit": {"episodes": len(audit),
            "successes": sum(e["success"] for e in audit),
            "success_rate": sum(e["success"] for e in audit) / len(audit),
            "environment_flags_removed": True},
        "historical_autonomous_episodes": count, "historical_autonomous_successes": good,
        "data_source": str(args.episodes_file.resolve()), "data_source_sha256": sha(args.episodes_file),
        "runtime_protocol": policy.episode_protocol,
        "scope": "One transfer-only replay round in a separate artifact store; historical autonomous collection"}
    write_json(store.root / "verification.json", summary)
    print(dumps(summary), flush=True)


if __name__ == "__main__":
    main()
