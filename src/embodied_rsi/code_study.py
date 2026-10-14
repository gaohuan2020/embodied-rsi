"""Frozen, paired code/model comparisons; these records never trigger deployment."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .code_policy import CODE_PROTOCOL, CodePolicy, HybridPolicy
from .deployment import load_policy
from .evaluation import fingerprint, paired_report
from .monitor import Monitor
from .scenes import DISTRIBUTION
from .sim import episode
from .storage import dumps, write_json
from .workflow import reserve_suite

PROFILES = ("normal", "missed-grasp")


def world_factory(profile):
    from .grasp import GraspWorld
    if profile == "normal":
        return GraspWorld
    if profile != "missed-grasp":
        raise ValueError("Unknown code study profile")

    class MissedGraspWorld(GraspWorld):
        def __init__(self, *args, **kwargs):
            self.injected = False
            self.interventions = []
            super().__init__(*args, **kwargs)

        def clone(self):
            other = super().clone()
            other.interventions = [dict(event) for event in self.interventions]
            return other

        def motion(self, target=None, gripper=None, seconds=.6, emit=True):
            if gripper == "close" and not self.injected:
                self.injected = True
                self.interventions.append({"kind": "missed_grasp", "sim_time_s": float(self.data.time),
                                           "requested": "close", "applied": "open"})
                gripper = "open"
            yield from super().motion(target, gripper, seconds, emit)

    return MissedGraspWorld


def source_hashes():
    from embodied_jev import physics, planning

    from . import (
        code_policy,
        deployment,
        evaluation,
        grasp,
        inference_worker,
        policies,
        scenes,
        sim,
        stage_prompts,
    )
    modules = (code_policy, grasp, scenes, sim, stage_prompts, physics, planning,
               deployment, evaluation, inference_worker, policies)
    return {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in modules} | {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def check_holdout(checkpoint, seeds):
    """Reject training/dev seeds even when the evaluation adds an actuator fault."""
    meta = json.loads((Path(checkpoint) / "meta.json").read_text())
    if meta.get("backend") == "compact-numpy":
        raise ValueError("Code study model comparison requires an RSI-Jev checkpoint, not Compact")
    configurations = [meta, meta.get("robot_training", {})]
    datasets = set()
    for config in configurations:
        for name in ("train_seed_range", "dev_seed_range"):
            bounds = config.get(name)
            if bounds and any(bounds[0] <= seed < bounds[1] for seed in seeds):
                raise ValueError("Code study seeds overlap model training/development")
        if config.get("dataset"):
            datasets.add(config["dataset"])
    from .dataset import validate_dataset
    for dataset in datasets:
        manifest = validate_dataset(dataset)
        for group, split in manifest["groups"].items():
            parts = group.split(":")
            if split in {"train", "dev"} and len(parts) >= 2 and int(parts[1]) in seeds:
                raise ValueError("Code study seeds overlap model dataset")


def summarize(rows, source):
    decisions = [r for ep in rows for r in ep["records"]]
    exposed = [ep for ep in rows if ep.get("interventions")]
    latencies = [r["latency_ms"] for r in decisions]
    calls = [sum(r.get("policy_decision", {}).get("model_calls", int(source == "rsi"))
                 for r in ep["records"]) for ep in rows]
    return {"episodes": len(rows), "successes": sum(ep["success"] for ep in rows),
            "success_rate": float(np.mean([ep["success"] for ep in rows])),
            "mean_steps": float(np.mean([ep["steps"] for ep in rows])),
            "mean_model_calls": float(np.mean(calls)),
            "decision_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.,
            "mean_wall_seconds": float(np.mean([ep["wall_seconds"] for ep in rows])),
            "risks": {key: sum(ep[key] for ep in rows) for key in ("collisions", "drops", "rejections")},
            "fault_exposed_episodes": len(exposed),
            "successful_after_fault": sum(ep["success"] for ep in exposed),
            "success_after_fault_rate": float(np.mean([ep["success"] for ep in exposed])) if exposed else None,
            "detected_failures": sum(bool(r.get("policy_feedback", {}).get("failure_reason"))
                                     for r in decisions)}


def run_code_study(store, checkpoint=None, episodes=90, seed_start=20000000, max_steps=30,
                   max_retries=2, profiles=PROFILES, staged_prompts=True, rsi_python=None):
    if not 1 <= episodes <= 1000 or max_steps < 1 or seed_start < 0:
        raise ValueError("Need 1–1000 episodes, positive max_steps and nonnegative seed_start")
    profiles = tuple(profiles)
    if not profiles or len(set(profiles)) != len(profiles) or not set(profiles) <= set(PROFILES):
        raise ValueError("Invalid code study profiles")
    if checkpoint:
        checkpoint = Path(checkpoint).resolve()
    policies = {"code": CodePolicy(max_retries), "code_no_recovery": CodePolicy(0)}
    policies["code_no_recovery"].source_policy = "code_no_recovery"
    # Reserve a fresh evaluation namespace, even when a run is interrupted.
    suite = reserve_suite(store, seed_start)
    seeds = list(range(suite["release"], suite["release"] + episodes))
    if checkpoint:
        check_holdout(checkpoint, seeds)
    model_sha = fingerprint(checkpoint) if checkpoint else None
    sources = source_hashes()
    protocol = {"schema_version": 1, "code_protocol": CODE_PROTOCOL,
                "task": "transfer", "profiles": list(profiles), "seeds": seeds,
                "scene_distribution": DISTRIBUTION, "success_definition": "strict_grasp_v1",
                "input_contract": "grasp-observation-v1", "observation_mode": "privileged",
                "staged_prompts": staged_prompts, "max_steps": max_steps, "max_retries": max_retries,
                "checkpoint": str(checkpoint) if checkpoint else None, "checkpoint_sha256": model_sha,
                "source_sha256": sources, "warmup_excluded_from_latency": True,
                "selection_or_training": False, "automatic_deployment": False,
                "fault": "First close command is replaced by open; previews do not consume the live fault.",
                "scope": "Code-policy experiment; simulator state and collision previews; no visual perception"}
    run = store.create("code-study", "mujoco", protocol)
    directory = store.run_dir(run)
    write_json(directory / "protocol.json", protocol)
    protocol_sha = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    model = None
    rows = {profile: {name: [] for name in policies} for profile in profiles}
    try:
        with Monitor(store, run):
            started = time.perf_counter()
            if checkpoint:
                model = load_policy(checkpoint, rsi_python)
                policies.update(rsi=model, hybrid=HybridPolicy(model, max_retries))
                for profile in profiles:
                    rows[profile].update(rsi=[], hybrid=[])
                # Warm up exactly the same observation/prompt/menu contract used below.
                from .grasp import grasp_state
                from .scenes import scene_config
                from .sim import criteria_for, menu_for, state_for
                from .stage_prompts import STAGES, PickPlaceStages
                world = world_factory("normal")("transfer", seeds[0], scene_config("transfer", seeds[0]))
                state = grasp_state(world, state_for(world, []))
                if staged_prompts:
                    state["control"] = PickPlaceStages().update(world.observe())
                options = menu_for(world, STAGES["acquire"][1]) if staged_prompts else menu_for(world)
                model.predict(state, criteria_for(options))
            load_seconds = time.perf_counter() - started
            with (directory / "episodes.jsonl").open("w") as output:
                count = 0
                from .grasp import grasp_state
                for profile in profiles:
                    factory = world_factory(profile)
                    for index, seed in enumerate(seeds):
                        names = list(policies)
                        names = names[index % len(names):] + names[:index % len(names)]
                        scene_hash = None
                        for name in names:
                            result = episode("transfer", seed, policies[name], max_steps=max_steps,
                                             label_mode="none", randomize_scene=True, world_factory=factory,
                                             state_transform=grasp_state, staged_prompts=staged_prompts)
                            if scene_hash is not None and result["scene_hash"] != scene_hash:
                                raise ValueError("Code study policies received different initial scenes")
                            scene_hash = result["scene_hash"]
                            result.update(source_policy=name, experiment_profile=profile,
                                          data_role="evaluation_only", study=run)
                            rows[profile][name].append(result)
                            output.write(dumps(result) + "\n")
                            output.flush()
                            count += 1
                            store.event(run, "episode", {"profile": profile, "policy": name,
                                        "seed": seed, "success": result["success"], "episodes_complete": count,
                                        "episodes_total": len(seeds) * len(profiles) * len(policies)})
            if (source_hashes() != sources or (checkpoint and fingerprint(checkpoint) != model_sha)
                    or hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest() != protocol_sha):
                raise ValueError("Code, model or frozen protocol changed during code study")
            summaries, comparisons = {}, {}
            for profile, experiments in rows.items():
                summaries[profile] = {name: summarize(results, name) for name, results in experiments.items()}
                pairs = [("code", "code_no_recovery")]
                if model:
                    pairs.extend([("code", "rsi"), ("hybrid", "rsi")])
                comparisons[profile] = {f"{new}_vs_{old}": paired_report(
                    experiments[new], experiments[old], expected_tasks=["transfer"]) for new, old in pairs}
            report = {"run": run, "protocol": protocol, "protocol_sha256": protocol_sha,
                      "model_load_and_warmup_seconds": load_seconds, "policies": summaries,
                      "paired_comparisons": comparisons, "deployment_supported": False,
                      "interpretation": "Paired gates are diagnostics only; no code deployment or training is performed."}
            write_json(directory / "report.json", report)
            write_json(store.root / "code-study-latest.json", report)
            store.finish(run, summary=report)
            return report
    except BaseException as exc:
        store.finish(run, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise
    finally:
        if hasattr(model, "close"):
            model.close()
