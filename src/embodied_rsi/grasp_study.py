"""Strict pick-and-place: demonstrations, fixed replay, self replay and corrections."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .evaluation import paired_report
from .grasp import GraspWorld, grasp_state
from .monitor import Monitor
from .sim import INSTRUCTIONS, episode, execute, menu_for, teacher
from .storage import dumps, write_json

REVISION = "f9248caceb89caf2e6c968ea33bf0d6eb7f957b0"
INPUT_CONTRACT = "grasp-observation-v1"


def case_from(row, ep, source):
    keys = list(row["criteria"])
    return {"case_id": f"{ep['episode_id']}:{row['step']}:{source}", "source": source,
            "state": dumps(row["state"]),
            "questions": [{"key": "action", "mode": "choice", "instructions": INSTRUCTIONS,
                           "options": keys, "criteria": row["criteria"]}],
            "gold": {"action": [float(k == row["choice"]) for k in keys]},
            "provenance": {"task": "transfer", "group": ep["group"], "seed": ep["seed"],
                           "episode_id": ep["episode_id"], "episode_success": ep["success"],
                           "choice": row["choice"], "label_source": source,
                           "input_contract": INPUT_CONTRACT}}


def successful_cases(episodes, source):
    cases, filtered = [], Counter()
    for ep in episodes:
        if not ep["success"]:
            continue
        for row in ep["records"]:
            if not row["executed"]:
                filtered["rejected"] += 1
                continue
            b, a = row["before"], row["after"]
            unchanged = (np.linalg.norm(np.asarray(b["tcp"]) - a["tcp"]) < .001
                         and np.linalg.norm(np.asarray(b["object"]) - a["object"]) < .001
                         and b["gripper"] == a["gripper"])
            if unchanged:
                # Keep waiting only if it actually advanced measured stability.
                stability = any(a[k] > b[k] + .05 for k in ("stable_seconds",))
                if not stability:
                    filtered["no_change"] += 1
                    continue
            if row["choice"] == "grasp" and not a["grasp_secured"]:
                filtered["empty_grasp"] += 1
                continue
            cases.append(case_from(row, ep, source))
    return cases, dict(filtered)


def save_dataset(store, name, pools, episode_counts=None, filtered=None):
    output = store.root / "datasets" / name
    output.mkdir(parents=True, exist_ok=False)
    cases = [c for pool in pools.values() for c in pool]
    unique = {c["case_id"]: c for c in cases}
    cases = list(unique.values())
    for split in ("train", "dev", "test"):
        (output / f"{split}.jsonl").write_text(
            "".join(dumps(c) + "\n" for c in cases) if split == "train" else "")
    manifest = {"schema_version": 2, "objective": "strict_grasp_feedback",
                "input_contract": INPUT_CONTRACT, "cases": {"train": len(cases), "dev": 0, "test": 0},
                "sources": dict(Counter(c["source"] for c in cases)),
                "actions": dict(Counter(c["provenance"]["choice"] for c in cases)),
                "filtered": filtered or {}, "development": "independent full strict tasks",
                "sha256": {s: hashlib.sha256((output / f"{s}.jsonl").read_bytes()).hexdigest()
                           for s in ("train", "dev", "test")}, **(episode_counts or {})}
    write_json(output / "manifest.json", manifest)
    return output, manifest


class BalancedSampler:
    """Fixed source proportions; balance skills and episodes within each source."""
    def __init__(self, pools, weights, seed):
        self.rng = np.random.default_rng(seed)
        self.pools = {k: v for k, v in pools.items() if v and weights.get(k, 0) > 0}
        self.names = list(self.pools)
        probabilities = np.asarray([weights[k] for k in self.names], dtype=float)
        if not len(probabilities):
            raise ValueError("No verified training samples")
        self.probabilities = probabilities / probabilities.sum()
        self.groups = {}
        for source, cases in self.pools.items():
            actions = defaultdict(lambda: defaultdict(list))
            for index, case in enumerate(cases):
                actions[case["provenance"]["choice"]][case["provenance"]["episode_id"]].append(index)
            self.groups[source] = {a: dict(g) for a, g in actions.items()}

    def indices(self, size):
        result = []
        for _ in range(size):
            source = self.names[int(self.rng.choice(len(self.names), p=self.probabilities))]
            actions = self.groups[source]
            action = list(actions)[int(self.rng.integers(len(actions)))]
            episodes = actions[action]
            group = list(episodes)[int(self.rng.integers(len(episodes)))]
            indices = episodes[group]
            result.append((source, indices[int(self.rng.integers(len(indices)))]))
        return result


class Preview:
    def __init__(self, root):
        self.root, self.renderer, self.model, self.last = root, None, None, 0.

    def update(self, world):
        if time.monotonic() - self.last < 1.:
            return
        self.last = time.monotonic()
        import mujoco
        from PIL import Image
        if self.model is not world.model:
            self.close()
            world.model.vis.global_.offwidth = max(640, world.model.vis.global_.offwidth)
            world.model.vis.global_.offheight = max(360, world.model.vis.global_.offheight)
            self.renderer = mujoco.Renderer(world.model, height=360, width=640)
            self.model = world.model
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.43, 0., .04]
        camera.azimuth, camera.elevation, camera.distance = 135, -25, 1.25
        self.renderer.update_scene(world.data, camera=camera)
        temporary = self.root / "grasp-frame.tmp"
        Image.fromarray(self.renderer.render()).save(temporary, format="JPEG", quality=80)
        temporary.replace(self.root / "grasp-frame.jpg")
        write_json(self.root / "grasp-observation.json", {
            "time": time.time(), "observation": world.observe(), "stages": world.diagnostics()})

    def close(self):
        if self.renderer:
            self.renderer.close()
        self.renderer, self.model = None, None


def verified_correction(world, state, criteria, step, remaining):
    """Label a visited state only if the teacher suffix physically finishes strictly."""
    branch = world.clone()
    choice = teacher(branch)
    if choice not in criteria:
        return None
    row = {"state": copy.deepcopy(state), "criteria": copy.deepcopy(criteria),
           "choice": choice, "step": step}
    for _ in range(remaining):
        if branch.success():
            return row
        options = menu_for(branch)
        action = teacher(branch)
        if action not in options:
            return None
        ok, _ = execute(branch, options[action])
        if not ok or branch.audit["drop"]:
            return None
    return row if branch.success() else None


class Study:
    def __init__(self, store, protocol):
        self.store, self.protocol = store, protocol
        self.run = store.create("grasp-study", "rsi", protocol)
        self.directory = store.run_dir(self.run)
        write_json(self.directory / "protocol.json", protocol)
        self.protocol_hash = hashlib.sha256((self.directory / "protocol.json").read_bytes()).hexdigest()
        self.preview = Preview(store.root)
        self.status = {"run": self.run, "status": "running", "protocol": protocol,
                       "stage": "pilot", "results": [], "audit": [], "comparisons": [],
                       "protocol_sha256": self.protocol_hash, "started": time.time()}
        self.publish()

    def publish(self, **values):
        self.status.update(values, updated=time.time())
        write_json(self.store.root / "grasp-study-latest.json", self.status)
        self.store.event(self.run, "stage", {k: v for k, v in self.status.items()
                                            if k not in {"results", "audit", "comparisons", "protocol"}})

    def rollout(self, policy, seed, rng=None, corrections=False):
        worlds, snapshots = [], []
        def factory(*args):
            world = GraspWorld(*args)
            worlds.append(world)
            return world
        def decision(world, state, criteria, step):
            if corrections and not world.audit["drop"]:
                snapshots.append((world.clone(), copy.deepcopy(state), copy.deepcopy(criteria), step))
        result = episode("transfer", seed, policy, max_steps=self.protocol["max_steps"],
                         label_mode="none", rng=rng, randomize_scene=True,
                         world_factory=factory, on_frame=self.preview.update,
                         on_decision=decision, state_transform=grasp_state,
                         staged_prompts=self.protocol["episode_protocol"]["staged_prompts"])
        result.update(stages=worlds[0].diagnostics(), input_contract=INPUT_CONTRACT,
                      success_definition="strict_grasp_v1", source_policy="teacher" if policy is None else "rsi")
        labels = []
        if corrections and not result["success"]:
            # Two late, recoverable, wrongly chosen states per failed episode, at most.
            for world, state, criteria, step in reversed(snapshots):
                if teacher(world) == result["records"][step]["choice"]:
                    continue
                row = verified_correction(world, state, criteria, step, self.protocol["max_steps"] - step)
                if row:
                    labels.append(case_from(row, result, "correction"))
                if len(labels) == 2:
                    break
        return result, labels

    def collect(self, policy, seeds, name, rng=None, corrections=False, successes_target=None):
        run = self.store.create("collect" if policy is None else "explore", "rsi", {
            "study": self.run, "label_source": "teacher" if policy is None else "model",
            "success_definition": "strict_grasp_v1", "phase": name, "scene_seeds": [seeds.start, seeds.stop]})
        rows, labels = [], []
        with Monitor(self.store, run) as monitor, (self.store.run_dir(run) / "episodes.jsonl").open("w") as output:
            for index, seed in enumerate(seeds):
                self.publish(phase=name, episode=index + 1, episode_budget=len(seeds), collection=run)
                result, correction = self.rollout(policy, seed, rng, corrections)
                rows.append(result)
                labels.extend(correction)
                output.write(dumps(result) + "\n")
                output.flush()
                monitor.metric(index + 1, episodes=index + 1,
                               successful_episodes=sum(r["success"] for r in rows),
                               exploration_success_rate=float(np.mean([r["success"] for r in rows])))
                if successes_target and sum(r["success"] for r in rows) >= successes_target:
                    break
        summary = {"episodes": len(rows), "successful_episodes": sum(r["success"] for r in rows),
                   "failed_episodes": sum(not r["success"] for r in rows),
                   "success_rate": float(np.mean([r["success"] for r in rows])), "corrections": len(labels)}
        self.store.finish(run, summary=summary)
        return rows, labels, run

    def validate(self, learner, run, step):
        first = self.protocol["dev_seed_start"]
        rows = []
        for index, seed in enumerate(range(first, first + self.protocol["dev_episodes"])):
            if index % 10 == 0:
                self.publish(stage="development", validation_step=step,
                             dev_completed=index, dev_budget=self.protocol["dev_episodes"])
            result, _ = self.rollout(learner, seed)
            rows.append(result)
        score = (float(np.mean([r["success"] for r in rows])),
                 -sum(r[k] for r in rows for k in ("collisions", "drops", "rejections")))
        self.store.event(run, "metric", {"step": step, "dev_success_rate": score[0], "dev_risk_count": -score[1],
                         **{f"dev_{k}_rate": float(np.mean([r["stages"][k] for r in rows]))
                            for k in ("grasp", "lift", "carry", "place", "drop")}})
        with (self.store.run_dir(run) / "development.jsonl").open("a") as output:
            for row in rows:
                output.write(dumps({"update_step": step, **row}) + "\n")
        return score

    def fit(self, learner, pools, weights, steps, seed, arm, round_index):
        sampler = BalancedSampler(pools, weights, seed)
        dataset, manifest = save_dataset(self.store, f"grasp-{arm}-{seed}-r{round_index}", pools)
        run = self.store.create("initialize" if arm == "S" else "self-train", "rsi", {
            "study": self.run, "arm": arm, "round": round_index, "training_seed": seed,
            "dataset": str(dataset), "dataset_sha256": manifest["sha256"], "steps": steps,
            "batch_size": 16, "lr": learner.lr, "source_weights": weights,
            "input_contract": INPUT_CONTRACT, "objective": "strict_grasp_feedback"})
        with Monitor(self.store, run) as monitor:
            self.publish(stage="development", training=run, arm=arm, round=round_index, training_seed=seed)
            best_score = self.validate(learner, run, 0)
            best, best_step = learner.snapshot(), 0
            self.publish(stage="encoding", training_samples=manifest["cases"]["train"])
            encoded = {}
            total = sum(len(v) for v in sampler.pools.values())
            count = 0
            for source, cases in sampler.pools.items():
                encoded[source] = []
                for c in cases:
                    row = {"state": json.loads(c["state"]), "criteria": c["questions"][0]["criteria"]}
                    encoded[source].append((learner.encode_success(row), int(np.argmax(c["gold"]["action"]))))
                    count += 1
                    if count % 20 == 0 or count == total:
                        monitor.metric(0, encoded_samples=count, training_samples=total)
                        self.publish(encoded_samples=count)
            learner.optimizer = learner.torch.optim.AdamW(learner.params, lr=learner.lr, weight_decay=0.)
            learner.torch.manual_seed(seed)
            first_loss = last_loss = None
            for step in range(1, steps + 1):
                if step == 1 or step % 10 == 0:
                    self.publish(stage="training", update_step=step, update_budget=steps)
                indices = sampler.indices(16)
                metrics = learner.imitate([encoded[source][index] for source, index in indices])
                first_loss = metrics["loss"] if first_loss is None else first_loss
                last_loss = metrics["loss"]
                if step == 1 or step % 10 == 0 or step == steps:
                    monitor.metric(step, **metrics, training_samples=total)
                if step % self.protocol["eval_every"] == 0 or step == steps:
                    score = self.validate(learner, run, step)
                    if score > best_score:
                        best_score, best, best_step = score, learner.snapshot(), step
            # Keep last-step head independently for honest diagnosis, not deployment.
            from safetensors.torch import save_file
            save_file({k: v.detach().float().cpu().contiguous() for k, v in learner.model.scorer.state_dict().items()},
                      str(self.store.run_dir(run) / "last-scorer.safetensors"))
            learner.restore(best)
            path = self.store.run_dir(run) / "checkpoint"
            metadata = {"objective": "strict_grasp_feedback", "steps": steps, "batch_size": 16,
                        "lr": learner.lr, "train_seed_range": [self.protocol["train_seed_start"],
                                                               self.protocol["dev_seed_start"]],
                        "dev_seed_range": [self.protocol["dev_seed_start"], self.protocol["dev_seed_start"] + self.protocol["dev_episodes"]],
                        "input_contract": INPUT_CONTRACT, "success_definition": "strict_grasp_v1",
                        "study": self.run, "arm": arm, "round": round_index, "training_seed": seed,
                        "training_samples": total, "selected_step": best_step, "success_rate": best_score[0],
                        "first_loss": first_loss, "last_loss": last_loss, "dataset": str(dataset)}
            learner.save(path, metadata)
            # Upstream pretraining counts are not counts of this experiment's samples.
            meta = json.loads((path / "meta.json").read_text())
            for key in ("n_train_cases", "final_loss", "train_seconds"):
                meta.pop(key, None)
            write_json(path / "meta.json", meta)
            write_json(self.store.run_dir(run) / "summary.json", {**metadata, "checkpoint": str(path)})
            self.store.finish(run, summary={**metadata, "checkpoint": str(path)})
        result = {**metadata, "checkpoint": str(path), "training": run,
                  "actions": manifest["actions"], "sources": manifest["sources"]}
        self.status["results"].append(result)
        self.publish(stage="training_complete")
        return learner.snapshot(), result


def run_study(store, rounds=3, explore_episodes=300, steps=300, warmup_steps=500,
              demos=100, dev_episodes=100, test_episodes=500, training_seeds=(17, 29, 43),
              checkpoint="v1.0-0.8b", max_steps=20, pilot_episodes=50, eval_every=50):
    if not all(0 < x <= 10000 for x in (rounds, explore_episodes, steps, warmup_steps,
                                       demos, dev_episodes, test_episodes, pilot_episodes, eval_every)):
        raise ValueError("Experiment budgets must be positive and bounded")
    if (store.root / "grasp-study-latest.json").exists():
        raise ValueError("Use a fresh artifact store; the existing experiment is immutable")
    protocol = {"version": "strict-grasp-study-v1", "rounds": rounds, "explore_episodes": explore_episodes,
                "steps": steps, "warmup_steps": warmup_steps, "demos": demos,
                "dev_episodes": dev_episodes, "test_episodes": test_episodes,
                "training_seeds": list(training_seeds), "pilot_episodes": pilot_episodes,
                "checkpoint": str(checkpoint), "revision": REVISION, "max_steps": max_steps,
                "episode_protocol": {"staged_prompts": False, "strict_grasp": True},
                "eval_every": eval_every, "train_seed_start": 2000000,
                "dev_seed_start": 2100000, "test_seed_start": 2200000,
                "scene_distribution": "workspace-random-v1", "input_contract": INPUT_CONTRACT,
                "success_definition": "strict_grasp_v1", "main_comparison": "R_final_vs_F_final",
                "gain_threshold": .05, "final_test_revealed": "after_all_training",
                "arms": ["V0", "S", "F", "R", "C"], "batch_size": 16, "lr": 1e-4,
                "source_weights": {"F": {"demo": 1.}, "R": {"demo": .5, "self": .5},
                                   "C": {"demo": .5, "self": .3, "correction": .2}},
                "drop_contact_grace_s": .05, "environment_information": "measured geometry, contacts, object properties; no future or scene seed"}
    write_json(store.root / "experiment-mode.json", {"strict_grasp": True, "input_contract": INPUT_CONTRACT})
    study = Study(store, protocol)
    try:
        with Monitor(store, study.run):
            pilot, _, _ = study.collect(None, range(2300000, 2300000 + pilot_episodes), "controller_pilot")
            pilot_rate = float(np.mean([r["success"] for r in pilot]))
            study.publish(pilot_success_rate=pilot_rate)
            if pilot_rate < .9:
                raise RuntimeError(f"Controller pilot strict success {pilot_rate:.1%} < 90%; formal training not started")
            study.publish(stage="demonstrations")
            demonstration_episodes, _, _ = study.collect(None, range(2000000, 2000000 + 3 * demos),
                                                       "demonstrations", successes_target=demos)
            good = [r for r in demonstration_episodes if r["success"]][:demos]
            if len(good) < demos:
                raise RuntimeError("Insufficient strict successful demonstrations")
            demo_cases, filtered = successful_cases(good, "demo")
            save_dataset(store, "grasp-demonstrations", {"demo": demo_cases},
                         {"episodes_total": len(demonstration_episodes), "successful_episodes": demos,
                          "failed_episodes": sum(not r["success"] for r in demonstration_episodes)}, filtered)
            from .task_rsi import RSILearner
            study.publish(stage="loading_model")
            learner = RSILearner(checkpoint, lr=1e-4, revision=REVISION)
            learner.episode_protocol = dict(protocol["episode_protocol"])
            original = learner.snapshot()
            frozen_meta = copy.deepcopy(learner.meta)
            models = [("V0", 0, original, None)]
            for seed_index, seed in enumerate(training_seeds):
                random.seed(seed)
                learner.torch.manual_seed(seed)
                learner.restore(original)
                learner.meta = copy.deepcopy(frozen_meta)
                seed_head, seed_result = study.fit(learner, {"demo": demo_cases}, {"demo": 1.}, warmup_steps, seed, "S", 0)
                models.append(("S", seed, seed_head, seed_result["checkpoint"]))
                for arm in ("F", "R", "C"):
                    learner.restore(seed_head)
                    learner.meta = copy.deepcopy(frozen_meta)
                    pools = {"demo": demo_cases, "self": [], "correction": []}
                    for round_index in range(1, rounds + 1):
                        study.publish(stage="collection" if arm != "F" else "fixed_replay",
                                      arm=arm, round=round_index, training_seed=seed)
                        if arm != "F":
                            first = 2010000 + seed_index * 20000 + (round_index - 1) * explore_episodes
                            collected, corrections, _ = study.collect(learner, range(first, first + explore_episodes),
                                f"{arm}_seed{seed}_round{round_index}", np.random.default_rng(seed * 100 + round_index), arm == "C")
                            own_cases, removed = successful_cases(collected, "self")
                            pools["self"].extend(own_cases)
                            pools["correction"].extend(corrections)
                            study.publish(new_successes=sum(r["success"] for r in collected),
                                          new_self_samples=len(own_cases), new_corrections=len(corrections), filtered=removed)
                            if not own_cases:
                                study.store.event(study.run, "alert", {"code": "no_new_self_success",
                                    "severity": "warning", "message": f"{arm} 第 {round_index} 轮无新增成功；实际数据配比按可用来源归一化，记录此限制"})
                        head, result = study.fit(learner, pools, protocol["source_weights"][arm], steps,
                                                 seed, arm, round_index)
                        models.append((f"{arm}{round_index}", seed, head, result["checkpoint"]))
            study.publish(stage="sealed_test", phase="final_audit")
            audited = {}
            for label, seed, head, checkpoint_path in models:
                # Final reports always use the physically saved/reloaded head.
                if checkpoint_path:
                    from safetensors.torch import load_file
                    learner.restore(load_file(str(Path(checkpoint_path) / "scorer.safetensors")))
                else:
                    learner.restore(head)
                audit_run = store.create("grasp-audit", "rsi", {"study": study.run, "label": label, "training_seed": seed})
                rows = []
                with Monitor(store, audit_run), (store.run_dir(audit_run) / "episodes.jsonl").open("w") as output:
                    for index in range(test_episodes):
                        study.publish(stage="sealed_test", audit_model=label, training_seed=seed,
                                      audit_completed=index, audit_budget=test_episodes)
                        row, _ = study.rollout(learner, protocol["test_seed_start"] + index)
                        output.write(dumps(row) + "\n")
                        output.flush()
                        rows.append(row)
                audited[label, seed] = rows
                summary = {"label": label, "training_seed": seed, "episodes": test_episodes,
                           "successful_episodes": sum(r["success"] for r in rows),
                           "success_rate": float(np.mean([r["success"] for r in rows])),
                           "checkpoint": checkpoint_path, "audit_run": audit_run,
                           "stages": {k: float(np.mean([r["stages"][k] for r in rows]))
                                      for k in ("grasp", "lift", "carry", "place", "drop")}}
                store.finish(audit_run, summary=summary)
                study.status["audit"].append(summary)
            for seed in training_seeds:
                for candidate, champion in ((f"R{rounds}", f"F{rounds}"),
                                           (f"C{rounds}", f"R{rounds}"), ("S", "V0")):
                    base_seed = 0 if champion == "V0" else seed
                    report = paired_report(audited[candidate, seed], audited[champion, base_seed],
                                           min_pairs=test_episodes, expected_tasks=["transfer"])
                    study.status["comparisons"].append({"candidate": candidate, "champion": champion,
                                                       "training_seed": seed, **report})
            main = [r for r in study.status["comparisons"] if r["candidate"] == f"R{rounds}"]
            conclusion = ("自主成功回放在所有训练种子均通过预设提升门槛" if all(r["gate"]["passed"] for r in main)
                          else "未在所有训练种子证明自主成功回放的可靠提升；请查看逐种子结果")
            if hashlib.sha256((study.directory / "protocol.json").read_bytes()).hexdigest() != study.protocol_hash:
                raise RuntimeError("Frozen protocol changed")
            study.publish(status="completed", stage="completed", conclusion=conclusion)
            write_json(study.directory / "report.json", study.status)
            store.finish(study.run, summary=study.status)
            return study.status
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        study.publish(status=status, error=f"{type(exc).__name__}: {exc}")
        for run in store.runs():
            if run["config"].get("study") == study.run and run["status"] == "running":
                store.finish(run["id"], status, {"error": str(exc)})
        store.finish(study.run, status, {"error": str(exc)})
        raise
    finally:
        study.preview.close()
