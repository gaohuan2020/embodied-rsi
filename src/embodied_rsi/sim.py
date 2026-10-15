from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .stage_prompts import instructions_for

PHASES = ("approach", "descend", "grasp", "lift", "carry", "lower", "release", "withdraw", "recover")
INSTRUCTIONS = "Choose the next robot skill to complete the goal. Use measured state and recent outcomes."
POLICY_VERSION = "skills-v1"


def state_for(world, history):
    # Only CURRENT observable facts; exclude simulator success, scene seed and future previews.
    o = world.observe()
    keep = ("task", "units", "tcp", "object", "destination", "gripper", "held",
            "finger_contacts", "grasp_secured", "support_contact")
    return {"task_id": world.task, "observation": {k: o[k] for k in keep},
            "recent_outcomes": history[-3:]}


def menu_for(world, phases=PHASES):
    from embodied_jev.planning import candidates
    options = {}
    for phase in phases:
        c = candidates(world, phase, preview=False)[0]
        if c.target is None or np.any(np.asarray(c.target) < [.20, -.32, .02]) or np.any(
                np.asarray(c.target) > [.65, .34, .42]):
            continue
        options[phase] = c
    return options


def criteria_for(options):
    return {key: json.dumps({"skill": key, "target_tcp": c.target,
                            "gripper_command": c.gripper or "unchanged", "duration_seconds": c.seconds},
                           separators=(",", ":")) for key, c in options.items()}


def teacher(world):
    from embodied_jev.planning import baseline_phase
    return baseline_phase(world)


def execute(world, candidate, on_frame=None):
    shadow = world.clone()
    unsafe = shadow.unsafe_contacts
    try:
        for _ in shadow.motion(candidate.target, candidate.gripper, candidate.seconds, emit=False):
            if shadow.unsafe_contacts > unsafe:
                return False, "collision_preview"
    except (ValueError, RuntimeError) as exc:
        return False, f"unreachable: {exc}"
    unsafe = world.unsafe_contacts
    for _ in world.motion(candidate.target, candidate.gripper, candidate.seconds,
                          emit=bool(on_frame) and not getattr(on_frame, "capture_only", False)):
        if on_frame:
            on_frame(world)
        if world.unsafe_contacts > unsafe:
            return False, "collision"
    return True, None


def rollout_labels(world, options, horizon=4):
    """Privileged TRAINING-only branch labels; never passed to the acting policy."""
    scores = {}
    for key, candidate in options.items():
        branch = world.clone()
        ok, _reason = execute(branch, candidate)
        score = -5. if not ok else 0.
        for _ in range(horizon if ok else 0):
            if branch.success():
                break
            next_options = menu_for(branch)
            next_key = teacher(branch)
            if next_key not in next_options:
                break
            ok, _reason = execute(branch, next_options[next_key])
            if not ok:
                score -= 5.
                break
        o = branch.observe()
        score += (10. * branch.success() + 2. * o["held"] + o["support_contact"]
                  - float(np.linalg.norm(branch.cube - branch.target)))
        scores[key] = score
    logits = np.asarray(list(scores.values())) / .5
    p = np.exp(logits - logits.max())
    return (p / p.sum()).tolist(), scores


def episode(task, seed, policy=None, max_steps=20, label_mode="teacher", intervention=False,
            rng=None, on_frame=None, on_step=None, randomize_scene=False,
            world_factory=None, on_decision=None, state_transform=None, replay_directory=None, on_action=None,
            staged_prompts=None):
    from embodied_jev.physics import RobotWorld

    from .episode_protocol import episode_protocol
    from .scenes import DISTRIBUTION, scene_config
    protocol = episode_protocol(policy)
    if world_factory is None and protocol["strict_grasp"]:
        from .grasp import GraspWorld, grasp_state
        world_factory, state_transform = GraspWorld, grasp_state
    world = (world_factory or RobotWorld)(task, seed, scene_config(task, seed) if randomize_scene else None)
    if hasattr(policy, "reset"):
        policy.reset()
    poses = None
    if replay_directory is not None:
        from .trajectory_replay import PoseRecorder
        poses, render_frame = PoseRecorder(), on_frame
        def capture_frame(current):
            poses.capture(current)
            if render_frame:
                render_frame(current)
        capture_frame.capture_only = render_frame is None
        on_frame = capture_frame
    controller = None
    staged_prompts = protocol["staged_prompts"] if staged_prompts is None else staged_prompts
    if staged_prompts and task == "transfer":
        from .stage_prompts import STAGES, PickPlaceStages
        controller = PickPlaceStages()
    history, records, latencies = [], [], []
    started = time.perf_counter()
    failure = None
    drops = 0
    if on_frame:
        on_frame(world)
    for step in range(max_steps):
        if poses:
            poses.step = step
        if world.success():
            break
        if intervention and step == 1:
            world.perturb("object_shift", [.015, -.01])
        state, options = state_for(world, history), menu_for(world)
        if state_transform:
            state = state_transform(world, state)
        if controller:
            state["control"] = controller.update(world.observe())
            options = menu_for(world, STAGES[controller.stage][1])
        gold_key = teacher(world) if policy is None or label_mode not in {"none", "rollout"} else None
        if not options or (policy is None and gold_key not in options):
            failure = "no_valid_action"
            break
        criteria = criteria_for(options)
        # Deterministic candidate-order augmentation; IDs remain bound to their actions.
        keys = list(criteria)
        np.random.default_rng(seed * 101 + step).shuffle(keys)
        criteria = {k: criteria[k] for k in keys}
        if on_decision:
            on_decision(world, state, criteria, step)
        before = world.observe()
        t = time.perf_counter()
        probabilities = policy.predict(state, criteria) if policy else None
        latency = (time.perf_counter() - t) * 1000
        latencies.append(latency)
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=float)
            if (len(probabilities) != len(keys) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or abs(probabilities.sum() - 1) > .001):
                raise ValueError("Invalid policy probabilities")
            probabilities = probabilities / probabilities.sum()
            chosen = keys[int(rng.choice(len(keys), p=probabilities)) if rng is not None
                          else int(np.argmax(probabilities))]
        else:
            chosen = gold_key
        if label_mode == "rollout":
            unordered_gold, branch_scores = rollout_labels(world, options)
            gold_by_key = dict(zip(options, unordered_gold))
            gold = [gold_by_key[k] for k in keys]
        elif label_mode == "none":
            branch_scores, gold = None, None
        else:
            branch_scores = None
            gold = [float(k == gold_key) for k in keys]
        if on_action:
            on_action({"step": step, "choice": chosen, "criteria": criteria,
                       "probabilities": probabilities.tolist() if probabilities is not None else None,
                       "latency_ms": latency})
        executed, rejection = execute(world, options[chosen], on_frame)
        after = world.observe()
        lost = before["held"] and not after["held"] and chosen not in {"release", "withdraw"}
        drops += int(lost)
        record = {"step": step, "state": state, "criteria": criteria, "choice": chosen,
                  "instructions": instructions_for(state, INSTRUCTIONS),
                  "probabilities": probabilities.tolist() if probabilities is not None else None,
                  "gold": gold, "label_source": label_mode, "branch_scores": branch_scores,
                  "action": options[chosen].serialise(), "before": before, "after": after,
                  "executed": executed, "rejection": rejection, "latency_ms": latency}
        if getattr(policy, "last_decision", None) is not None:
            record["policy_decision"] = dict(policy.last_decision)
        if hasattr(policy, "observe_outcome"):
            record["policy_feedback"] = policy.observe_outcome({
                "choice": chosen, "executed": executed, "rejection": rejection,
                "before": state["observation"], "after": state_for(world, [])["observation"]})
        if getattr(policy, "last_model_probabilities", None) is not None:
            record["model_probabilities"] = policy.last_model_probabilities.tolist()
            record["exploration"] = policy.last_details
        records.append(record)
        if on_step:
            on_step(record)
        history.append({"choice": chosen, "executed": executed, "rejection": rejection,
                        "tcp_delta": (np.asarray(after["tcp"]) - before["tcp"]).round(5).tolist(),
                        "object_delta": (np.asarray(after["object"]) - before["object"]).round(5).tolist(),
                        "held": after["held"]})
        if rejection == "collision":
            failure = "collision"
            break
        if getattr(policy, "stop_reason", None):
            failure = policy.stop_reason
            break
        if len(history) >= 3 and all(h["choice"] == chosen and
                np.linalg.norm(h["tcp_delta"]) < .001 and np.linalg.norm(h["object_delta"]) < .001
                for h in history[-3:]):
            failure = "stalled"
            break
    success = world.success()
    group = f"{task}:{seed}:{int(intervention)}"
    result = {"schema_version": 1, "task": task, "seed": seed, "group": group,
            "scene_distribution": DISTRIBUTION if randomize_scene else "upstream-default",
            "scene_config": world.scene_config, "scene_hash": world.scene_hash,
            "episode_id": hashlib.sha256(f"{group}:{started}".encode()).hexdigest()[:20],
            "policy_version": POLICY_VERSION, "observation_mode": "privileged", "control_mode": "skills",
            "episode_protocol": {"staged_prompts": bool(controller), "strict_grasp": hasattr(world, "diagnostics")},
            "success": success, "failure": None if success else failure or "budget_exhausted",
            "steps": len(records), "collisions": int(world.unsafe_contacts > 0), "drops": drops,
            "rejections": sum(not r["executed"] for r in records),
            "wall_seconds": time.perf_counter() - started,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            "records": records}
    if getattr(policy, "source_policy", None):
        result["source_policy"] = policy.source_policy
    if hasattr(world, "interventions"):
        result["interventions"] = world.interventions
    if hasattr(world, "diagnostics"):
        result.update(stages=world.diagnostics(), success_definition="strict_grasp_v1",
                      input_contract="grasp-observation-v1")
    if controller:
        result["control_protocol"] = state["control"]["protocol"]
        result["stage_transitions"] = controller.transitions
    if poses:
        from pathlib import Path
        result["replay"] = poses.save(Path(replay_directory), result)
    return result
