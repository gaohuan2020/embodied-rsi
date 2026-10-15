"""Keep a model's simulator input contract across training and service restarts."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .stage_prompts import SUPPORTED_PROTOCOLS


def episode_protocol(policy=None):
    saved = getattr(policy, "episode_protocol", None) or {}
    return {key: saved.get(key, os.environ.get(variable) == "1") for key, variable in (
        ("staged_prompts", "EMBODIED_RSI_STAGED_PROMPTS"),
        ("strict_grasp", "EMBODIED_RSI_STRICT_GRASP"),
    )}


def rsi_default_protocol():
    """New RSI robot runs use the measured pick/place contract by default."""
    return {key: os.environ.get(variable, "1") == "1" for key, variable in (
        ("staged_prompts", "EMBODIED_RSI_STAGED_PROMPTS"),
        ("strict_grasp", "EMBODIED_RSI_STRICT_GRASP"),
    )}


def replay_protocol(cases):
    states = [json.loads(c["state"]) if isinstance(c["state"], str) else c["state"] for c in cases]
    transfer = [s for s in states if s.get("task_id", "transfer") == "transfer"]
    staged = {s.get("control", {}).get("protocol") in SUPPORTED_PROTOCOLS for s in transfer}
    strict = {"required_stable_grasp_s" in s.get("environment", {}) for s in states}
    if len(staged) > 1 or len(strict) > 1:
        raise ValueError("Success replay mixes incompatible episode input protocols")
    return {"staged_prompts": next(iter(staged), False), "strict_grasp": next(iter(strict), False)}


def checkpoint_protocol(checkpoint, defaults=None):
    path = Path(checkpoint) / "meta.json"
    if not path.is_file():
        return defaults if defaults is not None else episode_protocol()
    meta = json.loads(path.read_text())
    if "episode_protocol" in meta:
        return meta["episode_protocol"]
    if "episode_protocol" in meta.get("robot_training", {}):
        return meta["robot_training"]["episode_protocol"]
    # Older releases did not save the flags. Recover them from the recorded
    # training inputs without rewriting the checkpoint or invalidating its hash.
    dataset = meta.get("dataset") or meta.get("robot_training", {}).get("dataset")
    if dataset:
        training = Path(dataset) / "train.jsonl"
        if training.is_file():
            return replay_protocol([json.loads(line) for line in training.read_text().splitlines() if line])
        raise ValueError("Legacy checkpoint input protocol is missing; restore its recorded training dataset")
    return defaults if defaults is not None else episode_protocol()
