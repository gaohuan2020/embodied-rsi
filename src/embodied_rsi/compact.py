"""CPU reference learner. Real supervised training, explicitly NOT RSI-Jev."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sim import PHASES


def features(state, criteria):
    if isinstance(state, str):
        state = json.loads(state)
    o = state["observation"]
    tcp, obj, dest = [np.asarray(o[k], dtype=float) for k in ("tcp", "object", "destination")]
    s = np.r_[1., tcp, obj - tcp, dest - obj, float(o["held"]), float(o["gripper"] == "closed"),
              float(o["support_contact"]), float(o["grasp_secured"])]
    # Numerical measurements + action interactions; no teacher phase or future result.
    rows = []
    for key, description in criteria.items():
        c = json.loads(description)
        phase = c.get("skill", key)
        if phase not in PHASES:
            raise ValueError(f"Unknown skill {phase}")
        onehot = np.asarray([float(phase == p) for p in PHASES])
        delta = np.asarray(c["target_tcp"]) - tcp
        rows.append(np.r_[np.outer(onehot, s).ravel(), delta, np.abs(delta), c["duration_seconds"]])
    return np.stack(rows)


def softmax(logits):
    p = np.exp(logits - np.max(logits))
    return p / p.sum()


class CompactPolicy:
    def __init__(self, weights=None, temperature=1.):
        self.weights = weights
        self.temperature = temperature

    @classmethod
    def load(cls, path):
        with np.load(Path(path) / "weights.npz", allow_pickle=False) as data:
            return cls(data["weights"], float(data["temperature"]))

    def predict(self, state, criteria):
        x = features(state, criteria)
        w = self.weights if self.weights is not None else np.zeros(x.shape[1])
        return softmax(x @ w / self.temperature)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "weights.npz", weights=self.weights, temperature=self.temperature)


def evaluate_cases(policy, cases):
    losses, correct, confidence = [], [], []
    for c in cases:
        p = policy.predict(c["state"], c["questions"][0]["criteria"])
        g = np.asarray(c["gold"]["action"])
        losses.append(-float(g @ np.log(np.maximum(p, 1e-12))))
        correct.append(float(p.argmax() == g.argmax()))
        confidence.append(float(p.max()))
    conf, hit = np.asarray(confidence), np.asarray(correct)
    ece = 0.
    for left in np.arange(0, 1, .1):
        mask = (conf >= left) & (conf < left + .1 if left < .9 else conf <= 1)
        if mask.any():
            ece += float(mask.mean() * abs(conf[mask].mean() - hit[mask].mean()))
    return {"loss": float(np.mean(losses)), "accuracy": float(hit.mean()), "ece": ece}
