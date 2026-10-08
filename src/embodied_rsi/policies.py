from __future__ import annotations

import json

import httpx
import numpy as np

from .sim import INSTRUCTIONS


class HTTPPolicy:
    def __init__(self, url, model="jev-latest", api_key=None):
        self.url, self.model = url, model
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def close(self):
        self.client.close()

    def predict(self, state, criteria):
        response = self.client.post(self.url, json={"model": self.model, "state": state,
            "questions": {"action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}})
        response.raise_for_status()
        answer = response.json()["answers"]["action"]
        probabilities = answer["probabilities"]
        if set(probabilities) != set(criteria):
            raise ValueError("Model returned a different candidate set")
        p = np.asarray([probabilities[k] for k in criteria], dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > .02:
            raise ValueError("Model returned invalid probabilities")
        if answer["choice"] not in criteria or probabilities[answer["choice"]] < max(p) - 1e-7:
            raise ValueError("Model choice does not match probability maximum")
        return p / p.sum()


class InProcessRSI:
    def __init__(self, checkpoint, device=None, revision=None):
        from rsijev import Decider
        self.decider = Decider(checkpoint, device=device, revision=revision)

    def predict(self, state, criteria):
        answers = self.decider.decide(json.dumps(state, ensure_ascii=False), {
            "action": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}})
        return np.asarray([answers["action"]["probabilities"][k] for k in criteria])
