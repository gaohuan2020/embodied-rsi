from __future__ import annotations

import httpx
import numpy as np

from .sim import INSTRUCTIONS
from .stage_prompts import decision_input


class HTTPPolicy:
    def __init__(self, url, model="jev-latest", api_key=None):
        self.url, self.model = url, model
        self.client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def close(self):
        self.client.close()

    def predict(self, state, criteria):
        prompt, instructions, descriptions = decision_input(state, criteria, INSTRUCTIONS)
        response = self.client.post(self.url, json={"model": self.model, "state": prompt,
            "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": descriptions}}})
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

        from .episode_protocol import checkpoint_protocol, rsi_default_protocol
        self.episode_protocol = checkpoint_protocol(checkpoint, rsi_default_protocol())
        # Match the differentiable learner: otherwise a bf16 serving tower can
        # choose a different argmax from the fp32 model used for development.
        self.decider = Decider(checkpoint, device=device, dtype="fp32", revision=revision)

    def predict(self, state, criteria):
        prompt, instructions, descriptions = decision_input(state, criteria, INSTRUCTIONS)
        answers = self.decider.decide(prompt, {
            "action": {"type": "choice", "instructions": instructions, "criteria": descriptions}})
        return np.asarray([answers["action"]["probabilities"][k] for k in criteria])
