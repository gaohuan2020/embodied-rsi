"""Observation-only pick/place rules with explicit, episode-local recovery state."""
from __future__ import annotations

import numpy as np

from .stage_prompts import PickPlaceStages

CODE_PROTOCOL = "stateful-transfer-v1"


class CodePolicy:
    source_policy = "code"

    def __init__(self, max_retries=2):
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.max_retries = max_retries
        self.reset()

    def reset(self):
        self.stages = PickPlaceStages()
        self.retry_count = 0
        self.recovery_phase = None
        self.stop_reason = None
        self.last_decision = None

    def _plan(self, state):
        if state.get("task_id") != "transfer":
            raise ValueError("CodePolicy currently supports transfer only")
        o = state["observation"]
        tcp, obj, dest = (np.asarray(o[k], dtype=float) for k in ("tcp", "object", "destination"))
        stage = self.stages.update(o)["stage"]
        centered = np.linalg.norm(obj[:2] - dest[:2]) < .025
        placed_height = abs(obj[2] - dest[2]) < .012
        if self.recovery_phase:
            choice, eligible = self.recovery_phase, (self.recovery_phase,)
        elif not o["held"]:
            if o["gripper"] == "closed":
                choice, eligible = "recover", ("recover",)
            elif centered and placed_height and o["support_contact"] and stage == "place":
                choice, eligible = "withdraw", ("withdraw",)
            elif np.linalg.norm(tcp[:2] - obj[:2]) >= .007:
                choice, eligible = "approach", ("approach",)
            elif abs(tcp[2] - obj[2]) >= .006:
                choice, eligible = "descend", ("descend", "approach")
            else:
                choice, eligible = "grasp", ("grasp", "approach")
        elif centered:
            if placed_height:
                choice, eligible = "release", ("release", "lower")
            else:
                choice, eligible = "lower", ("lower", "carry")
        else:
            travel = o.get("relative_geometry", {}).get("travel_tcp_height_m", .22)
            if tcp[2] < travel - .02 or not self.stages.acquired:
                choice, eligible = "lift", ("lift",)
            else:
                choice, eligible = "carry", ("carry", "lift")
        return choice, eligible, stage

    def predict(self, state, criteria):
        choice, eligible, stage = self._plan(state)
        if choice not in criteria:
            raise ValueError(f"Required code skill {choice!r} is unavailable")
        self.last_decision = {"protocol": CODE_PROTOCOL, "stage": stage,
                              "retry_count": self.retry_count, "recovering": bool(self.recovery_phase),
                              "eligible": list(eligible), "choice": choice, "model_calls": 0}
        return np.asarray([float(key == choice) for key in criteria])

    def observe_outcome(self, outcome):
        """Only action feedback and whitelisted measured facts enter this hook."""
        before, after = outcome["before"], outcome["after"]
        choice = outcome["choice"]
        reason = None
        centered = np.linalg.norm(np.asarray(before["object"])[:2] - before["destination"][:2]) < .025
        supported_release = (choice == "release" and centered
                             and abs(before["object"][2] - before["destination"][2]) < .012)
        if not outcome["executed"]:
            reason = "rejected_action"
        elif choice == "grasp" and not after["held"]:
            reason = "empty_grasp"
        elif before["held"] and not after["held"] and not supported_release:
            reason = "lost_grasp"
        if reason:
            self.retry_count += 1
            # A new acquisition must establish lift evidence again after a drop.
            if reason in {"empty_grasp", "lost_grasp"}:
                self.stages = PickPlaceStages()
            if self.retry_count > self.max_retries:
                self.stop_reason = "recovery_budget_exhausted"
            else:
                self.recovery_phase = ("lift" if after["held"] else
                                       "recover" if after["gripper"] == "closed" else "approach")
        elif self.recovery_phase and choice == self.recovery_phase:
            self.recovery_phase = "approach" if choice == "recover" else None
        return {"failure_reason": reason, "retry_count": self.retry_count,
                "next_recovery_skill": self.recovery_phase, "stop_reason": self.stop_reason}


class HybridPolicy(CodePolicy):
    """Code handles prerequisites/recovery; a model ranks remaining skill choices."""
    source_policy = "hybrid"

    def __init__(self, model, max_retries=2):
        self.model = model
        super().__init__(max_retries)

    def reset(self):
        super().reset()
        if hasattr(getattr(self, "model", None), "reset"):
            self.model.reset()

    def predict(self, state, criteria):
        result = super().predict(state, criteria)
        legal = [key for key in self.last_decision["eligible"] if key in criteria]
        if self.recovery_phase or len(legal) < 2:
            return result
        probabilities = np.asarray(self.model.predict(state, criteria), dtype=float)
        if (probabilities.shape != result.shape or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or abs(probabilities.sum() - 1) > .001):
            raise ValueError("Invalid hybrid model probabilities")
        masked = probabilities * np.asarray([key in legal for key in criteria])
        if masked.sum() > 0:
            result = masked / masked.sum()
        self.last_decision.update(choice=list(criteria)[int(np.argmax(result))], model_calls=1)
        return result
