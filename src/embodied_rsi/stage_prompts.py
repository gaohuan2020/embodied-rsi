"""Measured pick/place stages and a shared RSI prompt for acting and fitting."""
from __future__ import annotations

import json

import numpy as np

PROTOCOL = "pick-place-coarse-v2"
SUPPORTED_PROTOCOLS = {PROTOCOL, "pick-place-stages-v1"}
STAGES = {
    "acquire": (("Obtain a stable grasp on the object and raise it clear of the source surface. "
                 "Decide the next movement and finger command from the current observation."),
                ("approach", "descend", "grasp", "lift", "recover")),
    "transport": (("Move the held object to the destination while maintaining grasp and clearance. "
                   "Decide whether to raise, transport, reposition, or recover from the current observation."),
                  ("lift", "carry", "lower", "grasp", "recover")),
    "place": (("Leave the object resting stably on destination support, released from the fingers, "
               "with the empty gripper clear of it. Decide the next skill from the current observation."),
              ("carry", "lower", "release", "withdraw", "grasp", "recover")),
}
DESCRIPTIONS = {
    "approach": "Move the open empty gripper horizontally to align above the object on the table.",
    "descend": "Move the open empty gripper vertically down from above the object to surround it at grasp height.",
    "grasp": "Close the open fingers that are already surrounding the object at grasp height.",
    "lift": "Raise the object already held in the closed fingers from the table to safe travel height.",
    "carry": "Transport the object already held up at safe travel height horizontally to the destination.",
    "lower": "Lower the held object already above the destination onto its support.",
    "release": "Open the fingers to let go of the held object already lowered onto the destination support.",
    "withdraw": "Raise the empty open gripper after the object has been released on destination support.",
    "recover": "Open the closed empty gripper after a failed grasp before trying again.",
}


class PickPlaceStages:
    def __init__(self):
        self.stage = "acquire"
        self.transitions = []
        self.source_height = None
        self.acquired = False

    def update(self, observation):
        o = observation
        obj, dest = (np.asarray(o[k], dtype=float) for k in ("object", "destination"))
        if self.source_height is None:
            self.source_height = float(obj[2])
        centered = np.linalg.norm(obj[:2] - dest[:2]) < .025
        placement_height = abs(obj[2] - dest[2]) < .012
        # Transitions use measured effects, never the chosen action or teacher labels.
        if o["held"] and o["grasp_secured"] and obj[2] - self.source_height >= .03:
            self.acquired = True
        if not o["held"]:
            stage = ("place" if self.stage == "place" and centered and placement_height
                     and o["support_contact"] else "acquire")
        elif not o["grasp_secured"] or not self.acquired:
            stage = "acquire"
        elif centered:
            stage = "place"
        else:
            stage = "transport"
        if not self.transitions or stage != self.stage:
            self.transitions.append({"from": self.stage if self.transitions else None, "to": stage})
        self.stage = stage
        return {"protocol": PROTOCOL, "stage": stage, "goal": STAGES[stage][0]}


def instructions_for(state, fallback):
    control = state.get("control", {}) if isinstance(state, dict) else {}
    if control.get("protocol") not in SUPPORTED_PROTOCOLS:
        return fallback
    return "Choose the skill that achieves this stage goal: " + control["goal"]


def decision_input(state, criteria, fallback):
    """Project recorded facts into exactly the same prompt in serving and training."""
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
        except json.JSONDecodeError:
            return state, fallback, criteria
    else:
        parsed = state
    instructions = instructions_for(parsed, fallback)
    if (not isinstance(parsed, dict)
            or parsed.get("control", {}).get("protocol") not in SUPPORTED_PROTOCOLS):
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False), instructions, criteria
    o = parsed["observation"]
    tcp, obj, dest = (np.asarray(o[k], dtype=float) for k in ("tcp", "object", "destination"))
    centered = np.linalg.norm(obj[:2] - dest[:2]) < .025
    placement_height = abs(obj[2] - dest[2]) < .012
    travel = o.get("relative_geometry", {}).get("travel_tcp_height_m", .22)
    sentences = [("Goal: pick up the red block and place it in the blue tray, "
                  "then open the fingers and raise the empty gripper.")]
    if o["held"]:
        sentences.append("The gripper is CLOSED and is securely HOLDING the block between both fingers."
                         if o["grasp_secured"] else "The closed fingers contact the block; grasp is not yet stable.")
        sentences.append("The held block is directly over the destination tray." if centered else
                         "The held block is away from the destination tray.")
        if centered:
            sentences.append("The held block is lowered onto the destination support." if placement_height else
                             "The held block is still above the destination support.")
        else:
            sentences.append("The held block is up at safe travel height." if tcp[2] >= travel - .02 else
                             "The held block is still low near the table, below safe travel height.")
    else:
        sentences.append("The gripper is EMPTY and " + o["gripper"].upper() + ". The block is NOT held.")
        sentences.append("The block has been released and rests on the destination tray support."
                         if centered and o["support_contact"] else
                         "The block is away from the destination tray.")
        if np.linalg.norm(tcp[:2] - obj[:2]) >= .007:
            sentences.append("The gripper is horizontally away from the block, not aligned above it.")
        elif abs(tcp[2] - obj[2]) < .006:
            sentences.append("The open fingers are aligned around the block at grasp height."
                             if o["gripper"] == "open" else
                             "The closed empty fingers are aligned around the block at grasp height.")
        else:
            sentences.append("The gripper is aligned directly above the block, but its fingers are too high to grasp it.")
    return "\n".join(sentences), instructions, {key: DESCRIPTIONS[key] for key in criteria}
