"""Physical pick-and-place auditing, isolated from the legacy transfer protocol."""
from __future__ import annotations

import copy

import numpy as np
from embodied_jev.physics import DT, RobotWorld


class GraspWorld(RobotWorld):
    def __init__(self, task, seed=0, config=None):
        self.audit = {"grasp": False, "lift": False, "carry": False, "release": False, "drop": False}
        self.contact_gap = 0.
        super().__init__(task, seed, config)

    def tick(self):
        super().tick()
        fingers, _, _ = self.contacts()
        held = len(fingers) == 2 and self.closed
        if self.contact_seconds >= .16:
            self.audit["grasp"] = True
        if self.audit["grasp"] and held and self.cube[2] - self.source[2] >= .03:
            self.audit["lift"] = True
        if self.audit["lift"] and held and np.linalg.norm(self.cube[:2] - self.target[:2]) < .025:
            self.audit["carry"] = True
        if self.audit["lift"] and not self.audit["release"]:
            self.contact_gap = 0. if held else self.contact_gap + DT
            if self.contact_gap >= .05:
                self.audit["drop"] = True

    def motion(self, target=None, gripper=None, seconds=.6, emit=True):
        # A small clearance before opening is normal; support is checked after release.
        if (gripper == "open" and self.audit["carry"]
                and np.linalg.norm(self.cube[:2] - self.target[:2]) < .025
                and abs(self.cube[2] - self.target[2]) < .012):
            self.audit["release"] = True
        yield from super().motion(target, gripper, seconds, emit)

    def success(self):
        return bool(all(self.audit[k] for k in ("grasp", "lift", "carry", "release"))
                    and not self.audit["drop"] and self.unsafe_contacts == 0 and super().success())

    def clone(self):
        base = super().clone()
        other = object.__new__(type(self))
        other.__dict__ = base.__dict__
        other.audit = copy.deepcopy(self.audit)
        return other

    def diagnostics(self):
        return {**self.audit, "place": bool(super().success()), "success": self.success(),
                "max_lift_m": float(self.max_lift), "contact_gap_s": self.contact_gap}


def grasp_state(world, state):
    """One shared training/inference contract; no seed, return or future preview."""
    state = copy.deepcopy(state)
    state["observation"]["relative_geometry"] = world.observe()["relative_geometry"]
    state["environment"] = {
        "robot": "Franka Panda", "observation_source": "privileged MuJoCo state",
        "object_size_m": (2 * world.model.geom_size[world.cube_geom]).tolist(),
        "object_mass_kg": float(world.model.body_mass[world.model.geom_bodyid[world.cube_geom]]),
        "object_friction": world.model.geom_friction[world.cube_geom].tolist(),
        "tray_support_size_m": (2 * world.model.geom_size[world.support_id]).tolist(),
        "required_lift_m": .03, "required_stable_grasp_s": .16,
        "required_stable_place_s": .4,
        "workspace_min_m": [.20, -.32, .02], "workspace_max_m": [.65, .34, .42],
    }
    return state
