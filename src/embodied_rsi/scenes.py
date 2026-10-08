"""Reproducible domain randomization within the upstream robot workspace."""
import numpy as np

DISTRIBUTION = "workspace-random-v1"


def scene_config(task, seed):
    rng = np.random.default_rng(seed)
    config = {"name": DISTRIBUTION,
              "source_xy": (np.array([.43, -.17]) + rng.uniform([-.035, -.03], [.035, .03])).tolist(),
              "target_xy": (np.array([.43, .18]) + rng.uniform([-.025, -.02], [.025, .02])).tolist()}
    if task == "barrier":
        config["barrier_height"] = float(rng.uniform(.08, .12))
    return config


def task_for(seed):
    tasks = np.array(["transfer", "stack", "barrier"])
    np.random.default_rng(seed // 3).shuffle(tasks)
    return str(tasks[seed % 3])
