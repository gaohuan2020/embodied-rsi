"""On-policy REINFORCE: every decision receives the complete episode's terminal reward.

Teacher labels and single-step accuracy are never read by this optimizer or its
checkpoint selector. Supervised initialization remains a separate, optional stage.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .compact import CompactPolicy, features
from .monitor import Monitor
from .sim import episode
from .storage import dumps, write_json

TASKS = ("transfer", "stack", "barrier")


def terminal_reward(result):
    return float(bool(result["success"]))


def task_score(results):
    """Completion first; safety breaks ties. No teacher agreement or step loss."""
    return (float(np.mean([r["success"] for r in results])),
            -sum(r[k] for r in results for k in ("collisions", "drops", "rejections")))


class CompactLearner:
    def __init__(self, parent=None, lr=.03):
        self.policy = CompactPolicy.load(parent) if parent else CompactPolicy()
        self.lr = lr

    def predict(self, state, criteria):
        if self.policy.weights is None:
            self.policy.weights = np.zeros(features(state, criteria).shape[1])
        return self.policy.predict(state, criteria)

    def snapshot(self):
        return copy.deepcopy(self.policy)

    def restore(self, state):
        self.policy = copy.deepcopy(state)

    def update(self, trajectories, advantages):
        gradient = np.zeros_like(self.policy.weights)
        objective = 0.
        for trajectory, advantage in zip(trajectories, advantages):
            for row in trajectory["records"]:
                x = features(row["state"], row["criteria"])
                p = np.asarray(row["probabilities"])
                action = list(row["criteria"]).index(row["choice"])
                target = np.zeros(len(p))
                target[action] = 1.
                gradient += advantage * (x.T @ (target - p)) / self.policy.temperature
                objective += -advantage * np.log(max(p[action], 1e-12))
        gradient /= len(trajectories)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite task-policy gradient")
        self.policy.weights += self.lr * gradient / max(1., norm)
        return {"policy_loss": float(objective / len(trajectories)), "grad_norm": norm}

    def save(self, path, metadata):
        self.policy.save(path)
        write_json(path / "meta.json", {"backend": "compact-numpy", **metadata})

    def encode_success(self, row):
        return features(row["state"], row["criteria"])

    def imitate(self, batch):
        gradient = np.zeros_like(self.policy.weights)
        loss = 0.
        from .compact import softmax
        for x, choice in batch:
            p = softmax(x @ self.policy.weights / self.policy.temperature)
            target = np.zeros(len(p))
            target[choice] = 1.
            loss -= np.log(max(p[choice], 1e-12)) / len(batch)
            gradient += x.T @ (p - target) / (len(batch) * self.policy.temperature)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite successful-trajectory gradient")
        self.policy.weights -= self.lr * gradient / max(1., norm)
        return {"loss": float(loss), "grad_norm": norm}


def train_tasks(store, parent=None, backend="compact", steps=6, seed_start=1000000,
                dev_start=1010000, batch_size=3, dev_episodes=3, lr=None, revision=None):
    if not 1 <= steps <= 1000 or not 1 <= batch_size <= 30 or not 1 <= dev_episodes <= 30:
        raise ValueError("Task budgets out of range")
    if steps * batch_size > 10000 or abs(dev_start - seed_start) < 10000:
        raise ValueError("Training and development scene namespaces must be disjoint")
    config = {"objective": "episode_success", "algorithm": "REINFORCE",
              "parent": str(parent) if parent else None, "steps": steps, "batch_size": batch_size,
              "train_seed_range": [seed_start, seed_start + 10000],
              "dev_seed_range": [dev_start, dev_start + dev_episodes], "revision": revision}
    run_id = store.create("task-train", backend, config)
    directory = store.run_dir(run_id)
    started = time.perf_counter()
    try:
        with Monitor(store, run_id) as monitor:
            if backend == "compact":
                learner = CompactLearner(parent, lr or .03)
            else:
                from .task_rsi import RSILearner
                learner = RSILearner(parent or "v3.0-2b", lr or 1e-4, revision)
            rng = np.random.default_rng(seed_start)

            def validate(step):
                rows = [episode(task, seed, learner, label_mode="none") for task in TASKS
                        for seed in range(dev_start, dev_start + dev_episodes)]
                score = task_score(rows)
                monitor.metric(step, dev_success_rate=score[0], dev_risk_count=-score[1])
                store.event(run_id, "task_validation", {"step": step, "success_rate": score[0],
                    "by_task": {t: float(np.mean([r["success"] for r in rows if r["task"] == t]))
                                for t in TASKS}, "risk_count": -score[1]})
                return rows, score

            initial, best_score = validate(0)
            best, best_step = learner.snapshot(), 0
            # A smoothed, action-independent baseline avoids zero updates for the
            # first all-failure batch. It is a control variate, never a step label.
            baseline = {t: (sum(r["success"] for r in initial if r["task"] == t) + 1.) /
                        (dev_episodes + 2.) for t in TASKS}
            with (directory / "episodes.jsonl").open("w") as output:
                for step in range(1, steps + 1):
                    trajectories = []
                    for task in TASKS:
                        for index in range(batch_size):
                            seed = seed_start + (step - 1) * batch_size + index
                            result = episode(task, seed, learner, label_mode="none", rng=rng)
                            trajectories.append(result)
                            output.write(dumps(result) + "\n")
                            output.flush()
                            store.event(run_id, "episode", {k: v for k, v in result.items() if k != "records"})
                    rewards = [terminal_reward(r) for r in trajectories]
                    if not any(rewards):
                        monitor.alert("sparse_task_reward",
                                      "本批完整任务全部失败；稀疏奖励可能停滞，建议增加成功示范预热")
                    advantages = [reward - baseline[r["task"]] for reward, r in zip(rewards, trajectories)]
                    try:
                        metrics = learner.update(trajectories, advantages)
                    except (FloatingPointError, RuntimeError):
                        monitor.alert("task_update_failed", "策略梯度更新失败，已中止该轮训练", "critical")
                        raise
                    for task in TASKS:
                        observed = np.mean([terminal_reward(r) for r in trajectories if r["task"] == task])
                        baseline[task] = .8 * baseline[task] + .2 * float(observed)
                    monitor.metric(step, **metrics, train_success_rate=float(np.mean(rewards)),
                                   episode_reward=float(np.mean(rewards)),
                                   episodes=len(trajectories) * step,
                                   steps_per_second=step / (time.perf_counter() - started))
                    _rows, score = validate(step)
                    if score[0] < best_score[0] - .05:
                        monitor.alert("task_completion_regression",
                                      "开发场景任务完成率下降；保留完成率更高的检查点")
                    if score > best_score:
                        best_score, best, best_step = score, learner.snapshot(), step
            learner.restore(best)
            summary = {"objective": "episode_success", "initial_success_rate": task_score(initial)[0],
                       "success_rate": best_score[0], "selected_step": best_step,
                       "episodes": steps * batch_size * len(TASKS),
                       "checkpoint": str(directory / "checkpoint"),
                       "promotion": "pending_independent_task_evaluation",
                       "train_seconds": time.perf_counter() - started}
            learner.save(directory / "checkpoint", {**config, **summary})
            write_json(directory / "summary.json", summary)
        store.finish(run_id, summary=summary)
        return run_id
    except BaseException as exc:
        store.finish(run_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise
