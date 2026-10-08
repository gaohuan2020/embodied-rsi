"""Nonblocking collection thresholds; actual fitting runs in the existing job process."""
from pathlib import Path


class AutoTrainer:
    def __init__(self, store, jobs, request_type):
        self.store, self.jobs, self.request_type = store, jobs, request_type
        self.watermarks = {}
        self.state = {"status": "disabled", "round": 0}

    def observe(self, config, collection, backend, checkpoint, run):
        if not config.get("auto_train"):
            return {**self.state, "status": "disabled"}
        if backend == "teacher":
            return {"status": "disabled", "message": "规则示范不计为模型自采成功数据"}
        backlog = config.get("_auto_backlog", {})
        total = collection["episodes"] + backlog.get("episodes", 0)
        complete = collection["successful_episodes"] + backlog.get("successful_episodes", 0)
        episodes, successes = self.watermarks.get(run, config.get("_training_consumed", (0, 0)))
        fresh = total - episodes
        good = complete - successes
        self.state.update(new_episodes=fresh, new_successes=good,
                          episodes_threshold=config["train_every"], successes_threshold=config["min_successes"])
        if any(j["status"] == "running" for j in self.jobs.list()):
            self.state["status"] = "training"
            return dict(self.state)
        if fresh < config["train_every"] or good < config["min_successes"]:
            self.state["status"] = "collecting"
            return dict(self.state)
        parent = None
        if checkpoint and Path(checkpoint).is_dir():
            parent = Path(checkpoint).parent.name
        try:
            job = self.jobs.start(self.request_type(action="self-improve", backend=backend, checkpoint=parent,
                episodes=30, steps=config["train_steps"], rounds=1, replay_only=True))
        except ValueError as exc:
            self.state.update(status="waiting", message=str(exc))
            return dict(self.state)
        self.watermarks[run] = (total, complete)
        config["_training_consumed"] = [total, complete]
        self.state.update(status="training", job=job["id"], round=self.state["round"] + 1)
        self.store.event(run, "auto_training", {**self.state, "trigger": "new_completed_model_episodes"})
        return dict(self.state)
