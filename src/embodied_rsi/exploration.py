"""Exploration learns from observed failures; release evaluation uses the raw model."""
from __future__ import annotations

import json

import numpy as np


def context(state):
    o = state["observation"]
    delta = np.asarray(o["object"]) - o["tcp"]
    bins = np.clip(np.rint(delta / .035), -12, 12).astype(int).tolist()
    bucket = [bins, bool(o["held"]), o["gripper"], bool(o["support_contact"])]
    if state.get("control"):
        destination = np.asarray(o["destination"]) - o["object"]
        bucket.extend([state["control"]["protocol"], state["control"]["stage"],
                       np.clip(np.rint(destination / .05), -12, 12).astype(int).tolist()])
    return json.dumps(bucket)


class ExplorationMemory:
    def __init__(self, store, backend="rsi"):
        self.store, self.backend = store, backend
        with store.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS exploration_actions (backend TEXT, task TEXT, context TEXT, "
                       "action TEXT, trials INTEGER, successes INTEGER, failures REAL, "
                       "PRIMARY KEY(backend,task,context,action))")
            db.execute("CREATE TABLE IF NOT EXISTS explored_episodes (backend TEXT, episode_id TEXT, "
                       "PRIMARY KEY(backend,episode_id))")

    def import_past(self):
        for run in self.store.runs(limit=None):
            file = self.store.run_dir(run["id"]) / "episodes.jsonl"
            if run["kind"] == "explore" and run["backend"] == self.backend and file.exists():
                with file.open() as source:
                    for line in source:
                        if line.endswith("\n"):
                            self.observe(json.loads(line))

    def stats(self, state, keys):
        task = state.get("task_id", state["observation"].get("task", "transfer"))
        bucket = context(state)
        with self.store.connect() as db:
            rows = db.execute("SELECT action,trials,successes,failures FROM exploration_actions "
                              "WHERE backend=? AND task=? AND context=?", (self.backend, task, bucket)).fetchall()
        by_action = {r["action"]: dict(r) for r in rows}
        return [by_action.get(k, {"action": k, "trials": 0, "successes": 0, "failures": 0.}) for k in keys]

    def observe(self, result):
        diagnosed = 0
        rows = result["records"]
        with self.store.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO explored_episodes VALUES (?,?)",
                                  (self.backend, result["episode_id"]))
            if inserted.rowcount == 0:
                return {"diagnosed_failure_actions": 0, "already_observed": True, "backend": self.backend}
            for index, row in enumerate(rows):
                before, after = row["before"], row["after"]
                bad_grasp = row["choice"] == "grasp" and not after["held"]
                safe_placement = (np.linalg.norm(np.asarray(before["object"])[:2]
                                  - np.asarray(before["destination"])[:2]) < .025
                                  and abs(before["object"][2] - before["destination"][2]) < .012)
                drop = before["held"] and not after["held"] and not safe_placement
                no_change = (np.linalg.norm(np.asarray(after["tcp"]) - before["tcp"]) < .001 and
                             np.linalg.norm(np.asarray(after["object"]) - before["object"]) < .001 and
                             before["gripper"] == after["gripper"])
                failure = float(not row["executed"] or bad_grasp or drop or
                                (no_change and not result["success"]))
                if not result["success"] and index == len(rows) - 1:
                    failure = max(failure, .25)
                diagnosed += int(failure > 0)
                db.execute("INSERT INTO exploration_actions VALUES (?,?,?,?,1,?,?) "
                           "ON CONFLICT(backend,task,context,action) DO UPDATE SET "
                           "trials=trials+1,successes=successes+excluded.successes,failures=failures+excluded.failures",
                           (self.backend, result["task"], context(row["state"]), row["choice"],
                            int(result["success"]), failure))
        return {"diagnosed_failure_actions": diagnosed, "backend": self.backend,
                "strategy": "state-local failure penalty + uncertainty bonus + 5% exploration"}

    def overview(self):
        with self.store.connect() as db:
            row = db.execute("SELECT COUNT(*) contexts,SUM(trials) trials,SUM(failures) failures "
                             "FROM exploration_actions WHERE backend=?", (self.backend,)).fetchone()
        return dict(row)


class ExploringPolicy:
    def __init__(self, model, memory):
        self.model, self.memory = model, memory
        self.last_model_probabilities = None
        self.last_details = None

    @property
    def episode_protocol(self):
        from .episode_protocol import episode_protocol
        return episode_protocol(self.model)

    def predict(self, state, criteria):
        p = np.asarray(self.model.predict(state, criteria), dtype=float)
        stats = self.memory.stats(state, list(criteria))
        trials = np.asarray([r["trials"] for r in stats])
        failures = np.asarray([r["failures"] for r in stats])
        logits = np.log(np.maximum(p, 1e-12)) / 1.1 + .3 / np.sqrt(trials + 1) - 1.5 * failures / (trials + 1)
        adjusted = np.exp(logits - logits.max())
        adjusted = .95 * adjusted / adjusted.sum() + .05 / len(p)
        self.last_model_probabilities = p
        self.last_details = {"strategy": "failure_guided", "temperature": 1.1, "epsilon": .05,
                             "penalized_actions": [list(criteria)[i] for i, f in enumerate(failures) if f > 0]}
        return adjusted
