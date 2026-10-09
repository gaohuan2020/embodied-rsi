"""Rebuild comparable round records from durable simulator and training events."""
from __future__ import annotations

import csv
import io
import json
import threading
import time
from collections import defaultdict

from .storage import write_json


def round_history(store):
    with store.connect() as db:
        db.execute("BEGIN")
        runs = []
        for row in db.execute("SELECT * FROM runs ORDER BY created, id"):
            r = dict(row)
            r["config"], r["summary"] = json.loads(r["config"]), json.loads(r["summary"])
            runs.append(r)
        events = defaultdict(list)
        for row in db.execute("SELECT * FROM events WHERE kind IN ('metric','episode','pair','stage','auto_training') ORDER BY seq"):
            events[row["run_id"]].append({**dict(row), "payload": json.loads(row["payload"])})

    deployments = [r for r in runs if r["kind"] == "deploy" and r["status"] == "completed"
                   and r["summary"].get("backend") == "rsi"]
    versions = {r["summary"].get("version", r["id"]): i + 1 for i, r in enumerate(deployments)}
    by_checkpoint = {r["summary"].get("checkpoint"): r for r in deployments}
    evaluations = defaultdict(list)
    for r in runs:
        if r["kind"] == "evaluate":
            evaluations[r["config"].get("candidate")].append(r)
    training = []
    for r in runs:
        if r["kind"] != "self-train" or r["backend"] != "rsi":
            continue
        metrics = [e["payload"] for e in events[r["id"]] if e["kind"] == "metric"]
        losses = [{"step": m["step"], "loss": m["loss"]} for m in metrics if "loss" in m]
        dev = [{"step": m["step"], "success_rate": m["dev_success_rate"]}
               for m in metrics if "dev_success_rate" in m]
        s = r["summary"]
        checkpoint = str(store.run_dir(r["id"]) / "checkpoint")
        evaluation = (evaluations.get(checkpoint) or [None])[-1]
        deployment = by_checkpoint.get(checkpoint)
        pairs = [e["payload"] for e in events[evaluation["id"]] if e["kind"] == "pair"] if evaluation else []
        report = evaluation["summary"] if evaluation else {}
        finished = bool(evaluation and evaluation["status"] == "completed")
        selected = s.get("selected_step")
        row = {
            "round": len(training) + 1, "training": r["id"], "created": r["created"],
            "status": r["status"], "dataset": r["config"].get("dataset"),
            "training_samples": s.get("training_samples", next((m["training_samples"] for m in reversed(metrics)
                                                                  if "training_samples" in m), None)),
            "successful_episodes": s.get("successful_episodes", next((m["successful_episodes"] for m in reversed(metrics)
                                                                        if "successful_episodes" in m), None)),
            "updates": r["config"].get("steps"), "logged_updates": len(losses),
            "loss_first": losses[0]["loss"] if losses else None,
            "loss_last": losses[-1]["loss"] if losses else None,
            "loss_mean": sum(m["loss"] for m in losses) / len(losses) if losses else None,
            "loss_selected_step": next((m["loss"] for m in losses if m["step"] == selected), None),
            "selected_step": selected, "loss_curve": losses, "dev_curve": dev,
            "dev_before": s.get("initial_success_rate", dev[0]["success_rate"] if dev else None),
            "dev_selected": s.get("success_rate"),
            "evaluation": evaluation["id"] if evaluation else None,
            "evaluation_status": evaluation["status"] if evaluation else None,
            "evaluated_pairs": len(pairs),
            "total_pairs": len(evaluation["config"].get("tasks", [])) * len(evaluation["config"].get("seeds", [])) if evaluation else None,
            "candidate_success_rate": report.get("success_rate") if finished else
                sum(bool(p["candidate_success"]) for p in pairs) / len(pairs) if pairs else None,
            "parent_success_rate": report.get("champion_success_rate") if finished else
                sum(bool(p["champion_success"]) for p in pairs) / len(pairs) if pairs else None,
            "paired_gain": report.get("paired_gain"), "gain_ci95": report.get("gain_ci95"),
            "gate_passed": report.get("gate", {}).get("passed"),
            "deployment": deployment["id"] if deployment else None,
            "model_round": versions.get(deployment["id"]) if deployment else None,
        }
        training.append(row)

    # Use the model pinned to each completed episode, including across service restarts.
    # Never charge an interrupted episode or a teacher demonstration as a failure.
    groups, seen = {}, set()
    for r in runs:
        if r["kind"] != "explore" or r["backend"] != "rsi" or not r["config"].get("continuous"):
            continue
        for e in events[r["id"]]:
            p = e["payload"]
            if e["kind"] != "episode" or p.get("source_policy") != "rsi" or "success" not in p:
                continue
            key = p.get("episode_id") or (r["id"], e["seq"])
            if key in seen:
                continue
            seen.add(key)
            version = p.get("model_version", "unknown")
            group = groups.setdefault(version, {
                "model_version": version, "model_round": versions.get(version, 0 if "v1.0-0.8b" in version else None),
                "episodes": 0, "successes": 0, "failures": 0, "success_rate": None,
                "first_at": e["ts"], "last_at": e["ts"], "sessions": [], "curve": [],
                "protocols": [],
            })
            group["episodes"] += 1
            group["successes"] += int(bool(p["success"]))
            group["failures"] += int(not p["success"])
            group["success_rate"] = group["successes"] / group["episodes"]
            group["last_at"] = e["ts"]
            if r["id"] not in group["sessions"]:
                group["sessions"].append(r["id"])
            protocol = "stochastic" if r["config"].get("explore") else "greedy"
            if protocol not in group["protocols"]:
                group["protocols"].append(protocol)
            group["curve"].append({"episode": group["episodes"], "successes": group["successes"],
                                   "success_rate": group["success_rate"], "ts": e["ts"],
                                   "episode_id": p.get("episode_id"), "seed": p.get("seed"), "success": bool(p["success"])})

    active = [r for r in runs if r["status"] == "running" and r["kind"] in {"self-improve", "self-train", "evaluate", "deploy"}]
    phases = []
    for r in active:
        stage = next((e["payload"].get("stage") for e in reversed(events[r["id"]]) if e["kind"] == "stage"), None)
        phases.append({"run": r["id"], "kind": r["kind"], "stage": stage,
                       "last_update": r["updated"], "stale": time.time() - r["updated"] > 90})
    return {"generated_at": time.time(), "training": training, "collection": list(groups.values()), "active": phases,
            "notes": {"loss": "Minibatch loss; historical updates may have been logged every 5 steps. Missing points are not interpolated.",
                      "collection": "Completed RSI simulator episodes grouped by the actual pinned model version; stochastic exploration is not release evaluation.",
                      "evaluation": "Each round uses a fresh paired scene suite. Compare candidate and parent within that round."}}


class RoundRecorder:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.directory = store.root / "metrics"

    def refresh(self):
        with self.lock:
            report = round_history(self.store)
            write_json(self.directory / "rounds.json", report)
            for name, rows, omitted in (
                ("training-rounds", report["training"], {"loss_curve", "dev_curve"}),
                ("collection-rounds", report["collection"], {"curve"}),
            ):
                records = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                            for k, v in r.items() if k not in omitted} for r in rows]
                self._csv(name, records)
            self._csv("training-loss", [{"round": r["round"], "training": r["training"], **point}
                                         for r in report["training"] for point in r["loss_curve"]])
            self._csv("collection-episodes", [{"model_version": r["model_version"], "model_round": r["model_round"], **point}
                                               for r in report["collection"] for point in r["curve"]])
            return report

    def _csv(self, name, rows):
        stream = io.StringIO(newline="")
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        path = self.directory / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_text(stream.getvalue(), encoding="utf-8-sig")
        temporary.replace(path)
