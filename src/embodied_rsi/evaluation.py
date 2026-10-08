from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .sim import POLICY_VERSION
from .storage import write_json


def fingerprint(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            with file.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def paired_report(candidate, champion, min_pairs=90, gain=.05, regression=.02):
    def key(row):
        return row["group"]

    new, old = {key(r): r for r in candidate}, {key(r): r for r in champion}
    if not new or set(new) != set(old) or len(new) != len(candidate) or len(old) != len(champion):
        raise ValueError("Evaluation must contain unique matched episode groups")
    if any(r.get("policy_version") != POLICY_VERSION for r in candidate + champion):
        raise ValueError("Controller protocol mismatch")
    groups = sorted(new)
    delta = np.asarray([float(new[k]["success"]) - float(old[k]["success"]) for k in groups])
    rng = np.random.default_rng(42)
    samples = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)])
    ci = np.quantile(samples, [.025, .975]).tolist()
    by_task = {}
    for task in sorted({r["task"] for r in candidate}):
        keys = [k for k in groups if new[k]["task"] == task]
        by_task[task] = {"pairs": len(keys),
            "candidate_success_rate": float(np.mean([new[k]["success"] for k in keys])),
            "champion_success_rate": float(np.mean([old[k]["success"] for k in keys])),
            "delta": float(np.mean([float(new[k]["success"]) - float(old[k]["success"]) for k in keys]))}
    risks = {metric: {"candidate": sum(new[k][metric] for k in groups),
                       "champion": sum(old[k][metric] for k in groups)}
             for metric in ("collisions", "drops", "rejections")}
    checks = {"enough_pairs": len(groups) >= min_pairs,
              "task_coverage": set(by_task) == {"transfer", "stack", "barrier"} and
                               all(v["pairs"] >= 30 for v in by_task.values()),
              "success_gain": float(delta.mean()) >= gain,
              "confidence_interval": ci[0] > 0,
              "no_task_regression": all(v["delta"] >= -regression for v in by_task.values()),
              "no_risk_increase": all(v["candidate"] <= v["champion"] for v in risks.values())}
    return {"pairs": len(groups), "success_rate": float(np.mean([new[k]["success"] for k in groups])),
            "champion_success_rate": float(np.mean([old[k]["success"] for k in groups])),
            "paired_gain": float(delta.mean()), "gain_ci95": ci, "by_task": by_task, "risks": risks,
            "mean_steps": float(np.mean([new[k]["steps"] for k in groups])),
            "p95_latency_ms": float(np.percentile([new[k]["latency_p95_ms"] for k in groups], 95)),
            "gate": {"passed": all(checks.values()), "checks": checks,
                     "thresholds": {"min_pairs": min_pairs, "gain": gain, "regression": regression}},
            "scope": "privileged-state / code-generated skills / partial simulator collision checks"}


def promote(store, report_path):
    import json
    report = json.loads(Path(report_path).read_text())
    if not report["gate"]["passed"]:
        raise ValueError("Candidate failed release gates; champion is unchanged")
    path = Path(report["candidate_checkpoint"])
    if fingerprint(path) != report["candidate_sha256"]:
        raise ValueError("Checkpoint has changed since evaluation")
    champion = {"checkpoint": str(path.resolve()), "sha256": report["candidate_sha256"],
                "evaluation": str(Path(report_path).resolve()), "report": report}
    write_json(store.root / "champion.json", champion)
    return champion
