from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .compact import CompactPolicy, evaluate_cases, features, softmax
from .dataset import load_jsonl, validate_dataset
from .monitor import Monitor
from .storage import write_json


def train_compact(store, dataset, steps=500, seed=17, lr=.08, batch_size=16, parent=None):
    manifest = validate_dataset(dataset)
    train, dev = [load_jsonl(Path(dataset) / f"{s}.jsonl") for s in ("train", "dev")]
    config = {"dataset_sha256": manifest["sha256"], "steps": steps, "seed": seed,
              "lr": lr, "batch_size": batch_size, "parent": str(parent) if parent else None}
    run_id = store.create("train", "compact-numpy", config)
    path = store.run_dir(run_id)
    rng = np.random.default_rng(seed)
    xs = [features(c["state"], c["questions"][0]["criteria"]) for c in train]
    ys = [np.asarray(c["gold"]["action"]) for c in train]
    policy = CompactPolicy.load(parent) if parent else CompactPolicy(np.zeros(xs[0].shape[1]))
    policy.temperature = 1.
    best_loss, best_w = float("inf"), policy.weights.copy()
    started = time.perf_counter()
    try:
        with Monitor(store, run_id) as monitor:
            initial = evaluate_cases(policy, dev)
            monitor.metric(0, loss=evaluate_cases(policy, train)["loss"], val_loss=initial["loss"],
                           val_accuracy=initial["accuracy"], ece=initial["ece"])
            for step in range(1, steps + 1):
                gradient = np.zeros_like(policy.weights)
                loss = 0.
                for idx in rng.integers(0, len(xs), size=batch_size):
                    p = softmax(xs[idx] @ policy.weights)
                    loss -= float(ys[idx] @ np.log(np.maximum(p, 1e-12))) / batch_size
                    gradient += xs[idx].T @ (p - ys[idx]) / batch_size
                norm = float(np.linalg.norm(gradient))
                if not np.isfinite(gradient).all() or not np.isfinite(loss):
                    monitor.metric(step, loss=loss, grad_norm=norm)
                    raise FloatingPointError("Non-finite update")
                learning_rate = lr * (.1 + .9 * .5 * (1 + np.cos(np.pi * step / steps)))
                policy.weights -= learning_rate * gradient / max(1., norm)
                if step % 10 == 0 or step == steps:
                    val = evaluate_cases(policy, dev)
                    if val["loss"] < best_loss:
                        best_loss, best_w = val["loss"], policy.weights.copy()
                    monitor.metric(step, loss=loss, val_loss=val["loss"], val_accuracy=val["accuracy"],
                                   ece=val["ece"], grad_norm=norm, learning_rate=learning_rate,
                                   steps_per_second=step / (time.perf_counter() - started))
                    if val["loss"] > loss * 2 and step > 100:
                        monitor.alert("generalization_gap", "Validation loss exceeds 2× training loss")
            policy.weights = best_w
            # Temperature is fitted on DEV only. It calibrates option preference, not robot success.
            best_temp = min(np.geomspace(.3, 3., 25), key=lambda t: evaluate_cases(
                CompactPolicy(best_w, float(t)), dev)["loss"])
            policy.temperature = float(best_temp)
            policy.save(path / "checkpoint")
            summary = {"initial_dev": initial, "final_dev": evaluate_cases(policy, dev),
                       "checkpoint": str(path / "checkpoint"), "train_seconds": time.perf_counter() - started,
                       "calibration": "dev-temperature", "promotion": "pending_episode_evaluation"}
            write_json(path / "checkpoint" / "meta.json", {"backend": "compact-numpy", **config, **summary})
            write_json(path / "summary.json", summary)
        store.finish(run_id, summary=summary)
        return run_id
    except BaseException as exc:
        store.finish(run_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise


def train_rsi(store, dataset, checkpoint, steps=100, seed=17, lr=1e-4, batch_size=2,
              device="cuda", tune_tower=False, revision=None):
    """Train a released text decision head using the upstream encoder and objective.

    Uses v3.0 text checkpoints initially; multi-exit and visual releases require
    their own recipe and are intentionally rejected rather than silently altered.
    """
    validate_dataset(dataset)
    run_id = store.create("train", "rsi-jev", {"dataset": str(dataset), "checkpoint": str(checkpoint),
        "steps": steps, "seed": seed, "lr": lr, "batch_size": batch_size, "device": device,
        "tune_tower": tune_tower, "revision": revision})
    path = store.run_dir(run_id)
    try:
        with Monitor(store, run_id) as monitor:
            from .rsi_training import fit_rsi
            summary = fit_rsi(dataset, checkpoint, path / "checkpoint", monitor, steps=steps, seed=seed,
                              lr=lr, batch_size=batch_size, device=device, tune_tower=tune_tower,
                              revision=revision)
        write_json(path / "summary.json", summary)
        store.finish(run_id, summary=summary)
        return run_id
    except BaseException as exc:
        store.finish(run_id, "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                     {"error": type(exc).__name__, "message": str(exc)[:1000]})
        raise
