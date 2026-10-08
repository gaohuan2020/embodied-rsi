from __future__ import annotations

import argparse
import json
import math
import os

from .storage import Store


def parser():
    p = argparse.ArgumentParser(description="Embodied RSI learning workbench")
    p.add_argument("--artifacts", default="artifacts", help="Local records, datasets and checkpoints")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tasks", nargs="+", choices=["transfer", "stack", "barrier"],
                   default=["transfer", "stack", "barrier"])
    c.add_argument("--episodes", type=int, default=30, help="Seeds per task")
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--max-steps", type=int, default=20)
    c.add_argument("--label-mode", choices=["teacher", "rollout"], default="teacher")
    c.add_argument("--intervention", action="store_true")
    c.add_argument("--checkpoint")
    c.add_argument("--backend", choices=["compact", "rsi", "http"], default="compact")
    c.add_argument("--url")
    c.add_argument("--model", default="jev-latest")
    d = sub.add_parser("dataset")
    d.add_argument("--episodes-files", nargs="+", required=True)
    d.add_argument("--output", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--backend", choices=["compact", "rsi"], default="compact")
    t.add_argument("--steps", type=int, default=500)
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--lr", type=float)
    t.add_argument("--batch-size", type=int)
    t.add_argument("--checkpoint", help="Parent checkpoint; RSI default is v3.0-2b")
    t.add_argument("--revision", help="HF checkpoint revision (pin for reproducible RSI training)")
    t.add_argument("--device", default="cuda")
    t.add_argument("--tune-tower", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--backend", choices=["compact", "rsi"], default="compact")
    e.add_argument("--champion", help="Same-backend checkpoint; omit to compare with rule teacher")
    e.add_argument("--tasks", nargs="+", choices=["transfer", "stack", "barrier"],
                   default=["transfer", "stack", "barrier"])
    e.add_argument("--episodes", type=int, default=30)
    e.add_argument("--seed-start", type=int, default=10000)
    e.add_argument("--max-steps", type=int, default=20)
    e.add_argument("--dataset", required=True, help="Enables train/dev scene leakage checks")
    e.add_argument("--intervention", action="store_true")
    r = sub.add_parser("promote")
    r.add_argument("--report", required=True)
    s = sub.add_parser("dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8091)
    s.add_argument("--rsi-python", help="Optional separate interpreter for RSI training jobs")
    loop = sub.add_parser("cycle")
    loop.add_argument("--rounds", type=int, default=2)
    loop.add_argument("--episodes", type=int, default=30)
    loop.add_argument("--steps", type=int, default=500)
    loop.add_argument("--seed-start", type=int, default=0)
    sub.add_parser("status")
    return p


def policy_for(backend, checkpoint):
    if backend == "compact":
        from .compact import CompactPolicy
        return CompactPolicy.load(checkpoint)
    from .policies import InProcessRSI
    return InProcessRSI(checkpoint)


def main(argv=None):
    a = parser().parse_args(argv)
    for name in ("episodes", "steps", "rounds", "batch_size", "max_steps"):
        value = getattr(a, name, None)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if getattr(a, "seed_start", 0) < 0:
        raise SystemExit("--seed-start must be nonnegative")
    if getattr(a, "lr", None) is not None and (not math.isfinite(a.lr) or a.lr <= 0):
        raise SystemExit("--lr must be positive")
    store = Store(a.artifacts)
    if a.command == "collect":
        from .workflow import collect
        policy = None
        if a.backend == "http":
            from .policies import HTTPPolicy
            if not a.url:
                raise SystemExit("--url is required for HTTP collection")
            policy = HTTPPolicy(a.url, a.model, os.environ.get("RSIJEV_API_KEY"))
        elif a.checkpoint:
            policy = policy_for(a.backend, a.checkpoint)
        try:
            result = collect(store, a.tasks, range(a.seed_start, a.seed_start + a.episodes), policy,
                             a.checkpoint or a.backend if policy else "teacher", a.max_steps,
                             a.label_mode, a.intervention)
        finally:
            if hasattr(policy, "close"):
                policy.close()
    elif a.command == "dataset":
        from .dataset import build_dataset
        result = build_dataset(a.episodes_files, a.output)
    elif a.command == "train":
        from .training import train_compact, train_rsi
        if a.backend == "compact":
            result = train_compact(store, a.dataset, a.steps, a.seed, a.lr or .08,
                                   a.batch_size or 16, a.checkpoint)
        else:
            result = train_rsi(store, a.dataset, a.checkpoint or "v3.0-2b", a.steps, a.seed,
                               a.lr or 1e-4, a.batch_size or 2, a.device, a.tune_tower, a.revision)
    elif a.command == "evaluate":
        from .workflow import evaluate
        result = evaluate(store, policy_for(a.backend, a.checkpoint), a.checkpoint,
                          policy_for(a.backend, a.champion) if a.champion else None,
                          a.champion or "teacher", a.tasks, range(a.seed_start, a.seed_start + a.episodes),
                          a.max_steps, a.dataset, a.intervention)
    elif a.command == "promote":
        from .evaluation import promote
        result = promote(store, a.report)
    elif a.command == "cycle":
        from .workflow import cycle
        result = cycle(store, a.rounds, a.episodes, a.steps, a.seed_start)
    elif a.command == "status":
        result = store.runs()
    elif a.command == "dashboard":
        import uvicorn

        from .dashboard import create_app
        uvicorn.run(create_app(store, rsi_python=a.rsi_python), host=a.host, port=a.port)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
