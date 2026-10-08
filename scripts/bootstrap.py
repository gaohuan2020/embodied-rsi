"""Fetch exact upstream commits and install isolated environments with uv."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(*args, cwd=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsi", action="store_true", help="Also install the dedicated RSI / PyTorch environment")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    lock = json.loads((root / "upstreams.lock.json").read_text())
    for name, upstream in lock.items():
        directory = root / "vendor" / name
        if not directory.exists():
            directory.parent.mkdir(exist_ok=True)
            run("git", "clone", "--depth", "1", upstream["url"], directory)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if actual != upstream["commit"]:
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=directory, text=True)
            if dirty:
                raise RuntimeError(f"Refusing to overwrite modified upstream: {directory}")
            run("git", "fetch", "--depth", "1", "origin", upstream["commit"], cwd=directory)
        run("git", "checkout", "--detach", upstream["commit"], cwd=directory)
    env = root / ".venv"
    if not env.exists():
        run("uv", "venv", env)
    run("uv", "pip", "install", "--python", env / "bin/python", "-c", root / "constraints-simulation.txt", "-e", f"{root}[test]",
        "-e", root / "vendor/embodied-jev")
    if args.rsi:
        env_rsi = root / ".venv-rsi"
        if not env_rsi.exists():
            run("uv", "venv", env_rsi)
        run("uv", "pip", "install", "--python", env_rsi / "bin/python", "-c", root / "constraints-simulation.txt", "-e", root,
            "-e", root / "vendor/embodied-jev", "-e", root / "vendor/RSI-Jev")
    print("Ready: .venv/bin/embodied-rsi dashboard --port 8091")


if __name__ == "__main__":
    main()
