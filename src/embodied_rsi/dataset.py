from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .sim import INSTRUCTIONS
from .storage import dumps, write_json


def split_for(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "dev" if bucket < 85 else "test"


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(episode_files, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Dataset already exists: {output}")
    splits = {k: [] for k in ("train", "dev", "test")}
    hashes, groups, duplicates = {}, {}, 0
    sources = []
    for file in episode_files:
        path = Path(file)
        sources.append({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "name": path.name})
        for ep in load_jsonl(path):
            split = split_for(ep["group"])
            groups[ep["group"]] = split
            for record in ep["records"]:
                criteria = record["criteria"]
                if len(criteria) < 2:
                    continue
                state = dumps(record["state"])
                # Sort criteria only for dedup; training preserves the offered option order.
                fingerprint = hashlib.sha256(dumps([record["state"], sorted(criteria.items())]).encode()).hexdigest()
                if fingerprint in hashes:
                    if hashes[fingerprint] != split:
                        raise ValueError("Identical decision appears across splits; collect more distinct scenes")
                    duplicates += 1
                    continue
                hashes[fingerprint] = split
                gold = record["gold"]
                if len(gold) != len(criteria) or any(not math.isfinite(p) or p < 0 for p in gold) or abs(sum(gold) - 1) > 1e-5:
                    raise ValueError("Invalid training target distribution")
                splits[split].append({"case_id": fingerprint, "source": f"robot_{ep['task']}",
                    "state": state, "questions": [{"key": "action", "mode": "choice",
                    "instructions": INSTRUCTIONS, "options": list(criteria), "criteria": criteria}],
                    "gold": {"action": gold}, "provenance": {"group": ep["group"],
                    "episode_id": ep["episode_id"], "step": record["step"], "label_source": record["label_source"]}})
    if any(not cases for cases in splits.values()):
        raise ValueError("Need nonempty train/dev/test splits; collect more seeds (typically ≥30)")
    output.mkdir(parents=True)
    for split, cases in splits.items():
        (output / f"{split}.jsonl").write_text("".join(dumps(c) + "\n" for c in cases), encoding="utf-8")
    manifest = {"schema_version": 1, "split_policy": "sha256(group),70/15/15; episode-grouped",
                "cases": {k: len(v) for k, v in splits.items()}, "groups": groups,
                "sources": sources, "duplicates_removed": duplicates,
                "sha256": {k: hashlib.sha256((output / f"{k}.jsonl").read_bytes()).hexdigest() for k in splits}}
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    hashes, groups = {}, {}
    for split in ("train", "dev", "test"):
        if hashlib.sha256((path / f"{split}.jsonl").read_bytes()).hexdigest() != manifest["sha256"][split]:
            raise ValueError(f"Dataset checksum mismatch: {split}")
        for case in load_jsonl(path / f"{split}.jsonl"):
            group = case["provenance"]["group"]
            if (case["case_id"] in hashes and hashes[case["case_id"]] != split or
                    group in groups and groups[group] != split):
                raise ValueError("Cross-split contamination")
            hashes[case["case_id"]], groups[group] = split, split
    return manifest
