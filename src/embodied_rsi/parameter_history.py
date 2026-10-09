"""Record measured tensor changes without modifying the training objective."""
from __future__ import annotations

import json
import re
import struct
import time

import numpy as np

from .storage import dumps, write_json


class ParameterRecorder:
    def __init__(self, model, directory):
        self.model, self.directory = model, directory
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
        self.initial = {n: p.detach().clone() for n, p in self.params.items()}
        self.previous = self.initial
        self.versions = {n: p._version for n, p in self.frozen.items()}
        groups = {}
        for n, p in self.frozen.items():
            match = re.search(r"^(.*?layers\.\d+)", n)
            group = match[1] if match else n.rsplit(".", 1)[0]
            groups[group] = groups.get(group, 0) + p.numel()
        self.manifest = {"source": "optimizer_snapshots", "sample_method": "64 evenly spaced flat indices",
                         "trainable": [{"name": n, "shape": list(p.shape), "parameters": p.numel()}
                                       for n, p in self.params.items()],
                         "frozen": [{"name": n, "parameters": count, "trainable": False} for n, count in groups.items()]}
        write_json(directory / "parameters.json", self.manifest)
        self.capture(0, "initial")

    def capture(self, step, phase="update"):
        import torch

        tensors, next_previous = [], {}
        with torch.no_grad():
            for name, p in self.params.items():
                value = p.detach().float()
                previous = self.previous[name].float()
                initial = self.initial[name].float()
                indices = torch.linspace(0, p.numel() - 1, min(64, p.numel()), device=p.device).long()
                delta = value - previous
                start_delta = value - initial
                tensors.append({"name": name, "shape": list(p.shape), "parameters": p.numel(),
                                "norm": float(value.norm()), "delta_norm": float(delta.norm()),
                                "total_delta_norm": float(start_delta.norm()),
                                "max_delta": float(delta.abs().max()), "indices": indices.cpu().tolist(),
                                "values": value.flatten()[indices].cpu().tolist(),
                                "deltas": delta.flatten()[indices].cpu().tolist(),
                                "total_deltas": start_delta.flatten()[indices].cpu().tolist()})
                next_previous[name] = p.detach().clone()
        self.previous = next_previous
        row = {"step": step, "phase": phase, "ts": time.time(), "tensors": tensors,
               "frozen_changed": [n for n, p in self.frozen.items() if p._version != self.versions[n]]}
        with (self.directory / "parameters.jsonl").open("a") as file:
            file.write(dumps(row) + "\n")
        return row


def _tensor_file(path):
    with path.open("rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(length))
    return header, 8 + length


def _numpy_tensor(path, spec, offset):
    dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}.get(spec["dtype"])
    if dtype is None:
        raise ValueError(f"Unsupported parameter dtype {spec['dtype']}")
    value = np.memmap(path, dtype=dtype, mode="r", offset=offset + spec["data_offsets"][0], shape=tuple(spec["shape"]))
    if spec["dtype"] == "BF16":
        return (np.asarray(value).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(value, dtype=np.float32)


def checkpoint_history(store, run):
    """Historical checkpoints provide endpoints, never an invented optimizer trajectory."""
    directory = store.run_dir(run)
    checkpoint = directory / "checkpoint"
    path = checkpoint / "scorer.safetensors"
    if not path.exists():
        return {"source": "unavailable", "snapshots": [], "trainable": [], "frozen": []}
    with store.connect() as db:
        row = db.execute("SELECT config FROM runs WHERE kind='evaluate' AND json_extract(config,'$.candidate')=? ORDER BY created DESC LIMIT 1",
                         (str(checkpoint),)).fetchone()
    from pathlib import Path

    parent = Path(json.loads(row[0])["champion"]) if row else None
    baseline_path = parent / "scorer.safetensors" if parent else None
    header, offset = _tensor_file(path)
    base_header, base_offset = _tensor_file(baseline_path) if baseline_path and baseline_path.is_file() else ({}, 0)
    initial, selected, trainable = [], [], []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        value = _numpy_tensor(path, spec, offset)
        if name in base_header and base_header[name]["shape"] == spec["shape"]:
            base = _numpy_tensor(baseline_path, base_header[name], base_offset)
        else:
            base = None
        indices = np.linspace(0, value.size - 1, min(64, value.size)).astype(int)
        delta = value - base if base is not None else None
        common = {"name": "scorer." + name, "shape": spec["shape"], "parameters": value.size, "indices": indices.tolist()}
        trainable.append({k: common[k] for k in ("name", "shape", "parameters")})
        selected.append({**common, "norm": float(np.linalg.norm(value)),
                         "delta_norm": float(np.linalg.norm(delta)) if delta is not None else None,
                         "total_delta_norm": float(np.linalg.norm(delta)) if delta is not None else None,
                         "max_delta": float(np.max(np.abs(delta))) if delta is not None else None,
                         "values": value.flatten()[indices].tolist(),
                         "deltas": delta.flatten()[indices].tolist() if delta is not None else [],
                         "total_deltas": delta.flatten()[indices].tolist() if delta is not None else []})
        if base is not None:
            initial.append({**common, "norm": float(np.linalg.norm(base)), "delta_norm": 0., "total_delta_norm": 0.,
                            "max_delta": 0., "values": base.flatten()[indices].tolist(),
                            "deltas": [0.] * len(indices), "total_deltas": [0.] * len(indices)})
    meta = json.loads((checkpoint / "meta.json").read_text())
    frozen_groups = {}
    tower = checkpoint / "tower.safetensors"
    if tower.exists():
        tower_header, _ = _tensor_file(tower)
        for name, spec in tower_header.items():
            if name == "__metadata__":
                continue
            match = re.search(r"^(.*?layers\.\d+)", name)
            group = "tower." + (match[1] if match else name.rsplit(".", 1)[0])
            frozen_groups[group] = frozen_groups.get(group, 0) + int(np.prod(spec["shape"]))
    snapshots = ([{"step": 0, "phase": "parent_checkpoint", "tensors": initial, "frozen_changed": []}] if initial else [])
    snapshots.append({"step": meta.get("selected_step", 0), "phase": "selected_checkpoint", "tensors": selected,
                      "frozen_changed": [], "frozen_verification": "training configuration freeze_base"})
    return {"source": "checkpoint_endpoints", "sample_method": "64 evenly spaced flat indices",
            "trainable": trainable, "frozen": [{"name": n, "parameters": count, "trainable": False} for n, count in frozen_groups.items()],
            "snapshots": snapshots}


def parameter_history(store, run, after=-1):
    directory = store.run_dir(run)
    manifest = directory / "parameters.json"
    if not manifest.exists():
        return checkpoint_history(store, run)
    result = json.loads(manifest.read_text())
    rows = []
    with (directory / "parameters.jsonl").open() as file:
        for index, line in enumerate(file):
            if index > after and line.endswith("\n"):
                rows.append({"index": index, **json.loads(line)})
    result["snapshots"] = rows
    return result
