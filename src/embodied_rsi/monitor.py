from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from collections import deque

import psutil


def resources():
    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    result = {"cpu_percent": psutil.cpu_percent(), "ram_used_gib": vm.used / 2**30,
              "ram_total_gib": vm.total / 2**30, "ram_percent": vm.percent,
              "process_rss_gib": process.memory_info().rss / 2**30, "gpus": []}
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
        for line in output.stdout.splitlines():
            name, util, used, total, temp = [s.strip() for s in line.split(",")]

            def number(value):
                try:
                    return float(value)
                except ValueError:
                    return None

            result["gpus"].append({"name": name, "util_percent": number(util),
                                   "memory_used_mib": number(used), "memory_total_mib": number(total),
                                   "temperature_c": number(temp)})
    except (OSError, subprocess.SubprocessError):
        pass
    return result


class Monitor:
    def __init__(self, store, run_id, interval=5):
        self.store, self.run_id, self.interval = store, run_id, interval
        self.stop = threading.Event()
        self.losses = deque(maxlen=20)
        self.reported = set()
        self.last_metric = None
        self.last_training_loss = None
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=4)

    def alert(self, code, message, severity="warning"):
        if code not in self.reported:
            self.reported.add(code)
            self.store.event(self.run_id, "alert", {"code": code, "message": message, "severity": severity})

    def _sample(self):
        while not self.stop.is_set():
            r = resources()
            self.store.event(self.run_id, "resources", r)
            if self.last_metric is not None and time.monotonic() - self.last_metric > 120:
                self.alert("training_stall", "No training metric for more than 120 seconds")
            if r["ram_percent"] > 90:
                self.alert("ram_pressure", "Host memory exceeds 90%")
            for gpu in r["gpus"]:
                used, total = gpu["memory_used_mib"], gpu["memory_total_mib"]
                if used is not None and total and used / total > .95:
                    self.alert("gpu_memory", "GPU memory exceeds 95%")
                if gpu["temperature_c"] and gpu["temperature_c"] > 85:
                    self.alert("gpu_hot", "GPU temperature exceeds 85°C")
            self.stop.wait(self.interval)

    def metric(self, step, **values):
        self.last_metric = time.monotonic()
        bad = [k for k, v in values.items() if isinstance(v, (int, float)) and not math.isfinite(v)]
        if bad:
            self.alert("nonfinite", f"Non-finite training values: {', '.join(bad)}", "critical")
            raise FloatingPointError(f"Non-finite metrics: {bad}")
        loss = values.get("loss")
        if loss is not None:
            self.last_training_loss = loss
            if len(self.losses) >= 5 and loss > max(2., sum(self.losses) / len(self.losses) * 4):
                self.alert("loss_spike", "Loss exceeds 4× recent mean")
            self.losses.append(loss)
        if (step > 100 and self.last_training_loss is not None and
                values.get("val_loss", 0) > 2 * self.last_training_loss):
            self.alert("generalization_gap", "Validation loss exceeds 2× latest training loss")
        if values.get("grad_norm", 0) > 100:
            self.alert("gradient_spike", "Pre-clip gradient norm exceeds 100")
        self.store.event(self.run_id, "metric", {"step": step, **values})
