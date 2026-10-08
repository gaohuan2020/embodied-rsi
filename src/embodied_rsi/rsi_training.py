"""RSI-Jev integration, imported only inside the dedicated training environment."""
from __future__ import annotations

import dataclasses
import json
import random
import shutil
import time
from pathlib import Path

from .evaluation import fingerprint
from .storage import write_json


def fit_rsi(dataset, checkpoint, output, monitor, *, steps, seed, lr, batch_size, device,
            tune_tower=False, revision=None):
    import torch
    from rsijev.contract import load_cases
    from rsijev.encode import collate, encode_question, gold_tensor, unpermute_logits
    from rsijev.train import objective_loss
    from safetensors.torch import save_file
    from serve.release import load_release, resolve_ckpt

    torch.manual_seed(seed)
    rng = random.Random(seed)
    parent = resolve_ckpt(checkpoint, revision=revision)
    parent_sha256 = fingerprint(parent)
    raw_meta = json.loads((parent / "meta.json").read_text())
    spec = raw_meta["spec"]
    if spec.get("arch_extra", {}).get("aux_exits") or raw_meta.get("vision") or spec.get(
            "fit_extra", {}).get("vision"):
        raise ValueError("Initial RSI training supports text, single-exit releases (use v3.0-2b)")
    model, tok, enc, _ = load_release(parent, device=device, vision=False, adaptive="off")
    model.cal_mode = "none"  # Parent's calibration is stale after fine-tuning.
    model.cfg.freeze_base = not tune_tower
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("scorer") or (
            tune_tower and name.startswith("tower") and "embed_tokens" not in name))
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": list(model.scorer.parameters()), "lr": lr}]
    if tune_tower:
        groups.append({"params": [p for n, p in model.named_parameters()
                                  if n.startswith("tower") and p.requires_grad], "lr": lr * .01})
    optimizer = torch.optim.AdamW(groups)
    train, dev = [load_cases(str(Path(dataset) / f"{s}.jsonl")) for s in ("train", "dev")]
    max_options = max(len(c.questions[0].options) for c in train + dev)
    if max_options > model.cfg.max_options:
        raise ValueError("Robot option count exceeds parent architecture capacity")
    train_enc = dataclasses.replace(enc, option_order="shuffled")

    def forward(chunk, encoder):
        examples = [encode_question(tok, c.state, c.questions[0], encoder, rng=rng) for c in chunk]
        batch = collate(tok, examples, max_options=max_options, device=device)
        logits = unpermute_logits(model(**batch).float(), batch["option_perm"], batch["option_mask"])
        gold = gold_tensor(chunk, ["action"] * len(chunk), max_options, device)
        return logits, gold

    def validate():
        model.eval()
        total, correct = 0., 0
        with torch.no_grad():
            for start in range(0, len(dev), batch_size):
                chunk = dev[start:start + batch_size]
                logits, gold = forward(chunk, enc)
                total += float(objective_loss("soft_ce", logits, gold)) * len(chunk)
                correct += int((logits.argmax(-1) == gold.argmax(-1)).sum())
        return {"loss": total / len(dev), "accuracy": correct / len(dev)}

    initial = validate()
    monitor.metric(0, val_loss=initial["loss"], val_accuracy=initial["accuracy"])
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        chunk = [train[rng.randrange(len(train))] for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        logits, gold = forward(chunk, train_enc)
        loss = objective_loss("soft_ce", logits, gold)
        if not torch.isfinite(loss):
            monitor.metric(step, loss=float(loss.detach()))
        loss.backward()
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        except RuntimeError:
            monitor.alert("nonfinite_gradient", "Non-finite gradient detected; update aborted", "critical")
            raise
        # Emit BEFORE the optimizer update: a nonfinite update can never be applied.
        monitor.metric(step, loss=float(loss.detach()), grad_norm=float(grad_norm), learning_rate=lr,
                       steps_per_second=step / (time.perf_counter() - started),
                       gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30 if device.startswith("cuda") else 0)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            val = validate()
            monitor.metric(step, val_loss=val["loss"], val_accuracy=val["accuracy"])
    final = validate()
    output = Path(output)
    output.mkdir(parents=True)
    local = (parent / "config.json").exists()
    tower = {k: v.detach().float().contiguous().cpu() for k, v in model.tower.state_dict().items()
             if local or "embed_tokens" not in k}
    save_file(tower, str(output / "tower.safetensors"))
    save_file({k: v.detach().float().contiguous().cpu() for k, v in model.scorer.state_dict().items()},
              str(output / "scorer.safetensors"))
    if local:
        # Config + tokenizer are enough: the upstream loader builds from config then loads the tower.
        for file in parent.iterdir():
            if file.is_file() and (file.name.startswith(("tokenizer", "vocab", "merges", "special_tokens"))
                                  or file.name == "config.json"):
                shutil.copy2(file, output / file.name)
    tok.save_pretrained(output)
    raw_meta.pop("calibration", None)
    raw_meta["spec"] = {**spec, "steps": steps, "seed": seed, "batch_size": batch_size,
                        "lr_head": lr, "lr_base": lr * .01 if tune_tower else 0.,
                        "freeze_base": not tune_tower, "keep_last_k": 1, "option_order": "shuffled",
                        "grad_checkpointing": False, "fit_extra": {}, "rl_extra": {}}
    raw_meta["spec"].pop("min_trainable_tower", None)
    raw_meta["backend"] = "rsi-jev"
    raw_meta["linear_attn_kernel"] = "torch-reference/torch-" + torch.__version__
    raw_meta["train_seconds"] = time.perf_counter() - started
    raw_meta["n_train_cases"] = len(train)
    raw_meta["final_loss"] = float(loss.detach())
    raw_meta["robot_training"] = {"parent": str(checkpoint), "parent_sha256": parent_sha256,
                                   "revision": revision, "steps": steps,
                                   "seed": seed, "tune_tower": tune_tower, "calibration": "none"}
    write_json(output / "meta.json", raw_meta)
    # Reload from disk and compare outputs, not merely file existence.
    model.eval()
    with torch.no_grad():
        probe, _ = forward(dev[:1], enc)
    del optimizer
    reloaded, _, _, _ = load_release(output, device=device, vision=False, adaptive="off")
    original = model
    model = reloaded
    with torch.no_grad():
        restored, _ = forward(dev[:1], enc)
    if not torch.allclose(probe, restored, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Saved checkpoint failed prediction reload parity")
    del original
    return {"initial_dev": initial, "final_dev": final, "checkpoint": str(output),
            "train_seconds": time.perf_counter() - started, "reload_parity": True,
            "calibration": "none", "promotion": "pending_episode_evaluation"}
