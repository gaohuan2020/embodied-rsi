"""Differentiable RSI-Jev head with the same episode-return optimizer as Compact."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .sim import INSTRUCTIONS
from .storage import dumps, write_json


class RSILearner:
    def __init__(self, checkpoint, lr=1e-4, revision=None, device="cuda"):
        import torch
        from serve.release import load_release, resolve_ckpt

        self.torch, self.device, self.lr = torch, device, lr
        self.parent = resolve_ckpt(checkpoint, revision=revision)
        self.meta = json.loads((self.parent / "meta.json").read_text())
        spec = self.meta["spec"]
        if spec.get("arch_extra", {}).get("aux_exits") or self.meta.get("vision") or spec.get(
                "fit_extra", {}).get("vision"):
            raise ValueError("Task RL currently requires a text, single-exit RSI release")
        self.model, self.tok, self.enc, _ = load_release(self.parent, device=device,
                                                       vision=False, adaptive="off")
        self.model.cal_mode = "none"
        self.model.cfg.freeze_base = True
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith("scorer"))
        self.model.eval()  # Consistent sampling / gradient probabilities: no dropout.
        self.params = list(self.model.scorer.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.)

    def logits(self, state, criteria, capture=False):
        from rsijev.contract import Question
        from rsijev.encode import collate, encode_question, unpermute_logits

        q = Question("action", "choice", INSTRUCTIONS, tuple(criteria), criteria)
        example = encode_question(self.tok, dumps(state), q, self.enc, rng=random.Random(0))
        batch = collate(self.tok, [example], max_options=len(criteria), device=self.device)
        frozen = {}
        def remember(_module, _args, kwargs):
            frozen.update({k: v.detach() if isinstance(v, self.torch.Tensor) else v
                           for k, v in kwargs.items()})
        hook = self.model.scorer.register_forward_pre_hook(remember, with_kwargs=True) if capture else None
        try:
            logits = unpermute_logits(self.model(**batch).float(), batch["option_perm"],
                                      batch["option_mask"])[0, :len(criteria)]
        finally:
            if hook:
                hook.remove()
        if capture:
            if self.model.cfg.residual:
                raise ValueError("Cached success replay requires a non-residual RSI head")
            return frozen, batch["option_perm"], batch["option_mask"], len(criteria)
        return logits

    def encode_success(self, row):
        with self.torch.no_grad():
            return self.logits(row["state"], row["criteria"], capture=True)

    def imitate(self, batch):
        from rsijev.encode import unpermute_logits
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for encoded, choice in batch:
            frozen, perm, mask, count = encoded
            logits = self.model.scorer(**frozen).float()
            if self.model.cfg.logit_cap:
                cap = float(self.model.cfg.logit_cap)
                logits = cap * self.torch.tanh(logits / cap)
                logits = logits.masked_fill(~mask, float("-inf"))
            logits = unpermute_logits(logits, perm, mask)[0, :count]
            losses.append(-logits.log_softmax(-1)[choice])
        loss = self.torch.stack(losses).mean()
        if not self.torch.isfinite(loss):
            raise FloatingPointError("Nonfinite successful-trajectory loss")
        loss.backward()
        norm = self.torch.nn.utils.clip_grad_norm_(self.params, 1., error_if_nonfinite=True)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "grad_norm": float(norm)}

    def predict(self, state, criteria):
        with self.torch.no_grad():
            return self.logits(state, criteria).softmax(-1).cpu().numpy().astype(float)

    def snapshot(self):
        return {k: v.detach().clone().cpu() for k, v in self.model.scorer.state_dict().items()}

    def restore(self, state):
        self.model.scorer.load_state_dict(state)

    def update(self, trajectories, advantages):
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.
        for trajectory, advantage in zip(trajectories, advantages):
            if abs(advantage) < 1e-12:
                continue
            for row in trajectory["records"]:
                action = list(row["criteria"]).index(row["choice"])
                loss = -advantage * self.logits(row["state"], row["criteria"]).log_softmax(-1)[action]
                loss = loss / len(trajectories)
                if not self.torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite episode policy objective")
                loss.backward()
                total += float(loss.detach())
        norm = self.torch.nn.utils.clip_grad_norm_(self.params, 1., error_if_nonfinite=True)
        self.optimizer.step()
        return {"policy_loss": total, "grad_norm": float(norm)}

    def save(self, path, metadata):
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        local = (self.parent / "config.json").exists()
        save_file({k: v.detach().float().contiguous().cpu() for k, v in self.model.tower.state_dict().items()
                   if local or "embed_tokens" not in k}, str(path / "tower.safetensors"))
        save_file({k: v.detach().float().contiguous().cpu() for k, v in self.model.scorer.state_dict().items()},
                  str(path / "scorer.safetensors"))
        if local:
            shutil.copy2(self.parent / "config.json", path / "config.json")
        self.tok.save_pretrained(path)
        self.meta.pop("calibration", None)
        self.meta.update(backend="rsi-jev", robot_training=metadata)
        self.meta["spec"].update(freeze_base=True, option_order=self.enc.option_order,
                                 rl_extra={}, fit_extra={}, steps=metadata["steps"],
                                 batch_size=metadata["batch_size"], seed=metadata["train_seed_range"][0],
                                 lr_head=self.lr, lr_base=0.)
        self.meta.update(metadata)
        write_json(path / "meta.json", self.meta)
