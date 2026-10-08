"""Optional GPU check for cached scoring, trained-head reload and serving precision."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from embodied_rsi.policies import InProcessRSI
from embodied_rsi.sim import criteria_for, menu_for, state_for
from embodied_rsi.task_rsi import RSILearner


def check(learner, policy, scenes):
    errors = []
    for state, criteria in scenes:
        cached = learner.predict(state, criteria)
        with learner.torch.no_grad():
            raw = learner.logits(state, criteria).softmax(-1).cpu().numpy()
        online = policy.predict(state, criteria)
        np.testing.assert_allclose(cached, raw, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cached, online, atol=1e-5, rtol=1e-5)
        errors.append({"cached_error": float(np.max(np.abs(cached - raw))),
                       "serving_error": float(np.max(np.abs(cached - online)))})
    return errors


def main():
    from embodied_jev.physics import RobotWorld

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="v1.0-0.8b")
    parser.add_argument("--dataset", type=Path, help="Optional successful replay dataset for one update and reload check")
    args = parser.parse_args()
    learner = RSILearner(args.checkpoint,
                         revision="f9248caceb89caf2e6c968ea33bf0d6eb7f957b0" if args.checkpoint == "v1.0-0.8b" else None)
    scenes = []
    for seed in (0, 7, 1000000):
        world = RobotWorld("transfer", seed)
        state, criteria = state_for(world, []), criteria_for(menu_for(world))
        scenes.extend(((state, criteria), (state, dict(reversed(list(criteria.items()))))))
    result = {"initial": check(learner, InProcessRSI(str(learner.parent)), scenes)}
    if args.dataset:
        cases = [json.loads(line) for line in (args.dataset / "train.jsonl").read_text().splitlines() if line][:8]
        if not cases or any(not c["provenance"].get("episode_success") for c in cases):
            raise ValueError("The update probe requires actual successful episode samples")
        before = learner.predict(*scenes[0])
        batch = [(learner.encode_success({"state": json.loads(c["state"]),
                  "criteria": c["questions"][0]["criteria"]}), int(np.argmax(c["gold"]["action"]))) for c in cases]
        result["update"] = learner.imitate(batch)
        result["probability_change"] = float(np.max(np.abs(before - learner.predict(*scenes[0]))))
        if result["probability_change"] <= 1e-8:
            raise AssertionError("The head update did not change the probe probabilities")
        # Diagnostic checkpoint is temporary and never enters the deployment registry.
        with tempfile.TemporaryDirectory(prefix="rsi-consistency-") as directory:
            learner.save(directory, {"steps": 1, "batch_size": len(batch), "train_seed_range": [0, 1]})
            result["after_update_reload"] = check(learner, InProcessRSI(directory), scenes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
