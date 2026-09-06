"""Finish the independent test of a stopped, validation-qualified checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from retroagi.stages.block_smb.cli import _make_vision_factory, _normalize_config_values
from retroagi.stages.block_smb.train import BlockSMBTrainingConfig, make_block_smb_model
from scripts.block_smb_family_learning import evaluate, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    if (run / "result.json").exists():
        raise FileExistsError(run / "result.json")
    arguments = json.loads((run.parent / "arguments.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if (
        not arguments.get("autonomous")
        or len(events) < 2
        or not all(e["evaluation"]["passed"] for e in events[-2:])
    ):
        raise ValueError("Two successive autonomous validations are required before the test")
    last = events[-1]
    if arguments.get("method") == "demonstrations" and last["round"] * arguments[
        "updates"
    ] < arguments.get("min_updates", 0):
        raise ValueError("The requested minimum training budget has not been completed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(run / "policy.pth", weights_only=False)
    if checkpoint["epoch"] != last["round"]:
        raise ValueError("Checkpoint does not match the last validation round")
    config = BlockSMBTrainingConfig(**_normalize_config_values(checkpoint["config"]))
    model = make_block_smb_model(config).to(config.device)
    model.load_state_dict(checkpoint["states"]["model"])
    vision, _ = _make_vision_factory(config, None)
    test = evaluate(
        model,
        samples(last["family"], 817219, "test", max(10, arguments["eval_per_difficulty"]) * 3),
        config,
        vision,
        autonomous=True,
        batched=True,
    )
    result = dict(
        family=last["family"],
        seed=last["seed"],
        passed=test["passed"],
        evaluation=last["evaluation"],
        test_evaluation=test,
    )
    sources = sorted(Path("retroagi").rglob("*.py")) + [
        Path(__file__),
        Path("scripts/block_smb_batched_evaluation.py"),
    ]
    (run / "completion_manifest.json").write_text(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256((run / "policy.pth").read_bytes()).hexdigest(),
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                evaluator="batched",
                test_seed=817219,
            ),
            indent=2,
        )
    )
    (run / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                family=last["family"], seed=last["seed"], passed=test["passed"], rates=test["rates"]
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
