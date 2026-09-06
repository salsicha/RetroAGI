"""Evaluate one frozen shared policy on every family without training it."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from retroagi.stages.block_smb.cli import _make_vision_factory, _normalize_config_values
from retroagi.stages.block_smb.train import BlockSMBTrainingConfig, make_block_smb_model
from scripts.block_smb_joint_learning import evaluate_family_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    if args.count < 30 or args.count % 3:
        parser.error("count must be a multiple of three and at least 30")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = BlockSMBTrainingConfig(**_normalize_config_values(checkpoint["config"]))
    torch.use_deterministic_algorithms(config.deterministic)
    model = make_block_smb_model(config).to(config.device)
    model.load_state_dict(checkpoint["states"]["model"])
    vision, _ = _make_vision_factory(config, None)
    results = evaluate_family_set(
        model, config, vision, seed=args.seed, split="test", count=args.count, batched=True
    )
    sources = sorted(Path("retroagi").rglob("*.py")) + sorted(
        Path("scripts").glob("block_smb_*.py")
    )
    record = dict(
        passed=all(r["passed"] for r in results.values()),
        families=results,
        arguments=vars(args),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, default=str, indent=2))
    for family, result in results.items():
        print(
            json.dumps(dict(family=family, rates=result["rates"], passed=result["passed"])),
            flush=True,
        )


if __name__ == "__main__":
    main()
