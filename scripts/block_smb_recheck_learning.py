"""Recheck recorded family policies with the current environment and executor.

This reuses the recorded test split unless a different --test-seed is supplied.
It is a compatibility/retention check, not another training run.
"""

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
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--test-seed", type=int, default=817219)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    if args.count < 30 or args.count % 3:
        parser.error("count must be a multiple of three and at least 30")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(args.output_dir)
    audit = json.loads(args.audit.read_text())
    sources = sorted(Path("retroagi").rglob("*.py")) + sorted(
        Path("scripts").glob("block_smb_*.py")
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            dict(
                arguments=vars(args),
                audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            ),
            default=str,
            indent=2,
        )
    )
    torch.set_num_threads(1)
    results = []
    for row in audit["families"]:
        family = row["family"]
        if args.families and family not in args.families:
            continue
        for seed in args.seeds:
            entry = row["seeds"].get(str(seed))
            if not entry or not entry["passed"]:
                continue
            path = Path(entry["result_path"]).parent / "policy.pth"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            config = BlockSMBTrainingConfig(**_normalize_config_values(checkpoint["config"]))
            model = make_block_smb_model(config).to(config.device)
            model.load_state_dict(checkpoint["states"]["model"])
            vision, _ = _make_vision_factory(config, None)
            result = evaluate(
                model,
                samples(family, args.test_seed, "test", args.count),
                config,
                vision,
                autonomous=True,
                batched=True,
            )
            record = dict(
                family=family,
                seed=seed,
                checkpoint=str(path),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                evaluation=result,
            )
            (args.output_dir / f"{family}_seed{seed}.json").write_text(json.dumps(record, indent=2))
            print(
                json.dumps(
                    dict(family=family, seed=seed, rates=result["rates"], passed=result["passed"])
                ),
                flush=True,
            )
            results.append(record)
            del model, checkpoint
    (args.output_dir / "result.json").write_text(
        json.dumps(
            dict(
                passed=bool(results) and all(r["evaluation"]["passed"] for r in results),
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
