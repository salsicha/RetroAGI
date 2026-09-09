"""Bounded coaching checks; optional learning probe, never a training prerequisite."""

import argparse
import json
from pathlib import Path

import torch

from retroagi.core.smb_learning import (
    block_stage,
    collect_case,
    collect_reactive_case,
    make_model,
    rows_to_data,
)
from retroagi.core.smb_perception import DenseSMBPerception
from retroagi.stages.block_smb.demonstrations import fit_demonstrations
from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case
from scripts.smb_composable_training import evaluate, write_json


def audit(
    *, families, output, vision=None, device="cpu", seed=20260909, layouts=3, learning_updates=0
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = make_model(hidden_dim=64, device=device, provider="perceived" if vision else "oracle")
    rows, results, cases = [], [], []
    for family in families:
        for index in range(layouts):
            sample = sample_nes_case(
                family=family,
                split="train",
                seed=seed,
                index=100 + index,
                difficulty=("easy", "medium", "hard")[index % 3],
            )
            stage = block_stage(sample, vision=vision, device=device)
            try:
                data, result = collect_case(
                    model,
                    stage,
                    sample.oracle["actions"],
                    family=BLOCK_SMB_MC_FAMILIES.index(family),
                )
                if not data:
                    data, result = collect_reactive_case(
                        model, stage, family=BLOCK_SMB_MC_FAMILIES.index(family)
                    )
                rows.extend(data)
                result.update(
                    id=sample.scenario_id, family=family, difficulty=sample.difficulty_bin
                )
                results.append(result)
                cases.append(sample)
                print(json.dumps(result), flush=True)
                write_json(output / "coaching.json", dict(results=results))
            finally:
                stage.env.close()
    report = dict(
        results=results,
        successful=sum(r["success"] for r in results),
        attempted=len(results),
        observation_provider="perceived" if vision else "oracle",
        coaching_feasibility_only=not bool(learning_updates),
    )
    if learning_updates:
        data = rows_to_data(rows)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        before = evaluate(model, cases, vision)
        for start in range(0, learning_updates, 100):
            loss = fit_demonstrations(
                model,
                optimizer,
                data,
                steps=min(100, learning_updates - start),
                seed=seed + start,
                decision_durations_only=True,
                walk_durations=False,
                prioritized=True,
            )
            print(
                json.dumps(dict(updates=min(start + 100, learning_updates), loss=loss)), flush=True
            )
        after = evaluate(model, cases, vision)
        holdout = [
            sample_nes_case(
                family=f,
                split="validation",
                seed=seed,
                index=500 + i,
                difficulty=("easy", "medium", "hard")[i % 3],
            )
            for f in families
            for i in range(3)
        ]
        report["learning_probe"] = dict(
            updates=learning_updates,
            before=before,
            after=after,
            holdout=evaluate(model, holdout, vision),
        )
    write_json(output / "coaching.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--families", nargs="+", choices=BLOCK_SMB_MC_FAMILIES, default=BLOCK_SMB_MC_FAMILIES
    )
    parser.add_argument("--perception")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument("--learning-updates", type=int, default=0)
    args = parser.parse_args()
    vision = (
        DenseSMBPerception.load(args.perception, device=args.device) if args.perception else None
    )
    report = audit(
        families=args.families,
        output=args.output,
        vision=vision,
        device=args.device,
        layouts=args.layouts,
        learning_updates=args.learning_updates,
    )
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("results", "learning_probe")}),
        flush=True,
    )
    if report["successful"] != report["attempted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
