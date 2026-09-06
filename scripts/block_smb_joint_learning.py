"""Qualify retention in one shared policy after the isolated family checks."""

import argparse
import json
import os
import random
from dataclasses import asdict, fields, replace
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from retroagi.stages.block_smb.cli import _make_vision_factory, _normalize_config_values
from retroagi.stages.block_smb.demonstrations import (
    DEMONSTRATION_CONTRACT_VERSION,
    DemonstrationBatch,
    align_steady_demonstrations,
    collect_demonstrations,
    fit_demonstrations,
    varied_demonstration,
    with_robust_demonstrations,
    with_varied_demonstrations,
    without_walk_commitments,
)
from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
from retroagi.stages.block_smb.train import (
    BlockSMBTrainingConfig,
    make_block_smb_model,
    make_block_smb_optimizer,
    save_block_smb_checkpoint,
)
from scripts.block_smb_family_learning import evaluate, samples


def group_family_evaluation(cases, combined):
    """Partition actual episode outcomes without averaging family rates."""
    failures = {f["scenario"]: f for f in combined["failures"]}
    results = {}
    for case in cases:
        item = results.setdefault(
            case.family, dict(counts={d: [0, 0] for d in ("easy", "medium", "hard")}, failures=[])
        )
        counts = item["counts"][case.difficulty_bin]
        counts[1] += 1
        counts[0] += int(case.scenario_id not in failures)
        if case.scenario_id in failures:
            item["failures"].append(failures[case.scenario_id])
    for item in results.values():
        item["rates"] = {d: yes / total for d, (yes, total) in item["counts"].items() if total}
        item["passed"] = all(rate >= 0.9 for rate in item["rates"].values())
    return results


@lru_cache(maxsize=128)
def evaluation_cases(family, seed, split, count):
    # Fixed validation cases need their physics oracle constructed only once.
    # Stages deep-copy scenarios before execution, so these remain unchanged.
    return tuple(samples(family, seed, split, count))


def retained_practice_weights(previous, results, streaks, failure_weight, grace):
    weights = {}
    for family, result in results.items():
        index = BLOCK_SMB_MC_FAMILIES.index(family)
        if not result["passed"]:
            streaks[index] = 0
            weight = failure_weight
        else:
            streaks[index] = streaks.get(index, 0) + 1
            weight = previous.get(index, 1.0)
            if streaks[index] > grace:
                weight = max(1.0, weight - 1.0)
        if weight > 1:
            weights[index] = weight
    return weights


def evaluate_family_set(model, config, vision, *, seed, split, count, batched):
    if not batched:
        return {
            family: evaluate(
                model, evaluation_cases(family, seed, split, count), config, vision, autonomous=True
            )
            for family in BLOCK_SMB_MC_FAMILIES
        }
    from scripts.block_smb_batched_evaluation import evaluate_batched

    results = {}
    # Sharing forward passes across families avoids hundreds of tiny GPU
    # batches. Cap each group at roughly 180 simultaneous environments.
    per_batch = max(1, 180 // count)
    for start in range(0, len(BLOCK_SMB_MC_FAMILIES), per_batch):
        cases = [
            sample
            for family in BLOCK_SMB_MC_FAMILIES[start : start + per_batch]
            for sample in evaluation_cases(family, seed, split, count)
        ]
        combined = evaluate_batched(model, cases, config, vision)
        results.update(group_family_evaluation(cases, combined))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=303)
    p.add_argument("--batched-evaluation", action="store_true")
    p.add_argument("--fixed-duration", action="store_true")
    p.add_argument("--frame-walk", action="store_true")
    p.add_argument("--cases", type=int, default=12)
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--numeric-learning-rate", type=float, default=0.003)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--failure-practice-weight", type=float, default=1.0)
    p.add_argument("--practice-retention-grace", type=int, default=3)
    p.add_argument("--focus-families", nargs="*", default=[])
    p.add_argument("--eval-per-difficulty", type=int, default=3)
    p.add_argument("--varied-demonstrations", action="store_true")
    p.add_argument("--robust-demonstrations", action="store_true")
    p.add_argument("--prioritized-demonstrations", action="store_true")
    p.add_argument("--motion-observations", action="store_true")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--refresh-families", nargs="*", default=[])
    p.add_argument("--init-checkpoint", type=Path)
    args = p.parse_args()
    if (
        args.practice_retention_grace < 0
        or args.failure_practice_weight < 1
        or args.cases < 3
        or args.cases % 3
        or min(args.updates, args.rounds, args.eval_per_difficulty) < 1
    ):
        p.error(
            "cases must be a positive multiple of three and update/evaluation counts must be positive"
        )
    if (args.output_dir / "events.jsonl").exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    values = json.loads(Path("scripts/configs/block_smb_full_volume_revision2.json").read_text())
    values.update(
        device="cuda",
        motion_observations=args.motion_observations,
        walk_duration_primitives=not args.frame_walk,
        adaptive_duration_control=not args.fixed_duration,
        autonomous_policy=True,
        demonstration_varied_routes=args.varied_demonstrations,
        demonstration_robust_routes=args.robust_demonstrations,
        demonstration_prioritized=args.prioritized_demonstrations,
        save_checkpoints=False,
        checkpoint_path=None,
        log_path=None,
        ranked_candidate_search=False,
        numeric_policy_learning_rate=args.numeric_learning_rate,
    )
    config = BlockSMBTrainingConfig(**_normalize_config_values(values))
    config = replace(config, ablation=replace(config.ablation, recurrent_state_enabled=False))
    vision, _ = _make_vision_factory(config, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2))
    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), default=str, indent=2))
    import hashlib

    sources = (
        sorted(Path("retroagi").rglob("*.py"))
        + sorted(Path("scripts").glob("block_smb_*learning*.py"))
        + [Path("scripts/block_smb_batched_evaluation.py")]
    )
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)
    )
    dataset = args.dataset or (args.output_dir / "demonstrations.pth")
    if args.dataset:
        source_config = json.loads((dataset.parent / "config.json").read_text())
        if bool(source_config.get("motion_observations", False)) != config.motion_observations:
            raise ValueError("Cached demonstration observation layout does not match this run")
    if dataset.exists():
        data = torch.load(dataset, weights_only=False)
        metadata_path = dataset.parent / "demonstration_manifest.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("contract_version", 1) < DEMONSTRATION_CONTRACT_VERSION:
            data = align_steady_demonstrations(data)
        if metadata.get("bridge_goal_contract_version", 1) < 2:
            missing = {"bridge_wait", "wait_timing", "moving_bridge"} - set(args.refresh_families)
            if missing:
                raise ValueError(f"Cached bridge goals need refreshing: {sorted(missing)}")
        source_walk = json.loads((dataset.parent / "config.json").read_text()).get(
            "walk_duration_primitives", True
        )
        if source_walk and not config.walk_duration_primitives:
            data = without_walk_commitments(data)
        elif not source_walk and config.walk_duration_primitives:
            raise ValueError("Cannot restore walk commitments from frame-walk demonstration data")
    else:
        datasets = []
        for index, family in enumerate(BLOCK_SMB_MC_FAMILIES):
            cases = samples(family, args.seed, "train", args.cases)
            if args.robust_demonstrations:
                cases = [
                    s for _, s in with_robust_demonstrations([(index, s) for s in cases], args.seed)
                ]
            if args.varied_demonstrations:
                alternatives = [
                    varied_demonstration(
                        s, args.seed + i + 100000, robust=args.robust_demonstrations
                    )
                    for i, s in enumerate(cases)
                ]
                cases += [s for s in alternatives if s is not None]
            part = collect_demonstrations([(index, s) for s in cases], config, vision)
            datasets.append(part)
            print(
                json.dumps(dict(event="demonstrations", family=family, frames=len(part.action))),
                flush=True,
            )
        data = DemonstrationBatch(
            *(torch.cat([getattr(d, f.name) for d in datasets]) for f in fields(DemonstrationBatch))
        )
        torch.save(data, dataset)
    if args.refresh_families:
        if set(args.refresh_families) - set(BLOCK_SMB_MC_FAMILIES):
            raise ValueError("Unknown refreshed family")
        for family in args.refresh_families:
            index = BLOCK_SMB_MC_FAMILIES.index(family)
            keep = data.family != index
            refreshed = [(index, s) for s in samples(family, args.seed, "train", args.cases)]
            if args.robust_demonstrations:
                refreshed = with_robust_demonstrations(refreshed, args.seed)
            if args.varied_demonstrations:
                refreshed = with_varied_demonstrations(
                    refreshed, args.seed + 100000, robust=args.robust_demonstrations
                )
            part = collect_demonstrations(refreshed, config, vision)
            data = DemonstrationBatch(
                *(
                    torch.cat((getattr(data, f.name)[keep], getattr(part, f.name)))
                    for f in fields(data)
                )
            )
            print(
                json.dumps(
                    dict(event="refreshed_demonstrations", family=family, frames=len(part.action))
                ),
                flush=True,
            )
        torch.save(data, args.output_dir / "demonstrations.pth")
    torch.save(data, args.output_dir / "demonstrations.pth")
    (args.output_dir / "demonstration_manifest.json").write_text(
        json.dumps(
            dict(
                contract_version=DEMONSTRATION_CONTRACT_VERSION,
                bridge_goal_contract_version=2,
                motion_observations=config.motion_observations,
            ),
            indent=2,
        )
    )
    torch.manual_seed(args.seed)
    model = make_block_smb_model(config).cuda()
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cuda", weights_only=False)
        if (
            bool(checkpoint["config"].get("motion_observations", False))
            != config.motion_observations
        ):
            raise ValueError("Checkpoint observation layout does not match this run")
        model.load_state_dict(checkpoint["states"]["model"])
    optimizer = make_block_smb_optimizer(model, config)
    passes = 0
    practice_weights = {
        BLOCK_SMB_MC_FAMILIES.index(family): args.failure_practice_weight
        for family in set(args.refresh_families) | set(args.focus_families)
    }
    practice_streaks = {}
    for round_index in range(args.rounds):
        loss = fit_demonstrations(
            model,
            optimizer,
            data,
            steps=args.updates,
            seed=args.seed + round_index,
            decision_durations_only=not config.adaptive_duration_control,
            walk_durations=config.walk_duration_primitives,
            prioritized=config.demonstration_prioritized,
            family_weights=practice_weights,
        )
        print(
            json.dumps(
                dict(
                    event="trained",
                    round=round_index + 1,
                    loss=loss,
                    updates=args.updates,
                    **model.last_demonstration_metrics,
                )
            ),
            flush=True,
        )
        save_block_smb_checkpoint(
            args.output_dir / "policy.pth",
            model,
            optimizer,
            epoch=round_index + 1,
            global_step=(round_index + 1) * args.updates,
            config=config,
            metrics={"loss": loss},
        )
        results = evaluate_family_set(
            model,
            config,
            vision,
            seed=99173,
            split="validation",
            count=args.eval_per_difficulty * 3,
            batched=args.batched_evaluation,
        )
        for family, result in results.items():
            print(
                json.dumps(
                    dict(
                        event="evaluation",
                        round=round_index + 1,
                        family=family,
                        rates=result["rates"],
                    )
                ),
                flush=True,
            )
        passed = all(r["passed"] for r in results.values())
        passes = passes + 1 if passed else 0
        with (args.output_dir / "events.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        round=round_index + 1,
                        loss=loss,
                        families=results,
                        passed=passed,
                        practice_weights=practice_weights,
                    )
                )
                + "\n"
            )
        save_block_smb_checkpoint(
            args.output_dir / "policy.pth",
            model,
            optimizer,
            epoch=round_index + 1,
            global_step=(round_index + 1) * args.updates,
            config=config,
            metrics={"loss": loss},
        )
        if passes >= 2:
            break
        practice_weights = retained_practice_weights(
            practice_weights,
            results,
            practice_streaks,
            args.failure_practice_weight,
            args.practice_retention_grace,
        )
        if round_index % 2 == 1:
            extra_cases = []
            for index, family in enumerate(BLOCK_SMB_MC_FAMILIES):
                if not results[family]["passed"]:
                    extra_cases += [
                        (index, s)
                        for s in samples(
                            family, args.seed, "train", 12, args.cases + (round_index + 1) * 12
                        )
                    ]
            if extra_cases:
                if args.robust_demonstrations:
                    extra_cases = with_robust_demonstrations(extra_cases, args.seed + round_index)
                if args.varied_demonstrations:
                    extra_cases = with_varied_demonstrations(
                        extra_cases,
                        args.seed + round_index + 100000,
                        robust=args.robust_demonstrations,
                    )
                extra = collect_demonstrations(extra_cases, config, vision)
                data = DemonstrationBatch(
                    *(
                        torch.cat((getattr(data, f.name), getattr(extra, f.name)))
                        for f in fields(data)
                    )
                )
    tests = None
    if passes >= 2:
        tests = evaluate_family_set(
            model,
            config,
            vision,
            seed=817219,
            split="test",
            count=max(30, args.eval_per_difficulty * 3),
            batched=args.batched_evaluation,
        )
    (args.output_dir / "result.json").write_text(
        json.dumps(
            dict(
                passed=passes >= 2
                and tests is not None
                and all(r["passed"] for r in tests.values()),
                families=results,
                test_families=tests,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
