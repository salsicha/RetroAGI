"""Actual learning and held-out qualification, independently for each family.

Unlike physics preflight, evaluation never supplies demonstrations. Results are
per seed and difficulty; training and evaluation layouts use disjoint splits.
"""

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage
from retroagi.stages.block_smb.cli import _make_vision_factory, _normalize_config_values
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.train import (
    BlockSMBSuccessReplay,
    BlockSMBTrainingConfig,
    block_smb_policy_scenario,
    collect_trajectory,
    make_block_smb_model,
    make_block_smb_optimizer,
    save_block_smb_checkpoint,
    train_block_smb_epoch,
)

DIFFICULTIES = ("easy", "medium", "hard")


def samples(family, seed, split, count, offset=0):
    return [
        sample_block_smb_monte_carlo_scenario(
            family=family,
            seed=seed,
            split=split,
            sample_index=offset + i,
            difficulty=DIFFICULTIES[i % 3],
        )
        for i in range(count)
    ]


def autonomous_scenario(scenario):
    return block_smb_policy_scenario(scenario, True)


def evaluate(model, cases, config, vision_factory, *, autonomous=False, batched=False):
    if batched:
        if not autonomous:
            raise ValueError("Batched evaluation requires autonomous control")
        from scripts.block_smb_batched_evaluation import evaluate_batched

        return evaluate_batched(model, cases, config, vision_factory)
    model.eval()
    by_difficulty = {d: [] for d in DIFFICULTIES}
    failures = []
    with torch.no_grad():
        for sample in cases:
            scenario = autonomous_scenario(sample.scenario) if autonomous else sample.scenario
            stage = BlockSMBStage(
                env=MarioScenarioEnv(reward_config=config.reward_config),
                scenario=scenario,
                vision=vision_factory(),
                observation_config=BlockSMBObservationConfig(
                    motion_observations=config.motion_observations
                ),
            )
            try:
                trajectory = collect_trajectory(
                    model,
                    stage,
                    sample.scenario_id,
                    rollout_steps=config.evaluation_max_steps,
                    seed=sample.sample_seed % (2**31),
                    deterministic=True,
                    device=torch.device(config.device),
                    ablation=config.ablation,
                    adaptive_duration_control=config.adaptive_duration_control,
                    skill_goal_conditioning=config.skill_goal_conditioning,
                    steady_duration_primitives=config.steady_duration_primitives,
                    walk_duration_primitives=config.walk_duration_primitives,
                    engine_support=config.engine_support_override,
                )
                by_difficulty[sample.difficulty_bin].append(int(trajectory.success))
                if not trajectory.success:
                    from collections import Counter

                    last = trajectory.transitions[-1]
                    failures.append(
                        dict(
                            scenario=sample.scenario_id,
                            difficulty=sample.difficulty_bin,
                            actions=dict(Counter(t.action for t in trajectory.transitions)),
                            x=stage.env.mario["x"],
                            y=stage.env.mario["y"],
                            phase=last.info.get("skill_phase"),
                            death=last.info.get("death"),
                            mounted=last.info.get("pipe_mounted"),
                            frames=len(trajectory.transitions),
                        )
                    )
            finally:
                stage.env.close()
    rates = {d: sum(v) / len(v) for d, v in by_difficulty.items()}
    return dict(
        rates=rates,
        counts={d: [sum(v), len(v)] for d, v in by_difficulty.items()},
        passed=all(r >= 0.9 for r in rates.values()),
        failures=failures,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=list(BLOCK_SMB_MC_FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-per-difficulty", type=int, default=10)
    parser.add_argument("--teacher-rounds", type=int, default=0)
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--batched-evaluation", action="store_true")
    parser.add_argument("--method", choices=["rl", "demonstrations"], default="rl")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--min-updates", type=int, default=0)
    parser.add_argument("--numeric-learning-rate", type=float, default=0.003)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--fixed-duration", action="store_true")
    parser.add_argument("--frame-walk", action="store_true")
    parser.add_argument("--motion-observations", action="store_true")
    parser.add_argument("--varied-demonstrations", action="store_true")
    parser.add_argument("--robust-demonstrations", action="store_true")
    parser.add_argument("--prioritized-demonstrations", action="store_true")
    args = parser.parse_args()
    if (
        args.episodes < 3
        or args.episodes % 3
        or min(args.rounds, args.updates, args.eval_per_difficulty) < 1
    ):
        parser.error(
            "episodes must be a positive multiple of three; rounds, updates and evaluation counts must be positive"
        )
    if args.min_updates < 0 or (
        args.method == "demonstrations" and args.rounds * args.updates < args.min_updates
    ):
        parser.error("min-updates must fit the training budget")
    if set(args.families) - set(BLOCK_SMB_MC_FAMILIES):
        parser.error("Unknown family")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    values = json.loads(Path("scripts/configs/block_smb_full_volume_revision2.json").read_text())
    values.update(
        motion_observations=args.motion_observations,
        walk_duration_primitives=not args.frame_walk,
        autonomous_policy=args.autonomous,
        demonstration_varied_routes=args.varied_demonstrations,
        demonstration_robust_routes=args.robust_demonstrations,
        demonstration_prioritized=args.prioritized_demonstrations,
        numeric_policy_learning_rate=args.numeric_learning_rate,
        save_checkpoints=False,
        log_path=None,
        checkpoint_path=None,
        emit_temporal_spans=False,
        episodes_per_epoch=args.episodes,
        update_batch_episodes=4,
        success_replay_rehearsals_per_epoch=2,
        ranked_candidate_search=False,
    )
    if args.learning_rate is not None:
        values["learning_rate"] = args.learning_rate
    config = BlockSMBTrainingConfig(**_normalize_config_values(values))
    if args.fixed_duration:
        config = replace(config, adaptive_duration_control=False)
    if args.method == "demonstrations":
        config = replace(config, ablation=replace(config.ablation, recurrent_state_enabled=False))
    vision_factory, _ = _make_vision_factory(config, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2))
    from dataclasses import asdict

    sources = sorted(Path("retroagi").rglob("*.py")) + [Path(__file__)]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            dict(
                config=asdict(config),
                sources=manifest,
                torch_version=torch.__version__,
                qualification=dict(
                    min_rate=0.9,
                    successive_validations=2,
                    seeds=args.seeds,
                    validation_seed=99173,
                    test_seed=817219,
                    test_layouts_per_difficulty=max(10, args.eval_per_difficulty),
                ),
            ),
            default=str,
            indent=2,
        )
    )
    for family in args.families:
        for seed in args.seeds:
            run = args.output_dir / f"{family}_seed{seed}"
            run.mkdir(exist_ok=True)
            if (run / "events.jsonl").exists():
                raise FileExistsError(run)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = make_block_smb_model(config).to(config.device)
            if args.init_checkpoint is not None:
                checkpoint = torch.load(
                    args.init_checkpoint, map_location=config.device, weights_only=False
                )
                if (
                    bool(checkpoint["config"].get("motion_observations", False))
                    != config.motion_observations
                ):
                    raise ValueError("Checkpoint motion-observation layout does not match this run")
                model.load_state_dict(checkpoint["states"]["model"])
            optimizer = make_block_smb_optimizer(model, config)
            replay = BlockSMBSuccessReplay(seed=seed)
            cases = samples(family, 99173, "validation", args.eval_per_difficulty * 3)
            passed_rounds = 0
            if args.method == "demonstrations":
                from retroagi.stages.block_smb.demonstrations import (
                    collect_demonstrations,
                    fit_demonstrations,
                )

                training_cases = samples(family, seed, "train", max(12, args.episodes))
                if args.robust_demonstrations:
                    from retroagi.stages.block_smb.demonstrations import with_robust_demonstrations

                    training_cases = [
                        s
                        for _, s in with_robust_demonstrations(
                            [(0, s) for s in training_cases], seed
                        )
                    ]
                if args.varied_demonstrations:
                    from retroagi.stages.block_smb.demonstrations import varied_demonstration

                    alternatives = [
                        varied_demonstration(
                            s, seed + i + 100000, robust=args.robust_demonstrations
                        )
                        for i, s in enumerate(training_cases)
                    ]
                    training_cases += [s for s in alternatives if s is not None]
                data = collect_demonstrations(
                    [(0, s) for s in training_cases], config, vision_factory
                )
                print(
                    json.dumps(
                        dict(family=family, seed=seed, demonstration_frames=len(data.action))
                    ),
                    flush=True,
                )
            for round_index in range(args.rounds):
                start = time.monotonic()
                training = samples(
                    family, seed, "train", args.episodes, round_index * args.episodes
                )
                curriculum = [
                    (
                        s.scenario_id,
                        autonomous_scenario(s.scenario) if args.autonomous else s.scenario,
                    )
                    for s in training
                ]
                cfg = replace(
                    config, seed=seed, use_oracle_actions=round_index < args.teacher_rounds
                )
                if args.method == "demonstrations":
                    loss = fit_demonstrations(
                        model,
                        optimizer,
                        data,
                        steps=args.updates,
                        seed=seed + round_index,
                        decision_durations_only=not config.adaptive_duration_control,
                        walk_durations=config.walk_duration_primitives,
                        prioritized=config.demonstration_prioritized,
                    )
                    metrics = dict(
                        mean_return=None,
                        loss_total=loss,
                        optimizer_updates=args.updates,
                        demonstration_frames=len(data.action),
                        **model.last_demonstration_metrics,
                    )
                else:
                    metrics, _ = train_block_smb_epoch(
                        model,
                        optimizer,
                        curriculum,
                        cfg,
                        round_index,
                        device=torch.device(config.device),
                        vision_factory=vision_factory,
                        success_replay=replay,
                    )
                evaluation = evaluate(
                    model,
                    cases,
                    config,
                    vision_factory,
                    autonomous=args.autonomous,
                    batched=args.batched_evaluation,
                )
                passed_rounds = passed_rounds + 1 if evaluation["passed"] else 0
                if (
                    args.method == "demonstrations"
                    and not evaluation["passed"]
                    and round_index % 2 == 1
                ):
                    from dataclasses import fields

                    extra_cases = [
                        (0, s)
                        for s in samples(
                            family,
                            seed,
                            "train",
                            args.episodes,
                            (round_index + 1) * args.episodes,
                        )
                    ]
                    if args.robust_demonstrations:
                        from retroagi.stages.block_smb.demonstrations import (
                            with_robust_demonstrations,
                        )

                        extra_cases = with_robust_demonstrations(extra_cases, seed + round_index)
                    if args.varied_demonstrations:
                        from retroagi.stages.block_smb.demonstrations import (
                            with_varied_demonstrations,
                        )

                        extra_cases = with_varied_demonstrations(
                            extra_cases,
                            seed + round_index + 100000,
                            robust=args.robust_demonstrations,
                        )
                    extra = collect_demonstrations(extra_cases, config, vision_factory)
                    data = type(data)(
                        *(
                            torch.cat((getattr(data, f.name), getattr(extra, f.name)))
                            for f in fields(data)
                        )
                    )
                event = dict(
                    family=family,
                    seed=seed,
                    round=round_index + 1,
                    teacher=cfg.use_oracle_actions or args.method == "demonstrations",
                    method=args.method,
                    evaluation=evaluation,
                    metrics=metrics,
                    elapsed=time.monotonic() - start,
                )
                with (run / "events.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                print(
                    json.dumps(
                        {k: v for k, v in event.items() if k not in ("metrics", "evaluation")}
                        | {
                            "rates": evaluation["rates"],
                            "loss": metrics.get("loss_total"),
                            "mean_return": metrics["mean_return"],
                        }
                    ),
                    flush=True,
                )
                save_block_smb_checkpoint(
                    run / "policy.pth",
                    model,
                    optimizer,
                    epoch=round_index + 1,
                    global_step=(round_index + 1)
                    * (args.updates if args.method == "demonstrations" else args.episodes),
                    config=cfg,
                    metrics=metrics,
                )
                # A second successive evaluation after additional updates is
                # an initial retention check, not proof of multi-family retention.
                enough_updates = (
                    args.method != "demonstrations"
                    or (round_index + 1) * args.updates >= args.min_updates
                )
                if passed_rounds >= 2 and enough_updates and round_index >= args.teacher_rounds:
                    break
            test_evaluation = None
            if passed_rounds >= 2 and enough_updates:
                test_cases = samples(family, 817219, "test", max(10, args.eval_per_difficulty) * 3)
                test_evaluation = evaluate(
                    model,
                    test_cases,
                    config,
                    vision_factory,
                    autonomous=args.autonomous,
                    batched=args.batched_evaluation,
                )
            result = dict(
                family=family,
                seed=seed,
                passed=passed_rounds >= 2
                and test_evaluation is not None
                and test_evaluation["passed"],
                evaluation=evaluation,
                test_evaluation=test_evaluation,
            )
            (run / "result.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    dict(
                        family=family,
                        seed=seed,
                        final_passed=result["passed"],
                        test_rates=test_evaluation["rates"] if test_evaluation else None,
                    )
                ),
                flush=True,
            )
            del model, optimizer, replay
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
