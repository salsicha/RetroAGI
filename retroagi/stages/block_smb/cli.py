"""Command line entry point for Block SMB training and evaluation."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from retroagi.core import load_checkpoint, select_device, to_plain_data

from .env import BlockSMBRewardConfig, MarioScenarioEnv
from .train import BlockSMBTrainingConfig, train_and_evaluate_block_smb
from .vision import evaluate_block_vit_perception, load_block_vit_checkpoint

DEFAULT_RECORD_DIR = Path("artifacts/block_smb/recordings")
DEFAULT_VISION_DIAGNOSTIC_SAMPLES = 64
DEFAULT_VISION_DIAGNOSTIC_ROLLOUT_STEPS = 32


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed > 0:
        raise argparse.ArgumentTypeError("must be non-positive")
    return parsed


REWARD_CONFIG_ARGS = {
    "reward_progress_per_pixel": "progress_per_pixel",
    "reward_coin": "coin",
    "reward_enemy_stomp": "enemy_stomp",
    "reward_goal": "goal",
    "reward_fall_death": "fall_death",
    "reward_enemy_hit": "enemy_hit",
    "reward_frame_penalty": "frame_penalty",
}


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, help="write the resolved run summary JSON")
    parser.add_argument(
        "--vision-checkpoint",
        type=Path,
        help="Block ViT checkpoint to freeze and use for policy observations",
    )
    parser.add_argument("--seed", type=int, help="base random seed")
    parser.add_argument("--epochs", type=_positive_int, help="total training epochs")
    parser.add_argument("--episodes-per-epoch", type=_positive_int)
    parser.add_argument("--rollout-steps", type=_positive_int)
    parser.add_argument("--learning-rate", type=_positive_float)
    parser.add_argument("--gamma", type=_positive_float)
    parser.add_argument("--entropy-weight", type=_non_negative_float)
    parser.add_argument("--policy-loss-weight", type=_non_negative_float)
    parser.add_argument("--representation-weight", type=_non_negative_float)
    parser.add_argument("--world-model-weight", type=_non_negative_float)
    parser.add_argument("--reward-loss-weight", type=_non_negative_float)
    parser.add_argument("--value-loss-weight", type=_non_negative_float)
    parser.add_argument("--action-aux-weight", type=_non_negative_float)
    parser.add_argument("--critic-loss-weight", type=_non_negative_float)
    parser.add_argument("--reward-progress-per-pixel", type=_non_negative_float)
    parser.add_argument("--reward-coin", type=_non_negative_float)
    parser.add_argument("--reward-enemy-stomp", type=_non_negative_float)
    parser.add_argument("--reward-goal", type=_non_negative_float)
    parser.add_argument("--reward-fall-death", type=_non_positive_float)
    parser.add_argument("--reward-enemy-hit", type=_non_positive_float)
    parser.add_argument("--reward-frame-penalty", type=_non_positive_float)
    parser.add_argument("--gradient-clip-norm", type=_positive_float)
    parser.add_argument("--hidden-dim", type=_positive_int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--fixed-scenario", action="append", dest="fixed_scenarios")
    parser.add_argument("--generated-scenarios", type=_non_negative_int)
    parser.add_argument("--generated-seed", type=int)
    parser.add_argument("--evaluation-episodes", type=_positive_int)
    parser.add_argument("--evaluation-max-steps", type=_positive_int)
    parser.add_argument("--num-envs", type=_positive_int)
    parser.set_defaults(deterministic=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        dest="deterministic",
        help="enable deterministic Torch algorithms",
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_false",
        dest="deterministic",
        help="disable deterministic Torch algorithms",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi-block-smb",
        description="Train, evaluate, resume, and record the Block SMB agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train or resume a Block SMB policy")
    _add_common_config_args(train)
    train.add_argument("--checkpoint", type=Path, help="checkpoint path to save")
    train.add_argument("--resume", type=Path, help="checkpoint path to resume")
    train.add_argument(
        "--record",
        action="store_true",
        help="record deterministic evaluation trajectories after each epoch",
    )
    train.add_argument("--record-dir", type=Path, help="directory for recorded trajectories")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved Block SMB checkpoint")
    _add_common_config_args(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)

    record = subparsers.add_parser(
        "record", help="evaluate a saved Block SMB checkpoint and write trajectory files"
    )
    _add_common_config_args(record)
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)

    diagnose = subparsers.add_parser(
        "diagnose-vision",
        help="measure Block ViT semantic and position quality on procedural frames",
    )
    diagnose.add_argument("--output", type=Path, help="write the diagnostic JSON")
    diagnose.add_argument("--vision-checkpoint", type=Path)
    diagnose.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    diagnose.add_argument("--seed", type=int, default=7)
    diagnose.add_argument(
        "--samples", type=_positive_int, default=DEFAULT_VISION_DIAGNOSTIC_SAMPLES
    )
    diagnose.add_argument(
        "--rollout-steps",
        type=_positive_int,
        default=DEFAULT_VISION_DIAGNOSTIC_ROLLOUT_STEPS,
    )
    diagnose.add_argument("--batch-size", type=_positive_int, default=32)
    return parser


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "seed",
        "epochs",
        "episodes_per_epoch",
        "rollout_steps",
        "learning_rate",
        "gamma",
        "entropy_weight",
        "policy_loss_weight",
        "representation_weight",
        "world_model_weight",
        "reward_loss_weight",
        "value_loss_weight",
        "action_aux_weight",
        "critic_loss_weight",
        "gradient_clip_norm",
        "hidden_dim",
        "device",
        "fixed_scenarios",
        "generated_scenarios",
        "generated_seed",
        "evaluation_episodes",
        "evaluation_max_steps",
        "num_envs",
        "deterministic",
    )
    overrides = {
        name: getattr(args, name)
        for name in names
        if hasattr(args, name) and getattr(args, name) is not None
    }
    if "fixed_scenarios" in overrides:
        overrides["fixed_scenarios"] = tuple(overrides["fixed_scenarios"])
    return overrides


def _apply_reward_config_overrides(
    values: dict[str, Any], args: argparse.Namespace
) -> None:
    overrides = {
        field_name: getattr(args, arg_name)
        for arg_name, field_name in REWARD_CONFIG_ARGS.items()
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None
    }
    if not overrides:
        return
    current = values.get("reward_config", BlockSMBRewardConfig())
    if isinstance(current, BlockSMBRewardConfig):
        reward_values = asdict(current)
    elif isinstance(current, Mapping):
        reward_values = dict(current)
    else:
        raise TypeError("reward_config must be a BlockSMBRewardConfig or mapping")
    reward_values.update(overrides)
    values["reward_config"] = BlockSMBRewardConfig(**reward_values)


def _normalize_config_values(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    for name in ("checkpoint_path", "resume_path", "video_dir"):
        if normalized.get(name) is not None:
            normalized[name] = Path(normalized[name])
    if normalized.get("fixed_scenarios") is not None:
        normalized["fixed_scenarios"] = tuple(normalized["fixed_scenarios"])
    if normalized.get("reward_config") is not None:
        reward_config = normalized["reward_config"]
        if not isinstance(reward_config, BlockSMBRewardConfig):
            normalized["reward_config"] = BlockSMBRewardConfig(
                **dict(reward_config)
            )
    return normalized


def _checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path)
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint {path} does not contain a configuration mapping")
    values = _normalize_config_values(config)
    values["resume_path"] = path
    values["save_checkpoints"] = False
    values["record_videos"] = False
    if "epochs" not in values:
        values["epochs"] = max(1, int(checkpoint.get("epoch", 1)))
    else:
        values["epochs"] = max(int(values["epochs"]), int(checkpoint.get("epoch", 0)))
    return values


def _make_train_config(args: argparse.Namespace) -> BlockSMBTrainingConfig:
    values = _config_overrides(args)
    _apply_reward_config_overrides(values, args)
    if args.checkpoint is not None:
        values["checkpoint_path"] = args.checkpoint
        values["save_checkpoints"] = True
    if args.resume is not None:
        values["resume_path"] = args.resume
    record_dir = args.record_dir
    if args.record and record_dir is None:
        record_dir = DEFAULT_RECORD_DIR
    if record_dir is not None:
        values["video_dir"] = record_dir
        values["record_videos"] = True
    return BlockSMBTrainingConfig(**values)


def _make_checkpoint_config(
    args: argparse.Namespace, *, record: bool
) -> BlockSMBTrainingConfig:
    values = _checkpoint_config(args.checkpoint)
    values.update(_config_overrides(args))
    _apply_reward_config_overrides(values, args)
    values["resume_path"] = args.checkpoint
    values["save_checkpoints"] = False
    values["record_videos"] = record
    if record:
        values["video_dir"] = args.record_dir
    else:
        values["video_dir"] = None
    return BlockSMBTrainingConfig(**values)


def _make_vision_factory(
    config: BlockSMBTrainingConfig,
    checkpoint_path: Path | None,
) -> tuple[Any, dict[str, Any]]:
    device = select_device(config.device)
    cache: dict[str, Any] = {}
    vision_info: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "frozen": True,
    }

    def factory():
        if "model" not in cache:
            loaded = load_block_vit_checkpoint(
                checkpoint_path,
                device=device,
                freeze=True,
            )
            cache["model"] = loaded.model
            vision_info["checkpoint_path"] = str(loaded.path)
            vision_info["frozen"] = loaded.frozen
        return cache["model"]

    return factory, vision_info


def _sample_vision_diagnostic_action(rng: random.Random) -> int:
    return rng.choices((0, 1, 2, 3, 4, 5), weights=(5, 25, 35, 2, 3, 10), k=1)[0]


def _collect_vision_diagnostic_frames(
    *,
    samples: int,
    seed: int,
    rollout_steps: int,
) -> torch.Tensor:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    rng = random.Random(seed)
    env = MarioScenarioEnv()
    frames = []
    scenario_index = 0
    try:
        while len(frames) < samples:
            scenario_seed = seed + scenario_index
            scenario = MarioScenarioEnv.generate_scenario(
                num_screens=rng.randint(1, 3),
                enemy_density=rng.uniform(0.25, 0.9),
                moving_platform_chance=rng.uniform(0.1, 0.5),
                seed=scenario_seed,
            )
            observation, _info = env.reset(scenario=scenario, seed=scenario_seed)
            frames.append(observation.copy())
            for _ in range(rollout_steps - 1):
                if len(frames) >= samples:
                    break
                action = _sample_vision_diagnostic_action(rng)
                observation, _reward, terminated, truncated, _info = env.step(action)
                frames.append(observation.copy())
                if terminated or truncated:
                    break
            scenario_index += 1
    finally:
        env.close()
    return torch.from_numpy(np.stack(frames[:samples])).to(torch.uint8)


def _run_vision_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    device = select_device(args.device)
    loaded = load_block_vit_checkpoint(
        args.vision_checkpoint,
        device=device,
        freeze=True,
    )
    frames = _collect_vision_diagnostic_frames(
        samples=args.samples,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
    )
    metrics = evaluate_block_vit_perception(
        loaded.model,
        frames,
        batch_size=args.batch_size,
    )
    return {
        "config": {
            "samples": args.samples,
            "seed": args.seed,
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "vision": {
            "checkpoint_path": str(loaded.path),
            "frozen": loaded.frozen,
        },
        "perception": metrics,
    }


def _public_result(
    result: Mapping[str, Any],
    config: BlockSMBTrainingConfig,
    *,
    vision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "config": to_plain_data(config),
        "vision": dict(vision),
        "history": result.get("history", []),
        "metrics": result.get("metrics", {}),
        "evaluation": result.get("evaluation", {}),
        "curriculum": result.get("curriculum", []),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "train":
        config = _make_train_config(args)
    elif args.command == "evaluate":
        config = _make_checkpoint_config(args, record=False)
    elif args.command == "record":
        config = _make_checkpoint_config(args, record=True)
    elif args.command == "diagnose-vision":
        return _run_vision_diagnostic(args)
    else:
        raise ValueError(f"unknown command {args.command!r}")
    vision_factory, vision_info = _make_vision_factory(
        config,
        getattr(args, "vision_checkpoint", None),
    )
    result = train_and_evaluate_block_smb(config, vision_factory=vision_factory)
    return _public_result(result, config, vision=vision_info)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    output = json.dumps(result, indent=2, sort_keys=True)
    if getattr(args, "output", None) is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
