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

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    SUPPORTED_CONTROLLER_SCHEDULES,
    TRACKING_BACKENDS,
    architecture_names,
    load_checkpoint,
    select_device,
    to_plain_data,
)

from .env import BlockSMBRewardConfig, MarioScenarioEnv
from .monte_carlo import BLOCK_SMB_MC_FAMILIES, DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID
from .train import (
    TARGET_NETWORK_MODES,
    BlockSMBAblationConfig,
    BlockSMBTrainingConfig,
    block_smb_architecture_metadata,
    block_smb_monte_carlo_sweep_sample_count,
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
    restore_block_smb_checkpoint,
    train_and_evaluate_block_smb,
)
from .vision import (
    BlockVisionTransformer,
    evaluate_block_vit_perception,
    load_block_vit_checkpoint,
)

DEFAULT_RECORD_DIR = Path("artifacts/block_smb/recordings")
DEFAULT_VISION_DIAGNOSTIC_SAMPLES = 64
DEFAULT_VISION_DIAGNOSTIC_ROLLOUT_STEPS = 32
DEFAULT_ACTION_PROBE_OUTPUT = Path("artifacts/block_smb/action_probe.json")


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


def _architecture_name(value: str) -> str:
    normalized = value.strip()
    if normalized.lower() == "baseline":
        return BASELINE_ARCHITECTURE_NAME
    available = set(architecture_names())
    if normalized in available:
        return normalized
    choices = ", ".join(sorted({"baseline", *available}))
    raise argparse.ArgumentTypeError(f"unknown architecture {value!r}; expected one of: {choices}")


def _architecture_config_item(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use KEY=VALUE syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("architecture config key must be non-empty")
    return key, _parse_architecture_config_value(raw_value.strip())


def _slot_weight_item(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use SLOT=WEIGHT syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("slot name must be non-empty")
    weight = float(raw_value)
    if weight <= 0:
        raise argparse.ArgumentTypeError("slot weight must be positive")
    return key, weight


def _family_weight_item(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use FAMILY=WEIGHT syntax")
    family, raw_weight = value.split("=", 1)
    family = family.strip()
    if family not in BLOCK_SMB_MC_FAMILIES:
        choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
        raise argparse.ArgumentTypeError(f"unknown family {family!r}; expected one of: {choices}")
    weight = float(raw_weight)
    if weight < 0:
        raise argparse.ArgumentTypeError("family weight must be non-negative")
    return family, weight


def _parse_architecture_config_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


REWARD_CONFIG_ARGS = {
    "reward_progress_per_pixel": "progress_per_pixel",
    "reward_coin": "coin",
    "reward_enemy_stomp": "enemy_stomp",
    "reward_goal": "goal",
    "reward_fall_death": "fall_death",
    "reward_gap_jump": "gap_jump",
    "reward_enemy_hit": "enemy_hit",
    "reward_frame_penalty": "frame_penalty",
}

ABLATION_CONFIG_FIELDS = (
    "vision_enabled",
    "world_model_enabled",
    "critic_feedback_enabled",
    "hierarchy_enabled",
    "recurrent_state_enabled",
    "checkpoint_transfer_enabled",
)


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
    parser.add_argument(
        "--world-model-slot-weight",
        action="append",
        default=None,
        type=_slot_weight_item,
        metavar="SLOT=WEIGHT",
        help=(
            "weight a Block SMB C-stream dynamics slot; slots: position, "
            "semantic_probabilities, support_state, state, patch_tokens"
        ),
    )
    parser.add_argument("--reward-loss-weight", type=_non_negative_float)
    parser.add_argument("--value-loss-weight", type=_non_negative_float)
    parser.add_argument("--action-aux-weight", type=_non_negative_float)
    parser.add_argument("--critic-loss-weight", type=_non_negative_float)
    parser.add_argument("--imagined-rollout-weight", type=_non_negative_float)
    parser.add_argument("--imagined-rollout-horizon", type=_non_negative_int)
    parser.add_argument("--target-network-mode", choices=TARGET_NETWORK_MODES)
    parser.add_argument("--target-network-tau", type=_positive_float)
    parser.add_argument(
        "--target-network-instability-threshold",
        type=_non_negative_float,
    )
    parser.add_argument("--reward-progress-per-pixel", type=_non_negative_float)
    parser.add_argument("--reward-coin", type=_non_negative_float)
    parser.add_argument("--reward-enemy-stomp", type=_non_negative_float)
    parser.add_argument("--reward-goal", type=_non_negative_float)
    parser.add_argument("--reward-fall-death", type=_non_positive_float)
    parser.add_argument("--reward-gap-jump", type=_non_positive_float)
    parser.add_argument("--reward-enemy-hit", type=_non_positive_float)
    parser.add_argument("--reward-frame-penalty", type=_non_positive_float)
    parser.add_argument("--gradient-clip-norm", type=_positive_float)
    parser.add_argument("--hidden-dim", type=_positive_int)
    parser.add_argument(
        "--controller-schedule",
        choices=SUPPORTED_CONTROLLER_SCHEDULES,
        help="low-level controller gain schedule",
    )
    parser.add_argument(
        "--architecture",
        dest="architecture_name",
        type=_architecture_name,
        help="model architecture to instantiate; use 'baseline' for the default",
    )
    parser.add_argument(
        "--architecture-config",
        action="append",
        default=None,
        type=_architecture_config_item,
        metavar="KEY=VALUE",
        help="architecture-specific config override; may be repeated",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--fixed-scenario", action="append", dest="fixed_scenarios")
    parser.add_argument("--generated-scenarios", type=_non_negative_int)
    parser.add_argument("--generated-seed", type=int)
    parser.add_argument(
        "--monte-carlo-distribution",
        dest="monte_carlo_distribution_id",
        default=None,
        help=(
            "Block SMB Monte Carlo distribution ID; defaults to "
            f"{DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID}"
        ),
    )
    parser.add_argument(
        "--monte-carlo-train-samples-per-epoch",
        type=_non_negative_int,
        help="number of replayable Monte Carlo train samples in each curriculum epoch",
    )
    parser.add_argument("--monte-carlo-seed", type=int)
    parser.add_argument(
        "--monte-carlo-family-weight",
        action="append",
        default=None,
        type=_family_weight_item,
        metavar="FAMILY=WEIGHT",
        help="weighted family sampler override; may be repeated",
    )
    parser.set_defaults(monte_carlo_parameter_sweep=None)
    parser.add_argument(
        "--monte-carlo-parameter-sweep",
        action="store_true",
        dest="monte_carlo_parameter_sweep",
        help="use the deterministic full family x difficulty Monte Carlo sweep",
    )
    parser.add_argument(
        "--random-monte-carlo-sampling",
        action="store_false",
        dest="monte_carlo_parameter_sweep",
        help="use weighted/random Monte Carlo sampling instead of the full sweep",
    )
    parser.add_argument(
        "--monte-carlo-sweep-repeats-per-difficulty",
        type=_positive_int,
        help="number of deterministic sweep variants per family/difficulty bin",
    )
    parser.add_argument(
        "--monte-carlo-max-rejections",
        type=_non_negative_int,
        help="maximum unreachable samples to reject per Monte Carlo sample index",
    )
    parser.add_argument(
        "--monte-carlo-validation-samples",
        type=_non_negative_int,
        help="held-out validation samples to attach to Block SMB evaluation",
    )
    parser.add_argument(
        "--monte-carlo-test-samples",
        type=_non_negative_int,
        help="held-out test samples to attach to Block SMB evaluation",
    )
    parser.add_argument(
        "--monte-carlo-failure-replay-samples-per-epoch",
        type=_non_negative_int,
        help="additional train samples weighted by recent Monte Carlo validation failures",
    )
    parser.add_argument(
        "--monte-carlo-pass-rate-gate",
        type=float,
        help="minimum held-out Monte Carlo pass rate for promotion gating",
    )
    parser.add_argument(
        "--monte-carlo-family-pass-rate-gate",
        type=float,
        help="minimum per-family Monte Carlo pass rate for promotion gating",
    )
    parser.set_defaults(monte_carlo_validate_reachability=None)
    parser.add_argument(
        "--validate-monte-carlo-reachability",
        action="store_true",
        dest="monte_carlo_validate_reachability",
    )
    parser.add_argument(
        "--skip-monte-carlo-reachability-validation",
        action="store_false",
        dest="monte_carlo_validate_reachability",
    )
    parser.add_argument("--evaluation-episodes", type=_positive_int)
    parser.add_argument("--evaluation-max-steps", type=_positive_int)
    parser.add_argument("--evaluation-interval-epochs", type=_positive_int)
    parser.add_argument("--semantic-prediction-accuracy-threshold", type=float)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--tracking-backend", choices=TRACKING_BACKENDS)
    parser.add_argument("--tracking-log-dir", type=Path)
    parser.add_argument("--tracking-project")
    parser.add_argument("--tracking-run-name")
    parser.add_argument("--tracking-mode")
    parser.add_argument("--num-envs", type=_positive_int)
    parser.set_defaults(
        vision_enabled=None,
        world_model_enabled=None,
        critic_feedback_enabled=None,
        hierarchy_enabled=None,
        recurrent_state_enabled=None,
        checkpoint_transfer_enabled=None,
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        dest="vision_enabled",
        help="enable visual observation features in Block SMB policy input",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_false",
        dest="vision_enabled",
        help="zero visual A/B streams and visual C slots, preserving symbolic state",
    )
    parser.add_argument(
        "--enable-world-model",
        action="store_true",
        dest="world_model_enabled",
        help="use learned dynamics for next-state prediction and imagined rollouts",
    )
    parser.add_argument(
        "--disable-world-model",
        action="store_false",
        dest="world_model_enabled",
        help="bypass learned dynamics and use the current C state as prediction",
    )
    parser.add_argument(
        "--enable-critic-feedback",
        action="store_true",
        dest="critic_feedback_enabled",
        help="inject critic feedback into the actor's second pass",
    )
    parser.add_argument(
        "--disable-critic-feedback",
        action="store_false",
        dest="critic_feedback_enabled",
        help="run the actor's second pass without critic feedback injection",
    )
    parser.add_argument(
        "--enable-hierarchy",
        action="store_true",
        dest="hierarchy_enabled",
        help="use A/B semantic hierarchy streams",
    )
    parser.add_argument(
        "--disable-hierarchy",
        action="store_false",
        dest="hierarchy_enabled",
        help="collapse A/B hierarchy streams to background tokens",
    )
    parser.add_argument(
        "--enable-recurrent-state",
        action="store_true",
        dest="recurrent_state_enabled",
        help="carry world-model recurrent state across rollout steps",
    )
    parser.add_argument(
        "--disable-recurrent-state",
        action="store_false",
        dest="recurrent_state_enabled",
        help="reset world-model recurrent state for every rollout step",
    )
    parser.add_argument(
        "--enable-checkpoint-transfer",
        action="store_true",
        dest="checkpoint_transfer_enabled",
        help="load the Block ViT checkpoint for policy observations",
    )
    parser.add_argument(
        "--disable-checkpoint-transfer",
        action="store_false",
        dest="checkpoint_transfer_enabled",
        help="use a fresh randomly initialized Block ViT instead of loading a checkpoint",
    )
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

    evaluate_monte_carlo = subparsers.add_parser(
        "evaluate-monte-carlo",
        help="evaluate a saved Block SMB checkpoint on a held-out Monte Carlo split",
    )
    _add_common_config_args(evaluate_monte_carlo)
    evaluate_monte_carlo.add_argument("--checkpoint", type=Path, required=True)
    evaluate_monte_carlo.add_argument(
        "--split",
        choices=("train", "validation", "test", "stress"),
        default="validation",
    )
    evaluate_monte_carlo.add_argument(
        "--samples",
        type=_positive_int,
        help="number of sampled scenarios to evaluate; defaults to split config",
    )
    evaluate_monte_carlo.add_argument("--record-dir", type=Path)

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

    diagnose_actions = subparsers.add_parser(
        "diagnose-actions",
        help="probe policy logits at canonical pre-gap and pre-stair states",
    )
    diagnose_actions.add_argument("--checkpoint", type=Path, required=True)
    diagnose_actions.add_argument("--output", type=Path, default=DEFAULT_ACTION_PROBE_OUTPUT)
    diagnose_actions.add_argument("--vision-checkpoint", type=Path)
    diagnose_actions.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    diagnose_actions.add_argument("--seed", type=int, default=0)
    diagnose_actions.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        help="maximum scripted rollout steps used to reach probe states",
    )
    diagnose_actions.add_argument(
        "--points-per-scenario",
        type=_positive_int,
        default=None,
        help="number of scripted jump-transition states to log per scenario",
    )
    diagnose_actions.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help="fixed scenario to probe; may be repeated",
    )
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
        "world_model_slot_weight",
        "reward_loss_weight",
        "value_loss_weight",
        "action_aux_weight",
        "critic_loss_weight",
        "imagined_rollout_weight",
        "imagined_rollout_horizon",
        "target_network_mode",
        "target_network_tau",
        "target_network_instability_threshold",
        "gradient_clip_norm",
        "hidden_dim",
        "controller_schedule",
        "device",
        "fixed_scenarios",
        "generated_scenarios",
        "generated_seed",
        "monte_carlo_distribution_id",
        "monte_carlo_train_samples_per_epoch",
        "monte_carlo_seed",
        "monte_carlo_family_weight",
        "monte_carlo_parameter_sweep",
        "monte_carlo_sweep_repeats_per_difficulty",
        "monte_carlo_validate_reachability",
        "monte_carlo_max_rejections",
        "monte_carlo_validation_samples",
        "monte_carlo_test_samples",
        "monte_carlo_failure_replay_samples_per_epoch",
        "monte_carlo_pass_rate_gate",
        "monte_carlo_family_pass_rate_gate",
        "evaluation_episodes",
        "evaluation_max_steps",
        "evaluation_interval_epochs",
        "semantic_prediction_accuracy_threshold",
        "log_path",
        "tracking_backend",
        "tracking_log_dir",
        "tracking_project",
        "tracking_run_name",
        "tracking_mode",
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
    if "world_model_slot_weight" in overrides:
        overrides["world_model_slot_weights"] = dict(overrides.pop("world_model_slot_weight"))
    if "monte_carlo_family_weight" in overrides:
        overrides["monte_carlo_family_weights"] = dict(
            overrides.pop("monte_carlo_family_weight")
        )
    return overrides


def _apply_reward_config_overrides(values: dict[str, Any], args: argparse.Namespace) -> None:
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


def _apply_ablation_config_overrides(values: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        name: getattr(args, name)
        for name in ABLATION_CONFIG_FIELDS
        if hasattr(args, name) and getattr(args, name) is not None
    }
    if not overrides:
        return
    current = values.get("ablation", BlockSMBAblationConfig())
    if isinstance(current, BlockSMBAblationConfig):
        ablation_values = asdict(current)
    elif isinstance(current, Mapping):
        ablation_values = dict(current)
    else:
        raise TypeError("ablation must be a BlockSMBAblationConfig or mapping")
    ablation_values.update(overrides)
    values["ablation"] = BlockSMBAblationConfig(**ablation_values)


def _apply_architecture_overrides(values: dict[str, Any], args: argparse.Namespace) -> None:
    architecture_name = getattr(args, "architecture_name", None)
    if architecture_name is not None:
        values["architecture_name"] = architecture_name

    explicit_overrides = dict(getattr(args, "architecture_config", None) or ())
    legacy_overrides = {
        name: values[name]
        for name in ("hidden_dim", "controller_schedule")
        if name in values and name not in explicit_overrides
    }
    if not explicit_overrides and not legacy_overrides:
        return
    current = values.get("architecture_config", {})
    if current is None:
        current_values: dict[str, Any] = {}
    elif isinstance(current, Mapping):
        current_values = dict(current)
    else:
        raise TypeError("architecture_config must be a mapping")
    current_values.update(legacy_overrides)
    current_values.update(explicit_overrides)
    values["architecture_config"] = current_values


def _normalize_config_values(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    for name in (
        "checkpoint_path",
        "resume_path",
        "video_dir",
        "log_path",
        "vision_checkpoint_path",
        "tracking_log_dir",
    ):
        if normalized.get(name) is not None:
            normalized[name] = Path(normalized[name])
    if normalized.get("fixed_scenarios") is not None:
        normalized["fixed_scenarios"] = tuple(normalized["fixed_scenarios"])
    if normalized.get("reward_config") is not None:
        reward_config = normalized["reward_config"]
        if not isinstance(reward_config, BlockSMBRewardConfig):
            normalized["reward_config"] = BlockSMBRewardConfig(**dict(reward_config))
    if normalized.get("ablation") is not None:
        ablation = normalized["ablation"]
        if not isinstance(ablation, BlockSMBAblationConfig):
            normalized["ablation"] = BlockSMBAblationConfig(**dict(ablation))
    if normalized.get("architecture_config") is not None:
        architecture_config = normalized["architecture_config"]
        if not isinstance(architecture_config, Mapping):
            raise TypeError("architecture_config must be a mapping")
        normalized["architecture_config"] = dict(architecture_config)
    if normalized.get("monte_carlo_family_weights") is not None:
        family_weights = normalized["monte_carlo_family_weights"]
        if not isinstance(family_weights, Mapping):
            raise TypeError("monte_carlo_family_weights must be a mapping")
        normalized["monte_carlo_family_weights"] = dict(family_weights)
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
    _apply_ablation_config_overrides(values, args)
    _apply_architecture_overrides(values, args)
    if args.checkpoint is not None:
        values["checkpoint_path"] = args.checkpoint
        values["save_checkpoints"] = True
    if args.resume is not None:
        values["resume_path"] = args.resume
    if args.vision_checkpoint is not None:
        values["vision_checkpoint_path"] = args.vision_checkpoint
    record_dir = args.record_dir
    if args.record and record_dir is None:
        record_dir = DEFAULT_RECORD_DIR
    if record_dir is not None:
        values["video_dir"] = record_dir
        values["record_videos"] = True
    return BlockSMBTrainingConfig(**values)


def _make_checkpoint_config(args: argparse.Namespace, *, record: bool) -> BlockSMBTrainingConfig:
    values = _checkpoint_config(args.checkpoint)
    overrides = _config_overrides(args)
    values.update(overrides)
    if getattr(args, "vision_checkpoint", None) is not None:
        values["vision_checkpoint_path"] = args.vision_checkpoint
    _apply_reward_config_overrides(values, args)
    _apply_ablation_config_overrides(values, args)
    _apply_architecture_overrides(values, args)
    values["resume_path"] = args.checkpoint
    values["save_checkpoints"] = False
    values["record_videos"] = record
    if "log_path" not in overrides:
        values["log_path"] = None
    if "tracking_backend" not in overrides:
        values["tracking_backend"] = "none"
        values["tracking_log_dir"] = None
        values["tracking_run_name"] = None
        values["tracking_mode"] = None
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
    resolved_checkpoint_path = checkpoint_path or config.vision_checkpoint_path
    vision_info: dict[str, Any] = {
        "checkpoint_path": (
            str(resolved_checkpoint_path)
            if resolved_checkpoint_path is not None and config.ablation.checkpoint_transfer_enabled
            else None
        ),
        "frozen": True,
        "checkpoint_transfer": config.ablation.checkpoint_transfer_enabled,
    }

    def factory():
        if "model" not in cache:
            if config.ablation.checkpoint_transfer_enabled:
                loaded = load_block_vit_checkpoint(
                    resolved_checkpoint_path,
                    device=device,
                    freeze=True,
                )
                cache["model"] = loaded.model
                vision_info["checkpoint_path"] = str(loaded.path)
                vision_info["frozen"] = loaded.frozen
            else:
                model = BlockVisionTransformer().to(device)
                model.requires_grad_(False)
                model.eval()
                cache["model"] = model
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


def _run_action_probe(args: argparse.Namespace) -> dict[str, Any]:
    from retroagi.stages.block_smb.action_diagnostics import (
        DEFAULT_BLOCK_SMB_ACTION_PROBE_MAX_STEPS,
        DEFAULT_BLOCK_SMB_ACTION_PROBE_POINTS_PER_SCENARIO,
        DEFAULT_BLOCK_SMB_ACTION_PROBE_SCENARIOS,
        run_block_smb_action_probe,
    )

    config = _make_checkpoint_config(args, record=False)
    device = select_device(config.device)
    model = make_block_smb_model(config).to(device)
    restore_block_smb_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
        architecture_name=config.architecture_name,
        architecture_config=config.architecture_config,
    )
    vision_factory, vision_info = _make_vision_factory(
        config,
        getattr(args, "vision_checkpoint", None),
    )
    result = run_block_smb_action_probe(
        model,
        device=device,
        vision_factory=vision_factory,
        scenarios=tuple(args.scenarios or DEFAULT_BLOCK_SMB_ACTION_PROBE_SCENARIOS),
        seed=args.seed,
        max_steps=args.max_steps or DEFAULT_BLOCK_SMB_ACTION_PROBE_MAX_STEPS,
        points_per_scenario=(
            args.points_per_scenario
            or DEFAULT_BLOCK_SMB_ACTION_PROBE_POINTS_PER_SCENARIO
        ),
        ablation=config.ablation,
    )
    return {
        "checkpoint": {"path": str(args.checkpoint)},
        "architecture": block_smb_architecture_metadata(config),
        "vision": vision_info,
        **result,
    }


def _monte_carlo_cli_sample_count(
    config: BlockSMBTrainingConfig,
    *,
    split: str,
    explicit_samples: int | None,
) -> int:
    if explicit_samples is not None:
        return explicit_samples
    if config.monte_carlo_parameter_sweep:
        return block_smb_monte_carlo_sweep_sample_count(config)
    if split == "validation" and config.monte_carlo_validation_samples > 0:
        return config.monte_carlo_validation_samples
    if split == "test" and config.monte_carlo_test_samples > 0:
        return config.monte_carlo_test_samples
    if config.monte_carlo_train_samples_per_epoch > 0:
        return config.monte_carlo_train_samples_per_epoch
    if config.generated_scenarios > 0:
        return config.generated_scenarios
    return len(BLOCK_SMB_MC_FAMILIES)


def _run_monte_carlo_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    config = _make_checkpoint_config(args, record=False)
    device = select_device(config.device)
    model = make_block_smb_model(config).to(device)
    restore_block_smb_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
        architecture_name=config.architecture_name,
        architecture_config=config.architecture_config,
    )
    vision_factory, vision_info = _make_vision_factory(
        config,
        getattr(args, "vision_checkpoint", None),
    )
    evaluation = evaluate_block_smb_monte_carlo(
        model,
        config,
        split=args.split,
        sample_count=_monte_carlo_cli_sample_count(
            config,
            split=args.split,
            explicit_samples=args.samples,
        ),
        device=device,
        vision_factory=vision_factory,
        record_dir=args.record_dir,
    )
    return {
        "config": to_plain_data(config),
        "checkpoint": {"path": str(args.checkpoint)},
        "architecture": block_smb_architecture_metadata(config),
        "vision": vision_info,
        "evaluation": evaluation,
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
        "evaluations": result.get("evaluations", []),
        "metrics": result.get("metrics", {}),
        "evaluation": result.get("evaluation", {}),
        "curriculum": result.get("curriculum", []),
        "curriculum_summary": result.get("curriculum_summary", {}),
        "architecture": result.get("architecture", {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "train":
        config = _make_train_config(args)
    elif args.command == "evaluate":
        config = _make_checkpoint_config(args, record=False)
    elif args.command == "record":
        config = _make_checkpoint_config(args, record=True)
    elif args.command == "evaluate-monte-carlo":
        return _run_monte_carlo_evaluation(args)
    elif args.command == "diagnose-vision":
        return _run_vision_diagnostic(args)
    elif args.command == "diagnose-actions":
        return _run_action_probe(args)
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
