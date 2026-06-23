"""Direct Full SMB policy training, evaluation, resume, and checkpointing."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.optim as optim

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    SMB_ACTIONS,
    StageBatch,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    select_device,
    to_plain_data,
)
from retroagi.stages.full_smb.adapter import FULL_SMB_SPEC, FullSMBStage
from retroagi.stages.full_smb.transfer import (
    FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    FULL_SMB_TRANSFER_MODEL_NAME,
    load_transferred_full_smb_policy,
    make_full_smb_policy_model,
    policy_architecture_from_checkpoint,
)
from retroagi.stages.full_smb.vision import (
    DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    FullSMBSegmentationVision,
)

FULL_SMB_POLICY_MODEL_NAME = "full_smb_policy"
FULL_SMB_POLICY_CHECKPOINT_KIND = "full_smb_policy_trainer"
FULL_SMB_ACTION_COUNT = len(SMB_ACTIONS)


@dataclass(frozen=True)
class FullSMBTrainingConfig:
    """Configuration for direct emulator-level Full SMB policy updates."""

    seed: int = 0
    architecture_name: str = BASELINE_ARCHITECTURE_NAME
    architecture_config: Mapping[str, Any] = field(default_factory=dict)
    epochs: int = 1
    episodes_per_epoch: int = 1
    max_steps_per_episode: int = 64
    learning_rate: float = 1e-4
    entropy_weight: float = 0.01
    reward_scale: float = 1.0
    gradient_clip_norm: float = 1.0
    deterministic_actions: bool = False
    device: str = "auto"
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 64
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    init_checkpoint: Optional[Path] = None
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT
    freeze_vision: bool = True
    save_checkpoints: bool = False
    output_summary: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.architecture_name:
            raise ValueError("architecture_name must be non-empty")
        if any(not str(key) for key in self.architecture_config):
            raise ValueError("architecture_config keys must be non-empty")
        for name in (
            "epochs",
            "episodes_per_epoch",
            "max_steps_per_episode",
            "evaluation_episodes",
            "evaluation_max_steps",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("learning_rate", "reward_scale", "gradient_clip_norm"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative")
        if self.resume_path is not None and self.init_checkpoint is not None:
            raise ValueError("resume_path and init_checkpoint are mutually exclusive")


@dataclass(frozen=True)
class FullSMBEvaluationResult:
    """Deterministic Full SMB policy evaluation summary."""

    episodes: int
    max_steps_per_episode: int
    steps: int
    returns: tuple[float, ...]
    mean_return: float
    success_rate: float
    terminated_count: int
    truncated_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullSMBTrainingResult:
    """Artifacts from one direct Full SMB training run."""

    checkpoint: dict[str, Any]
    history: Mapping[str, list[float]]
    evaluation: FullSMBEvaluationResult
    checkpoint_path: Optional[Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": {
                "stage": self.checkpoint["stage"],
                "model_name": self.checkpoint["model_name"],
                "checkpoint_kind": self.checkpoint["checkpoint_kind"],
                "epoch": self.checkpoint["epoch"],
                "global_step": self.checkpoint["global_step"],
                "metrics": to_plain_data(self.checkpoint.get("metrics", {})),
                "config": to_plain_data(self.checkpoint.get("config", {})),
                "architecture": to_plain_data(self.checkpoint.get("architecture", {})),
            },
            "history": to_plain_data(self.history),
            "evaluation": self.evaluation.as_dict(),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


StageFactory = Callable[[FullSMBSegmentationVision], FullSMBStage]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_full_smb_policy(
    config: FullSMBTrainingConfig = FullSMBTrainingConfig(),
    *,
    make_stage: Optional[StageFactory] = None,
) -> FullSMBTrainingResult:
    """Run online policy-gradient updates directly against the Full SMB stage."""

    seed_everything(config.seed)
    device = select_device(config.device)
    model, optimizer, start_epoch, global_step, architecture_name, architecture_config = (
        _load_training_state(config, device)
    )
    history: dict[str, list[float]] = {"episode_return": [], "policy_loss": []}
    vision = FullSMBSegmentationVision(
        checkpoint=config.full_smb_vision_checkpoint,
        device=device,
        freeze=config.freeze_vision,
    )
    stage = _make_stage(make_stage, vision)
    try:
        model.train()
        for epoch in range(start_epoch, config.epochs):
            for episode_index in range(config.episodes_per_epoch):
                episode_seed = config.seed + epoch * config.episodes_per_epoch + episode_index
                metrics = _train_episode(
                    model,
                    optimizer,
                    stage,
                    seed=episode_seed,
                    max_steps=config.max_steps_per_episode,
                    device=device,
                    reward_scale=config.reward_scale,
                    entropy_weight=config.entropy_weight,
                    gradient_clip_norm=config.gradient_clip_norm,
                    deterministic_actions=config.deterministic_actions,
                )
                global_step += int(metrics["steps"])
                history["episode_return"].append(float(metrics["return"]))
                history["policy_loss"].append(float(metrics["loss"]))
        evaluation = evaluate_full_smb_policy(
            model,
            config=config,
            make_stage=make_stage,
            vision=vision,
            device=device,
        )
    finally:
        stage.close()

    checkpoint = build_full_smb_policy_checkpoint(
        model,
        optimizer,
        epoch=config.epochs,
        global_step=global_step,
        config=config,
        metrics={
            "mean_train_return": _mean(history["episode_return"]),
            "mean_policy_loss": _mean(history["policy_loss"]),
            "evaluation_mean_return": evaluation.mean_return,
            "evaluation_success_rate": evaluation.success_rate,
        },
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    )
    checkpoint_path = config.checkpoint_path
    if checkpoint_path is not None and config.save_checkpoints:
        save_checkpoint(checkpoint_path, checkpoint)
    result = FullSMBTrainingResult(
        checkpoint=checkpoint,
        history=history,
        evaluation=evaluation,
        checkpoint_path=checkpoint_path if config.save_checkpoints else None,
    )
    if config.output_summary is not None:
        save_full_smb_training_summary(config.output_summary, result)
    return result


@torch.no_grad()
def evaluate_full_smb_policy(
    model: torch.nn.Module,
    *,
    config: FullSMBTrainingConfig = FullSMBTrainingConfig(),
    make_stage: Optional[StageFactory] = None,
    vision: Optional[FullSMBSegmentationVision] = None,
    device: Optional[torch.device] = None,
) -> FullSMBEvaluationResult:
    """Evaluate a Full SMB policy with deterministic action selection."""

    resolved_device = device or select_device(config.device)
    model.eval()
    owns_vision = vision is None
    if vision is None:
        vision = FullSMBSegmentationVision(
            checkpoint=config.full_smb_vision_checkpoint,
            device=resolved_device,
            freeze=config.freeze_vision,
        )
    stage = _make_stage(make_stage, vision)
    returns: list[float] = []
    steps = 0
    terminated_count = 0
    truncated_count = 0
    try:
        for episode_index in range(config.evaluation_episodes):
            observation = stage.reset(seed=config.seed + 10_000 + episode_index)
            episode_return = 0.0
            for _step in range(config.evaluation_max_steps):
                batch = stage.encode_observation(observation)
                logits = _policy_action_logits(model, batch, device=resolved_device)
                action = int(logits.argmax(dim=-1).item())
                observation, reward, terminated, truncated, _info = stage.step(action)
                episode_return += float(reward)
                steps += 1
                if terminated:
                    terminated_count += 1
                if truncated:
                    truncated_count += 1
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
    finally:
        stage.close()
        if owns_vision:
            del vision

    episodes = len(returns)
    successes = terminated_count
    return FullSMBEvaluationResult(
        episodes=episodes,
        max_steps_per_episode=config.evaluation_max_steps,
        steps=steps,
        returns=tuple(returns),
        mean_return=_mean(returns),
        success_rate=float(successes / episodes) if episodes else 0.0,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
    )


def load_full_smb_policy_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, optim.Optimizer, dict[str, Any]]:
    """Load a direct Full SMB policy checkpoint with optimizer state."""

    checkpoint = load_checkpoint(path, map_location=device)
    _validate_full_smb_policy_checkpoint(checkpoint)
    architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
    model = make_full_smb_policy_model(
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    ).to(device)
    model.load_state_dict(checkpoint["states"]["model"])
    optimizer = optim.AdamW(model.parameters())
    if "optimizer" in checkpoint["states"]:
        optimizer.load_state_dict(checkpoint["states"]["optimizer"])
    return model, optimizer, checkpoint


def build_full_smb_policy_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    config: FullSMBTrainingConfig,
    metrics: Mapping[str, float],
    architecture_name: str,
    architecture_config: Mapping[str, Any],
) -> dict[str, Any]:
    return build_checkpoint(
        stage=FULL_SMB_SPEC.name,
        model_name=FULL_SMB_POLICY_MODEL_NAME,
        checkpoint_kind=FULL_SMB_POLICY_CHECKPOINT_KIND,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        config={
            **to_plain_data(config),
            "architecture_name": architecture_name,
            "architecture_config": dict(architecture_config),
        },
        specs={
            "stage": {
                "name": FULL_SMB_SPEC.name,
                "seq_len_a": FULL_SMB_SPEC.seq_len_a,
                "seq_len_b": FULL_SMB_SPEC.seq_len_b,
                "seq_len_c": FULL_SMB_SPEC.seq_len_c,
                "ratio_bc": FULL_SMB_SPEC.ratio_bc,
                "vocab_size": FULL_SMB_SPEC.vocab_size,
            }
        },
        states={
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        },
    )


def save_full_smb_training_summary(path: Path, result: FullSMBTrainingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_training_state(
    config: FullSMBTrainingConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, optim.Optimizer, int, int, str, dict[str, Any]]:
    if config.resume_path is not None:
        model, optimizer, checkpoint = load_full_smb_policy_checkpoint(
            config.resume_path,
            device=device,
        )
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        return (
            model,
            optimizer,
            int(checkpoint.get("epoch", 0)),
            int(checkpoint.get("global_step", 0)),
            architecture_name,
            architecture_config,
        )

    if config.init_checkpoint is not None:
        transferred = load_transferred_full_smb_policy(
            config.init_checkpoint,
            full_smb_vision_checkpoint=config.full_smb_vision_checkpoint,
            device=device,
            freeze_vision=config.freeze_vision,
        )
        model = transferred.model
        architecture_name, architecture_config = policy_architecture_from_checkpoint(
            transferred.checkpoint
        )
    else:
        architecture_name = config.architecture_name
        architecture_config = dict(config.architecture_config)
        model = make_full_smb_policy_model(
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer, 0, 0, architecture_name, dict(architecture_config)


def _train_episode(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    stage: FullSMBStage,
    *,
    seed: int,
    max_steps: int,
    device: torch.device,
    reward_scale: float,
    entropy_weight: float,
    gradient_clip_norm: float,
    deterministic_actions: bool,
) -> dict[str, float]:
    observation = stage.reset(seed=seed)
    total_return = 0.0
    losses: list[float] = []
    for _step in range(max_steps):
        batch = stage.encode_observation(observation)
        logits = _policy_action_logits(model, batch, device=device)
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic_actions:
            action_tensor = logits.argmax(dim=-1)
        else:
            action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        observation, reward, terminated, truncated, _info = stage.step(int(action_tensor.item()))
        scaled_reward = torch.as_tensor(
            float(reward) * reward_scale,
            dtype=log_prob.dtype,
            device=log_prob.device,
        )
        entropy = distribution.entropy().mean()
        loss = -(log_prob.mean() * scaled_reward.detach()) - entropy_weight * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        total_return += float(reward)
        if terminated or truncated:
            break
    return {"return": total_return, "loss": _mean(losses), "steps": float(len(losses))}


def _policy_action_logits(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    src_a = batch.src_a.to(device)
    src_b = batch.src_b.to(device)
    src_c = batch.src_c.to(device)
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(episode_mask, dtype=src_c.dtype, device=src_c.device)
    outputs = model(src_a, src_b, src_c, tau=1.0, episode_mask=episode_mask)
    logits_a = outputs[4]
    return logits_a[:, -1, :FULL_SMB_ACTION_COUNT]


def _make_stage(
    make_stage: Optional[StageFactory],
    vision: FullSMBSegmentationVision,
) -> FullSMBStage:
    if make_stage is not None:
        return make_stage(vision)
    return FullSMBStage(vision=vision)


def _validate_full_smb_policy_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint["stage"] != FULL_SMB_SPEC.name:
        raise ValueError("checkpoint stage does not match Full SMB")
    if checkpoint["model_name"] not in {
        FULL_SMB_POLICY_MODEL_NAME,
        FULL_SMB_TRANSFER_MODEL_NAME,
    }:
        raise ValueError("checkpoint model does not match Full SMB policy")
    if checkpoint["checkpoint_kind"] not in {
        FULL_SMB_POLICY_CHECKPOINT_KIND,
        FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    }:
        raise ValueError("checkpoint kind does not match Full SMB policy")
    if "model" not in checkpoint["states"]:
        raise ValueError("checkpoint is missing states.model")


def _mean(values: list[float] | tuple[float, ...]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _architecture_config_item(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("architecture config must use KEY=VALUE syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("architecture config key must be non-empty")
    return key, _parse_value(raw_value.strip())


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_common_args(train)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--episodes-per-epoch", type=int, default=1)
    train.add_argument("--max-steps-per-episode", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--entropy-weight", type=float, default=0.01)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--deterministic-actions", action="store_true")
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--init-checkpoint", type=Path)
    train.add_argument("--save-checkpoints", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    _add_common_args(evaluate)
    evaluate.add_argument("--policy-checkpoint", type=Path, required=True)

    args = parser.parse_args(argv)
    config = _config_from_args(args)
    if args.command == "train":
        result = train_full_smb_policy(config)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        model, _optimizer, checkpoint = load_full_smb_policy_checkpoint(
            args.policy_checkpoint,
            device=select_device(config.device),
        )
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        eval_config = FullSMBTrainingConfig(
            **{
                **to_plain_data(config),
                "architecture_name": architecture_name,
                "architecture_config": architecture_config,
            }
        )
        result = evaluate_full_smb_policy(model, config=eval_config)
        if config.output_summary is not None:
            config.output_summary.parent.mkdir(parents=True, exist_ok=True)
            config.output_summary.write_text(
                json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architecture", default=BASELINE_ARCHITECTURE_NAME)
    parser.add_argument(
        "--architecture-config",
        action="append",
        type=_architecture_config_item,
        default=[],
    )
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument("--fine-tune-vision", action="store_true")
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--evaluation-max-steps", type=int, default=64)
    parser.add_argument("--output-summary", type=Path)


def _config_from_args(args: argparse.Namespace) -> FullSMBTrainingConfig:
    architecture_config = dict(args.architecture_config or ())
    return FullSMBTrainingConfig(
        seed=args.seed,
        architecture_name=args.architecture,
        architecture_config=architecture_config,
        epochs=getattr(args, "epochs", 0),
        episodes_per_epoch=getattr(args, "episodes_per_epoch", 0),
        max_steps_per_episode=getattr(args, "max_steps_per_episode", 0),
        learning_rate=getattr(args, "learning_rate", 1e-4),
        entropy_weight=getattr(args, "entropy_weight", 0.01),
        reward_scale=getattr(args, "reward_scale", 1.0),
        gradient_clip_norm=getattr(args, "gradient_clip_norm", 1.0),
        deterministic_actions=getattr(args, "deterministic_actions", False),
        device=args.device,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps,
        checkpoint_path=getattr(args, "checkpoint", None),
        resume_path=getattr(args, "resume", None),
        init_checkpoint=getattr(args, "init_checkpoint", None),
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        freeze_vision=not args.fine_tune_vision,
        save_checkpoints=getattr(args, "save_checkpoints", False)
        or getattr(args, "checkpoint", None) is not None,
        output_summary=args.output_summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())
