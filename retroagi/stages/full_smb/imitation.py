"""Real-emulator imitation warm starts for Full SMB policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core import (
    DEFAULT_PRIMITIVE_DURATION_BINS,
    SMBAction,
    StageBatch,
    save_checkpoint,
    select_device,
    smb_jump_release_action,
    to_plain_data,
)
from retroagi.stages.full_smb.adapter import (
    DEFAULT_FULL_SMB_CONTENT,
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.train import (
    FullSMBTrainingConfig,
    _build_full_smb_perception,
    _policy_action_logits_and_state,
    build_full_smb_policy_checkpoint,
    load_full_smb_policy_checkpoint,
)
from retroagi.stages.full_smb.transfer import policy_architecture_from_checkpoint
from retroagi.stages.full_smb.vision import DEFAULT_FULL_SMB_VIT_CHECKPOINT

DEFAULT_FULL_SMB_IMITATION_STEPS = 600
DEFAULT_FULL_SMB_IMITATION_BATCH_SIZE = 32
DEFAULT_FULL_SMB_IMITATION_EPOCHS = 3
DEFAULT_FULL_SMB_IMITATION_LR = 5e-4
DEFAULT_FULL_SMB_IMITATION_TRAINABLE_PREFIXES = (
    "agent.fc_out_A",
    "agent.fc_controller_params",
    "agent.fc_primitive_",
)


def full_smb_opening_imitation_script(
    max_steps: int = DEFAULT_FULL_SMB_IMITATION_STEPS,
) -> tuple[int, ...]:
    """Timed real-emulator opening script for Level 1-1 warm-start imitation."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    pattern = (
        [int(SMBAction.RIGHT)] * 160
        + [int(SMBAction.RIGHT_JUMP)] * 20
        + [int(SMBAction.RIGHT)] * 120
        + [int(SMBAction.RIGHT_JUMP)] * 20
    )
    actions: list[int] = []
    while len(actions) < max_steps:
        actions.extend(pattern)
    return tuple(actions[:max_steps])


@torch.no_grad()
def collect_full_smb_imitation_dataset(
    stage: FullSMBStage,
    actions: Sequence[int],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Collect encoded observations and scripted actions from the real stage."""

    if not actions:
        raise ValueError("actions must be non-empty")
    observation = stage.reset(seed=seed)
    src_a: list[torch.Tensor] = []
    src_b: list[torch.Tensor] = []
    src_c: list[torch.Tensor] = []
    targets: list[int] = []
    rewards: list[float] = []
    progress_values: list[float] = []
    terminated = False
    truncated = False
    for action in actions:
        batch = stage.encode_observation(observation)
        src_a.append(batch.src_a.detach().cpu())
        src_b.append(batch.src_b.detach().cpu())
        src_c.append(batch.src_c.detach().cpu())
        targets.append(int(action))
        observation, reward, terminated, truncated, info = stage.step(int(action))
        rewards.append(float(reward))
        progress = _progress_from_info(info)
        if progress is not None:
            progress_values.append(progress)
        if terminated or truncated:
            break
    return {
        "src_a": torch.cat(src_a, dim=0),
        "src_b": torch.cat(src_b, dim=0),
        "src_c": torch.cat(src_c, dim=0),
        "actions": torch.as_tensor(targets, dtype=torch.long),
        "metrics": {
            "samples": float(len(targets)),
            "script_steps_requested": float(len(actions)),
            "return": float(sum(rewards)),
            "max_progress": float(max(progress_values, default=0.0)),
            "last_progress": float(progress_values[-1]) if progress_values else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        },
    }


def train_full_smb_imitation_warm_start(
    model: torch.nn.Module,
    dataset: Mapping[str, Any],
    *,
    device: torch.device,
    epochs: int = DEFAULT_FULL_SMB_IMITATION_EPOCHS,
    batch_size: int = DEFAULT_FULL_SMB_IMITATION_BATCH_SIZE,
    learning_rate: float = DEFAULT_FULL_SMB_IMITATION_LR,
    trainable_prefixes: Sequence[str] = DEFAULT_FULL_SMB_IMITATION_TRAINABLE_PREFIXES,
    seed: int = 0,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
    """Distill scripted Full SMB action timing into the policy/controller head."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long)
    if actions.numel() <= 0:
        raise ValueError("dataset must contain at least one action")
    primitive_targets = _full_smb_imitation_primitive_targets(actions)
    parameters = _select_trainable_parameters(model, trainable_prefixes)
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    model.train()
    losses: list[float] = []
    accuracies: list[float] = []
    sample_count = int(actions.numel())
    for _epoch in range(epochs):
        order = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            batch = StageBatch(
                src_a=dataset["src_a"][indices].to(device),
                target_a=None,
                src_b=dataset["src_b"][indices].to(device),
                target_b=None,
                src_c=dataset["src_c"][indices].to(device),
                target_c=None,
                metadata={},
            )
            target = actions[indices].to(device)
            forward = _policy_action_logits_and_state(model, batch, device=device)
            logits = forward.logits
            loss_action = F.cross_entropy(logits, target)
            loss_primitive = _full_smb_imitation_primitive_loss(
                forward.motor_primitives,
                target,
                duration_targets=primitive_targets["duration_bin"][indices].to(device),
                duration_mask=primitive_targets["duration_mask"][indices].to(device),
                release_targets=primitive_targets["release"][indices].to(device),
                post_release_targets=primitive_targets["post_release"][indices].to(device),
            )
            loss = loss_action + 0.25 * loss_primitive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            prediction = logits.detach().argmax(dim=-1)
            accuracies.append(float((prediction == target).float().mean().cpu().item()))
    return (
        {
            "samples": float(sample_count),
            "epochs": float(epochs),
            "batch_size": float(batch_size),
            "learning_rate": float(learning_rate),
            "trainable_prefixes": tuple(trainable_prefixes),
            "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "final_loss": float(losses[-1]) if losses else 0.0,
            "mean_action_accuracy": float(sum(accuracies) / len(accuracies))
            if accuracies
            else 0.0,
            "final_action_accuracy": float(accuracies[-1]) if accuracies else 0.0,
        },
        optimizer,
    )


def _full_smb_imitation_primitive_targets(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    duration_bins = torch.as_tensor(DEFAULT_PRIMITIVE_DURATION_BINS, dtype=torch.float32)
    duration_targets = torch.zeros_like(actions)
    duration_mask = torch.zeros(actions.shape, dtype=torch.float32)
    release_targets = torch.zeros(actions.shape, dtype=torch.float32)
    post_release_targets = torch.zeros_like(actions)
    jump_actions = {
        int(SMBAction.RIGHT_JUMP),
        int(SMBAction.LEFT_JUMP),
        int(SMBAction.JUMP),
    }
    action_values = [int(action) for action in actions.detach().cpu().tolist()]
    for index, action in enumerate(action_values):
        post_release_targets[index] = int(smb_jump_release_action(action))
        if action not in jump_actions:
            continue
        run_length = 1
        for next_index in range(index + 1, len(action_values)):
            if action_values[next_index] != action:
                break
            run_length += 1
        duration_mask[index] = 1.0
        duration_targets[index] = int(
            torch.abs(duration_bins - float(run_length)).argmin().item()
        )
        if index + 1 >= len(action_values) or action_values[index + 1] != action:
            release_targets[index] = 1.0
    return {
        "duration_bin": duration_targets.long(),
        "duration_mask": duration_mask,
        "release": release_targets,
        "post_release": post_release_targets.long(),
    }


def _full_smb_imitation_primitive_loss(
    motor_primitives: Any,
    target_actions: torch.Tensor,
    *,
    duration_targets: torch.Tensor,
    duration_mask: torch.Tensor,
    release_targets: torch.Tensor,
    post_release_targets: torch.Tensor,
) -> torch.Tensor:
    if motor_primitives is None:
        return target_actions.new_zeros((), dtype=torch.float32)
    losses: list[torch.Tensor] = []
    combo_logits = getattr(motor_primitives, "button_combo_logits", None)
    if combo_logits is not None and combo_logits.ndim == 3:
        losses.append(F.cross_entropy(combo_logits[:, -1, : len(SMBAction)], target_actions))
    post_release_logits = getattr(motor_primitives, "post_release_logits", None)
    if post_release_logits is not None and post_release_logits.ndim == 3:
        losses.append(
            F.cross_entropy(
                post_release_logits[:, -1, : len(SMBAction)],
                post_release_targets,
            )
        )
    release_logit = getattr(motor_primitives, "release_logit", None)
    if release_logit is not None and release_logit.ndim == 2:
        losses.append(
            F.binary_cross_entropy_with_logits(release_logit[:, -1], release_targets)
        )
    hold_duration_logits = getattr(motor_primitives, "hold_duration_logits", None)
    if (
        hold_duration_logits is not None
        and hold_duration_logits.ndim == 3
        and bool((duration_mask > 0).any().item())
    ):
        per_sample = F.cross_entropy(
            hold_duration_logits[:, -1, :],
            duration_targets,
            reduction="none",
        )
        losses.append((per_sample * duration_mask).sum() / duration_mask.sum().clamp_min(1.0))
    if not losses:
        return target_actions.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def run_full_smb_imitation_warm_start(
    *,
    policy_checkpoint: Path,
    output_checkpoint: Path,
    full_smb_vision_checkpoint: Path = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    output_summary: Optional[Path] = None,
    device: str | torch.device = "auto",
    seed: int = 0,
    steps: int = DEFAULT_FULL_SMB_IMITATION_STEPS,
    epochs: int = DEFAULT_FULL_SMB_IMITATION_EPOCHS,
    batch_size: int = DEFAULT_FULL_SMB_IMITATION_BATCH_SIZE,
    learning_rate: float = DEFAULT_FULL_SMB_IMITATION_LR,
    game_id: str = DEFAULT_FULL_SMB_CONTENT.game,
    state: str = "Level1-1",
    frame_skip: int = 1,
    make_stage: Optional[Callable[[Any], FullSMBStage]] = None,
) -> dict[str, Any]:
    """Run collection, imitation training, checkpoint save, and summary write."""

    resolved_device = select_device(device)
    model, _optimizer, source_checkpoint = load_full_smb_policy_checkpoint(
        policy_checkpoint,
        device=resolved_device,
    )
    architecture_name, architecture_config = policy_architecture_from_checkpoint(source_checkpoint)
    config = FullSMBTrainingConfig(
        seed=seed,
        device=str(resolved_device),
        architecture_name=architecture_name,
        architecture_config=architecture_config,
        full_smb_vision_checkpoint=full_smb_vision_checkpoint,
        game_id=game_id,
        emulator_state=state,
        frame_skip=frame_skip,
        evaluation_episodes=0,
        evaluation_max_steps=0,
    )
    vision = _build_full_smb_perception(config, resolved_device)
    stage = (
        make_stage(vision)
        if make_stage is not None
        else FullSMBStage(
            env_config=FullSMBEnvConfig(game=game_id, state=state),
            vision=vision,
            observation_config=FullSMBObservationConfig(frame_skip=frame_skip),
        )
    )
    try:
        script = full_smb_opening_imitation_script(steps)
        dataset = collect_full_smb_imitation_dataset(stage, script, seed=seed)
    finally:
        stage.close()

    training_metrics, optimizer = train_full_smb_imitation_warm_start(
        model,
        dataset,
        device=resolved_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    checkpoint = build_full_smb_policy_checkpoint(
        model,
        optimizer,
        epoch=epochs,
        global_step=int(dataset["metrics"]["samples"]) * int(epochs),
        config=config,
        metrics={
            "imitation_loss": float(training_metrics["final_loss"]),
            "imitation_action_accuracy": float(training_metrics["final_action_accuracy"]),
            "imitation_dataset_max_progress": float(dataset["metrics"]["max_progress"]),
        },
        architecture_name=architecture_name,
        architecture_config=architecture_config,
        vision=vision,
        training_source={
            "mode": "full_smb_real_emulator_imitation",
            "source_checkpoint": str(policy_checkpoint),
            "script": "full_smb_opening_imitation_script",
        },
    )
    save_checkpoint(output_checkpoint, checkpoint)
    result = {
        "policy_checkpoint": str(policy_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "checkpoint_summary": str(output_checkpoint.with_suffix(".json")),
        "script": {
            "steps": int(steps),
            "right_count": int(sum(1 for action in script if action == int(SMBAction.RIGHT))),
            "right_jump_count": int(
                sum(1 for action in script if action == int(SMBAction.RIGHT_JUMP))
            ),
        },
        "dataset": dataset["metrics"],
        "training": training_metrics,
    }
    if output_summary is not None:
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(
            json.dumps(to_plain_data(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _select_trainable_parameters(
    model: torch.nn.Module,
    prefixes: Sequence[str],
) -> tuple[torch.nn.Parameter, ...]:
    prefixes = tuple(str(prefix) for prefix in prefixes if str(prefix))
    selected = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise ValueError(f"no trainable parameters matched prefixes {prefixes!r}")
    return tuple(selected)


def _progress_from_info(info: Mapping[str, Any]) -> Optional[float]:
    signals = info.get("full_smb_signals")
    source = signals if isinstance(signals, Mapping) else info
    value = source.get("progress") if isinstance(source, Mapping) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retroagi imitate --stage full")
    parser.add_argument("--policy-checkpoint", "--checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        "--vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=DEFAULT_FULL_SMB_IMITATION_STEPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_FULL_SMB_IMITATION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_FULL_SMB_IMITATION_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_FULL_SMB_IMITATION_LR)
    parser.add_argument("--game-id", default=DEFAULT_FULL_SMB_CONTENT.game)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_full_smb_imitation_warm_start(
        policy_checkpoint=args.policy_checkpoint,
        output_checkpoint=args.output_checkpoint,
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        output_summary=args.output_summary,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        game_id=args.game_id,
        state=args.state,
        frame_skip=args.frame_skip,
    )
    print(json.dumps(to_plain_data(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
