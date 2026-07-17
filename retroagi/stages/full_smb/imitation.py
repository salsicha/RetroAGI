"""Real-emulator imitation warm starts for Full SMB policies."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
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
DEFAULT_FULL_SMB_OBSTACLE_WINDOW_HOLD_CANDIDATES = (2, 3, 4, 6, 8, 12, 16)


@dataclass(frozen=True)
class FullSMBObstacleWindowDurationSpec:
    """Save-state sweep recipe for one explicit jump-duration label."""

    name: str
    save_state_artifact: str
    obstacle_kind: str
    warmup_script: tuple[tuple[int, int], ...] = ()
    candidate_hold_decisions: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_FULL_SMB_OBSTACLE_WINDOW_HOLD_CANDIDATES
    )
    post_release_action: int = int(SMBAction.RIGHT)
    settle_frames: int = 96
    minimum_progress_delta: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("obstacle window name must be non-empty")
        if not self.save_state_artifact:
            raise ValueError("save_state_artifact must be non-empty")
        if not self.obstacle_kind:
            raise ValueError("obstacle_kind must be non-empty")
        warmup = []
        for action, frames in self.warmup_script:
            if int(frames) <= 0:
                raise ValueError("warmup_script frames must be positive")
            warmup.append((int(SMBAction(action)), int(frames)))
        candidates = tuple(int(value) for value in self.candidate_hold_decisions)
        if not candidates:
            raise ValueError("candidate_hold_decisions must not be empty")
        if any(value <= 0 for value in candidates):
            raise ValueError("candidate_hold_decisions must be positive")
        if int(self.settle_frames) <= 0:
            raise ValueError("settle_frames must be positive")
        object.__setattr__(self, "warmup_script", tuple(warmup))
        object.__setattr__(self, "candidate_hold_decisions", candidates)
        object.__setattr__(self, "post_release_action", int(SMBAction(self.post_release_action)))
        object.__setattr__(self, "settle_frames", int(self.settle_frames))
        object.__setattr__(self, "minimum_progress_delta", float(self.minimum_progress_delta))


DEFAULT_FULL_SMB_OBSTACLE_WINDOW_DURATION_SPECS = (
    FullSMBObstacleWindowDurationSpec(
        name="first_enemy_approach",
        save_state_artifact="section_1_1_first_enemy_approach",
        obstacle_kind="enemy",
        candidate_hold_decisions=(2, 3, 4, 6, 8, 12),
        settle_frames=84,
        minimum_progress_delta=2.0,
    ),
    FullSMBObstacleWindowDurationSpec(
        name="first_pipe_midpipe",
        save_state_artifact="section_1_1_midpipe",
        obstacle_kind="pipe",
        warmup_script=((int(SMBAction.RIGHT), 16),),
        candidate_hold_decisions=(3, 4, 6, 8, 12, 16),
        settle_frames=108,
        minimum_progress_delta=2.0,
    ),
)


def full_smb_opening_imitation_script(
    max_steps: int = DEFAULT_FULL_SMB_IMITATION_STEPS,
    *,
    decision_frame_skip: int = 1,
) -> tuple[int, ...]:
    """Timed real-emulator opening script for Level 1-1 warm-start imitation."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if int(decision_frame_skip) <= 0:
        raise ValueError("decision_frame_skip must be positive")
    pattern = (
        [int(SMBAction.RIGHT)] * 160
        + [int(SMBAction.RIGHT_JUMP)] * 20
        + [int(SMBAction.RIGHT)] * 120
        + [int(SMBAction.RIGHT_JUMP)] * 20
    )
    actions: list[int] = []
    decision_step = int(decision_frame_skip)
    frame_steps = int(max_steps) * decision_step
    while len(actions) < frame_steps:
        actions.extend(pattern)
    if decision_step == 1:
        return tuple(actions[:max_steps])
    sampled = [
        _full_smb_imitation_window_action(actions[index : index + decision_step])
        for index in range(0, frame_steps, decision_step)
    ]
    return tuple(sampled[:max_steps])


def _full_smb_imitation_window_action(window: Sequence[int]) -> int:
    jump_priority = (
        int(SMBAction.RIGHT_JUMP),
        int(SMBAction.LEFT_JUMP),
        int(SMBAction.JUMP),
    )
    for action in jump_priority:
        if action in window:
            return action
    return int(window[0])


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


@torch.no_grad()
def collect_full_smb_obstacle_window_duration_dataset(
    stage: FullSMBStage,
    *,
    repository_root: Path | str = Path("."),
    decision_frame_skip: int = 1,
    specs: Sequence[FullSMBObstacleWindowDurationSpec] = (
        DEFAULT_FULL_SMB_OBSTACLE_WINDOW_DURATION_SPECS
    ),
    seed: int = 0,
) -> dict[str, Any]:
    """Collect explicit duration-bin labels from local save-state sweeps."""

    if int(decision_frame_skip) <= 0:
        raise ValueError("decision_frame_skip must be positive")
    root = Path(repository_root)
    selected_specs = tuple(specs)
    if not selected_specs:
        raise ValueError("specs must not be empty")

    from retroagi.stages.full_smb.save_states import (
        full_smb_save_state_plan,
        load_full_smb_save_state_payload,
    )

    plan = full_smb_save_state_plan()
    src_a: list[torch.Tensor] = []
    src_b: list[torch.Tensor] = []
    src_c: list[torch.Tensor] = []
    actions: list[int] = []
    duration_bins: list[int] = []
    duration_masks: list[float] = []
    release_targets: list[float] = []
    release_masks: list[float] = []
    post_release_targets: list[int] = []
    labels: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped: list[dict[str, str]] = []
    trial_count = 0

    try:
        stage.reset(seed=seed)
    except Exception as exc:  # pragma: no cover - defensive for real backends.
        return _empty_obstacle_window_duration_dataset(
            selected_specs,
            missing=(),
            skipped=(
                {
                    "name": "stage_reset",
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            ),
            trial_count=0,
        )

    for spec in selected_specs:
        try:
            artifact = plan.artifact(spec.save_state_artifact)
        except KeyError:
            skipped.append({"name": spec.name, "reason": "unknown_save_state_artifact"})
            continue
        path = root / artifact.path
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            payload = load_full_smb_save_state_payload(path)
            label = _collect_obstacle_window_duration_label(
                stage,
                payload["state"],
                spec,
                decision_frame_skip=int(decision_frame_skip),
            )
        except _FullSMBObstacleWindowWarmupTerminated:
            skipped.append({"name": spec.name, "reason": "warmup_terminated"})
            continue
        except Exception as exc:
            skipped.append({"name": spec.name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        trial_count += len(label["trials"])
        if not label["accepted"]:
            skipped.append({"name": spec.name, "reason": "no_candidate_progressed"})
            continue
        batch = label["batch"]
        src_a.append(batch.src_a.detach().cpu())
        src_b.append(batch.src_b.detach().cpu())
        src_c.append(batch.src_c.detach().cpu())
        actions.append(int(label["action"]))
        duration_bins.append(int(label["duration_bin"]))
        duration_masks.append(1.0)
        release_targets.append(0.0)
        release_masks.append(0.0)
        post_release_targets.append(int(label["post_release_action"]))
        labels.append({key: value for key, value in label.items() if key not in {"batch", "state"}})

    metrics = {
        "source": "full_smb_obstacle_window_save_state_sweeps",
        "samples": float(len(actions)),
        "windows_attempted": float(len(selected_specs)),
        "windows_labeled": float(len(labels)),
        "trial_count": float(trial_count),
        "missing_save_state_count": float(len(missing)),
        "skipped_count": float(len(skipped)),
        "missing_save_states": tuple(missing),
        "skipped": tuple(skipped),
        "labels": tuple(labels),
    }
    if not actions:
        return {
            "src_a": torch.empty((0, stage.spec.seq_len_a), dtype=torch.long),
            "src_b": torch.empty((0, stage.spec.seq_len_b), dtype=torch.long),
            "src_c": torch.empty((0, stage.spec.seq_len_c), dtype=torch.float32),
            "actions": torch.empty((0,), dtype=torch.long),
            "metrics": metrics,
        }
    return {
        "src_a": torch.cat(src_a, dim=0),
        "src_b": torch.cat(src_b, dim=0),
        "src_c": torch.cat(src_c, dim=0),
        "actions": torch.as_tensor(actions, dtype=torch.long),
        "primitive_duration_bin": torch.as_tensor(duration_bins, dtype=torch.long),
        "primitive_duration_mask": torch.as_tensor(duration_masks, dtype=torch.float32),
        "primitive_release": torch.as_tensor(release_targets, dtype=torch.float32),
        "primitive_release_mask": torch.as_tensor(release_masks, dtype=torch.float32),
        "primitive_post_release": torch.as_tensor(post_release_targets, dtype=torch.long),
        # Every obstacle-window sample carries authoritative primitive targets
        # (including deliberate zero masks) and is an independent decision point.
        "primitive_explicit_mask": torch.ones(len(actions), dtype=torch.float32),
        "sample_trajectory_ids": torch.arange(len(actions), dtype=torch.long),
        "metrics": metrics,
    }


def merge_full_smb_imitation_datasets(
    datasets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Concatenate imitation datasets while preserving optional primitive labels."""

    non_empty = [
        dataset
        for dataset in datasets
        if int(torch.as_tensor(dataset.get("actions", ())).numel()) > 0
    ]
    if not non_empty:
        raise ValueError("at least one imitation dataset must contain samples")
    merged: dict[str, Any] = {
        "src_a": torch.cat([torch.as_tensor(dataset["src_a"]) for dataset in non_empty], dim=0),
        "src_b": torch.cat([torch.as_tensor(dataset["src_b"]) for dataset in non_empty], dim=0),
        "src_c": torch.cat([torch.as_tensor(dataset["src_c"]) for dataset in non_empty], dim=0),
        "actions": torch.cat(
            [torch.as_tensor(dataset["actions"], dtype=torch.long) for dataset in non_empty],
            dim=0,
        ),
    }
    optional_specs = {
        "primitive_duration_bin": torch.long,
        "primitive_duration_mask": torch.float32,
        "primitive_release": torch.float32,
        "primitive_release_mask": torch.float32,
        "primitive_post_release": torch.long,
        "primitive_explicit_mask": torch.float32,
    }
    for key, dtype in optional_specs.items():
        if not any(key in dataset for dataset in non_empty):
            continue
        values = []
        for dataset in non_empty:
            sample_count = int(torch.as_tensor(dataset["actions"]).numel())
            if key in dataset:
                values.append(torch.as_tensor(dataset[key], dtype=dtype))
            elif key == "primitive_post_release":
                values.append(
                    torch.as_tensor(
                        [
                            int(smb_jump_release_action(int(action)))
                            for action in torch.as_tensor(dataset["actions"]).tolist()
                        ],
                        dtype=dtype,
                    )
                )
            else:
                values.append(torch.zeros((sample_count,), dtype=dtype))
        merged[key] = torch.cat(values, dim=0)
    trajectory_ids: list[torch.Tensor] = []
    next_trajectory_id = 0
    for dataset in non_empty:
        sample_count = int(torch.as_tensor(dataset["actions"]).numel())
        if "sample_trajectory_ids" in dataset:
            ids = torch.as_tensor(dataset["sample_trajectory_ids"], dtype=torch.long)
            ids = ids + next_trajectory_id
        else:
            ids = torch.full((sample_count,), next_trajectory_id, dtype=torch.long)
        trajectory_ids.append(ids)
        next_trajectory_id = int(ids.max().item()) + 1
    merged["sample_trajectory_ids"] = torch.cat(trajectory_ids, dim=0)
    merged["metrics"] = {
        "samples": float(int(merged["actions"].numel())),
        "max_progress": float(
            max(
                (
                    float(dataset.get("metrics", {}).get("max_progress", 0.0))
                    for dataset in non_empty
                ),
                default=0.0,
            )
        ),
        "components": tuple(
            {
                "source": dataset.get("metrics", {}).get("source", "scripted_opening"),
                "samples": float(torch.as_tensor(dataset["actions"]).numel()),
                "metrics": dataset.get("metrics", {}),
            }
            for dataset in non_empty
        ),
    }
    return merged


def _empty_obstacle_window_duration_dataset(
    specs: Sequence[FullSMBObstacleWindowDurationSpec],
    *,
    missing: Sequence[str],
    skipped: Sequence[Mapping[str, str]],
    trial_count: int,
) -> dict[str, Any]:
    return {
        "src_a": torch.empty((0, 0), dtype=torch.long),
        "src_b": torch.empty((0, 0), dtype=torch.long),
        "src_c": torch.empty((0, 0), dtype=torch.float32),
        "actions": torch.empty((0,), dtype=torch.long),
        "metrics": {
            "source": "full_smb_obstacle_window_save_state_sweeps",
            "samples": 0.0,
            "windows_attempted": float(len(specs)),
            "windows_labeled": 0.0,
            "trial_count": float(trial_count),
            "missing_save_state_count": float(len(missing)),
            "skipped_count": float(len(skipped)),
            "missing_save_states": tuple(missing),
            "skipped": tuple(dict(item) for item in skipped),
            "labels": (),
        },
    }


class _FullSMBObstacleWindowWarmupTerminated(RuntimeError):
    """The warmup script hit a terminal state before the sweep snapshot."""


def _collect_obstacle_window_duration_label(
    stage: FullSMBStage,
    state: Any,
    spec: FullSMBObstacleWindowDurationSpec,
    *,
    decision_frame_skip: int,
) -> dict[str, Any]:
    _load_obstacle_window_state(stage, state)
    terminated = False
    truncated = False
    for action, frames in spec.warmup_script:
        for _ in range(_decision_count(frames, decision_frame_skip)):
            _observation, _reward, terminated, truncated, _info = stage.step(action)
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    if terminated or truncated:
        raise _FullSMBObstacleWindowWarmupTerminated(
            f"obstacle window {spec.name!r} warmup script terminated before its sweep"
        )
    snapshot = stage.save_emulator_state()
    label_observation = _load_obstacle_window_state(stage, snapshot)
    label_batch = stage.encode_observation(label_observation)
    start_info = dict(getattr(stage, "last_info", {}))
    start_progress = _progress_from_info(start_info)
    trials = []
    for hold_decisions in spec.candidate_hold_decisions:
        trial = _run_obstacle_window_candidate(
            stage,
            snapshot,
            spec,
            hold_decisions=int(hold_decisions),
            decision_frame_skip=decision_frame_skip,
            start_progress=start_progress,
        )
        trials.append(trial)
    best = _select_obstacle_window_trial(trials)
    accepted = bool(best is not None and best["success"])
    if best is None:
        best = {
            "hold_decisions": 0,
            "duration_bin": 0,
            "progress_delta": 0.0,
            "success": False,
            "score": float("-inf"),
        }
    return {
        "name": spec.name,
        "save_state_artifact": spec.save_state_artifact,
        "obstacle_kind": spec.obstacle_kind,
        "action": int(SMBAction.RIGHT_JUMP),
        "post_release_action": int(spec.post_release_action),
        "hold_decisions": int(best["hold_decisions"]),
        "duration_bin": int(best["duration_bin"]),
        "progress_delta": float(best["progress_delta"]),
        "score": float(best["score"]),
        "success": bool(best["success"]),
        "accepted": accepted,
        "trials": tuple(trials),
        "batch": label_batch,
    }


def _run_obstacle_window_candidate(
    stage: FullSMBStage,
    snapshot: Any,
    spec: FullSMBObstacleWindowDurationSpec,
    *,
    hold_decisions: int,
    decision_frame_skip: int,
    start_progress: Optional[float],
) -> dict[str, Any]:
    _load_obstacle_window_state(stage, snapshot)
    terminated = False
    truncated = False
    info: Mapping[str, Any] = dict(getattr(stage, "last_info", {}))
    for _ in range(int(hold_decisions)):
        _observation, _reward, terminated, truncated, info = stage.step(SMBAction.RIGHT_JUMP)
        if terminated or truncated:
            break
    if not (terminated or truncated):
        for _ in range(_decision_count(spec.settle_frames, decision_frame_skip)):
            _observation, _reward, terminated, truncated, info = stage.step(
                spec.post_release_action
            )
            if terminated or truncated:
                break
    end_progress = _progress_from_info(info)
    progress_delta = (
        float(end_progress - start_progress)
        if end_progress is not None and start_progress is not None
        else 0.0
    )
    failure = bool(truncated or _full_smb_terminal_failure(info))
    success = (not failure) and progress_delta >= float(spec.minimum_progress_delta)
    duration_bin = _nearest_duration_bin_index(float(hold_decisions))
    score = progress_delta
    if failure:
        score -= 10_000.0
    if not success:
        score -= 100.0
    return {
        "hold_decisions": int(hold_decisions),
        "duration_bin": int(duration_bin),
        "progress_delta": float(progress_delta),
        "end_progress": float(end_progress) if end_progress is not None else None,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure": bool(failure),
        "success": bool(success),
        "score": float(score),
    }


def _load_obstacle_window_state(stage: FullSMBStage, state: Any) -> Any:
    observation = stage.load_emulator_state(state)
    reset_frame_stack = getattr(stage, "_reset_frame_stack", None)
    if callable(reset_frame_stack):
        reset_frame_stack(observation)
    return observation


def _select_obstacle_window_trial(
    trials: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if not trials:
        return None
    return max(
        trials,
        key=lambda trial: (
            bool(trial.get("success")),
            float(trial.get("score", float("-inf"))),
            float(trial.get("progress_delta", 0.0)),
            -abs(int(trial.get("hold_decisions", 0)) - 6),
        ),
    )


def _decision_count(frames: int, decision_frame_skip: int) -> int:
    return max(1, int(math.ceil(float(frames) / float(decision_frame_skip))))


def _nearest_duration_bin_index(duration: float) -> int:
    duration_bins = torch.as_tensor(DEFAULT_PRIMITIVE_DURATION_BINS, dtype=torch.float32)
    return int(torch.abs(duration_bins - float(duration)).argmin().item())


def _full_smb_terminal_failure(info: Mapping[str, Any]) -> bool:
    source = info.get("full_smb_signals")
    signals = source if isinstance(source, Mapping) else info
    for key in ("death", "game_over", "timeout", "time_up"):
        if bool(signals.get(key, False)):
            return True
    reason = signals.get("termination_reason") or signals.get("reason")
    if reason is None:
        return False
    normalized = str(reason).strip().lower()
    return any(token in normalized for token in ("death", "dead", "game_over", "timeout"))


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
    primitive_targets = _full_smb_imitation_primitive_targets(actions, dataset=dataset)
    parameters = _select_trainable_parameters(model, trainable_prefixes)
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    model.train()
    losses: list[float] = []
    action_losses: list[float] = []
    primitive_losses: list[float] = []
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
                release_mask=primitive_targets["release_mask"][indices].to(device),
                post_release_targets=primitive_targets["post_release"][indices].to(device),
            )
            loss = loss_action + 0.25 * loss_primitive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            action_losses.append(float(loss_action.detach().cpu().item()))
            primitive_losses.append(float(loss_primitive.detach().cpu().item()))
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
            "mean_action_loss": (
                float(sum(action_losses) / len(action_losses)) if action_losses else 0.0
            ),
            "final_action_loss": float(action_losses[-1]) if action_losses else 0.0,
            "mean_primitive_loss": (
                float(sum(primitive_losses) / len(primitive_losses)) if primitive_losses else 0.0
            ),
            "final_primitive_loss": float(primitive_losses[-1]) if primitive_losses else 0.0,
            "mean_action_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
            "final_action_accuracy": float(accuracies[-1]) if accuracies else 0.0,
            "duration_supervision_count": float(primitive_targets["duration_mask"].sum().item()),
            "release_supervision_count": float(primitive_targets["release_mask"].sum().item()),
            "release_positive_count": float(primitive_targets["release"].sum().item()),
            "explicit_duration_supervision_count": float(
                primitive_targets["explicit_duration_mask"].sum().item()
            ),
        },
        optimizer,
    )


def _full_smb_imitation_primitive_targets(
    actions: torch.Tensor,
    *,
    dataset: Optional[Mapping[str, Any]] = None,
) -> dict[str, torch.Tensor]:
    duration_bins = torch.as_tensor(DEFAULT_PRIMITIVE_DURATION_BINS, dtype=torch.float32)
    duration_targets = torch.zeros_like(actions)
    duration_mask = torch.zeros(actions.shape, dtype=torch.float32)
    release_targets = torch.zeros(actions.shape, dtype=torch.float32)
    release_mask = torch.zeros(actions.shape, dtype=torch.float32)
    post_release_targets = torch.zeros_like(actions)
    jump_actions = {
        int(SMBAction.RIGHT_JUMP),
        int(SMBAction.LEFT_JUMP),
        int(SMBAction.JUMP),
    }
    action_values = [int(action) for action in actions.detach().cpu().tolist()]
    sample_count = len(action_values)
    explicit_values = _explicit_sample_flags(dataset, sample_count=sample_count)
    trajectory_values = _sample_trajectory_values(dataset, sample_count=sample_count)

    def _same_run(index: int, other: int) -> bool:
        return (
            0 <= other < sample_count
            and action_values[other] == action_values[index]
            and trajectory_values[other] == trajectory_values[index]
            and not explicit_values[other]
        )

    for index, action in enumerate(action_values):
        post_release_targets[index] = int(smb_jump_release_action(action))
        if explicit_values[index] or action not in jump_actions:
            continue
        release_mask[index] = 1.0
        is_run_start = not _same_run(index, index - 1)
        run_length = 1
        next_index = index + 1
        while _same_run(index, next_index):
            run_length += 1
            next_index += 1
        if is_run_start:
            duration_mask[index] = 1.0
            duration_targets[index] = int(
                torch.abs(duration_bins - float(run_length)).argmin().item()
            )
        if not _same_run(index, index + 1):
            release_targets[index] = 1.0
    targets = {
        "duration_bin": duration_targets.long(),
        "duration_mask": duration_mask,
        "release": release_targets,
        "release_mask": release_mask,
        "post_release": post_release_targets.long(),
        "explicit_duration_mask": torch.zeros(actions.shape, dtype=torch.float32),
    }
    if dataset is None:
        return targets
    _overlay_explicit_primitive_targets(targets, dataset, sample_count=int(actions.numel()))
    return targets


def _overlay_explicit_primitive_targets(
    targets: dict[str, torch.Tensor],
    dataset: Mapping[str, Any],
    *,
    sample_count: int,
) -> None:
    explicit_samples = _optional_target_tensor(
        dataset,
        "primitive_explicit_mask",
        sample_count=sample_count,
        dtype=torch.float32,
    )
    explicit_duration_mask = _optional_target_tensor(
        dataset,
        "primitive_duration_mask",
        sample_count=sample_count,
        dtype=torch.float32,
    )
    explicit_duration_bin = _optional_target_tensor(
        dataset,
        "primitive_duration_bin",
        sample_count=sample_count,
        dtype=torch.long,
    )
    if explicit_duration_mask is not None and explicit_duration_bin is not None:
        # Explicit samples are authoritative for every target, including zero
        # masks that deliberately withhold supervision.
        mask = explicit_samples > 0 if explicit_samples is not None else explicit_duration_mask > 0
        targets["duration_bin"][mask] = explicit_duration_bin[mask]
        targets["duration_mask"][mask] = explicit_duration_mask[mask]
        targets["explicit_duration_mask"][mask] = explicit_duration_mask[mask]

    explicit_release_mask = _optional_target_tensor(
        dataset,
        "primitive_release_mask",
        sample_count=sample_count,
        dtype=torch.float32,
    )
    explicit_release = _optional_target_tensor(
        dataset,
        "primitive_release",
        sample_count=sample_count,
        dtype=torch.float32,
    )
    if explicit_release_mask is not None and explicit_release is not None:
        mask = explicit_samples > 0 if explicit_samples is not None else explicit_release_mask > 0
        targets["release"][mask] = explicit_release[mask]
        targets["release_mask"][mask] = explicit_release_mask[mask]

    explicit_post_release = _optional_target_tensor(
        dataset,
        "primitive_post_release",
        sample_count=sample_count,
        dtype=torch.long,
    )
    if explicit_post_release is not None:
        targets["post_release"] = explicit_post_release


def _explicit_sample_flags(
    dataset: Optional[Mapping[str, Any]],
    *,
    sample_count: int,
) -> list[bool]:
    if dataset is None:
        return [False] * sample_count
    tensor = _optional_target_tensor(
        dataset,
        "primitive_explicit_mask",
        sample_count=sample_count,
        dtype=torch.float32,
    )
    if tensor is None:
        return [False] * sample_count
    return [bool(value > 0.0) for value in tensor.tolist()]


def _sample_trajectory_values(
    dataset: Optional[Mapping[str, Any]],
    *,
    sample_count: int,
) -> list[int]:
    if dataset is None:
        return [0] * sample_count
    tensor = _optional_target_tensor(
        dataset,
        "sample_trajectory_ids",
        sample_count=sample_count,
        dtype=torch.long,
    )
    if tensor is None:
        return [0] * sample_count
    return [int(value) for value in tensor.tolist()]


def _optional_target_tensor(
    dataset: Mapping[str, Any],
    key: str,
    *,
    sample_count: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if key not in dataset:
        return None
    tensor = torch.as_tensor(dataset[key], dtype=dtype)
    if tensor.shape != (sample_count,):
        raise ValueError(f"{key} must have shape ({sample_count},), got {tuple(tensor.shape)}")
    return tensor


def _full_smb_imitation_primitive_loss(
    motor_primitives: Any,
    target_actions: torch.Tensor,
    *,
    duration_targets: torch.Tensor,
    duration_mask: torch.Tensor,
    release_targets: torch.Tensor,
    release_mask: torch.Tensor,
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
    if (
        release_logit is not None
        and release_logit.ndim == 2
        and bool((release_mask > 0).any().item())
    ):
        per_sample = F.binary_cross_entropy_with_logits(
            release_logit[:, -1],
            release_targets,
            reduction="none",
        )
        losses.append((per_sample * release_mask).sum() / release_mask.sum().clamp_min(1.0))
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
    obstacle_window_labels: bool = True,
    obstacle_window_repository_root: Path | str = Path("."),
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
        script = full_smb_opening_imitation_script(
            steps,
            decision_frame_skip=frame_skip,
        )
        dataset = collect_full_smb_imitation_dataset(stage, script, seed=seed)
    finally:
        stage.close()
    obstacle_window_metrics: Mapping[str, Any] = {
        "enabled": bool(obstacle_window_labels),
        "samples": 0.0,
        "windows_attempted": 0.0,
        "windows_labeled": 0.0,
    }
    datasets: list[Mapping[str, Any]] = [dataset]
    if obstacle_window_labels:
        obstacle_stage = (
            make_stage(vision)
            if make_stage is not None
            else FullSMBStage(
                env_config=FullSMBEnvConfig(game=game_id, state=state),
                vision=vision,
                observation_config=FullSMBObservationConfig(frame_skip=frame_skip),
            )
        )
        try:
            obstacle_dataset = collect_full_smb_obstacle_window_duration_dataset(
                obstacle_stage,
                repository_root=obstacle_window_repository_root,
                decision_frame_skip=frame_skip,
                seed=seed,
            )
        finally:
            obstacle_stage.close()
        obstacle_window_metrics = {
            "enabled": True,
            **dict(obstacle_dataset.get("metrics", {})),
        }
        if int(torch.as_tensor(obstacle_dataset["actions"]).numel()) > 0:
            datasets.append(obstacle_dataset)
    dataset = merge_full_smb_imitation_datasets(datasets)

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
            "imitation_primitive_loss": float(training_metrics["final_primitive_loss"]),
            "imitation_duration_supervision_count": float(
                training_metrics["duration_supervision_count"]
            ),
            "imitation_release_supervision_count": float(
                training_metrics["release_supervision_count"]
            ),
            "imitation_release_positive_count": float(training_metrics["release_positive_count"]),
            "imitation_explicit_duration_supervision_count": float(
                training_metrics["explicit_duration_supervision_count"]
            ),
            "imitation_obstacle_window_label_count": float(
                obstacle_window_metrics.get("windows_labeled", 0.0)
            ),
            "imitation_obstacle_window_trial_count": float(
                obstacle_window_metrics.get("trial_count", 0.0)
            ),
            "imitation_decision_frame_skip": float(frame_skip),
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
            "decision_frame_skip": int(frame_skip),
            "right_count": int(sum(1 for action in script if action == int(SMBAction.RIGHT))),
            "right_jump_count": int(
                sum(1 for action in script if action == int(SMBAction.RIGHT_JUMP))
            ),
        },
        "dataset": dataset["metrics"],
        "obstacle_window_duration_labels": obstacle_window_metrics,
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
    parser.set_defaults(obstacle_window_labels=True)
    parser.add_argument(
        "--obstacle-window-labels",
        action="store_true",
        dest="obstacle_window_labels",
        help="add save-state obstacle-window duration labels",
    )
    parser.add_argument(
        "--no-obstacle-window-labels",
        action="store_false",
        dest="obstacle_window_labels",
        help="disable save-state obstacle-window duration labels",
    )
    parser.add_argument(
        "--obstacle-window-root",
        type=Path,
        default=Path("."),
        help="repository root used to locate local/full_smb/states save-state files",
    )
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
        obstacle_window_labels=args.obstacle_window_labels,
        obstacle_window_repository_root=args.obstacle_window_root,
    )
    print(json.dumps(to_plain_data(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
