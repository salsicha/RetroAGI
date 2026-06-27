"""Perception diagnostics for Full SMB emulator frames."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from retroagi.core import SMB_ACTIONS, select_device, to_plain_data
from retroagi.stages.full_smb.adapter import FullSMBObservationConfig, FullSMBStage
from retroagi.stages.full_smb.vision import (
    DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    load_full_smb_vit_checkpoint,
)


@dataclass(frozen=True)
class FullSMBPerceptionDiagnosticThresholds:
    """Minimum quality gates for unlabeled Full SMB emulator-frame diagnostics."""

    min_semantic_confidence: float = 0.35
    min_class_coverage: float = 0.15
    min_temporal_stability: float = 0.40
    max_position_rmse: float = 0.35
    min_position_within_tolerance: float = 0.30
    position_tolerance: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "min_semantic_confidence",
            "min_class_coverage",
            "min_temporal_stability",
            "min_position_within_tolerance",
            "position_tolerance",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_position_rmse < 0:
            raise ValueError("max_position_rmse must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class FullSMBPerceptionDiagnosticTrace:
    """Frames and metadata collected from a Full SMB rollout."""

    observations: np.ndarray
    infos: tuple[Mapping[str, Any], ...]
    action_ids: tuple[int, ...]
    reset_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "samples": int(self.observations.shape[0]),
            "observation_shape": tuple(int(value) for value in self.observations.shape),
            "action_ids": self.action_ids,
            "reset_count": int(self.reset_count),
        }


def collect_full_smb_perception_diagnostic_frames(
    stage: FullSMBStage,
    *,
    samples: int = 32,
    seed: int = 0,
    rollout_steps: int = 64,
) -> FullSMBPerceptionDiagnosticTrace:
    """Collect real emulator frames and adapter info for perception diagnostics."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")

    rng = random.Random(seed)
    observations: list[np.ndarray] = []
    infos: list[Mapping[str, Any]] = []
    action_ids: list[int] = []
    reset_count = 0

    while len(observations) < samples:
        reset_seed = seed + reset_count
        observation = stage.reset(seed=reset_seed)
        reset_count += 1
        observations.append(np.asarray(observation).copy())
        infos.append(dict(stage.last_info))
        if len(observations) >= samples:
            break

        for _ in range(rollout_steps):
            action = rng.choice(SMB_ACTIONS)
            observation, _reward, terminated, truncated, info = stage.step(action)
            observations.append(np.asarray(observation).copy())
            infos.append(dict(info))
            action_ids.append(int(action))
            if len(observations) >= samples or terminated or truncated:
                break

    return FullSMBPerceptionDiagnosticTrace(
        observations=np.stack(observations[:samples]).astype(np.uint8, copy=False),
        infos=tuple(infos[:samples]),
        action_ids=tuple(action_ids),
        reset_count=reset_count,
    )


@torch.no_grad()
def evaluate_full_smb_perception(
    vision: Any,
    observations: Any,
    infos: Sequence[Mapping[str, Any]],
    *,
    thresholds: FullSMBPerceptionDiagnosticThresholds = FullSMBPerceptionDiagnosticThresholds(),
    batch_size: int = 8,
) -> dict[str, Any]:
    """Evaluate Full SMB ViT outputs on unlabeled emulator frames."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frames = torch.as_tensor(observations)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError("observations must have shape [N,H,W,C] or [N,C,H,W]")
    if frames.shape[0] <= 0:
        raise ValueError("observations must contain at least one frame")
    if len(infos) != int(frames.shape[0]):
        raise ValueError("infos length must match the number of observations")

    class_names = tuple(getattr(vision.spec, "semantic_classes", ()))
    if not class_names:
        raise ValueError("vision spec must declare semantic_classes")

    was_training = bool(getattr(vision, "training", False))
    if hasattr(vision, "eval"):
        vision.eval()

    confidence_total = 0.0
    patch_count = 0
    semantic_ids: list[torch.Tensor] = []
    predicted_positions: list[torch.Tensor] = []

    for start in range(0, frames.shape[0], batch_size):
        batch = frames[start : start + batch_size]
        output = vision.encode(batch)
        probabilities = output.semantic_logits.float().softmax(dim=1)
        confidence_total += probabilities.max(dim=1).values.sum().item()
        patch_count += probabilities.shape[0] * probabilities.shape[2] * probabilities.shape[3]
        semantic_ids.append(output.semantic_ids.detach().cpu().long())
        predicted_positions.append(output.position.detach().cpu().float())

    if was_training and hasattr(vision, "train"):
        vision.train()

    ids = torch.cat(semantic_ids, dim=0)
    positions = torch.cat(predicted_positions, dim=0)
    if ids.shape[0] != frames.shape[0]:
        raise ValueError("vision semantic_ids batch dimension must match observations")
    if positions.shape[0] != frames.shape[0] or positions.shape[1] < 2:
        raise ValueError("vision position must have shape [N, P>=2]")

    covered_class_ids = sorted(int(value) for value in torch.unique(ids).tolist())
    covered_classes = [
        class_names[index] for index in covered_class_ids if 0 <= index < len(class_names)
    ]
    missing_classes = [name for name in class_names if name not in set(covered_classes)]
    per_class_coverage = {
        class_name: float((ids == index).float().mean().item())
        for index, class_name in enumerate(class_names)
    }
    class_coverage = len(covered_classes) / max(len(class_names), 1)
    semantic_confidence = confidence_total / max(patch_count, 1)
    temporal_stability = _temporal_stability(ids)

    target_positions, valid_position = _position_targets_from_infos(infos)
    position_metrics = _position_metrics(
        positions[:, :2],
        target_positions,
        valid_position,
        tolerance=thresholds.position_tolerance,
    )

    semantic_bottleneck_reasons = []
    if semantic_confidence < thresholds.min_semantic_confidence:
        semantic_bottleneck_reasons.append("semantic_confidence")
    if class_coverage < thresholds.min_class_coverage:
        semantic_bottleneck_reasons.append("class_coverage")
    if temporal_stability < thresholds.min_temporal_stability:
        semantic_bottleneck_reasons.append("temporal_stability")

    signal_extraction_bottleneck = position_metrics["position_samples"] <= 0.0
    signal_extraction_bottleneck_reasons = (
        ["missing_position_targets"] if signal_extraction_bottleneck else []
    )

    vision_position_bottleneck_reasons = []
    if not signal_extraction_bottleneck:
        if position_metrics["position_rmse"] > thresholds.max_position_rmse:
            vision_position_bottleneck_reasons.append("position_rmse")
        if position_metrics["position_within_tolerance"] < thresholds.min_position_within_tolerance:
            vision_position_bottleneck_reasons.append("position_consistency")

    bottleneck_reasons = [
        *semantic_bottleneck_reasons,
        *vision_position_bottleneck_reasons,
        *signal_extraction_bottleneck_reasons,
    ]

    return {
        "samples": float(frames.shape[0]),
        "patches": float(patch_count),
        "semantic_confidence": float(semantic_confidence),
        "class_coverage": float(class_coverage),
        "covered_classes": covered_classes,
        "missing_classes": missing_classes,
        "per_class_coverage": per_class_coverage,
        "temporal_stability": float(temporal_stability),
        **position_metrics,
        "thresholds": thresholds.as_dict(),
        "semantic_bottleneck": bool(semantic_bottleneck_reasons),
        "semantic_bottleneck_reasons": semantic_bottleneck_reasons,
        "vision_position_bottleneck": bool(vision_position_bottleneck_reasons),
        "vision_position_bottleneck_reasons": vision_position_bottleneck_reasons,
        "signal_extraction_bottleneck": bool(signal_extraction_bottleneck_reasons),
        "signal_extraction_bottleneck_reasons": signal_extraction_bottleneck_reasons,
        "bottleneck": bool(bottleneck_reasons),
        "bottleneck_reasons": bottleneck_reasons,
    }


def run_full_smb_perception_diagnostic(
    stage: FullSMBStage,
    vision: Any,
    *,
    samples: int = 32,
    seed: int = 0,
    rollout_steps: int = 64,
    batch_size: int = 8,
    thresholds: FullSMBPerceptionDiagnosticThresholds = FullSMBPerceptionDiagnosticThresholds(),
) -> dict[str, Any]:
    trace = collect_full_smb_perception_diagnostic_frames(
        stage,
        samples=samples,
        seed=seed,
        rollout_steps=rollout_steps,
    )
    metrics = evaluate_full_smb_perception(
        vision,
        trace.observations,
        trace.infos,
        thresholds=thresholds,
        batch_size=batch_size,
    )
    return {"trace": trace.summary(), "perception": metrics}


def _temporal_stability(ids: torch.Tensor) -> float:
    if ids.shape[0] < 2:
        return 1.0
    matches = []
    for index in range(1, ids.shape[0]):
        current = ids[index]
        previous = ids[index - 1]
        if current.shape != previous.shape:
            continue
        matches.append(float((current == previous).float().mean().item()))
    return float(sum(matches) / len(matches)) if matches else 0.0


def _position_targets_from_infos(
    infos: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    valid = []
    for info in infos:
        explicit_target = _array_from_info(info, "vision_position_target")
        if explicit_target is not None and explicit_target.size >= 2:
            targets.append((float(explicit_target[0]), float(explicit_target[1])))
            valid.append(True)
            continue
        state = _array_from_info(info, "state_vec")
        camera = _array_from_info(info, "camera_vec")
        if state is not None and state.size >= 2:
            target_y = float(state[1])
            target_x = (
                float(camera[3]) if camera is not None and camera.size >= 4 else float(state[0])
            )
            targets.append((target_x, target_y))
            valid.append(True)
        else:
            targets.append((0.0, 0.0))
            valid.append(False)
    return (
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(valid, dtype=torch.bool),
    )


def _position_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if bool(valid.any()):
        delta = predicted[valid] - target[valid]
        distance = torch.linalg.vector_norm(delta, dim=1)
        mse = delta.pow(2).mean().item()
        rmse = float(mse**0.5)
        within = float((distance <= tolerance).float().mean().item())
        mean_error = float(distance.mean().item())
        samples = int(valid.sum().item())
    else:
        rmse = 0.0
        within = 0.0
        mean_error = 0.0
        samples = 0
    return {
        "position_samples": float(samples),
        "position_rmse": float(rmse),
        "position_mean_error": float(mean_error),
        "position_within_tolerance": float(within),
    }


def _array_from_info(info: Mapping[str, Any], name: str) -> Optional[np.ndarray]:
    if name not in info:
        return None
    try:
        array = np.asarray(info[name], dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retroagi diagnose-vision --stage full")
    parser.add_argument("--vision-checkpoint", type=Path, default=DEFAULT_FULL_SMB_VIT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = select_device(args.device)
    loaded = load_full_smb_vit_checkpoint(
        args.vision_checkpoint,
        device=device,
        freeze=True,
    )
    stage = FullSMBStage(
        vision=loaded.model,
        observation_config=FullSMBObservationConfig(frame_skip=args.frame_skip),
    )
    try:
        result = run_full_smb_perception_diagnostic(
            stage,
            loaded.model,
            samples=args.samples,
            seed=args.seed,
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
        )
    finally:
        stage.close()

    payload = {
        "config": {
            "samples": args.samples,
            "seed": args.seed,
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "frame_skip": args.frame_skip,
            "device": str(device),
        },
        "vision": {
            "checkpoint_path": str(loaded.path),
            "frozen": loaded.frozen,
        },
        **result,
    }
    output = json.dumps(to_plain_data(payload), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
