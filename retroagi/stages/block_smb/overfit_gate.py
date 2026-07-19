"""Tiny-overfit sanity gate for Block SMB policy training.

A policy that cannot memorize a handful of teacher-forced scenarios will never
learn the full curriculum — it means the actor/optimizer/supervision wiring is
broken, not that more training is needed (this is exactly how the historical
"always hold RIGHT" collapse should have been caught before a full run). The
gate trains a fresh policy up to a bounded epoch ceiling, checks deterministic
accuracy periodically, stops as soon as it passes, and reports jump-boundary
errors when it does not.

Run standalone:
    python -m retroagi.stages.block_smb.overfit_gate --output gate.json

or call :func:`run_block_smb_overfit_gate` before an expensive curriculum run.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.optim as optim

from retroagi.core import SMBAction, VisionEncoder, is_smb_jump_action, select_device
from retroagi.stages.block_smb.distill import (
    BlockSMBDistillationConfig,
    _action_logits_with_state,
    _cached_vision_factory,
    _example_sequences,
    _stack_examples,
    _train_behavior_cloning,
    _training_config_from_distillation,
    collect_scripted_distillation_examples,
)
from retroagi.stages.block_smb.train import make_block_smb_model, seed_everything
from retroagi.stages.block_smb.vision import DEFAULT_BLOCK_VIT_CHECKPOINT, BlockVisionTransformer

DEFAULT_OVERFIT_SCENARIOS = ("level_1_flat.json", "level_2_gap.json")
DEFAULT_OVERFIT_EPOCHS = 240
DEFAULT_OVERFIT_CHECK_INTERVAL = 10
# Memorizing 120 examples wants a hotter optimizer than the full curriculum:
# at the curriculum default (1e-3) convergence is too slow for a useful gate.
DEFAULT_OVERFIT_LEARNING_RATE = 3e-3
DEFAULT_OVERFIT_ACCURACY_THRESHOLD = 0.95


def _untrained_vision_factory(device: torch.device) -> Callable[[], VisionEncoder]:
    cache: dict[str, VisionEncoder] = {}

    def factory() -> VisionEncoder:
        if "model" not in cache:
            model = BlockVisionTransformer().to(device)
            model.requires_grad_(False)
            model.eval()
            cache["model"] = model
        return cache["model"]

    return factory


def _jump_boundary_error(
    teacher_actions: list[int],
    step_index: int,
    predicted_action: int,
) -> tuple[str, Optional[int]]:
    teacher_action = teacher_actions[step_index]
    teacher_is_jump = is_smb_jump_action(teacher_action)
    prediction_is_jump = is_smb_jump_action(predicted_action)
    if teacher_is_jump and not prediction_is_jump:
        return "missed_jump", 0
    if prediction_is_jump and not teacher_is_jump:
        previous_jump = next(
            (
                index
                for index in range(step_index - 1, -1, -1)
                if is_smb_jump_action(teacher_actions[index])
            ),
            None,
        )
        next_jump = next(
            (
                index
                for index in range(step_index + 1, len(teacher_actions))
                if is_smb_jump_action(teacher_actions[index])
            ),
            None,
        )
        previous_distance = None if previous_jump is None else step_index - previous_jump
        next_distance = None if next_jump is None else next_jump - step_index
        if next_distance is not None and (
            previous_distance is None or next_distance <= previous_distance
        ):
            return "early_jump", next_distance
        if previous_distance is not None:
            return "late_jump", previous_distance
        return "spurious_jump", None
    return "other", None


@torch.no_grad()
def _teacher_action_diagnostics(
    model,
    dataset,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    """Evaluate teacher actions with recurrent state and report boundary errors."""

    model.eval()
    correct = 0
    seen = 0
    errors: list[dict[str, Any]] = []
    confusion: Counter[str] = Counter()
    for sequence in _example_sequences(dataset):
        world_model_state = None
        teacher_actions = [int(example.action) for example in sequence]
        for sequence_index, example in enumerate(sequence):
            src_a, src_b, src_c, actions, _next_c = _stack_examples([example], device)
            logits, _next_state_pred, world_model_state, _motor = _action_logits_with_state(
                model,
                src_a,
                src_b,
                src_c,
                world_model_state=world_model_state,
            )
            predicted_action = int(logits.argmax(dim=-1).item())
            teacher_action = int(actions.item())
            correct += int(predicted_action == teacher_action)
            seen += 1
            if predicted_action == teacher_action:
                continue
            category, boundary_distance = _jump_boundary_error(
                teacher_actions,
                sequence_index,
                predicted_action,
            )
            teacher_name = SMBAction(teacher_action).name
            predicted_name = SMBAction(predicted_action).name
            confusion[f"{teacher_name}->{predicted_name}"] += 1
            errors.append(
                {
                    "scenario": example.scenario_name,
                    "episode": int(example.episode),
                    "step": int(example.step_index),
                    "teacher": teacher_name,
                    "predicted": predicted_name,
                    "category": category,
                    "frames_from_jump_boundary": boundary_distance,
                }
            )

    boundary_counts = Counter(error["category"] for error in errors)
    accuracy = correct / seen if seen else 0.0
    return accuracy, {
        "misclassified_count": len(errors),
        "confusion": dict(sorted(confusion.items())),
        "jump_boundary": {
            "early_jump_count": int(boundary_counts["early_jump"]),
            "late_jump_count": int(boundary_counts["late_jump"]),
            "missed_jump_count": int(boundary_counts["missed_jump"]),
            "spurious_jump_count": int(boundary_counts["spurious_jump"]),
            "other_count": int(boundary_counts["other"]),
        },
        "errors": errors,
    }


def _teacher_action_accuracy(model, dataset, device: torch.device) -> float:
    """Return deterministic recurrent teacher-action accuracy."""

    accuracy, _diagnostics = _teacher_action_diagnostics(model, dataset, device)
    return accuracy


def run_block_smb_overfit_gate(
    *,
    scenarios: tuple[str, ...] = DEFAULT_OVERFIT_SCENARIOS,
    epochs: int = DEFAULT_OVERFIT_EPOCHS,
    check_interval_epochs: int = DEFAULT_OVERFIT_CHECK_INTERVAL,
    accuracy_threshold: float = DEFAULT_OVERFIT_ACCURACY_THRESHOLD,
    seed: int = 0,
    device: str = "cpu",
    learning_rate: Optional[float] = None,
    vision_checkpoint: Optional[Path] = None,
    vision_factory: Optional[Callable[[], VisionEncoder]] = None,
    architecture_name: Optional[str] = None,
    architecture_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train on a tiny scenario set and require near-perfect memorization.

    Returns a summary dict with ``passed``, ``teacher_action_accuracy``, and
    the configuration used. ``passed`` is False when the freshly trained
    policy cannot reproduce the teacher's actions on the identical states it
    was just trained on — a definitive signal that supervision is broken and
    a full curriculum run would be wasted.
    """

    if not 0.0 < accuracy_threshold <= 1.0:
        raise ValueError("accuracy_threshold must be in (0, 1]")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if check_interval_epochs <= 0:
        raise ValueError("check_interval_epochs must be positive")
    config_values: dict[str, Any] = dict(
        fixed_scenarios=tuple(scenarios),
        monte_carlo_samples=0,
        required_monte_carlo_families=(),
        rollout_steps=60,
        episodes_per_scenario=1,
        epochs=int(epochs),
        evaluation_episodes=1,
        evaluation_max_steps=60,
        dagger_iterations=0,
        seed=int(seed),
        device=device,
        # The gate measures memorization capacity, so class weighting must be
        # neutral: the curriculum's jump upweighting makes visually ambiguous
        # run-up frames resolve to jump, which is not a supervision failure.
        jump_weight_multiplier=1.0,
    )
    config_values["learning_rate"] = float(
        learning_rate if learning_rate is not None else DEFAULT_OVERFIT_LEARNING_RATE
    )
    if architecture_name is not None:
        config_values["architecture_name"] = architecture_name
    if architecture_config is not None:
        config_values["architecture_config"] = dict(architecture_config)
    config = BlockSMBDistillationConfig(**config_values)

    seed_everything(config.seed, deterministic=True)
    resolved_device = select_device(config.device)
    training_config = _training_config_from_distillation(config)
    model = make_block_smb_model(training_config).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    if vision_factory is None:
        if vision_checkpoint is not None:
            vision_factory = _cached_vision_factory(vision_checkpoint, resolved_device)
        else:
            # A trained ViT separates adjacent frames far better than random
            # features, so prefer the shipped checkpoint — but fall back to an
            # untrained encoder when the blob is absent (e.g. a Git LFS
            # pointer in CI checkouts).
            vision_factory = _untrained_vision_factory(resolved_device)
            if DEFAULT_BLOCK_VIT_CHECKPOINT.exists():
                candidate = _cached_vision_factory(DEFAULT_BLOCK_VIT_CHECKPOINT, resolved_device)
                try:
                    candidate()
                except Exception:  # noqa: BLE001 - any load failure means fallback
                    pass
                else:
                    vision_checkpoint = DEFAULT_BLOCK_VIT_CHECKPOINT
                    vision_factory = candidate

    dataset = collect_scripted_distillation_examples(config, vision_factory=vision_factory)
    initial_accuracy = _teacher_action_accuracy(model, dataset, resolved_device)
    accuracy_checks = [{"epoch": 0, "teacher_action_accuracy": float(initial_accuracy)}]
    latest_accuracy = initial_accuracy

    def stop_when_memorized(
        current_model: torch.nn.Module,
        completed_epoch: int,
        _record: dict[str, Any],
    ) -> bool:
        nonlocal latest_accuracy
        if completed_epoch % check_interval_epochs != 0 and completed_epoch != config.epochs:
            return False
        latest_accuracy = _teacher_action_accuracy(current_model, dataset, resolved_device)
        accuracy_checks.append(
            {
                "epoch": int(completed_epoch),
                "teacher_action_accuracy": float(latest_accuracy),
            }
        )
        return latest_accuracy >= accuracy_threshold

    history = []
    if initial_accuracy < accuracy_threshold:
        history = _train_behavior_cloning(
            model,
            optimizer,
            dataset,
            config,
            resolved_device,
            epochs=config.epochs,
            phase="overfit_gate",
            epoch_end_callback=stop_when_memorized,
        )
    accuracy, action_diagnostics = _teacher_action_diagnostics(
        model,
        dataset,
        resolved_device,
    )
    passed = accuracy >= accuracy_threshold
    return {
        "gate": "block_smb_tiny_overfit",
        "passed": bool(passed),
        "teacher_action_accuracy": float(accuracy),
        "initial_teacher_action_accuracy": float(initial_accuracy),
        "accuracy_threshold": float(accuracy_threshold),
        "example_count": len(dataset),
        "epochs": int(config.epochs),
        "max_epochs": int(config.epochs),
        "epochs_trained": len(history),
        "stopped_early": bool(passed and len(history) < config.epochs),
        "check_interval_epochs": int(check_interval_epochs),
        "accuracy_checks": accuracy_checks,
        "scenarios": list(scenarios),
        "seed": int(config.seed),
        "architecture_name": training_config.architecture_name,
        "action_diagnostics": action_diagnostics,
        "final_training_loss": (float(history[-1].get("loss", 0.0)) if history else None),
        "message": (
            "policy memorized the tiny scenario set"
            if passed
            else (
                "policy FAILED to memorize a tiny scenario set — supervision or "
                "optimization is broken; do not start a full curriculum run"
            )
        ),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Overfit a tiny Block SMB scenario set as a pre-training sanity gate."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_OVERFIT_SCENARIOS),
        help="fixed scenario file names to memorize",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_OVERFIT_EPOCHS)
    parser.add_argument(
        "--check-interval-epochs",
        type=int,
        default=DEFAULT_OVERFIT_CHECK_INTERVAL,
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=DEFAULT_OVERFIT_ACCURACY_THRESHOLD,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--vision-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON summary here")
    args = parser.parse_args(argv)

    summary = run_block_smb_overfit_gate(
        scenarios=tuple(args.scenarios),
        epochs=args.epochs,
        check_interval_epochs=args.check_interval_epochs,
        accuracy_threshold=args.accuracy_threshold,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        vision_checkpoint=args.vision_checkpoint,
    )
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
