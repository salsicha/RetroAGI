"""Distill the scripted known-good Block SMB policy into neural weights."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import StageBatch, VisionEncoder, select_device, to_plain_data

from .adapter import BlockSMBStage
from .env import MarioScenarioEnv
from .scripted_policy import BlockSMBScriptedPolicy, fixed_scenario_action_scripts
from .train import (
    BLOCK_SMB_ACTION_COUNT,
    BlockSMBTrainingConfig,
    evaluate_block_smb,
    load_fixed_scenarios,
    make_block_smb_model,
    restore_block_smb_checkpoint,
    save_block_smb_checkpoint,
    seed_everything,
)
from .vision import DEFAULT_BLOCK_VIT_CHECKPOINT, load_block_vit_checkpoint


@dataclass(frozen=True)
class BlockSMBDistillationConfig:
    """Configuration for scripted-policy behavioral cloning."""

    seed: int = 20260627
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    jump_weight_multiplier: float = 8.0
    sequence_training: bool = True
    dagger_iterations: int = 0
    dagger_epochs_per_iteration: int = 20
    rollout_steps: int = 200
    episodes_per_scenario: int = 3
    evaluation_episodes: int = 3
    evaluation_max_steps: int = 200
    fixed_scenarios: tuple[str, ...] = (
        "level_1_flat.json",
        "level_2_gap.json",
        "level_3_stairs.json",
        "level_4_platforms.json",
        "level_5_enemy_hop.json",
        "level_6_enemy_patrol.json",
    )
    device: str = "auto"
    deterministic: bool = True
    checkpoint_path: Optional[Path] = None
    init_checkpoint: Optional[Path] = None
    vision_checkpoint: Optional[Path] = DEFAULT_BLOCK_VIT_CHECKPOINT
    output_summary: Optional[Path] = None
    log_path: Optional[Path] = None
    hidden_dim: int = 32
    controller_schedule: str = "constant"

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "dagger_iterations",
            "dagger_epochs_per_iteration",
            "rollout_steps",
            "episodes_per_scenario",
            "evaluation_episodes",
            "evaluation_max_steps",
            "hidden_dim",
        ):
            if name == "dagger_iterations":
                if int(getattr(self, name)) < 0:
                    raise ValueError(f"{name} must be non-negative")
                continue
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.jump_weight_multiplier <= 0:
            raise ValueError("jump_weight_multiplier must be positive")
        for path_name in ("checkpoint_path", "init_checkpoint", "vision_checkpoint", "output_summary", "log_path"):
            value = getattr(self, path_name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, path_name, Path(value))
        object.__setattr__(
            self,
            "fixed_scenarios",
            tuple(str(name) for name in self.fixed_scenarios),
        )


@dataclass(frozen=True)
class BlockSMBDistillationExample:
    """One teacher-forced state/action pair."""

    batch: StageBatch
    action: int
    scenario_name: str
    episode: int
    step_index: int


def train_distilled_block_smb_policy(
    config: Optional[BlockSMBDistillationConfig] = None,
    *,
    vision_factory: Optional[Callable[[], VisionEncoder]] = None,
) -> dict[str, Any]:
    """Train and evaluate a neural policy that imitates scripted fixed scenarios."""

    config = config or BlockSMBDistillationConfig()
    seed_everything(config.seed, deterministic=config.deterministic)
    device = select_device(config.device)
    training_config = _training_config_from_distillation(config)
    model = make_block_smb_model(training_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.init_checkpoint is not None:
        restore_block_smb_checkpoint(
            config.init_checkpoint,
            model,
            optimizer=None,
            map_location=device,
            architecture_name=training_config.architecture_name,
            architecture_config=training_config.architecture_config,
        )
    vision_factory = vision_factory or _cached_vision_factory(config.vision_checkpoint, device)
    dataset = collect_scripted_distillation_examples(
        config,
        vision_factory=vision_factory,
    )
    all_history: list[dict[str, float]] = []
    evaluations: list[dict[str, Any]] = []
    all_history.extend(
        _train_behavior_cloning(
            model,
            optimizer,
            dataset,
            config,
            device,
            epochs=config.epochs,
            phase="teacher_forced",
        )
    )
    evaluation = _evaluate_distilled_policy(
        model,
        training_config,
        device=device,
        vision_factory=vision_factory,
        phase="teacher_forced",
        iteration=0,
    )
    evaluations.append(evaluation)
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = _evaluation_metrics(evaluation)
    best_phase = "teacher_forced"

    for dagger_iteration in range(1, config.dagger_iterations + 1):
        dagger_examples = collect_dagger_distillation_examples(
            model,
            config,
            vision_factory=vision_factory,
            device=device,
            iteration=dagger_iteration,
        )
        dataset.extend(dagger_examples)
        _append_jsonl(
            config.log_path,
            {
                "event": "dagger_dataset",
                "iteration": dagger_iteration,
                "new_examples": len(dagger_examples),
                "total_examples": len(dataset),
                "dataset": _dataset_summary(dataset),
            },
        )
        all_history.extend(
            _train_behavior_cloning(
                model,
                optimizer,
                dataset,
                config,
                device,
                epochs=config.dagger_epochs_per_iteration,
                phase="dagger",
                iteration=dagger_iteration,
            )
        )
        evaluation = _evaluate_distilled_policy(
            model,
            training_config,
            device=device,
            vision_factory=vision_factory,
            phase="dagger",
            iteration=dagger_iteration,
        )
        evaluations.append(evaluation)
        metrics = _evaluation_metrics(evaluation)
        if _evaluation_sort_key(metrics) > _evaluation_sort_key(best_metrics):
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            best_phase = f"dagger_{dagger_iteration}"

    model.load_state_dict(best_state)
    evaluation = evaluations[-1]
    if best_phase != evaluations[-1]["phase"]:
        evaluation = _evaluate_distilled_policy(
            model,
            training_config,
            device=device,
            vision_factory=vision_factory,
            phase=f"best_{best_phase}",
            iteration=config.dagger_iterations,
        )
    final_metrics = {
        **all_history[-1],
        "eval_mean_return": float(evaluation["mean_return"]),
        "eval_success_rate": float(evaluation["success_rate"]),
        "eval_threshold_pass_rate": float(
            evaluation.get("tuning_metrics", {}).get("threshold_pass_rate", 0.0)
        ),
        "eval_tuning_score": float(evaluation.get("tuning_metrics", {}).get("score", 0.0)),
    }
    checkpoint = None
    if config.checkpoint_path is not None:
        checkpoint = save_block_smb_checkpoint(
            config.checkpoint_path,
            model,
            optimizer,
            epoch=config.epochs + config.dagger_iterations * config.dagger_epochs_per_iteration,
            global_step=len(dataset)
            * (config.epochs + config.dagger_iterations * config.dagger_epochs_per_iteration),
            config=training_config,
            metrics=final_metrics,
        )
    result = {
        "config": to_plain_data(config),
        "training_config": to_plain_data(training_config),
        "dataset": _dataset_summary(dataset),
        "history": all_history,
        "evaluations": evaluations,
        "best_phase": best_phase,
        "metrics": final_metrics,
        "evaluation": evaluation,
        "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
        "checkpoint": _checkpoint_summary(checkpoint),
    }
    _write_json(config.output_summary, result)
    return result


def collect_scripted_distillation_examples(
    config: BlockSMBDistillationConfig,
    *,
    vision_factory: Callable[[], VisionEncoder],
) -> list[BlockSMBDistillationExample]:
    """Collect teacher-forced observations from the scripted fixed scenarios."""

    policy = BlockSMBScriptedPolicy(
        fixed_scenario_action_scripts(max_steps=config.rollout_steps)
    )
    scenarios = load_fixed_scenarios(config.fixed_scenarios)
    examples: list[BlockSMBDistillationExample] = []
    for scenario_index, (scenario_name, scenario) in enumerate(scenarios):
        for episode in range(config.episodes_per_scenario):
            env = MarioScenarioEnv()
            stage = BlockSMBStage(env=env, scenario=scenario, vision=vision_factory())
            try:
                observation = stage.reset(
                    seed=config.seed + scenario_index * 10_000 + episode
                )
                for step_index in range(config.rollout_steps):
                    action = policy.action(scenario_name, step_index)
                    batch = _detach_batch(stage.encode_observation(observation))
                    examples.append(
                        BlockSMBDistillationExample(
                            batch=batch,
                            action=int(action),
                            scenario_name=scenario_name,
                            episode=episode,
                            step_index=step_index,
                        )
                    )
                    observation, _reward, terminated, truncated, _info = stage.step(action)
                    if terminated or truncated:
                        break
            finally:
                env.close()
    if not examples:
        raise ValueError("scripted distillation dataset is empty")
    return examples


def collect_dagger_distillation_examples(
    model: torch.nn.Module,
    config: BlockSMBDistillationConfig,
    *,
    vision_factory: Callable[[], VisionEncoder],
    device: torch.device,
    iteration: int,
) -> list[BlockSMBDistillationExample]:
    """Collect corrective labels from states visited by the current neural policy."""

    policy = BlockSMBScriptedPolicy(
        fixed_scenario_action_scripts(max_steps=config.rollout_steps)
    )
    scenarios = load_fixed_scenarios(config.fixed_scenarios)
    examples: list[BlockSMBDistillationExample] = []
    model.eval()
    with torch.no_grad():
        for scenario_index, (scenario_name, scenario) in enumerate(scenarios):
            for episode in range(config.episodes_per_scenario):
                env = MarioScenarioEnv()
                stage = BlockSMBStage(env=env, scenario=scenario, vision=vision_factory())
                try:
                    observation = stage.reset(
                        seed=config.seed
                        + 2_000_000
                        + iteration * 100_000
                        + scenario_index * 10_000
                        + episode
                    )
                    world_model_state = None
                    for step_index in range(config.rollout_steps):
                        teacher_action = policy.action(scenario_name, step_index)
                        batch = _detach_batch(stage.encode_observation(observation))
                        examples.append(
                            BlockSMBDistillationExample(
                                batch=batch,
                                action=int(teacher_action),
                                scenario_name=scenario_name,
                                episode=config.episodes_per_scenario * iteration
                                + episode,
                                step_index=step_index,
                            )
                        )
                        src_a, src_b, src_c, _actions = _stack_examples([examples[-1]], device)
                        logits, next_world_model_state = _action_logits_with_state(
                            model,
                            src_a,
                            src_b,
                            src_c,
                            world_model_state=world_model_state,
                        )
                        model_action = int(logits.argmax(dim=-1).item())
                        observation, _reward, terminated, truncated, _info = stage.step(
                            model_action
                        )
                        if terminated or truncated:
                            break
                        world_model_state = (
                            next_world_model_state.detach()
                            if next_world_model_state is not None
                            else None
                        )
                finally:
                    env.close()
    if not examples:
        raise ValueError("DAgger distillation dataset is empty")
    return examples


def _train_behavior_cloning(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    dataset: list[BlockSMBDistillationExample],
    config: BlockSMBDistillationConfig,
    device: torch.device,
    *,
    epochs: int,
    phase: str,
    iteration: int = 0,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    rng = random.Random(config.seed + iteration)
    class_weights = _action_class_weights(
        dataset,
        device,
        jump_weight_multiplier=config.jump_weight_multiplier,
    )
    sequences = _example_sequences(dataset)
    for epoch in range(epochs):
        model.train()
        if config.sequence_training:
            total_loss, total_correct, total_seen = _train_behavior_cloning_epoch_recurrent(
                model,
                optimizer,
                sequences,
                class_weights,
                rng,
                device,
            )
        else:
            total_loss, total_correct, total_seen = _train_behavior_cloning_epoch_independent(
                model,
                optimizer,
                dataset,
                class_weights,
                rng,
                config.batch_size,
                device,
            )
        record = {
            "phase": phase,
            "iteration": float(iteration),
            "epoch": float(epoch + 1),
            "loss": total_loss / max(total_seen, 1),
            "accuracy": total_correct / max(total_seen, 1),
        }
        history.append(record)
        _append_jsonl(config.log_path, {"event": "distill_epoch", **record})
    return history


def _evaluate_distilled_policy(
    model: torch.nn.Module,
    training_config: BlockSMBTrainingConfig,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder],
    phase: str,
    iteration: int,
) -> dict[str, Any]:
    evaluation = evaluate_block_smb(
        model,
        training_config,
        device=device,
        vision_factory=vision_factory,
    )
    evaluation["phase"] = phase
    evaluation["iteration"] = iteration
    return evaluation


def _evaluation_metrics(evaluation: Mapping[str, Any]) -> dict[str, float]:
    tuning = evaluation.get("tuning_metrics", {})
    return {
        "threshold_pass_rate": float(tuning.get("threshold_pass_rate", 0.0)),
        "success_rate": float(evaluation.get("success_rate", 0.0)),
        "mean_return": float(evaluation.get("mean_return", 0.0)),
        "score": float(tuning.get("score", 0.0)),
    }


def _evaluation_sort_key(metrics: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("threshold_pass_rate", 0.0)),
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("mean_return", 0.0)),
        float(metrics.get("score", 0.0)),
    )


def _train_behavior_cloning_epoch_recurrent(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    sequences: list[list[BlockSMBDistillationExample]],
    class_weights: torch.Tensor,
    rng: random.Random,
    device: torch.device,
) -> tuple[float, int, int]:
    rng.shuffle(sequences)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for sequence in sequences:
        losses = []
        world_model_state = None
        sequence_correct = 0
        for example in sequence:
            src_a, src_b, src_c, actions = _stack_examples([example], device)
            logits, next_world_model_state = _action_logits_with_state(
                model,
                src_a,
                src_b,
                src_c,
                world_model_state=world_model_state,
            )
            action_loss = F.cross_entropy(logits, actions, reduction="none")
            losses.append(action_loss.squeeze(0) * class_weights[actions].squeeze(0))
            with torch.no_grad():
                prediction = int(logits.argmax(dim=-1).item())
                sequence_correct += int(prediction == int(actions.item()))
            world_model_state = (
                next_world_model_state.detach()
                if next_world_model_state is not None
                else None
            )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_correct += sequence_correct
        total_seen += len(sequence)
        total_loss += float(loss.detach().cpu().item()) * len(sequence)
    return total_loss, total_correct, total_seen


def _train_behavior_cloning_epoch_independent(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    dataset: list[BlockSMBDistillationExample],
    class_weights: torch.Tensor,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int, int]:
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        src_a, src_b, src_c, actions = _stack_examples(
            [dataset[index] for index in batch_indices],
            device,
        )
        logits = _action_logits(model, src_a, src_b, src_c)
        action_loss = F.cross_entropy(logits, actions, reduction="none")
        loss = (action_loss * class_weights[actions]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            total_correct += int((predictions == actions).sum().item())
            total_seen += int(actions.numel())
            total_loss += float(loss.detach().cpu().item()) * int(actions.numel())
    return total_loss, total_correct, total_seen


def _action_logits(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
) -> torch.Tensor:
    *_prefix, logits_a, _w_b, _b_b = model(src_a, src_b, src_c, tau=1.0)
    return logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT]


def _action_logits_with_state(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
    *,
    world_model_state: Any,
) -> tuple[torch.Tensor, Any]:
    episode_mask = torch.ones((src_c.size(0),), dtype=src_c.dtype, device=src_c.device)
    (
        _actions1,
        _next_state_pred,
        _criticism,
        _actions2,
        logits_a,
        _w_b,
        _b_b,
        next_world_model_state,
    ) = model(
        src_a,
        src_b,
        src_c,
        tau=1.0,
        world_model_state=world_model_state,
        episode_mask=episode_mask,
        return_world_model_state=True,
    )
    return logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT], next_world_model_state


def _example_sequences(
    dataset: list[BlockSMBDistillationExample],
) -> list[list[BlockSMBDistillationExample]]:
    grouped: dict[tuple[str, int], list[BlockSMBDistillationExample]] = {}
    for example in dataset:
        grouped.setdefault((example.scenario_name, example.episode), []).append(example)
    return [
        sorted(sequence, key=lambda example: example.step_index)
        for _key, sequence in sorted(grouped.items())
    ]


def _stack_examples(
    examples: list[BlockSMBDistillationExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_a = torch.cat([example.batch.src_a for example in examples], dim=0).to(device)
    src_b = torch.cat([example.batch.src_b for example in examples], dim=0).to(device)
    src_c = torch.cat([example.batch.src_c for example in examples], dim=0).to(device)
    actions = torch.tensor(
        [example.action for example in examples],
        dtype=torch.long,
        device=device,
    )
    return src_a, src_b, src_c, actions


def _detach_batch(batch: StageBatch) -> StageBatch:
    return StageBatch(
        src_a=batch.src_a.detach().cpu(),
        target_a=batch.target_a.detach().cpu() if batch.target_a is not None else None,
        src_b=batch.src_b.detach().cpu(),
        target_b=batch.target_b.detach().cpu() if batch.target_b is not None else None,
        src_c=batch.src_c.detach().cpu(),
        target_c=batch.target_c.detach().cpu() if batch.target_c is not None else None,
        metadata=None,
    )


def _action_class_weights(
    dataset: list[BlockSMBDistillationExample],
    device: torch.device,
    *,
    jump_weight_multiplier: float,
) -> torch.Tensor:
    counts = torch.zeros(BLOCK_SMB_ACTION_COUNT, dtype=torch.float32)
    for example in dataset:
        counts[int(example.action)] += 1.0
    nonzero = counts > 0
    weights = torch.zeros_like(counts)
    weights[nonzero] = counts[nonzero].sum() / (
        float(nonzero.sum().item()) * counts[nonzero]
    )
    if BLOCK_SMB_ACTION_COUNT > 2:
        weights[2] *= float(jump_weight_multiplier)
    weights[~nonzero] = 0.0
    return weights.to(device)


def _training_config_from_distillation(
    config: BlockSMBDistillationConfig,
) -> BlockSMBTrainingConfig:
    return BlockSMBTrainingConfig(
        seed=config.seed,
        architecture_config={
            "hidden_dim": config.hidden_dim,
            "controller_schedule": config.controller_schedule,
        },
        epochs=config.epochs,
        episodes_per_epoch=max(1, len(config.fixed_scenarios) * config.episodes_per_scenario),
        rollout_steps=config.rollout_steps,
        learning_rate=config.learning_rate,
        fixed_scenarios=config.fixed_scenarios,
        generated_scenarios=0,
        evaluation_episodes=config.evaluation_episodes,
        evaluation_max_steps=config.evaluation_max_steps,
        checkpoint_path=config.checkpoint_path,
        save_checkpoints=config.checkpoint_path is not None,
        resume_path=config.init_checkpoint,
        device=config.device,
        deterministic=config.deterministic,
        tracking_backend="none",
    )


def _cached_vision_factory(
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> Callable[[], VisionEncoder]:
    cache: dict[str, VisionEncoder] = {}

    def factory() -> VisionEncoder:
        if "model" not in cache:
            cache["model"] = load_block_vit_checkpoint(
                checkpoint_path,
                device=device,
                freeze=True,
            ).model
        return cache["model"]

    return factory


def _dataset_summary(dataset: list[BlockSMBDistillationExample]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    actions = torch.zeros(BLOCK_SMB_ACTION_COUNT, dtype=torch.long)
    max_step = 0
    for example in dataset:
        counts[example.scenario_name] = counts.get(example.scenario_name, 0) + 1
        actions[example.action] += 1
        max_step = max(max_step, example.step_index)
    return {
        "examples": len(dataset),
        "scenario_examples": counts,
        "action_counts": {
            str(index): int(count.item()) for index, count in enumerate(actions)
        },
        "max_step_index": max_step,
    }


def _checkpoint_summary(checkpoint: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if checkpoint is None:
        return None
    return {
        "stage": checkpoint.get("stage"),
        "model_name": checkpoint.get("model_name"),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }


def _write_json(path: Optional[Path], payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_data(payload), indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Optional[Path], payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_plain_data(payload), sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi-block-smb-distill",
        description="Distill scripted fixed-scenario Block SMB behavior into neural weights.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--vision-checkpoint", type=Path, default=DEFAULT_BLOCK_VIT_CHECKPOINT)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=BlockSMBDistillationConfig.seed)
    parser.add_argument("--epochs", type=int, default=BlockSMBDistillationConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BlockSMBDistillationConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=BlockSMBDistillationConfig.learning_rate)
    parser.add_argument(
        "--dagger-iterations",
        type=int,
        default=BlockSMBDistillationConfig.dagger_iterations,
    )
    parser.add_argument(
        "--dagger-epochs-per-iteration",
        type=int,
        default=BlockSMBDistillationConfig.dagger_epochs_per_iteration,
    )
    parser.add_argument(
        "--jump-weight-multiplier",
        type=float,
        default=BlockSMBDistillationConfig.jump_weight_multiplier,
    )
    parser.add_argument("--rollout-steps", type=int, default=BlockSMBDistillationConfig.rollout_steps)
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=BlockSMBDistillationConfig.episodes_per_scenario,
    )
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--evaluation-max-steps", type=int, default=200)
    parser.add_argument("--fixed-scenario", action="append", dest="fixed_scenarios")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--controller-schedule", default="constant")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--deterministic", action="store_true", dest="deterministic")
    parser.add_argument("--nondeterministic", action="store_false", dest="deterministic")
    parser.set_defaults(sequence_training=True)
    parser.add_argument("--sequence-training", action="store_true", dest="sequence_training")
    parser.add_argument(
        "--independent-training",
        action="store_false",
        dest="sequence_training",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = BlockSMBDistillationConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dagger_iterations=args.dagger_iterations,
        dagger_epochs_per_iteration=args.dagger_epochs_per_iteration,
        jump_weight_multiplier=args.jump_weight_multiplier,
        rollout_steps=args.rollout_steps,
        episodes_per_scenario=args.episodes_per_scenario,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps,
        fixed_scenarios=tuple(
            args.fixed_scenarios
            or BlockSMBDistillationConfig.__dataclass_fields__["fixed_scenarios"].default
        ),
        device=args.device,
        deterministic=args.deterministic,
        checkpoint_path=args.checkpoint,
        init_checkpoint=args.init_checkpoint,
        vision_checkpoint=args.vision_checkpoint,
        output_summary=args.output_summary,
        log_path=args.log_path,
        hidden_dim=args.hidden_dim,
        controller_schedule=args.controller_schedule,
        sequence_training=args.sequence_training,
    )
    result = train_distilled_block_smb_policy(config)
    print(json.dumps(to_plain_data(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
