"""Distill the scripted known-good Block SMB policy into neural weights."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import StageBatch, VisionEncoder, select_device, to_plain_data

from .adapter import BlockSMBStage
from .env import MarioScenarioEnv
from .monte_carlo import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    block_smb_monte_carlo_oracle_actions,
    sample_block_smb_monte_carlo_parameter_sweep,
    sample_block_smb_monte_carlo_split,
)
from .scripted_policy import BlockSMBScriptedPolicy, fixed_scenario_action_scripts
from .train import (
    BLOCK_SMB_ACTION_COUNT,
    DEFAULT_BLOCK_SMB_SEMANTIC_PREDICTION_ACCURACY_THRESHOLD,
    BlockSMBTrainingConfig,
    apply_block_smb_semantic_prediction_gate,
    block_smb_action_count_metric_values,
    block_smb_c_stream_dynamics_metrics,
    block_smb_c_stream_dynamics_slot_losses,
    block_smb_dynamics_loss,
    block_smb_monte_carlo_eval_metrics,
    evaluate_block_smb,
    load_fixed_scenarios,
    make_block_smb_model,
    normalize_block_smb_world_model_slot_weights,
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
    dynamics_loss_weight: float = 1.0
    world_model_slot_weights: Mapping[str, float] = field(
        default_factory=lambda: {"semantic_probabilities": 4.0}
    )
    semantic_prediction_accuracy_threshold: float = (
        DEFAULT_BLOCK_SMB_SEMANTIC_PREDICTION_ACCURACY_THRESHOLD
    )
    require_semantic_prediction_gate: bool = True
    sequence_training: bool = True
    dagger_iterations: int = 0
    dagger_epochs_per_iteration: int = 20
    rollout_steps: int = 200
    episodes_per_scenario: int = 3
    evaluation_episodes: int = 3
    evaluation_max_steps: int = 200
    monte_carlo_distribution_id: str = DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID
    monte_carlo_samples: int = 0
    monte_carlo_seed: int = 50_000
    monte_carlo_family_weights: Mapping[str, float] = field(default_factory=dict)
    monte_carlo_parameter_sweep: bool = False
    monte_carlo_sweep_repeats_per_difficulty: int = 1
    monte_carlo_validation_samples: int = 0
    monte_carlo_test_samples: int = 0
    monte_carlo_pass_rate_gate: float = 0.95
    monte_carlo_family_pass_rate_gate: float = 0.90
    fixed_scenarios: tuple[str, ...] = (
        "level_1_flat.json",
        "level_2_gap.json",
        "level_3_stairs.json",
        "level_4_platforms.json",
        "level_5_enemy_hop.json",
        "level_6_enemy_patrol.json",
        "level_7_moving_bridge.json",
        "level_8_enemy_gap.json",
        "level_9_enemy_stomp.json",
        "level_10_left_retreat.json",
        "level_11_left_jump_recovery.json",
        "level_12_wait_bridge.json",
        "level_13_variable_pits.json",
        "level_14_under_enemy_platform.json",
        "level_15_wait_long_bridge.json",
        "level_16_wait_enemy_gate.json",
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
        if self.dynamics_loss_weight < 0:
            raise ValueError("dynamics_loss_weight must be non-negative")
        if self.monte_carlo_samples < 0:
            raise ValueError("monte_carlo_samples must be non-negative")
        if not isinstance(self.monte_carlo_parameter_sweep, bool):
            raise TypeError("monte_carlo_parameter_sweep must be a bool")
        if self.monte_carlo_sweep_repeats_per_difficulty <= 0:
            raise ValueError("monte_carlo_sweep_repeats_per_difficulty must be positive")
        if self.monte_carlo_validation_samples < 0:
            raise ValueError("monte_carlo_validation_samples must be non-negative")
        if self.monte_carlo_test_samples < 0:
            raise ValueError("monte_carlo_test_samples must be non-negative")
        if not self.monte_carlo_distribution_id:
            raise ValueError("monte_carlo_distribution_id must be non-empty")
        object.__setattr__(
            self,
            "monte_carlo_family_weights",
            _normalize_distillation_monte_carlo_family_weights(
                self.monte_carlo_family_weights
            ),
        )
        if not 0.0 <= self.monte_carlo_pass_rate_gate <= 1.0:
            raise ValueError("monte_carlo_pass_rate_gate must be between 0 and 1")
        if not 0.0 <= self.monte_carlo_family_pass_rate_gate <= 1.0:
            raise ValueError("monte_carlo_family_pass_rate_gate must be between 0 and 1")
        object.__setattr__(
            self,
            "world_model_slot_weights",
            normalize_block_smb_world_model_slot_weights(self.world_model_slot_weights),
        )
        if not 0.0 <= self.semantic_prediction_accuracy_threshold <= 1.0:
            raise ValueError("semantic_prediction_accuracy_threshold must be between 0 and 1")
        if not isinstance(self.require_semantic_prediction_gate, bool):
            raise TypeError("require_semantic_prediction_gate must be a bool")
        for path_name in (
            "checkpoint_path",
            "init_checkpoint",
            "vision_checkpoint",
            "output_summary",
            "log_path",
        ):
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
    next_batch: StageBatch
    action: int
    scenario_name: str
    episode: int
    step_index: int


def _normalize_distillation_monte_carlo_family_weights(
    weights: Mapping[str, Any] | None,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_family, raw_weight in dict(weights or {}).items():
        family = str(raw_family)
        if family not in BLOCK_SMB_MC_FAMILIES:
            choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
            raise ValueError(f"unknown Block SMB Monte Carlo family {raw_family!r}; expected {choices}")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError("monte_carlo_family_weights must be non-negative")
        if weight > 0.0:
            normalized[family] = weight
    return normalized


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
    _source_scenarios, _source_scripts, distillation_sources = (
        build_block_smb_distillation_scenarios(config)
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
    final_metrics.update(
        block_smb_action_count_metric_values(
            "eval_fixed",
            evaluation.get("action_counts", {}),
        )
    )
    final_metrics.update(block_smb_monte_carlo_eval_metrics(evaluation))
    gated_evaluation = apply_block_smb_semantic_prediction_gate(
        evaluation,
        final_metrics,
        threshold=config.semantic_prediction_accuracy_threshold,
    )
    if config.require_semantic_prediction_gate:
        evaluation = gated_evaluation
    else:
        evaluation = {
            **evaluation,
            "semantic_prediction_gate_met": gated_evaluation["semantic_prediction_gate_met"],
            "semantic_prediction_accuracy": gated_evaluation["semantic_prediction_accuracy"],
            "semantic_prediction_accuracy_threshold": gated_evaluation[
                "semantic_prediction_accuracy_threshold"
            ],
            "tuning_metrics": {
                **dict(evaluation.get("tuning_metrics", {})),
                **{
                    key: value
                    for key, value in gated_evaluation.get("tuning_metrics", {}).items()
                    if key.startswith("semantic_prediction")
                },
            },
        }
    final_metrics.update(
        {
            "eval_success_thresholds_met_after_semantic_gate": float(
                bool(evaluation.get("success_thresholds_met"))
            ),
            "semantic_prediction_gate_met": float(
                bool(evaluation.get("semantic_prediction_gate_met"))
            ),
            "semantic_prediction_accuracy_threshold": float(
                config.semantic_prediction_accuracy_threshold
            ),
        }
    )
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
        "distillation_sources": distillation_sources,
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


def build_block_smb_distillation_scenarios(
    config: BlockSMBDistillationConfig,
) -> tuple[list[tuple[str, dict]], dict[str, list[int]], dict[str, Any]]:
    """Return distillation scenarios and teacher/oracle action scripts."""

    scenarios: list[tuple[str, dict]] = []
    action_scripts: dict[str, list[int]] = {}
    fixed_scripts = fixed_scenario_action_scripts(max_steps=config.rollout_steps)
    for scenario_name, scenario in load_fixed_scenarios(config.fixed_scenarios):
        if scenario_name not in fixed_scripts:
            raise ValueError(f"fixed scenario {scenario_name!r} does not have a scripted oracle")
        scenarios.append((scenario_name, scenario))
        action_scripts[scenario_name] = list(fixed_scripts[scenario_name])

    monte_carlo_manifest: dict[str, Any] = {}
    if config.monte_carlo_parameter_sweep or config.monte_carlo_samples > 0:
        if config.monte_carlo_parameter_sweep:
            sample_set = sample_block_smb_monte_carlo_parameter_sweep(
                distribution_id=config.monte_carlo_distribution_id,
                split="train",
                seed=config.monte_carlo_seed,
                repeats_per_difficulty=config.monte_carlo_sweep_repeats_per_difficulty,
            )
        else:
            sample_set = sample_block_smb_monte_carlo_split(
                distribution_id=config.monte_carlo_distribution_id,
                split="train",
                seed=config.monte_carlo_seed,
                sample_count=config.monte_carlo_samples,
                family_weights=config.monte_carlo_family_weights,
            )
        for sample in sample_set.samples:
            scenarios.append((sample.scenario_id, copy.deepcopy(dict(sample.scenario))))
            action_scripts[sample.scenario_id] = block_smb_monte_carlo_oracle_actions(
                sample.scenario,
                max_steps=config.rollout_steps,
            )
        monte_carlo_manifest = sample_set.manifest(include_scenarios=False)

    if not scenarios:
        raise ValueError("distillation requires at least one fixed or Monte Carlo scenario")

    summary = {
        "fixed_scenarios": list(config.fixed_scenarios),
        "fixed_scenario_count": len(config.fixed_scenarios),
        "monte_carlo": monte_carlo_manifest,
        "monte_carlo_coverage": monte_carlo_manifest.get("coverage", {}),
        "scenario_count": len(scenarios),
    }
    return scenarios, action_scripts, summary


def collect_scripted_distillation_examples(
    config: BlockSMBDistillationConfig,
    *,
    vision_factory: Callable[[], VisionEncoder],
) -> list[BlockSMBDistillationExample]:
    """Collect teacher-forced observations from the scripted fixed scenarios."""

    scenarios, action_scripts, _summary = build_block_smb_distillation_scenarios(config)
    policy = BlockSMBScriptedPolicy(action_scripts)
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
                    next_observation, _reward, terminated, truncated, info = stage.step(action)
                    next_batch = _detach_batch(
                        stage.encode_observation(next_observation, dict(info))
                    )
                    examples.append(
                        BlockSMBDistillationExample(
                            batch=batch,
                            next_batch=next_batch,
                            action=int(action),
                            scenario_name=scenario_name,
                            episode=episode,
                            step_index=step_index,
                        )
                    )
                    observation = next_observation
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

    scenarios, action_scripts, _summary = build_block_smb_distillation_scenarios(config)
    policy = BlockSMBScriptedPolicy(action_scripts)
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
                        src_a, src_b, src_c, _actions, _next_c = _stack_examples(
                            [
                                BlockSMBDistillationExample(
                                    batch=batch,
                                    next_batch=batch,
                                    action=int(teacher_action),
                                    scenario_name=scenario_name,
                                    episode=0,
                                    step_index=step_index,
                                )
                            ],
                            device,
                        )
                        (
                            logits,
                            _next_state_pred,
                            next_world_model_state,
                        ) = _action_logits_with_state(
                            model,
                            src_a,
                            src_b,
                            src_c,
                            world_model_state=world_model_state,
                        )
                        model_action = int(logits.argmax(dim=-1).item())
                        next_observation, _reward, terminated, truncated, info = stage.step(
                            model_action
                        )
                        next_batch = _detach_batch(
                            stage.encode_observation(next_observation, dict(info))
                        )
                        examples.append(
                            BlockSMBDistillationExample(
                                batch=batch,
                                next_batch=next_batch,
                                action=int(teacher_action),
                                scenario_name=scenario_name,
                                episode=config.episodes_per_scenario * iteration
                                + episode,
                                step_index=step_index,
                            )
                        )
                        if terminated or truncated:
                            break
                        observation = next_observation
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
            epoch_metrics = _train_behavior_cloning_epoch_recurrent(
                model,
                optimizer,
                sequences,
                class_weights,
                config,
                rng,
                device,
            )
        else:
            epoch_metrics = _train_behavior_cloning_epoch_independent(
                model,
                optimizer,
                dataset,
                class_weights,
                config,
                rng,
                config.batch_size,
                device,
            )
        record = {
            "phase": phase,
            "iteration": float(iteration),
            "epoch": float(epoch + 1),
            **epoch_metrics,
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
    config: BlockSMBDistillationConfig,
    rng: random.Random,
    device: torch.device,
) -> dict[str, float]:
    rng.shuffle(sequences)
    total_loss = 0.0
    total_action_loss = 0.0
    total_dynamics_loss = 0.0
    total_semantic_accuracy = 0.0
    total_semantic_gate = 0.0
    total_slot_losses = {slot_name: 0.0 for slot_name in config.world_model_slot_weights}
    for slot_name in (
        "position",
        "semantic_probabilities",
        "support_state",
        "state",
        "patch_tokens",
    ):
        total_slot_losses.setdefault(slot_name, 0.0)
    total_correct = 0
    total_seen = 0
    for sequence in sequences:
        losses = []
        action_losses = []
        dynamics_losses = []
        semantic_accuracies = []
        semantic_gates = []
        slot_losses_by_name: dict[str, list[torch.Tensor]] = {
            slot_name: [] for slot_name in total_slot_losses
        }
        world_model_state = None
        sequence_correct = 0
        for example in sequence:
            src_a, src_b, src_c, actions, next_c = _stack_examples([example], device)
            logits, next_state_pred, next_world_model_state = _action_logits_with_state(
                model,
                src_a,
                src_b,
                src_c,
                world_model_state=world_model_state,
            )
            action_loss = F.cross_entropy(logits, actions, reduction="none")
            weighted_action_loss = action_loss.squeeze(0) * class_weights[actions].squeeze(0)
            slot_losses = block_smb_c_stream_dynamics_slot_losses(
                next_state_pred,
                next_c.detach(),
                example.next_batch,
            )
            dynamics_loss = block_smb_dynamics_loss(
                next_state_pred,
                next_c.detach(),
                slot_losses,
                world_model_slot_weights=config.world_model_slot_weights,
            )
            metrics = block_smb_c_stream_dynamics_metrics(
                next_state_pred.detach(),
                next_c.detach(),
                example.next_batch,
                semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
            )
            losses.append(weighted_action_loss + config.dynamics_loss_weight * dynamics_loss)
            action_losses.append(weighted_action_loss)
            dynamics_losses.append(dynamics_loss)
            semantic_accuracies.append(metrics["dynamics_semantic_prediction_accuracy"])
            semantic_gates.append(metrics["dynamics_semantic_prediction_gate_met"])
            for slot_name, slot_loss in slot_losses.items():
                slot_losses_by_name.setdefault(slot_name, []).append(slot_loss.detach())
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
        total_action_loss += _tensor_list_mean(action_losses) * len(sequence)
        total_dynamics_loss += _tensor_list_mean(dynamics_losses) * len(sequence)
        total_semantic_accuracy += _tensor_list_mean(semantic_accuracies) * len(sequence)
        total_semantic_gate += _tensor_list_mean(semantic_gates) * len(sequence)
        for slot_name, values in slot_losses_by_name.items():
            total_slot_losses[slot_name] += _tensor_list_mean(values) * len(sequence)
    return _behavior_cloning_epoch_summary(
        total_loss=total_loss,
        total_action_loss=total_action_loss,
        total_dynamics_loss=total_dynamics_loss,
        total_slot_losses=total_slot_losses,
        total_semantic_accuracy=total_semantic_accuracy,
        total_semantic_gate=total_semantic_gate,
        total_correct=total_correct,
        total_seen=total_seen,
        semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
    )


def _train_behavior_cloning_epoch_independent(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    dataset: list[BlockSMBDistillationExample],
    class_weights: torch.Tensor,
    config: BlockSMBDistillationConfig,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    total_loss = 0.0
    total_action_loss = 0.0
    total_dynamics_loss = 0.0
    total_semantic_accuracy = 0.0
    total_semantic_gate = 0.0
    total_slot_losses = {slot_name: 0.0 for slot_name in config.world_model_slot_weights}
    for slot_name in (
        "position",
        "semantic_probabilities",
        "support_state",
        "state",
        "patch_tokens",
    ):
        total_slot_losses.setdefault(slot_name, 0.0)
    total_correct = 0
    total_seen = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        examples = [dataset[index] for index in batch_indices]
        src_a, src_b, src_c, actions, next_c = _stack_examples(examples, device)
        logits, next_state_pred = _action_logits_and_prediction(model, src_a, src_b, src_c)
        action_loss = F.cross_entropy(logits, actions, reduction="none")
        weighted_action_loss = (action_loss * class_weights[actions]).mean()
        slot_losses = _batched_distillation_slot_losses(next_state_pred, next_c, examples)
        dynamics_loss = block_smb_dynamics_loss(
            next_state_pred,
            next_c.detach(),
            slot_losses,
            world_model_slot_weights=config.world_model_slot_weights,
        )
        loss = weighted_action_loss + config.dynamics_loss_weight * dynamics_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            total_correct += int((predictions == actions).sum().item())
            total_seen += int(actions.numel())
            total_loss += float(loss.detach().cpu().item()) * int(actions.numel())
            total_action_loss += (
                float(weighted_action_loss.detach().cpu().item()) * int(actions.numel())
            )
            total_dynamics_loss += (
                float(dynamics_loss.detach().cpu().item()) * int(actions.numel())
            )
            metrics = _batched_distillation_dynamics_metrics(
                next_state_pred.detach(),
                next_c.detach(),
                examples,
                semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
            )
            total_semantic_accuracy += (
                float(metrics["dynamics_semantic_prediction_accuracy"].detach().cpu().item())
                * int(actions.numel())
            )
            total_semantic_gate += (
                float(metrics["dynamics_semantic_prediction_gate_met"].detach().cpu().item())
                * int(actions.numel())
            )
            for slot_name, slot_loss in slot_losses.items():
                total_slot_losses[slot_name] += (
                    float(slot_loss.detach().cpu().item()) * int(actions.numel())
                )
    return _behavior_cloning_epoch_summary(
        total_loss=total_loss,
        total_action_loss=total_action_loss,
        total_dynamics_loss=total_dynamics_loss,
        total_slot_losses=total_slot_losses,
        total_semantic_accuracy=total_semantic_accuracy,
        total_semantic_gate=total_semantic_gate,
        total_correct=total_correct,
        total_seen=total_seen,
        semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
    )


def _action_logits(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
) -> torch.Tensor:
    logits, _next_state_pred = _action_logits_and_prediction(model, src_a, src_b, src_c)
    return logits


def _action_logits_and_prediction(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _actions1, next_state_pred, _criticism, _actions2, logits_a, _w_b, _b_b = model(
        src_a,
        src_b,
        src_c,
        tau=1.0,
    )
    return logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT], next_state_pred


def _action_logits_with_state(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
    *,
    world_model_state: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    episode_mask = torch.ones((src_c.size(0),), dtype=src_c.dtype, device=src_c.device)
    (
        _actions1,
        next_state_pred,
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
    return logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT], next_state_pred, next_world_model_state


def _batched_distillation_slot_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    examples: list[BlockSMBDistillationExample],
) -> dict[str, torch.Tensor]:
    losses_by_slot: dict[str, list[torch.Tensor]] = {}
    for index, example in enumerate(examples):
        slot_losses = block_smb_c_stream_dynamics_slot_losses(
            prediction[index : index + 1],
            target[index : index + 1],
            example.next_batch,
        )
        for slot_name, slot_loss in slot_losses.items():
            losses_by_slot.setdefault(slot_name, []).append(slot_loss)
    return {
        slot_name: torch.stack(values).mean()
        for slot_name, values in losses_by_slot.items()
        if values
    }


def _batched_distillation_dynamics_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    examples: list[BlockSMBDistillationExample],
    *,
    semantic_accuracy_threshold: float,
) -> dict[str, torch.Tensor]:
    metrics_by_name: dict[str, list[torch.Tensor]] = {}
    for index, example in enumerate(examples):
        metrics = block_smb_c_stream_dynamics_metrics(
            prediction[index : index + 1],
            target[index : index + 1],
            example.next_batch,
            semantic_accuracy_threshold=semantic_accuracy_threshold,
        )
        for name, value in metrics.items():
            metrics_by_name.setdefault(name, []).append(value)
    metrics = {
        name: torch.stack(values).mean()
        for name, values in metrics_by_name.items()
        if values
    }
    if "dynamics_semantic_prediction_accuracy" in metrics:
        metrics["dynamics_semantic_prediction_gate_met"] = (
            metrics["dynamics_semantic_prediction_accuracy"] >= semantic_accuracy_threshold
        ).to(dtype=prediction.dtype)
    return metrics


def _behavior_cloning_epoch_summary(
    *,
    total_loss: float,
    total_action_loss: float,
    total_dynamics_loss: float,
    total_slot_losses: Mapping[str, float],
    total_semantic_accuracy: float,
    total_semantic_gate: float,
    total_correct: int,
    total_seen: int,
    semantic_accuracy_threshold: float,
) -> dict[str, float]:
    denom = max(total_seen, 1)
    summary = {
        "loss": total_loss / denom,
        "loss_action": total_action_loss / denom,
        "loss_dynamics": total_dynamics_loss / denom,
        "accuracy": total_correct / denom,
        "dynamics_semantic_prediction_accuracy": total_semantic_accuracy / denom,
        "dynamics_semantic_prediction_gate_met": float(
            (total_semantic_accuracy / denom) >= semantic_accuracy_threshold
        ),
        "dynamics_semantic_prediction_step_gate_rate": total_semantic_gate / denom,
    }
    summary.update(
        {
            f"loss_dynamics_{slot_name}": float(total_slot_loss) / denom
            for slot_name, total_slot_loss in total_slot_losses.items()
        }
    )
    return summary


def _tensor_list_mean(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    stacked = torch.stack([value.detach().float() for value in values])
    return float(stacked.mean().cpu().item())


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_a = torch.cat([example.batch.src_a for example in examples], dim=0).to(device)
    src_b = torch.cat([example.batch.src_b for example in examples], dim=0).to(device)
    src_c = torch.cat([example.batch.src_c for example in examples], dim=0).to(device)
    next_c = torch.cat([example.next_batch.src_c for example in examples], dim=0).to(device)
    actions = torch.tensor(
        [example.action for example in examples],
        dtype=torch.long,
        device=device,
    )
    return src_a, src_b, src_c, actions, next_c


def _detach_batch(batch: StageBatch) -> StageBatch:
    metadata = None
    if isinstance(batch.metadata, Mapping):
        fusion = batch.metadata.get("vision_fusion")
        if isinstance(fusion, Mapping):
            metadata = {"vision_fusion": dict(fusion)}
    return StageBatch(
        src_a=batch.src_a.detach().cpu(),
        target_a=batch.target_a.detach().cpu() if batch.target_a is not None else None,
        src_b=batch.src_b.detach().cpu(),
        target_b=batch.target_b.detach().cpu() if batch.target_b is not None else None,
        src_c=batch.src_c.detach().cpu(),
        target_c=batch.target_c.detach().cpu() if batch.target_c is not None else None,
        metadata=metadata,
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
    monte_carlo_count = (
        len(BLOCK_SMB_MC_FAMILIES)
        * len(BLOCK_SMB_MC_DIFFICULTY_BINS)
        * int(config.monte_carlo_sweep_repeats_per_difficulty)
        if config.monte_carlo_parameter_sweep
        else int(config.monte_carlo_samples)
    )
    scenario_count = len(config.fixed_scenarios) + monte_carlo_count
    return BlockSMBTrainingConfig(
        seed=config.seed,
        architecture_config={
            "hidden_dim": config.hidden_dim,
            "controller_schedule": config.controller_schedule,
        },
        epochs=config.epochs,
        episodes_per_epoch=max(1, scenario_count * config.episodes_per_scenario),
        rollout_steps=config.rollout_steps,
        learning_rate=config.learning_rate,
        world_model_slot_weights=config.world_model_slot_weights,
        fixed_scenarios=config.fixed_scenarios,
        generated_scenarios=0,
        monte_carlo_distribution_id=config.monte_carlo_distribution_id,
        monte_carlo_train_samples_per_epoch=monte_carlo_count,
        monte_carlo_seed=config.monte_carlo_seed,
        monte_carlo_family_weights=config.monte_carlo_family_weights,
        monte_carlo_parameter_sweep=config.monte_carlo_parameter_sweep,
        monte_carlo_sweep_repeats_per_difficulty=(
            config.monte_carlo_sweep_repeats_per_difficulty
        ),
        monte_carlo_validation_samples=config.monte_carlo_validation_samples,
        monte_carlo_test_samples=config.monte_carlo_test_samples,
        monte_carlo_pass_rate_gate=config.monte_carlo_pass_rate_gate,
        monte_carlo_family_pass_rate_gate=config.monte_carlo_family_pass_rate_gate,
        evaluation_episodes=config.evaluation_episodes,
        evaluation_max_steps=config.evaluation_max_steps,
        checkpoint_path=config.checkpoint_path,
        save_checkpoints=config.checkpoint_path is not None,
        resume_path=config.init_checkpoint,
        device=config.device,
        deterministic=config.deterministic,
        vision_checkpoint_path=config.vision_checkpoint,
        tracking_backend="none",
        semantic_prediction_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
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


def _slot_weight_arg(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use SLOT=WEIGHT syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("slot name must be non-empty")
    weight = float(raw_value)
    if weight <= 0.0:
        raise argparse.ArgumentTypeError("slot weight must be positive")
    return key, weight


def _family_weight_arg(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use FAMILY=WEIGHT syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if key not in BLOCK_SMB_MC_FAMILIES:
        choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
        raise argparse.ArgumentTypeError(f"unknown family {key!r}; expected one of: {choices}")
    weight = float(raw_value)
    if weight < 0.0:
        raise argparse.ArgumentTypeError("family weight must be non-negative")
    return key, weight


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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=BlockSMBDistillationConfig.learning_rate,
    )
    parser.add_argument(
        "--dynamics-loss-weight",
        type=float,
        default=BlockSMBDistillationConfig.dynamics_loss_weight,
    )
    parser.add_argument(
        "--world-model-slot-weight",
        action="append",
        default=None,
        type=_slot_weight_arg,
        metavar="SLOT=WEIGHT",
        help=(
            "weight a Block SMB C-stream dynamics slot; slots: position, "
            "semantic_probabilities, support_state, state, terminal_outcome, "
            "patch_tokens"
        ),
    )
    parser.add_argument(
        "--semantic-prediction-accuracy-threshold",
        type=float,
        default=BlockSMBDistillationConfig.semantic_prediction_accuracy_threshold,
    )
    parser.set_defaults(require_semantic_prediction_gate=True)
    parser.add_argument(
        "--require-semantic-prediction-gate",
        action="store_true",
        dest="require_semantic_prediction_gate",
    )
    parser.add_argument(
        "--no-require-semantic-prediction-gate",
        action="store_false",
        dest="require_semantic_prediction_gate",
    )
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
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=BlockSMBDistillationConfig.rollout_steps,
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=BlockSMBDistillationConfig.episodes_per_scenario,
    )
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--evaluation-max-steps", type=int, default=200)
    parser.add_argument("--fixed-scenario", action="append", dest="fixed_scenarios")
    parser.add_argument(
        "--monte-carlo-distribution",
        default=BlockSMBDistillationConfig.monte_carlo_distribution_id,
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=BlockSMBDistillationConfig.monte_carlo_samples,
    )
    parser.add_argument("--monte-carlo-seed", type=int, default=50_000)
    parser.add_argument(
        "--monte-carlo-family-weight",
        action="append",
        default=None,
        type=_family_weight_arg,
        metavar="FAMILY=WEIGHT",
    )
    parser.set_defaults(monte_carlo_parameter_sweep=False)
    parser.add_argument(
        "--monte-carlo-parameter-sweep",
        action="store_true",
        dest="monte_carlo_parameter_sweep",
        help="use a deterministic full family x difficulty Monte Carlo sweep",
    )
    parser.add_argument(
        "--monte-carlo-sweep-repeats-per-difficulty",
        type=int,
        default=BlockSMBDistillationConfig.monte_carlo_sweep_repeats_per_difficulty,
    )
    parser.add_argument("--monte-carlo-validation-samples", type=int, default=0)
    parser.add_argument("--monte-carlo-test-samples", type=int, default=0)
    parser.add_argument("--monte-carlo-pass-rate-gate", type=float, default=0.95)
    parser.add_argument("--monte-carlo-family-pass-rate-gate", type=float, default=0.90)
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
        dynamics_loss_weight=args.dynamics_loss_weight,
        world_model_slot_weights=dict(
            args.world_model_slot_weight
            or BlockSMBDistillationConfig.__dataclass_fields__[
                "world_model_slot_weights"
            ].default_factory()
        ),
        semantic_prediction_accuracy_threshold=args.semantic_prediction_accuracy_threshold,
        require_semantic_prediction_gate=args.require_semantic_prediction_gate,
        dagger_iterations=args.dagger_iterations,
        dagger_epochs_per_iteration=args.dagger_epochs_per_iteration,
        jump_weight_multiplier=args.jump_weight_multiplier,
        rollout_steps=args.rollout_steps,
        episodes_per_scenario=args.episodes_per_scenario,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps,
        monte_carlo_distribution_id=args.monte_carlo_distribution,
        monte_carlo_samples=args.monte_carlo_samples,
        monte_carlo_seed=args.monte_carlo_seed,
        monte_carlo_family_weights=dict(args.monte_carlo_family_weight or ()),
        monte_carlo_parameter_sweep=args.monte_carlo_parameter_sweep,
        monte_carlo_sweep_repeats_per_difficulty=(
            args.monte_carlo_sweep_repeats_per_difficulty
        ),
        monte_carlo_validation_samples=args.monte_carlo_validation_samples,
        monte_carlo_test_samples=args.monte_carlo_test_samples,
        monte_carlo_pass_rate_gate=args.monte_carlo_pass_rate_gate,
        monte_carlo_family_pass_rate_gate=args.monte_carlo_family_pass_rate_gate,
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
