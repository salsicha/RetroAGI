"""Distill the scripted known-good Block SMB policy into neural weights."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import (
    DEFAULT_PRIMITIVE_DURATION_BINS,
    PrimitiveOutcomePrediction,
    SMBAction,
    StageBatch,
    VisionEncoder,
    is_smb_jump_action,
    select_device,
    smb_jump_release_action,
    to_plain_data,
)

from .adapter import BlockSMBStage
from .env import MarioScenarioEnv
from .monte_carlo import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    BlockSMBMonteCarloSampleSet,
    BlockSMBScenarioSample,
    block_smb_monte_carlo_oracle_actions,
    sample_block_smb_monte_carlo_parameter_sweep,
    sample_block_smb_monte_carlo_split,
    summarize_block_smb_monte_carlo_samples,
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
    block_smb_c_stream_slot_spans,
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

DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES = BLOCK_SMB_MC_FAMILIES


@dataclass(frozen=True)
class BlockSMBDistillationConfig:
    """Configuration for scripted-policy behavioral cloning."""

    seed: int = 20260627
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    jump_weight_multiplier: float = 8.0
    dynamics_loss_weight: float = 1.0
    primitive_loss_weight: float = 0.25
    primitive_hazard_weight_multiplier: float = 2.0
    primitive_outcome_loss_weight: float = 0.25
    primitive_outcome_horizon: int = 8
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
    required_monte_carlo_families: tuple[str, ...] = DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES
    required_monte_carlo_repeats_per_difficulty: int = 1
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
        if self.primitive_loss_weight < 0:
            raise ValueError("primitive_loss_weight must be non-negative")
        if self.primitive_hazard_weight_multiplier <= 0:
            raise ValueError("primitive_hazard_weight_multiplier must be positive")
        if self.primitive_outcome_loss_weight < 0:
            raise ValueError("primitive_outcome_loss_weight must be non-negative")
        if self.primitive_outcome_horizon <= 0:
            raise ValueError("primitive_outcome_horizon must be positive")
        if self.monte_carlo_samples < 0:
            raise ValueError("monte_carlo_samples must be non-negative")
        if not isinstance(self.monte_carlo_parameter_sweep, bool):
            raise TypeError("monte_carlo_parameter_sweep must be a bool")
        if self.monte_carlo_sweep_repeats_per_difficulty <= 0:
            raise ValueError("monte_carlo_sweep_repeats_per_difficulty must be positive")
        if self.required_monte_carlo_repeats_per_difficulty <= 0:
            raise ValueError("required_monte_carlo_repeats_per_difficulty must be positive")
        if self.monte_carlo_validation_samples < 0:
            raise ValueError("monte_carlo_validation_samples must be non-negative")
        if self.monte_carlo_test_samples < 0:
            raise ValueError("monte_carlo_test_samples must be non-negative")
        if not self.monte_carlo_distribution_id:
            raise ValueError("monte_carlo_distribution_id must be non-empty")
        object.__setattr__(
            self,
            "monte_carlo_family_weights",
            _normalize_distillation_monte_carlo_family_weights(self.monte_carlo_family_weights),
        )
        object.__setattr__(
            self,
            "required_monte_carlo_families",
            _normalize_distillation_monte_carlo_families(self.required_monte_carlo_families),
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
    primitive_button_combo: int = int(SMBAction.NOOP)
    primitive_button_combo_mask: float = 0.0
    primitive_duration_bin: int = 0
    primitive_duration_mask: float = 0.0
    primitive_release: float = 0.0
    primitive_release_mask: float = 0.0
    primitive_post_release: int = int(SMBAction.NOOP)
    primitive_post_release_mask: float = 0.0
    primitive_cancel: float = 0.0
    primitive_cancel_mask: float = 0.0
    primitive_replan: float = 0.0
    primitive_replan_mask: float = 0.0
    primitive_hazard_window: float = 0.0
    primitive_hazard_window_mask: float = 0.0
    primitive_weight: float = 1.0
    primitive_outcome_mask: float = 0.0
    primitive_outcome_progress_delta: float = 0.0
    primitive_outcome_support_loss: float = 0.0
    primitive_outcome_collision_death_risk: float = 0.0
    primitive_outcome_terminal: float = 0.0
    primitive_outcome_continue: float = 0.0
    primitive_outcome_cancel: float = 0.0
    primitive_outcome_replan: float = 0.0


_PRIMITIVE_HAZARD_SCENARIO_TOKENS = (
    "bridge",
    "chained",
    "enemy",
    "gap",
    "opening",
    "pipe",
    "pit",
    "platform",
    "stair",
)


def _scripted_primitive_labels(
    actions: Sequence[int],
    *,
    scenario_name: str,
    scenario: Mapping[str, Any],
    hazard_weight_multiplier: float,
) -> list[dict[str, float | int]]:
    labels = [
        {
            "primitive_button_combo": int(action),
            "primitive_button_combo_mask": 1.0,
            "primitive_duration_bin": 0,
            "primitive_duration_mask": 0.0,
            "primitive_release": 0.0,
            "primitive_release_mask": 0.0,
            "primitive_post_release": int(SMBAction.NOOP),
            "primitive_post_release_mask": 0.0,
            "primitive_cancel": 0.0,
            "primitive_cancel_mask": 0.0,
            "primitive_replan": (
                1.0 if index == 0 or int(action) != int(actions[index - 1]) else 0.0
            ),
            "primitive_replan_mask": 1.0,
            "primitive_hazard_window": 0.0,
            "primitive_hazard_window_mask": 0.0,
            "primitive_weight": 1.0,
        }
        for index, action in enumerate(actions)
    ]
    has_timing_hazard = _scenario_has_primitive_timing_hazard(scenario_name, scenario)
    hazard_weight = float(hazard_weight_multiplier) if has_timing_hazard else 1.0
    index = 0
    while index < len(actions):
        action = int(actions[index])
        end = index + 1
        while end < len(actions) and int(actions[end]) == action:
            end += 1
        if has_timing_hazard and action == int(SMBAction.NOOP):
            for wait_index in range(index, end):
                labels[wait_index]["primitive_hazard_window"] = 1.0
                labels[wait_index]["primitive_hazard_window_mask"] = 1.0
                labels[wait_index]["primitive_weight"] = hazard_weight
            labels[end - 1]["primitive_replan"] = 1.0
        if not is_smb_jump_action(action):
            index = end
            continue
        duration_bin = _nearest_primitive_duration_bin(end - index)
        release_action = int(smb_jump_release_action(action))
        hazard_start = max(0, index - 1) if has_timing_hazard else index
        for hazard_index in range(hazard_start, end):
            labels[hazard_index]["primitive_hazard_window"] = 1.0
            labels[hazard_index]["primitive_hazard_window_mask"] = 1.0
            labels[hazard_index]["primitive_weight"] = hazard_weight
        for jump_index in range(index, end):
            labels[jump_index]["primitive_release_mask"] = 1.0
            labels[jump_index]["primitive_post_release"] = release_action
            labels[jump_index]["primitive_post_release_mask"] = 1.0
            labels[jump_index]["primitive_weight"] = hazard_weight
        labels[index]["primitive_duration_bin"] = duration_bin
        labels[index]["primitive_duration_mask"] = 1.0
        labels[index]["primitive_replan"] = 1.0
        labels[end - 1]["primitive_release"] = 1.0
        if has_timing_hazard:
            labels[end - 1]["primitive_cancel"] = 1.0
            labels[end - 1]["primitive_cancel_mask"] = 1.0
            labels[end - 1]["primitive_replan"] = 1.0
        for jump_index in range(index, end - 1):
            labels[jump_index]["primitive_cancel_mask"] = 1.0
        index = end
    return labels


def _nearest_primitive_duration_bin(duration: int | float) -> int:
    best_index = 0
    best_distance = float("inf")
    for index, bin_value in enumerate(DEFAULT_PRIMITIVE_DURATION_BINS):
        distance = abs(float(bin_value) - float(duration))
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _scenario_has_primitive_timing_hazard(
    scenario_name: str,
    scenario: Mapping[str, Any],
) -> bool:
    lowered_name = str(scenario_name).lower()
    if any(token in lowered_name for token in _PRIMITIVE_HAZARD_SCENARIO_TOKENS):
        return True
    if bool(scenario.get("enemies")):
        return True
    platforms = scenario.get("platforms")
    if isinstance(platforms, Sequence) and len(platforms) > 1:
        return True
    return False


def _annotate_primitive_outcomes(
    examples: Sequence[BlockSMBDistillationExample],
    *,
    horizon: int,
) -> list[BlockSMBDistillationExample]:
    if not examples:
        return []
    resolved_horizon = max(1, int(horizon))
    return [
        replace(
            example,
            **_primitive_outcome_target_fields(
                examples,
                index,
                horizon=resolved_horizon,
            ),
        )
        for index, example in enumerate(examples)
    ]


def _primitive_outcome_target_fields(
    examples: Sequence[BlockSMBDistillationExample],
    index: int,
    *,
    horizon: int,
) -> dict[str, float]:
    current = examples[index]
    future_end = min(len(examples) - 1, index + max(1, int(horizon)) - 1)
    future_examples = examples[index + 1 : future_end + 1]
    future_terminal_examples = examples[index : future_end + 1]
    future_batch = examples[future_end].next_batch

    progress_delta = _c_stream_progress_x(future_batch) - _c_stream_progress_x(current.batch)
    support_loss = _c_stream_support_loss(
        current.batch,
        [example.next_batch for example in future_terminal_examples],
    )
    collision_death_risk = max(
        _batch_collision_death_risk(example.next_batch) for example in future_terminal_examples
    )
    terminal_outcome = max(
        _batch_terminal_outcome(example.next_batch) for example in future_terminal_examples
    )
    future_replan = (
        max(float(example.primitive_replan) for example in future_examples)
        if future_examples
        else 0.0
    )
    future_cancel = (
        max(float(example.primitive_cancel) for example in future_examples)
        if future_examples
        else 0.0
    )
    bad_progress = 1.0 if progress_delta <= 0.0 and terminal_outcome <= 0.0 else 0.0
    cancel = max(future_cancel, collision_death_risk)
    replan = max(future_replan, bad_progress, terminal_outcome)
    should_continue = (
        1.0
        if future_examples
        and cancel <= 0.0
        and replan <= 0.0
        and terminal_outcome <= 0.0
        and support_loss < 0.75
        else 0.0
    )
    return {
        "primitive_outcome_mask": 1.0,
        "primitive_outcome_progress_delta": float(progress_delta),
        "primitive_outcome_support_loss": float(support_loss),
        "primitive_outcome_collision_death_risk": float(collision_death_risk),
        "primitive_outcome_terminal": float(terminal_outcome),
        "primitive_outcome_continue": float(should_continue),
        "primitive_outcome_cancel": float(cancel),
        "primitive_outcome_replan": float(replan),
    }


def _c_stream_progress_x(batch: StageBatch) -> float:
    info = _batch_info(batch)
    state_vec = info.get("state_vec")
    if state_vec is not None:
        try:
            return float(state_vec[0])
        except (TypeError, ValueError, IndexError):
            pass
    spans = block_smb_c_stream_slot_spans(batch)
    start, end = spans.get("position", (0, 0))
    if end > start:
        return float(batch.src_c[:, start].detach().float().mean().cpu().item())
    if batch.src_c.numel() == 0:
        return 0.0
    return float(batch.src_c[:, 0].detach().float().mean().cpu().item())


def _c_stream_support_loss(
    current_batch: StageBatch,
    future_batches: Sequence[StageBatch],
) -> float:
    spans = block_smb_c_stream_slot_spans(current_batch)
    start, end = spans.get("support_state", (0, 0))
    if end - start < 2:
        return 0.0
    current_support = current_batch.src_c[:, start:end].detach().float().clamp(0.0, 1.0)
    current_air = current_support[:, 0]
    current_stable = current_support[:, 1:].amax(dim=1)
    risks = []
    for batch in future_batches:
        future_spans = block_smb_c_stream_slot_spans(batch)
        future_start, future_end = future_spans.get("support_state", (0, 0))
        if future_end - future_start != end - start:
            continue
        future_support = batch.src_c[:, future_start:future_end].detach().float().clamp(0.0, 1.0)
        future_air = future_support[:, 0]
        future_stable = future_support[:, 1:].amax(dim=1)
        risks.append(
            torch.maximum(
                (future_air - current_air).clamp_min(0.0),
                (current_stable - future_stable).clamp_min(0.0),
            )
        )
    if not risks:
        return 0.0
    return float(torch.stack(risks, dim=0).amax().clamp(0.0, 1.0).cpu().item())


def _batch_info(batch: StageBatch) -> Mapping[str, Any]:
    metadata = batch.metadata if isinstance(batch.metadata, Mapping) else {}
    info = metadata.get("info", {})
    return info if isinstance(info, Mapping) else {}


def _batch_episode(batch: StageBatch) -> Mapping[str, Any]:
    metadata = batch.metadata if isinstance(batch.metadata, Mapping) else {}
    episode = metadata.get("episode", {})
    return episode if isinstance(episode, Mapping) else {}


def _batch_collision_death_risk(batch: StageBatch) -> float:
    info = _batch_info(batch)
    if bool(info.get("death", False)):
        return 1.0
    reward_terms = info.get("reward_terms", {})
    if isinstance(reward_terms, Mapping):
        if float(reward_terms.get("fall_death", 0.0) or 0.0) < 0.0:
            return 1.0
        if float(reward_terms.get("enemy_hit", 0.0) or 0.0) < 0.0:
            return 1.0
    return 0.0


def _batch_terminal_outcome(batch: StageBatch) -> float:
    info = _batch_info(batch)
    episode = _batch_episode(batch)
    if bool(info.get("terminated", False)) or bool(info.get("truncated", False)):
        return 1.0
    if bool(episode.get("terminated", False)) or bool(episode.get("truncated", False)):
        return 1.0
    spans = block_smb_c_stream_slot_spans(batch)
    start, end = spans.get("terminal_outcome", (0, 0))
    if end > start:
        terminal_score = batch.src_c[:, start:end].detach().float().clamp(0.0, 1.0).amax()
        return float((terminal_score > 0.5).to(dtype=torch.float32).cpu().item())
    return 0.0


def _normalize_distillation_monte_carlo_family_weights(
    weights: Mapping[str, Any] | None,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_family, raw_weight in dict(weights or {}).items():
        family = str(raw_family)
        if family not in BLOCK_SMB_MC_FAMILIES:
            choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
            raise ValueError(
                f"unknown Block SMB Monte Carlo family {raw_family!r}; expected {choices}"
            )
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError("monte_carlo_family_weights must be non-negative")
        if weight > 0.0:
            normalized[family] = weight
    return normalized


def _normalize_distillation_monte_carlo_families(
    families: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_family in families or ():
        family = str(raw_family)
        if family not in BLOCK_SMB_MC_FAMILIES:
            choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
            raise ValueError(
                f"unknown Block SMB Monte Carlo family {raw_family!r}; expected {choices}"
            )
        if family not in normalized:
            normalized.append(family)
    return tuple(normalized)


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

    monte_carlo_samples, monte_carlo_manifest = _distillation_monte_carlo_samples(config)
    for sample in monte_carlo_samples:
        scenarios.append((sample.scenario_id, copy.deepcopy(dict(sample.scenario))))
        action_scripts[sample.scenario_id] = block_smb_monte_carlo_oracle_actions(
            sample.scenario,
            max_steps=config.rollout_steps,
        )

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


def _distillation_monte_carlo_samples(
    config: BlockSMBDistillationConfig,
) -> tuple[list[BlockSMBScenarioSample], dict[str, Any]]:
    sample_sets: list[tuple[str, BlockSMBMonteCarloSampleSet]] = []
    if config.required_monte_carlo_families:
        sample_sets.append(
            (
                "required_warm_start",
                sample_block_smb_monte_carlo_parameter_sweep(
                    distribution_id=config.monte_carlo_distribution_id,
                    split="train",
                    seed=config.monte_carlo_seed,
                    repeats_per_difficulty=(config.required_monte_carlo_repeats_per_difficulty),
                    families=config.required_monte_carlo_families,
                ),
            )
        )
    if config.monte_carlo_parameter_sweep:
        sample_sets.append(
            (
                "parameter_sweep",
                sample_block_smb_monte_carlo_parameter_sweep(
                    distribution_id=config.monte_carlo_distribution_id,
                    split="train",
                    seed=config.monte_carlo_seed,
                    repeats_per_difficulty=config.monte_carlo_sweep_repeats_per_difficulty,
                ),
            )
        )
    elif config.monte_carlo_samples > 0:
        sample_sets.append(
            (
                "sampled",
                sample_block_smb_monte_carlo_split(
                    distribution_id=config.monte_carlo_distribution_id,
                    split="train",
                    seed=config.monte_carlo_seed,
                    sample_count=config.monte_carlo_samples,
                    family_weights=config.monte_carlo_family_weights,
                ),
            )
        )

    if not sample_sets:
        return [], {}

    selected_samples: list[BlockSMBScenarioSample] = []
    source_manifests: dict[str, Any] = {}
    source_selected_counts: dict[str, int] = {}
    seen_keys: set[tuple[Any, ...]] = set()
    for source_name, sample_set in sample_sets:
        source_manifests[source_name] = sample_set.manifest(include_scenarios=False)
        source_selected_counts[source_name] = 0
        for sample in sample_set.samples:
            sample_key = _distillation_monte_carlo_sample_key(sample)
            if sample_key in seen_keys:
                continue
            seen_keys.add(sample_key)
            selected_samples.append(sample)
            source_selected_counts[source_name] += 1

    manifest = {
        "schema_version": selected_samples[0].schema_version if selected_samples else None,
        "distribution_id": config.monte_carlo_distribution_id,
        "split": "train",
        "seed": int(config.monte_carlo_seed),
        "sample_count": len(selected_samples),
        "required_families": list(config.required_monte_carlo_families),
        "required_repeats_per_difficulty": int(config.required_monte_carlo_repeats_per_difficulty),
        "source_selected_counts": source_selected_counts,
        "sources": source_manifests,
        "coverage": summarize_block_smb_monte_carlo_samples(selected_samples),
        "scenario_ids": [sample.scenario_id for sample in selected_samples],
    }
    return selected_samples, manifest


def _distillation_monte_carlo_sample_key(
    sample: BlockSMBScenarioSample,
) -> tuple[Any, ...]:
    parameters = sample.parameters if isinstance(sample.parameters, Mapping) else {}
    if bool(parameters.get("parameter_sweep", False)):
        return (
            "sweep",
            sample.family,
            str(parameters.get("difficulty_bin", "default")),
            int(parameters.get("sweep_repeat", 0)),
        )
    return ("scenario", sample.scenario_id)


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
        primitive_labels = _scripted_primitive_labels(
            action_scripts[scenario_name],
            scenario_name=scenario_name,
            scenario=scenario,
            hazard_weight_multiplier=config.primitive_hazard_weight_multiplier,
        )
        for episode in range(config.episodes_per_scenario):
            env = MarioScenarioEnv()
            stage = BlockSMBStage(env=env, scenario=scenario, vision=vision_factory())
            episode_examples: list[BlockSMBDistillationExample] = []
            try:
                observation = stage.reset(seed=config.seed + scenario_index * 10_000 + episode)
                for step_index in range(config.rollout_steps):
                    action = policy.action(scenario_name, step_index)
                    batch = _detach_batch(stage.encode_observation(observation))
                    next_observation, _reward, terminated, truncated, info = stage.step(action)
                    next_batch = _detach_batch(
                        stage.encode_observation(next_observation, dict(info))
                    )
                    primitive_label = primitive_labels[min(step_index, len(primitive_labels) - 1)]
                    episode_examples.append(
                        BlockSMBDistillationExample(
                            batch=batch,
                            next_batch=next_batch,
                            action=int(action),
                            scenario_name=scenario_name,
                            episode=episode,
                            step_index=step_index,
                            **primitive_label,
                        )
                    )
                    observation = next_observation
                    if terminated or truncated:
                        break
            finally:
                env.close()
            examples.extend(
                _annotate_primitive_outcomes(
                    episode_examples,
                    horizon=config.primitive_outcome_horizon,
                )
            )
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
            primitive_labels = _scripted_primitive_labels(
                action_scripts[scenario_name],
                scenario_name=scenario_name,
                scenario=scenario,
                hazard_weight_multiplier=config.primitive_hazard_weight_multiplier,
            )
            for episode in range(config.episodes_per_scenario):
                env = MarioScenarioEnv()
                stage = BlockSMBStage(env=env, scenario=scenario, vision=vision_factory())
                episode_examples: list[BlockSMBDistillationExample] = []
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
                            _motor_primitives,
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
                        primitive_label = primitive_labels[
                            min(step_index, len(primitive_labels) - 1)
                        ]
                        episode_examples.append(
                            BlockSMBDistillationExample(
                                batch=batch,
                                next_batch=next_batch,
                                action=int(teacher_action),
                                scenario_name=scenario_name,
                                episode=config.episodes_per_scenario * iteration + episode,
                                step_index=step_index,
                                **primitive_label,
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
                examples.extend(
                    _annotate_primitive_outcomes(
                        episode_examples,
                        horizon=config.primitive_outcome_horizon,
                    )
                )
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
    total_primitive_loss = 0.0
    total_primitive_outcome_loss = 0.0
    total_semantic_accuracy = 0.0
    total_semantic_gate = 0.0
    primitive_counts = _empty_primitive_supervision_counts()
    total_slot_losses = {slot_name: 0.0 for slot_name in config.world_model_slot_weights}
    for slot_name in (
        "position",
        "semantic_probabilities",
        "support_state",
        "state",
        "terminal_outcome",
        "patch_tokens",
    ):
        total_slot_losses.setdefault(slot_name, 0.0)
    total_correct = 0
    total_seen = 0
    for sequence in sequences:
        losses = []
        action_losses = []
        dynamics_losses = []
        primitive_losses = []
        primitive_outcome_losses = []
        semantic_accuracies = []
        semantic_gates = []
        slot_losses_by_name: dict[str, list[torch.Tensor]] = {
            slot_name: [] for slot_name in total_slot_losses
        }
        world_model_state = None
        sequence_correct = 0
        for example in sequence:
            src_a, src_b, src_c, actions, next_c = _stack_examples([example], device)
            (
                logits,
                next_state_pred,
                next_world_model_state,
                motor_primitives,
            ) = _action_logits_with_state(
                model,
                src_a,
                src_b,
                src_c,
                world_model_state=world_model_state,
            )
            action_loss = F.cross_entropy(logits, actions, reduction="none")
            weighted_action_loss = action_loss.squeeze(0) * class_weights[actions].squeeze(0)
            primitive_loss = _block_smb_distillation_primitive_loss(
                motor_primitives,
                [example],
                device=device,
            )
            primitive_outcome_loss = _block_smb_distillation_primitive_outcome_loss(
                getattr(model, "last_primitive_outcome", None),
                [example],
                device=device,
            )
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
            losses.append(
                weighted_action_loss
                + config.dynamics_loss_weight * dynamics_loss
                + config.primitive_loss_weight * primitive_loss
                + config.primitive_outcome_loss_weight * primitive_outcome_loss
            )
            action_losses.append(weighted_action_loss)
            dynamics_losses.append(dynamics_loss)
            primitive_losses.append(primitive_loss)
            primitive_outcome_losses.append(primitive_outcome_loss)
            _add_primitive_supervision_counts(primitive_counts, [example])
            semantic_accuracies.append(metrics["dynamics_semantic_prediction_accuracy"])
            semantic_gates.append(metrics["dynamics_semantic_prediction_gate_met"])
            for slot_name, slot_loss in slot_losses.items():
                slot_losses_by_name.setdefault(slot_name, []).append(slot_loss.detach())
            with torch.no_grad():
                prediction = int(logits.argmax(dim=-1).item())
                sequence_correct += int(prediction == int(actions.item()))
            world_model_state = (
                next_world_model_state.detach() if next_world_model_state is not None else None
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
        total_primitive_loss += _tensor_list_mean(primitive_losses) * len(sequence)
        total_primitive_outcome_loss += _tensor_list_mean(primitive_outcome_losses) * len(sequence)
        total_semantic_accuracy += _tensor_list_mean(semantic_accuracies) * len(sequence)
        total_semantic_gate += _tensor_list_mean(semantic_gates) * len(sequence)
        for slot_name, values in slot_losses_by_name.items():
            total_slot_losses[slot_name] += _tensor_list_mean(values) * len(sequence)
    return _behavior_cloning_epoch_summary(
        total_loss=total_loss,
        total_action_loss=total_action_loss,
        total_dynamics_loss=total_dynamics_loss,
        total_primitive_loss=total_primitive_loss,
        total_primitive_outcome_loss=total_primitive_outcome_loss,
        total_slot_losses=total_slot_losses,
        total_semantic_accuracy=total_semantic_accuracy,
        total_semantic_gate=total_semantic_gate,
        total_correct=total_correct,
        total_seen=total_seen,
        primitive_counts=primitive_counts,
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
    total_primitive_loss = 0.0
    total_primitive_outcome_loss = 0.0
    total_semantic_accuracy = 0.0
    total_semantic_gate = 0.0
    primitive_counts = _empty_primitive_supervision_counts()
    total_slot_losses = {slot_name: 0.0 for slot_name in config.world_model_slot_weights}
    for slot_name in (
        "position",
        "semantic_probabilities",
        "support_state",
        "state",
        "terminal_outcome",
        "patch_tokens",
    ):
        total_slot_losses.setdefault(slot_name, 0.0)
    total_correct = 0
    total_seen = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        examples = [dataset[index] for index in batch_indices]
        src_a, src_b, src_c, actions, next_c = _stack_examples(examples, device)
        logits, next_state_pred, motor_primitives = _action_logits_and_prediction(
            model,
            src_a,
            src_b,
            src_c,
        )
        action_loss = F.cross_entropy(logits, actions, reduction="none")
        weighted_action_loss = (action_loss * class_weights[actions]).mean()
        primitive_loss = _block_smb_distillation_primitive_loss(
            motor_primitives,
            examples,
            device=device,
        )
        primitive_outcome_loss = _block_smb_distillation_primitive_outcome_loss(
            getattr(model, "last_primitive_outcome", None),
            examples,
            device=device,
        )
        slot_losses = _batched_distillation_slot_losses(next_state_pred, next_c, examples)
        dynamics_loss = block_smb_dynamics_loss(
            next_state_pred,
            next_c.detach(),
            slot_losses,
            world_model_slot_weights=config.world_model_slot_weights,
        )
        loss = (
            weighted_action_loss
            + config.dynamics_loss_weight * dynamics_loss
            + config.primitive_loss_weight * primitive_loss
            + config.primitive_outcome_loss_weight * primitive_outcome_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            total_correct += int((predictions == actions).sum().item())
            total_seen += int(actions.numel())
            total_loss += float(loss.detach().cpu().item()) * int(actions.numel())
            total_action_loss += float(weighted_action_loss.detach().cpu().item()) * int(
                actions.numel()
            )
            total_dynamics_loss += float(dynamics_loss.detach().cpu().item()) * int(actions.numel())
            total_primitive_loss += float(primitive_loss.detach().cpu().item()) * int(
                actions.numel()
            )
            total_primitive_outcome_loss += float(
                primitive_outcome_loss.detach().cpu().item()
            ) * int(actions.numel())
            _add_primitive_supervision_counts(primitive_counts, examples)
            metrics = _batched_distillation_dynamics_metrics(
                next_state_pred.detach(),
                next_c.detach(),
                examples,
                semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
            )
            total_semantic_accuracy += float(
                metrics["dynamics_semantic_prediction_accuracy"].detach().cpu().item()
            ) * int(actions.numel())
            total_semantic_gate += float(
                metrics["dynamics_semantic_prediction_gate_met"].detach().cpu().item()
            ) * int(actions.numel())
            for slot_name, slot_loss in slot_losses.items():
                total_slot_losses[slot_name] += float(slot_loss.detach().cpu().item()) * int(
                    actions.numel()
                )
    return _behavior_cloning_epoch_summary(
        total_loss=total_loss,
        total_action_loss=total_action_loss,
        total_dynamics_loss=total_dynamics_loss,
        total_primitive_loss=total_primitive_loss,
        total_primitive_outcome_loss=total_primitive_outcome_loss,
        total_slot_losses=total_slot_losses,
        total_semantic_accuracy=total_semantic_accuracy,
        total_semantic_gate=total_semantic_gate,
        total_correct=total_correct,
        total_seen=total_seen,
        primitive_counts=primitive_counts,
        semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
    )


def _action_logits(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
) -> torch.Tensor:
    logits, _next_state_pred, _motor_primitives = _action_logits_and_prediction(
        model,
        src_a,
        src_b,
        src_c,
    )
    return logits


def _action_logits_and_prediction(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    _actions1, next_state_pred, _criticism, _actions2, logits_a, _w_b, _b_b = model(
        src_a,
        src_b,
        src_c,
        tau=1.0,
    )
    return (
        logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT],
        next_state_pred,
        getattr(model, "last_motor_primitives", None),
    )


def _action_logits_with_state(
    model: torch.nn.Module,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    src_c: torch.Tensor,
    *,
    world_model_state: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
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
    return (
        logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT],
        next_state_pred,
        next_world_model_state,
        getattr(model, "last_motor_primitives", None),
    )


def _block_smb_distillation_primitive_loss(
    motor_primitives: Any,
    examples: list[BlockSMBDistillationExample],
    *,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    if motor_primitives is None or not examples:
        return zero
    sample_weights = torch.tensor(
        [float(example.primitive_weight) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    losses: list[torch.Tensor] = []

    combo_logits = getattr(motor_primitives, "button_combo_logits", None)
    combo_mask = torch.tensor(
        [float(example.primitive_button_combo_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if combo_logits is not None and combo_logits.ndim == 3 and bool((combo_mask > 0).any().item()):
        combo_targets = torch.tensor(
            [int(example.primitive_button_combo) for example in examples],
            dtype=torch.long,
            device=device,
        )
        per_sample = F.cross_entropy(
            combo_logits[:, -1, :BLOCK_SMB_ACTION_COUNT],
            combo_targets,
            reduction="none",
        )
        weights = combo_mask * sample_weights
        losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))

    hold_duration_logits = getattr(motor_primitives, "hold_duration_logits", None)
    duration_mask = torch.tensor(
        [float(example.primitive_duration_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if (
        hold_duration_logits is not None
        and hold_duration_logits.ndim == 3
        and bool((duration_mask > 0).any().item())
    ):
        duration_targets = torch.tensor(
            [int(example.primitive_duration_bin) for example in examples],
            dtype=torch.long,
            device=device,
        )
        per_sample = F.cross_entropy(
            hold_duration_logits[:, -1, :],
            duration_targets,
            reduction="none",
        )
        weights = duration_mask * sample_weights
        losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))

    release_logit = getattr(motor_primitives, "release_logit", None)
    release_mask = torch.tensor(
        [float(example.primitive_release_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if (
        release_logit is not None
        and release_logit.ndim == 2
        and bool((release_mask > 0).any().item())
    ):
        release_targets = torch.tensor(
            [float(example.primitive_release) for example in examples],
            dtype=torch.float32,
            device=device,
        )
        per_sample = F.binary_cross_entropy_with_logits(
            release_logit[:, -1],
            release_targets,
            reduction="none",
        )
        weights = release_mask * sample_weights
        losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))

    post_release_logits = getattr(motor_primitives, "post_release_logits", None)
    post_release_mask = torch.tensor(
        [float(example.primitive_post_release_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if (
        post_release_logits is not None
        and post_release_logits.ndim == 3
        and bool((post_release_mask > 0).any().item())
    ):
        post_release_targets = torch.tensor(
            [int(example.primitive_post_release) for example in examples],
            dtype=torch.long,
            device=device,
        )
        per_sample = F.cross_entropy(
            post_release_logits[:, -1, :BLOCK_SMB_ACTION_COUNT],
            post_release_targets,
            reduction="none",
        )
        weights = post_release_mask * sample_weights
        losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))

    cancel_logit = getattr(motor_primitives, "cancel_logit", None)
    cancel_mask = torch.tensor(
        [float(example.primitive_cancel_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if cancel_logit is not None and cancel_logit.ndim == 2 and bool((cancel_mask > 0).any().item()):
        cancel_targets = torch.tensor(
            [float(example.primitive_cancel) for example in examples],
            dtype=torch.float32,
            device=device,
        )
        per_sample = F.binary_cross_entropy_with_logits(
            cancel_logit[:, -1],
            cancel_targets,
            reduction="none",
        )
        weights = cancel_mask * sample_weights
        losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))

    replan_signal = getattr(motor_primitives, "interrupt_logit", None)
    replan_uses_logits = replan_signal is not None
    if replan_signal is None:
        replan_signal = getattr(motor_primitives, "replan_probability", None)
    replan_mask = torch.tensor(
        [float(example.primitive_replan_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if (
        replan_signal is not None
        and replan_signal.ndim == 2
        and bool((replan_mask > 0).any().item())
    ):
        replan_targets = torch.tensor(
            [float(example.primitive_replan) for example in examples],
            dtype=torch.float32,
            device=device,
        )
        if replan_uses_logits:
            per_sample = F.binary_cross_entropy_with_logits(
                replan_signal[:, -1],
                replan_targets,
                reduction="none",
            )
        else:
            probabilities = replan_signal[:, -1].clamp(1e-6, 1.0 - 1e-6)
            per_sample = F.binary_cross_entropy(
                probabilities,
                replan_targets,
                reduction="none",
            )
        weights = replan_mask * sample_weights
        losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1.0))

    if not losses:
        return zero
    return torch.stack(losses).mean()


def _block_smb_distillation_primitive_outcome_loss(
    primitive_outcome: PrimitiveOutcomePrediction | None,
    examples: list[BlockSMBDistillationExample],
    *,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    if primitive_outcome is None or not examples:
        return zero
    mask = torch.tensor(
        [float(example.primitive_outcome_mask) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    if not bool((mask > 0).any().item()):
        return zero
    progress_targets = torch.tensor(
        [float(example.primitive_outcome_progress_delta) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    support_targets = torch.tensor(
        [float(example.primitive_outcome_support_loss) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    collision_death_targets = torch.tensor(
        [float(example.primitive_outcome_collision_death_risk) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    terminal_targets = torch.tensor(
        [float(example.primitive_outcome_terminal) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    continue_targets = torch.tensor(
        [float(example.primitive_outcome_continue) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    cancel_targets = torch.tensor(
        [float(example.primitive_outcome_cancel) for example in examples],
        dtype=torch.float32,
        device=device,
    )
    replan_targets = torch.tensor(
        [float(example.primitive_outcome_replan) for example in examples],
        dtype=torch.float32,
        device=device,
    )

    def masked_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    losses = [
        masked_mean(
            F.mse_loss(primitive_outcome.progress_delta, progress_targets, reduction="none")
        ),
        masked_mean(
            F.binary_cross_entropy_with_logits(
                primitive_outcome.support_loss_logit,
                support_targets,
                reduction="none",
            )
        ),
        masked_mean(
            F.binary_cross_entropy_with_logits(
                primitive_outcome.collision_death_logit,
                collision_death_targets,
                reduction="none",
            )
        ),
        masked_mean(
            F.binary_cross_entropy_with_logits(
                primitive_outcome.terminal_logit,
                terminal_targets,
                reduction="none",
            )
        ),
        masked_mean(
            F.binary_cross_entropy_with_logits(
                primitive_outcome.continue_logit,
                continue_targets,
                reduction="none",
            )
        ),
        masked_mean(
            F.binary_cross_entropy_with_logits(
                primitive_outcome.cancel_logit,
                cancel_targets,
                reduction="none",
            )
        ),
        masked_mean(
            F.binary_cross_entropy_with_logits(
                primitive_outcome.replan_logit,
                replan_targets,
                reduction="none",
            )
        ),
    ]
    return torch.stack(losses).mean()


def _empty_primitive_supervision_counts() -> dict[str, float]:
    return {
        "primitive_button_combo_supervision_count": 0.0,
        "primitive_duration_supervision_count": 0.0,
        "primitive_release_supervision_count": 0.0,
        "primitive_release_positive_count": 0.0,
        "primitive_post_release_supervision_count": 0.0,
        "primitive_cancel_supervision_count": 0.0,
        "primitive_cancel_positive_count": 0.0,
        "primitive_replan_supervision_count": 0.0,
        "primitive_replan_positive_count": 0.0,
        "primitive_hazard_window_supervision_count": 0.0,
        "primitive_hazard_window_positive_count": 0.0,
        "primitive_weighted_supervision_count": 0.0,
        "primitive_outcome_supervision_count": 0.0,
        "primitive_outcome_continue_positive_count": 0.0,
        "primitive_outcome_cancel_positive_count": 0.0,
        "primitive_outcome_replan_positive_count": 0.0,
        "primitive_outcome_collision_death_positive_count": 0.0,
        "primitive_outcome_terminal_positive_count": 0.0,
    }


def _add_primitive_supervision_counts(
    counts: dict[str, float],
    examples: list[BlockSMBDistillationExample],
) -> None:
    for example in examples:
        combo_mask = float(example.primitive_button_combo_mask)
        duration_mask = float(example.primitive_duration_mask)
        release_mask = float(example.primitive_release_mask)
        post_release_mask = float(example.primitive_post_release_mask)
        cancel_mask = float(example.primitive_cancel_mask)
        replan_mask = float(example.primitive_replan_mask)
        hazard_window_mask = float(example.primitive_hazard_window_mask)
        outcome_mask = float(example.primitive_outcome_mask)
        any_timing_mask = max(
            duration_mask,
            release_mask,
            post_release_mask,
            cancel_mask,
            hazard_window_mask,
            replan_mask * float(example.primitive_replan),
        )
        counts["primitive_button_combo_supervision_count"] += combo_mask
        counts["primitive_duration_supervision_count"] += duration_mask
        counts["primitive_release_supervision_count"] += release_mask
        counts["primitive_release_positive_count"] += release_mask * float(
            example.primitive_release
        )
        counts["primitive_post_release_supervision_count"] += post_release_mask
        counts["primitive_cancel_supervision_count"] += cancel_mask
        counts["primitive_cancel_positive_count"] += cancel_mask * float(example.primitive_cancel)
        counts["primitive_replan_supervision_count"] += replan_mask
        counts["primitive_replan_positive_count"] += replan_mask * float(example.primitive_replan)
        counts["primitive_hazard_window_supervision_count"] += hazard_window_mask
        counts["primitive_hazard_window_positive_count"] += hazard_window_mask * float(
            example.primitive_hazard_window
        )
        counts["primitive_weighted_supervision_count"] += any_timing_mask * float(
            example.primitive_weight
        )
        counts["primitive_outcome_supervision_count"] += outcome_mask
        counts["primitive_outcome_continue_positive_count"] += outcome_mask * float(
            example.primitive_outcome_continue
        )
        counts["primitive_outcome_cancel_positive_count"] += outcome_mask * float(
            example.primitive_outcome_cancel
        )
        counts["primitive_outcome_replan_positive_count"] += outcome_mask * float(
            example.primitive_outcome_replan
        )
        counts["primitive_outcome_collision_death_positive_count"] += outcome_mask * float(
            example.primitive_outcome_collision_death_risk
        )
        counts["primitive_outcome_terminal_positive_count"] += outcome_mask * float(
            example.primitive_outcome_terminal
        )


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
        name: torch.stack(values).mean() for name, values in metrics_by_name.items() if values
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
    total_primitive_loss: float,
    total_primitive_outcome_loss: float,
    total_slot_losses: Mapping[str, float],
    total_semantic_accuracy: float,
    total_semantic_gate: float,
    total_correct: int,
    total_seen: int,
    primitive_counts: Mapping[str, float],
    semantic_accuracy_threshold: float,
) -> dict[str, float]:
    denom = max(total_seen, 1)
    summary = {
        "loss": total_loss / denom,
        "loss_action": total_action_loss / denom,
        "loss_dynamics": total_dynamics_loss / denom,
        "loss_primitive": total_primitive_loss / denom,
        "loss_primitive_outcome": total_primitive_outcome_loss / denom,
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
    summary.update({key: float(value) for key, value in primitive_counts.items()})
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
    weights[nonzero] = counts[nonzero].sum() / (float(nonzero.sum().item()) * counts[nonzero])
    if BLOCK_SMB_ACTION_COUNT > 2:
        weights[2] *= float(jump_weight_multiplier)
    weights[~nonzero] = 0.0
    return weights.to(device)


def _training_config_from_distillation(
    config: BlockSMBDistillationConfig,
) -> BlockSMBTrainingConfig:
    explicit_monte_carlo_count = (
        len(BLOCK_SMB_MC_FAMILIES)
        * len(BLOCK_SMB_MC_DIFFICULTY_BINS)
        * int(config.monte_carlo_sweep_repeats_per_difficulty)
        if config.monte_carlo_parameter_sweep
        else int(config.monte_carlo_samples)
    )
    required_monte_carlo_count = (
        0
        if config.monte_carlo_parameter_sweep
        else len(config.required_monte_carlo_families)
        * len(BLOCK_SMB_MC_DIFFICULTY_BINS)
        * int(config.required_monte_carlo_repeats_per_difficulty)
    )
    monte_carlo_count = required_monte_carlo_count + explicit_monte_carlo_count
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
        monte_carlo_sweep_repeats_per_difficulty=(config.monte_carlo_sweep_repeats_per_difficulty),
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
    primitive_counts = _empty_primitive_supervision_counts()
    for example in dataset:
        counts[example.scenario_name] = counts.get(example.scenario_name, 0) + 1
        actions[example.action] += 1
        max_step = max(max_step, example.step_index)
        _add_primitive_supervision_counts(primitive_counts, [example])
    return {
        "examples": len(dataset),
        "scenario_examples": counts,
        "action_counts": {str(index): int(count.item()) for index, count in enumerate(actions)},
        "max_step_index": max_step,
        **primitive_counts,
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


def _family_arg(value: str) -> str:
    family = str(value)
    if family not in BLOCK_SMB_MC_FAMILIES:
        choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
        raise argparse.ArgumentTypeError(f"unknown family {family!r}; expected one of: {choices}")
    return family


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
        "--primitive-loss-weight",
        type=float,
        default=BlockSMBDistillationConfig.primitive_loss_weight,
        help=(
            "weight for scripted Level-B button-combo, duration, release, "
            "post-release, cancel, replan, and hazard-window supervision"
        ),
    )
    parser.add_argument(
        "--primitive-hazard-weight-multiplier",
        type=float,
        default=BlockSMBDistillationConfig.primitive_hazard_weight_multiplier,
        help="extra primitive-supervision weight for enemy, pipe, pit, gap, and chain windows",
    )
    parser.add_argument(
        "--primitive-outcome-loss-weight",
        type=float,
        default=BlockSMBDistillationConfig.primitive_outcome_loss_weight,
        help=(
            "weight for k-step B-primitive outcome supervision: progress, support loss, "
            "collision/death risk, terminal outcome, continue, cancel, and replan"
        ),
    )
    parser.add_argument(
        "--primitive-outcome-horizon",
        type=int,
        default=BlockSMBDistillationConfig.primitive_outcome_horizon,
        help="number of future Block SMB steps used for B-primitive outcome targets",
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
    parser.set_defaults(required_monte_carlo_families=None)
    parser.add_argument(
        "--required-monte-carlo-family",
        action="append",
        default=None,
        type=_family_arg,
        dest="required_monte_carlo_families",
        metavar="FAMILY",
        help=(
            "required MC family for scripted warm start; defaults to the full "
            "MC family set, including chained and Full-SMB-opening proxy families; "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--required-monte-carlo-repeats-per-difficulty",
        type=int,
        default=BlockSMBDistillationConfig.required_monte_carlo_repeats_per_difficulty,
    )
    parser.set_defaults(include_required_monte_carlo_families=True)
    parser.add_argument(
        "--no-required-monte-carlo-families",
        action="store_false",
        dest="include_required_monte_carlo_families",
        help="disable default chained/proxy MC scripted warm-start coverage",
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
        primitive_loss_weight=args.primitive_loss_weight,
        primitive_hazard_weight_multiplier=args.primitive_hazard_weight_multiplier,
        primitive_outcome_loss_weight=args.primitive_outcome_loss_weight,
        primitive_outcome_horizon=args.primitive_outcome_horizon,
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
        monte_carlo_sweep_repeats_per_difficulty=(args.monte_carlo_sweep_repeats_per_difficulty),
        required_monte_carlo_families=(
            tuple(args.required_monte_carlo_families)
            if args.required_monte_carlo_families is not None
            else (
                DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES
                if args.include_required_monte_carlo_families
                else ()
            )
        ),
        required_monte_carlo_repeats_per_difficulty=(
            args.required_monte_carlo_repeats_per_difficulty
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
