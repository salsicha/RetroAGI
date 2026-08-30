"""Training utilities for the Block SMB stage."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import (
    ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    BASELINE_ARCHITECTURE_NAME,
    DEFAULT_EVALUATION_SEED_COUNT,
    POLICY_TUPLE_OUTPUT_CONTRACTS,
    SUPPORTED_CONTROLLER_SCHEDULES,
    TRACKING_BACKENDS,
    ExperimentTrackerConfig,
    SMBAction,
    SMBParameterizedPrimitiveExecutor,
    SMBPrimitiveExecution,
    StageBatch,
    VisionEncoder,
    WorldModelState,
    action_distribution_stats,
    action_level_world_model_state_dict,
    build_architecture,
    build_checkpoint,
    evaluate_over_seeds,
    get_architecture,
    is_smb_jump_action,
    load_checkpoint,
    make_experiment_tracker,
    save_checkpoint,
    select_device,
    smb_jump_release_action,
    to_plain_data,
)

from .adapter import (
    BLOCK_SMB_SPEC,
    SCENARIOS_DIR,
    BlockSMBStage,
    block_smb_deterministic_critic_slots,
)
from .env import BlockSMBRewardConfig, MarioScenarioEnv
from .temporal_spans import build_block_smb_temporal_spans
from .skills import requested_block_smb_skill_goal
from .monte_carlo import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    block_smb_monte_carlo_metadata,
    block_smb_monte_carlo_oracle_actions,
    evaluate_block_smb_monte_carlo_gates,
    sample_block_smb_monte_carlo_parameter_sweep,
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
    summarize_block_smb_monte_carlo_action_counts,
    summarize_block_smb_monte_carlo_samples,
)
from .success import evaluate_fixed_success_thresholds, summarize_fixed_success_metrics
from .vision import BlockVisionTransformer

BLOCK_SMB_MODEL_NAME = "block_smb_actor_world_model_critic"
BLOCK_SMB_CHECKPOINT_KIND = "block_smb_trainer"
BLOCK_SMB_ACTION_COUNT = 6
TARGET_NETWORK_MODES = ("off", "on", "auto")
BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_NAMES = (
    "position",
    "semantic_probabilities",
    "support_state",
    "state",
    "terminal_outcome",
    "patch_tokens",
)
_BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_ALIASES = {
    "position": "position",
    "pos": "position",
    "semantic": "semantic_probabilities",
    "semantics": "semantic_probabilities",
    "semantic_probabilities": "semantic_probabilities",
    "semantic-probabilities": "semantic_probabilities",
    "support": "support_state",
    "support_state": "support_state",
    "support-state": "support_state",
    "grounded": "support_state",
    "ground_state": "support_state",
    "ground-state": "support_state",
    "state": "state",
    "symbolic": "state",
    "symbolic_state": "state",
    "symbolic-state": "state",
    "terminal": "terminal_outcome",
    "terminal_outcome": "terminal_outcome",
    "terminal-outcome": "terminal_outcome",
    "outcome": "terminal_outcome",
    "death": "terminal_outcome",
    "tokens": "patch_tokens",
    "patch": "patch_tokens",
    "patch_tokens": "patch_tokens",
    "patch-tokens": "patch_tokens",
}
DEFAULT_BLOCK_SMB_SEMANTIC_PREDICTION_ACCURACY_THRESHOLD = 0.8
DEFAULT_BLOCK_SMB_MC_TRAIN_SAMPLES = 512
DEFAULT_BLOCK_SMB_MC_VALIDATION_SAMPLES = 128
DEFAULT_BLOCK_SMB_MC_TEST_SAMPLES = 256
# Real-volume training defaults for fresh (non-smoke, non-sweep) CLI runs.
# The dataclass defaults below stay tiny so unit tests and directly-constructed
# configs remain fast; these are injected by the CLI's real-volume path only.
# They raise the training budget ~3 orders of magnitude over the previous
# real-volume shape (epochs 1, rollout 32 -> 200 x (160/32) = ~1000x the env
# frames / gradient signal). rollout_steps 32 was also too short to reach most
# scenario goals (Mario travels ~3px/step, goals sit at x>=230), so 160 is both
# a volume increase and a correctness fix. Evaluation is spaced out so the
# held-out gate sweeps do not dominate the longer run.
# 200 -> 100 -> 60 -> 50 -> 70: fifty proved too short — the composite
# families historically break through at rounds 40-70, and the 50-round
# waitfix run produced the weakest frontier of the series (the chained
# families never banked a single training success). Seventy covers the
# full climb window plus retention observation.
DEFAULT_BLOCK_SMB_REAL_VOLUME_EPOCHS = 70
DEFAULT_BLOCK_SMB_REAL_VOLUME_ROLLOUT_STEPS = 160
DEFAULT_BLOCK_SMB_REAL_VOLUME_EVALUATION_INTERVAL_EPOCHS = 25
DEFAULT_BLOCK_SMB_MC_PASS_RATE_GATE = 0.95
DEFAULT_BLOCK_SMB_MC_FAMILY_PASS_RATE_GATE = 0.90
DEFAULT_BLOCK_SMB_FAILURE_FOCUS_MC_FAMILIES = (
    "single_gap",
    "stair_climb",
    "platform_chain",
    "enemy_gap",
    "retreat_recovery",
    "wait_timing",
    "mixed_section",
    "full_smb_opening_proxy",
    "tall_pipe_jump",
    "pipe_mount",
    "pit_leap",
    "stomp_mount",
    "platform_hop",
)
DEFAULT_BLOCK_SMB_FAILURE_FOCUS_MC_FAMILY_WEIGHT_ITEMS = (
    ("single_gap", 1.0),
    ("stair_climb", 1.0),
    ("platform_chain", 1.0),
    ("enemy_gap", 1.0),
    ("retreat_recovery", 1.0),
    ("wait_timing", 1.0),
    ("mixed_section", 1.0),
    ("full_smb_opening_proxy", 4.0),
    ("tall_pipe_jump", 2.0),
    ("pipe_mount", 2.0),
    ("pit_leap", 2.0),
    ("stomp_mount", 2.0),
    ("platform_hop", 2.0),
)
DEFAULT_BLOCK_SMB_MC_FAILURE_REPLAY_SAMPLES = 64
ROUTINE_BLOCK_SMB_MC_REQUIRED_TRAIN_FAMILIES = (
    "chained_obstacles",
    "chained_enemy_gauntlet",
    "full_smb_opening_proxy",
)
BLOCK_SMB_NOOP_ACTION = 0
FIXED_BLOCK_SMB_NOOP_WINDOWS = {
    "level_12_wait_bridge.json": ((0, 20),),
    "level_15_wait_long_bridge.json": ((0, 28),),
    "level_16_wait_enemy_gate.json": ((0, 50),),
}


def normalize_block_smb_world_model_slot_weights(
    weights: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Normalize user-facing C-stream slot weight aliases."""

    normalized: dict[str, float] = {}
    for raw_name, raw_weight in dict(weights or {}).items():
        slot_name = _BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_ALIASES.get(str(raw_name).strip().lower())
        if slot_name is None:
            choices = ", ".join(BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_NAMES)
            raise ValueError(f"unknown Block SMB world-model slot {raw_name!r}; expected {choices}")
        slot_weight = float(raw_weight)
        if not np.isfinite(slot_weight) or slot_weight <= 0.0:
            raise ValueError("world_model_slot_weights must contain finite positive values")
        normalized[slot_name] = slot_weight
    return normalized


def default_block_smb_failure_focus_monte_carlo_family_weights() -> dict[str, float]:
    """Return train sampling weights for recently failing Block SMB MC families."""

    return dict(DEFAULT_BLOCK_SMB_FAILURE_FOCUS_MC_FAMILY_WEIGHT_ITEMS)


@dataclass(frozen=True)
class BlockSMBAblationConfig:
    """Switches for measuring Block SMB architectural contributions."""

    vision_enabled: bool = True
    world_model_enabled: bool = True
    critic_feedback_enabled: bool = True
    hierarchy_enabled: bool = True
    recurrent_state_enabled: bool = True
    checkpoint_transfer_enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vision_enabled",
            "world_model_enabled",
            "critic_feedback_enabled",
            "hierarchy_enabled",
            "recurrent_state_enabled",
            "checkpoint_transfer_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True)
class BlockSMBTrainingConfig:
    seed: int = 0
    architecture_name: str = BASELINE_ARCHITECTURE_NAME
    architecture_config: Mapping[str, Any] = field(default_factory=dict)
    epochs: int = 1
    episodes_per_epoch: int = 2
    rollout_steps: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.95
    reward_config: BlockSMBRewardConfig = field(default_factory=BlockSMBRewardConfig)
    ablation: BlockSMBAblationConfig = field(default_factory=BlockSMBAblationConfig)
    entropy_weight: float = 0.01
    policy_loss_weight: float = 1.0
    representation_weight: float = 0.05
    world_model_weight: float = 0.1
    world_model_slot_weights: Mapping[str, float] = field(default_factory=dict)
    reward_loss_weight: float = 0.01
    value_loss_weight: float = 0.25
    action_aux_weight: float = 0.01
    oracle_action_loss_weight: float = 1.0
    # 0.0, retired from 0.25: the anti-freeze guard predates committed
    # primitives and the energy regulator, which now prevent NOOP-collapse
    # by construction — while the guard actively fought the wait familes'
    # one required behavior. Opt back in per run via --noop-loss-weight.
    noop_loss_weight: float = 0.0
    critic_loss_weight: float = 0.001
    imagined_rollout_weight: float = 0.0
    imagined_rollout_horizon: int = 0
    target_network_mode: str = "off"
    target_network_tau: float = 0.01
    target_network_instability_threshold: float = 1.0
    gradient_clip_norm: float = 1.0
    # 32 -> 64 -> 128 across the full-volume series: each doubling widened
    # the set of skills retained simultaneously (32: constant churn; 64:
    # core of 8 held perfectly while the narrow-timing frontier still
    # collapsed under interference). 128 widens every family's basin; the
    # environment, not the network, dominates wall-clock per round.
    hidden_dim: int = 128
    controller_schedule: str = "constant"
    device: str = "auto"
    deterministic: bool = True
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
    generated_scenarios: int = 0
    generated_seed: int = 50_000
    monte_carlo_distribution_id: str = DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID
    monte_carlo_train_samples_per_epoch: int = 0
    monte_carlo_seed: int = 50_000
    monte_carlo_family_weights: Mapping[str, float] = field(default_factory=dict)
    monte_carlo_parameter_sweep: bool = False
    monte_carlo_sweep_repeats_per_difficulty: int = 1
    monte_carlo_validate_reachability: bool = True
    monte_carlo_max_rejections: int = 32
    monte_carlo_validation_samples: int = 0
    monte_carlo_test_samples: int = 0
    monte_carlo_failure_replay_samples_per_epoch: int = 0
    monte_carlo_pass_rate_gate: float = DEFAULT_BLOCK_SMB_MC_PASS_RATE_GATE
    monte_carlo_family_pass_rate_gate: float = DEFAULT_BLOCK_SMB_MC_FAMILY_PASS_RATE_GATE
    # Mastery-gated schedule: focus MC train sampling on families that have not
    # yet cleared the family pass-rate gate, keep a small retention share for
    # mastered families, and unlock difficulties per family (easy -> medium ->
    # hard) as each bin clears the gate. Every family always has nonzero weight.
    mastery_gated_schedule: bool = False
    mastery_retention_weight: float = 0.25
    # Graduated retention: a newly-mastered family keeps elevated practice
    # for this many evaluations, ramping linearly from full weight down to
    # mastery_retention_weight, instead of dropping to the floor instantly.
    mastery_retention_grace_evals: int = 3
    # Scenario rehearsal: layouts the policy has solved are stored per family
    # and a balanced sample is re-rolled live each epoch through the normal
    # on-policy losses. Fresh practice on known-solvable layouts cannot go
    # stale (the frozen-record replay it replaces provably did), and the
    # rehearsal success rate is a per-epoch retention gauge.
    success_replay_episodes_per_family: int = 8
    # 12, up from 4: the diagnostic dose measured a flat ~50% retention rate
    # without moving it; a third of practice as retention work is the
    # therapeutic-dose experiment.
    success_replay_rehearsals_per_epoch: int = 12
    # Execute scripted oracle actions during training rollouts on Monte Carlo
    # scenarios that carry them (fixed scenarios have no oracle and stay
    # on-policy). This is the in-loop demonstration channel that supervises the
    # action and primitive heads; evaluation rollouts never use oracle actions.
    use_oracle_actions: bool = False
    # Ranked-candidate critic search: sort the A-level next-action logits and
    # evaluate candidates from most to least likely through the LSTM world
    # model and critic, executing the first one predicted to progress without
    # death. Walking into an obstacle predicts no motion, which the critic
    # treats as no progress, so blocked actions are rejected and the next
    # most likely token is tried.
    ranked_candidate_search: bool = True
    # Deterministic critic gates: would_progress is the mechanistic decrease of
    # the predicted normalized goal distance and predicts_death reads the LSTM
    # world model's predicted death flag directly, bypassing the learned
    # progress/death MLP heads for gating.
    deterministic_critic_gates: bool = True
    # Adaptive in-flight duration control: while a jump is active the executor
    # tracks B-level's current duration head as a slew-limited setpoint,
    # shifting the hold by the change in B's expected duration since
    # initiation. This lets B re-parameterize a jump mid-air to intercept
    # moving targets (enemy stomps, moving platforms) without reintroducing
    # noise-driven truncation of committed holds.
    adaptive_duration_control: bool = True
    # Per-frame primitive-outcome loss: each completed horizontal jump is
    # hindsight-relabeled with the hold that would have hit the goal (which
    # rides the enemy under goal_on_stomp), scaling the realized hold by
    # target-distance / realized-displacement, and every frame's expected
    # hold regresses quadratically toward it. This is the dense
    # geometry-conditioned gradient the duration head does not get from
    # episode-level REINFORCE, and it is bounded and self-terminating —
    # unlike a raw signed-error push, whose asymmetric magnitudes collapse
    # the head to the shortest bin.
    primitive_outcome_weight: float = 0.5
    # HSP0: write every rollout's temporal spans (one JSONL next to the train
    # log) so episodes can be reconstructed as goals and spans after the run.
    emit_temporal_spans: bool = True
    # HSP1: span-supervised release-timing prediction (BCE toward "the
    # hindsight-correct hold is over"); the executor still ignores the
    # release head at runtime.
    release_timing_weight: float = 0.1
    # HSP2: condition B-level on the scenario family's requested skill goal
    # and train the skill outcome head from hindsight-relabeled spans.
    skill_goal_conditioning: bool = True
    # Universal duration primitives: walking and waiting run through the
    # executor as committed multi-frame actions with adaptive setpoints.
    steady_duration_primitives: bool = True
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 200
    cover_curriculum_per_epoch: bool = True
    update_batch_episodes: int = 16
    action_gate_min_distinct_actions: int = 2
    action_gate_max_dominant_fraction: float = 0.95
    action_gate_required_actions: tuple[int, ...] = (1, 2)
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    # Weights-only warm start: load model weights from a Block SMB checkpoint
    # (for example a B-level jump-suite policy) while keeping a fresh
    # optimizer, epoch counter, and curriculum. Mutually exclusive with resume.
    init_checkpoint: Optional[Path] = None
    save_checkpoints: bool = False
    video_dir: Optional[Path] = None
    record_videos: bool = False
    num_envs: int = 1
    evaluation_interval_epochs: int = 1
    log_path: Optional[Path] = None
    vision_checkpoint_path: Optional[Path] = None
    tracking_backend: str = "none"
    tracking_log_dir: Optional[Path] = None
    tracking_project: str = "retroagi"
    tracking_run_name: Optional[str] = None
    tracking_mode: Optional[str] = None
    semantic_prediction_accuracy_threshold: float = (
        DEFAULT_BLOCK_SMB_SEMANTIC_PREDICTION_ACCURACY_THRESHOLD
    )

    def __post_init__(self) -> None:
        if not self.architecture_name:
            raise ValueError("architecture_name must be non-empty")
        if any(not str(key) for key in self.architecture_config):
            raise ValueError("architecture_config keys must be non-empty")
        resolved_architecture_config = dict(self.architecture_config)
        resolved_architecture_config.setdefault("hidden_dim", self.hidden_dim)
        resolved_architecture_config.setdefault("controller_schedule", self.controller_schedule)
        object.__setattr__(
            self,
            "architecture_config",
            resolved_architecture_config,
        )
        object.__setattr__(self, "hidden_dim", int(resolved_architecture_config["hidden_dim"]))
        object.__setattr__(
            self,
            "controller_schedule",
            str(resolved_architecture_config["controller_schedule"]),
        )
        if isinstance(self.reward_config, Mapping):
            object.__setattr__(self, "reward_config", BlockSMBRewardConfig(**self.reward_config))
        elif not isinstance(self.reward_config, BlockSMBRewardConfig):
            raise TypeError("reward_config must be a BlockSMBRewardConfig or mapping")
        if isinstance(self.ablation, Mapping):
            object.__setattr__(self, "ablation", BlockSMBAblationConfig(**self.ablation))
        elif not isinstance(self.ablation, BlockSMBAblationConfig):
            raise TypeError("ablation must be a BlockSMBAblationConfig or mapping")
        object.__setattr__(
            self,
            "world_model_slot_weights",
            normalize_block_smb_world_model_slot_weights(self.world_model_slot_weights),
        )
        positive_ints = (
            "epochs",
            "episodes_per_epoch",
            "rollout_steps",
            "hidden_dim",
            "evaluation_episodes",
            "evaluation_max_steps",
            "num_envs",
            "evaluation_interval_epochs",
            "update_batch_episodes",
            "action_gate_min_distinct_actions",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gamma", "gradient_clip_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.generated_scenarios < 0:
            raise ValueError("generated_scenarios must be non-negative")
        if self.monte_carlo_train_samples_per_epoch < 0:
            raise ValueError("monte_carlo_train_samples_per_epoch must be non-negative")
        if not isinstance(self.monte_carlo_parameter_sweep, bool):
            raise TypeError("monte_carlo_parameter_sweep must be a bool")
        if not isinstance(self.cover_curriculum_per_epoch, bool):
            raise TypeError("cover_curriculum_per_epoch must be a bool")
        if self.monte_carlo_sweep_repeats_per_difficulty <= 0:
            raise ValueError("monte_carlo_sweep_repeats_per_difficulty must be positive")
        if self.monte_carlo_max_rejections < 0:
            raise ValueError("monte_carlo_max_rejections must be non-negative")
        if self.monte_carlo_validation_samples < 0:
            raise ValueError("monte_carlo_validation_samples must be non-negative")
        if self.monte_carlo_test_samples < 0:
            raise ValueError("monte_carlo_test_samples must be non-negative")
        if self.monte_carlo_failure_replay_samples_per_epoch < 0:
            raise ValueError("monte_carlo_failure_replay_samples_per_epoch must be non-negative")
        if not self.monte_carlo_distribution_id:
            raise ValueError("monte_carlo_distribution_id must be non-empty")
        object.__setattr__(
            self,
            "monte_carlo_family_weights",
            normalize_block_smb_monte_carlo_family_weights(self.monte_carlo_family_weights),
        )
        if not isinstance(self.monte_carlo_validate_reachability, bool):
            raise TypeError("monte_carlo_validate_reachability must be a bool")
        if not 0.0 <= self.monte_carlo_pass_rate_gate <= 1.0:
            raise ValueError("monte_carlo_pass_rate_gate must be between 0 and 1")
        if not 0.0 <= self.monte_carlo_family_pass_rate_gate <= 1.0:
            raise ValueError("monte_carlo_family_pass_rate_gate must be between 0 and 1")
        if not isinstance(self.mastery_gated_schedule, bool):
            raise TypeError("mastery_gated_schedule must be a bool")
        if float(self.mastery_retention_weight) <= 0.0:
            raise ValueError("mastery_retention_weight must be positive")
        if not isinstance(self.use_oracle_actions, bool):
            raise TypeError("use_oracle_actions must be a bool")
        if not isinstance(self.ranked_candidate_search, bool):
            raise TypeError("ranked_candidate_search must be a bool")
        if self.resume_path is not None and self.init_checkpoint is not None:
            raise ValueError("resume_path and init_checkpoint are mutually exclusive")
        if not isinstance(self.deterministic_critic_gates, bool):
            raise TypeError("deterministic_critic_gates must be a bool")
        if not 0.0 <= self.action_gate_max_dominant_fraction <= 1.0:
            raise ValueError("action_gate_max_dominant_fraction must be between 0 and 1")
        required_actions = tuple(int(action) for action in self.action_gate_required_actions)
        if any(action < 0 or action >= BLOCK_SMB_ACTION_COUNT for action in required_actions):
            raise ValueError(
                "action_gate_required_actions must contain valid Block SMB action indices"
            )
        object.__setattr__(self, "action_gate_required_actions", required_actions)
        if self.imagined_rollout_horizon < 0:
            raise ValueError("imagined_rollout_horizon must be non-negative")
        if self.target_network_mode not in TARGET_NETWORK_MODES:
            raise ValueError(f"target_network_mode must be one of {TARGET_NETWORK_MODES}")
        if not 0 < self.target_network_tau <= 1:
            raise ValueError("target_network_tau must be in (0, 1]")
        if self.target_network_instability_threshold < 0:
            raise ValueError("target_network_instability_threshold must be non-negative")
        if not 0.0 <= self.semantic_prediction_accuracy_threshold <= 1.0:
            raise ValueError("semantic_prediction_accuracy_threshold must be between 0 and 1")
        if self.controller_schedule not in SUPPORTED_CONTROLLER_SCHEDULES:
            raise ValueError(
                "controller_schedule must be one of " f"{SUPPORTED_CONTROLLER_SCHEDULES}"
            )
        object.__setattr__(self, "tracking_backend", self.tracking_backend.lower())
        if self.tracking_backend not in TRACKING_BACKENDS:
            raise ValueError(f"tracking_backend must be one of {TRACKING_BACKENDS}")
        if not self.tracking_project:
            raise ValueError("tracking_project must be non-empty")
        loss_weights = (
            self.entropy_weight,
            self.policy_loss_weight,
            self.representation_weight,
            self.world_model_weight,
            self.reward_loss_weight,
            self.value_loss_weight,
            self.action_aux_weight,
            self.oracle_action_loss_weight,
            self.noop_loss_weight,
            self.critic_loss_weight,
            self.imagined_rollout_weight,
        )
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("loss weights must be non-negative")
        for path_name in (
            "checkpoint_path",
            "resume_path",
            "video_dir",
            "log_path",
            "vision_checkpoint_path",
            "tracking_log_dir",
        ):
            path_value = getattr(self, path_name)
            if path_value is not None and not isinstance(path_value, Path):
                object.__setattr__(self, path_name, Path(path_value))
        if self.save_checkpoints and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required when save_checkpoints is true")


@dataclass
class BlockSMBTransition:
    batch: StageBatch
    next_batch: StageBatch
    action: int
    reward: float
    done: bool
    episode_mask: float
    scenario_name: str
    info: Mapping[str, Any]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    actions1: torch.Tensor
    actions2: torch.Tensor
    next_state_pred: torch.Tensor
    criticism: torch.Tensor
    logits_a: torch.Tensor
    primitive_aux_loss: torch.Tensor | None = None
    oracle_action: int | None = None
    step_index: int = 0
    noop_allowed: bool = False
    # Normalized expected hold duration (graph-attached) emitted while a jump
    # primitive was engaged this frame; paired post-hoc with the primitive's
    # hindsight landing error (info["primitive_outcome"]) for the per-frame
    # primitive-outcome loss.
    expected_hold: torch.Tensor | None = None
    # HSP1: graph-attached release logit while the primitive is engaged,
    # span-supervised toward "the hindsight-correct hold has elapsed".
    release_logit: torch.Tensor | None = None


@dataclass
class BlockSMBTrajectory:
    scenario_name: str
    transitions: list[BlockSMBTransition] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)
    # HSP0 temporal spans covering every frame of the rollout, built by
    # build_block_smb_temporal_spans at collection time.
    spans: list[Any] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return float(sum(step.reward for step in self.transitions))

    @property
    def success(self) -> bool:
        return bool(
            self.transitions
            and self.transitions[-1].done
            and self.transitions[-1].info.get("goal_reached", False)
        )


class BlockSMBSuccessReplay:
    """Balanced per-family store of scenarios the policy has solved.

    Live practice is a zero-sum budget: boosting one family's samples takes
    from every other, which is how narrow-timing skills (bridges, stairs)
    decayed once their practice eased off. The first fix — replaying frozen
    per-step records as a supervised loss — went stale as the policy
    improved: its loss rose monotonically all run while it dragged the
    network back toward outdated behavior. This version therefore stores
    only the SCENARIO of each success (a known-solvable layout, deduplicated
    per family, FIFO-capped) and rehearsal re-rolls those scenarios live
    each epoch through the normal on-policy losses. Fresh practice cannot
    go stale, and the rehearsal success rate is a direct retention gauge.
    """

    def __init__(self, max_episodes_per_family: int = 8, seed: int = 0) -> None:
        if int(max_episodes_per_family) <= 0:
            raise ValueError("max_episodes_per_family must be positive")
        self.max_episodes_per_family = int(max_episodes_per_family)
        self._scenarios: dict[str, dict[str, dict[str, Any]]] = {}
        self._rng = random.Random(seed)

    def add(
        self,
        trajectory: "BlockSMBTrajectory",
        family: str | None,
        scenario_id: str | None = None,
        scenario: Mapping[str, Any] | None = None,
    ) -> None:
        if not family or not trajectory.success or not scenario_id or scenario is None:
            return
        bucket = self._scenarios.setdefault(family, {})
        # Re-solving a stored scenario refreshes its recency; new scenarios
        # evict the oldest entry once the family bucket is full.
        bucket.pop(scenario_id, None)
        bucket[scenario_id] = {
            "scenario_id": scenario_id,
            "scenario": copy.deepcopy(dict(scenario)),
            "family": family,
        }
        while len(bucket) > self.max_episodes_per_family:
            oldest = next(iter(bucket))
            bucket.pop(oldest)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._scenarios.values())

    def families(self) -> list[str]:
        return sorted(family for family, bucket in self._scenarios.items() if bucket)

    def sample_scenarios(self, count: int) -> list[dict[str, Any]]:
        """Balanced sample without replacement: round-robin across families."""

        families = self.families()
        if not families or count <= 0:
            return []
        order = list(families)
        self._rng.shuffle(order)
        pools = {
            family: self._rng.sample(
                list(self._scenarios[family].values()),
                len(self._scenarios[family]),
            )
            for family in order
        }
        picks: list[dict[str, Any]] = []
        while len(picks) < int(count) and any(pools.values()):
            for family in order:
                if pools[family] and len(picks) < int(count):
                    record = pools[family].pop()
                    picks.append(
                        {
                            "scenario_id": record["scenario_id"],
                            "scenario": copy.deepcopy(record["scenario"]),
                            "family": record["family"],
                        }
                    )
        return picks


class BlockSMBReplayBuffer:
    def __init__(self) -> None:
        self.trajectories: list[BlockSMBTrajectory] = []

    def add(self, trajectory: BlockSMBTrajectory) -> None:
        if trajectory.transitions:
            self.trajectories.append(trajectory)

    def clear(self) -> None:
        self.trajectories.clear()

    def transitions(self) -> list[BlockSMBTransition]:
        return [step for trajectory in self.trajectories for step in trajectory.transitions]

    def episode_masks(self) -> torch.Tensor:
        values = [step.episode_mask for step in self.transitions()]
        return torch.tensor(values, dtype=torch.float32)


class SequentialBlockSMBVectorEnv:
    """Deterministic vector-env scaffold that steps independent envs sequentially."""

    def __init__(
        self,
        scenarios: list[tuple[str, dict]],
        num_envs: int = 1,
        reward_config: BlockSMBRewardConfig = BlockSMBRewardConfig(),
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not scenarios:
            raise ValueError("scenarios must be non-empty")
        self.scenarios = scenarios
        self.envs = [MarioScenarioEnv(reward_config=reward_config) for _ in range(num_envs)]

    def reset(self, seed: int = 0) -> list[tuple[np.ndarray, Mapping[str, Any]]]:
        outputs = []
        for index, env in enumerate(self.envs):
            _name, scenario = self.scenarios[index % len(self.scenarios)]
            outputs.append(env.reset(scenario=scenario, seed=seed + index))
        return outputs

    def step(
        self, actions: list[int]
    ) -> list[tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]]:
        if len(actions) != len(self.envs):
            raise ValueError("actions length must match num_envs")
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def close(self) -> None:
        for env in self.envs:
            env.close()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def load_fixed_scenarios(names: tuple[str, ...]) -> list[tuple[str, dict]]:
    scenarios = []
    for name in names:
        path = SCENARIOS_DIR / name
        with path.open("r", encoding="utf-8") as handle:
            scenarios.append((name, json.load(handle)))
    return scenarios


def block_smb_noop_allowed_for_step(
    scenario_name: str,
    scenario: Mapping[str, Any] | None,
    step_index: int,
) -> bool:
    """Return true when a scenario's known-good plan explicitly waits now."""

    step = max(0, int(step_index))
    if isinstance(scenario, Mapping):
        metadata = block_smb_monte_carlo_metadata(scenario)
        if metadata:
            try:
                oracle_actions = block_smb_monte_carlo_oracle_actions(
                    scenario,
                    max_steps=step + 1,
                )
            except ValueError:
                oracle_actions = []
            if step < len(oracle_actions):
                return int(oracle_actions[step]) == BLOCK_SMB_NOOP_ACTION

        generic_metadata = scenario.get("metadata")
        if isinstance(generic_metadata, Mapping):
            windows = generic_metadata.get("noop_allowed_windows", ())
            for window in windows if isinstance(windows, (list, tuple)) else ():
                try:
                    start, end = int(window[0]), int(window[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if start <= step < end:
                    return True

    for start, end in FIXED_BLOCK_SMB_NOOP_WINDOWS.get(str(scenario_name), ()):
        if int(start) <= step < int(end):
            return True
    return False


def normalize_block_smb_monte_carlo_family_weights(
    weights: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Validate optional Monte Carlo family sampling weights."""

    normalized: dict[str, float] = {}
    for raw_family, raw_weight in dict(weights or {}).items():
        family = str(raw_family)
        if family not in BLOCK_SMB_MC_FAMILIES:
            choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
            raise ValueError(
                f"unknown Block SMB Monte Carlo family {raw_family!r}; expected {choices}"
            )
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("monte_carlo_family_weights must be finite non-negative values")
        if weight > 0.0:
            normalized[family] = weight
    return normalized


def block_smb_monte_carlo_train_sample_count(config: BlockSMBTrainingConfig) -> int:
    """Return the requested Monte Carlo train samples for one curriculum epoch."""

    explicit = int(config.monte_carlo_train_samples_per_epoch)
    legacy = int(config.generated_scenarios)
    if explicit <= 0:
        return legacy
    if config.monte_carlo_family_weights:
        return explicit
    return max(explicit, routine_block_smb_monte_carlo_train_min_sample_count())


def routine_block_smb_monte_carlo_train_min_sample_count(
    required_families: tuple[str, ...] = ROUTINE_BLOCK_SMB_MC_REQUIRED_TRAIN_FAMILIES,
) -> int:
    """Return the minimum default-order sample count that covers routine train families."""

    if not required_families:
        return 0
    missing = [family for family in required_families if family not in BLOCK_SMB_MC_FAMILIES]
    if missing:
        choices = ", ".join(BLOCK_SMB_MC_FAMILIES)
        raise ValueError(
            f"unknown required Block SMB Monte Carlo family {missing!r}; expected {choices}"
        )
    return max(BLOCK_SMB_MC_FAMILIES.index(family) for family in required_families) + 1


def block_smb_monte_carlo_sweep_sample_count(
    config: BlockSMBTrainingConfig,
) -> int:
    """Return the number of scenarios in one full family/difficulty sweep."""

    return (
        len(BLOCK_SMB_MC_FAMILIES)
        * len(BLOCK_SMB_MC_DIFFICULTY_BINS)
        * int(config.monte_carlo_sweep_repeats_per_difficulty)
    )


def build_monte_carlo_curriculum(
    config: BlockSMBTrainingConfig,
    *,
    split: str = "train",
    sample_count: int | None = None,
    seed: int | None = None,
    family_weights: Mapping[str, float] | None = None,
) -> list[tuple[str, dict]]:
    """Build replayable Monte Carlo scenarios for a Block SMB split."""

    resolved_count = (
        block_smb_monte_carlo_train_sample_count(config)
        if sample_count is None
        else int(sample_count)
    )
    if config.monte_carlo_parameter_sweep and family_weights is None:
        sample_set = sample_block_smb_monte_carlo_parameter_sweep(
            distribution_id=config.monte_carlo_distribution_id,
            split=split,
            seed=int(config.monte_carlo_seed if seed is None else seed),
            repeats_per_difficulty=config.monte_carlo_sweep_repeats_per_difficulty,
            validate_reachability=config.monte_carlo_validate_reachability,
            max_rejections=config.monte_carlo_max_rejections,
        )
        return sample_set.scenarios()
    if resolved_count <= 0:
        return []
    sample_set = sample_block_smb_monte_carlo_split(
        distribution_id=config.monte_carlo_distribution_id,
        split=split,
        seed=int(config.monte_carlo_seed if seed is None else seed),
        sample_count=resolved_count,
        family_weights=(
            config.monte_carlo_family_weights if family_weights is None else family_weights
        ),
        validate_reachability=config.monte_carlo_validate_reachability,
        max_rejections=config.monte_carlo_max_rejections,
    )
    return sample_set.scenarios()


def build_curriculum(config: BlockSMBTrainingConfig) -> list[tuple[str, dict]]:
    scenarios = load_fixed_scenarios(config.fixed_scenarios)
    scenarios.extend(build_monte_carlo_curriculum(config))
    return scenarios


def build_adaptive_monte_carlo_replay_curriculum(
    config: BlockSMBTrainingConfig,
    failure_bins: Mapping[str, Any],
    *,
    epoch: int,
) -> list[tuple[str, dict]]:
    """Sample train scenarios weighted by recent held-out failure families."""

    sample_count = int(config.monte_carlo_failure_replay_samples_per_epoch)
    if sample_count <= 0 or not failure_bins:
        return []
    family_weights: dict[str, float] = {}
    for bin_name, bin_result in failure_bins.items():
        family = str(bin_name).split(":", 1)[0]
        if family not in BLOCK_SMB_MC_FAMILIES:
            continue
        failure_count = 1.0
        if isinstance(bin_result, Mapping):
            try:
                failure_count = max(1.0, float(bin_result.get("failure_count", 1.0)))
            except (TypeError, ValueError):
                failure_count = 1.0
        family_weights[family] = family_weights.get(family, 0.0) + failure_count
    if not family_weights:
        return []
    return build_monte_carlo_curriculum(
        config,
        split="train",
        sample_count=sample_count,
        seed=int(config.monte_carlo_seed) + 900_000 + int(epoch),
        family_weights=family_weights,
    )


def build_epoch_curriculum(
    base_curriculum: list[tuple[str, dict]],
    replay_curriculum: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    if not replay_curriculum:
        return list(base_curriculum)
    fixed = [
        (name, scenario)
        for name, scenario in base_curriculum
        if not block_smb_monte_carlo_metadata(scenario)
    ]
    monte_carlo = [
        (name, scenario)
        for name, scenario in base_curriculum
        if block_smb_monte_carlo_metadata(scenario)
    ]
    return [*fixed, *replay_curriculum, *monte_carlo]


def initial_block_smb_mastery_state() -> dict[str, dict[str, Any]]:
    """Return the starting mastery record for every Monte Carlo family."""

    return {
        family: {
            "pass_rate": 0.0,
            "bin_pass_rates": {},
            "unlocked_difficulties": ["easy"],
            "mastered": False,
            "mastered_evals": 0,
        }
        for family in BLOCK_SMB_MC_FAMILIES
    }


def update_block_smb_mastery_state(
    state: Mapping[str, Mapping[str, Any]],
    monte_carlo_validation: Mapping[str, Any],
    *,
    family_pass_rate_gate: float,
) -> dict[str, dict[str, Any]]:
    """Fold one held-out evaluation into the per-family mastery record.

    Difficulty unlocks are monotonic: once a bin clears the gate the next
    difficulty stays unlocked even if a later evaluation regresses, so the
    training distribution does not thrash between difficulty mixes.
    """

    families = monte_carlo_validation.get("families", {})
    bins = monte_carlo_validation.get("difficulty_bins", {})
    updated: dict[str, dict[str, Any]] = {}
    for family in BLOCK_SMB_MC_FAMILIES:
        previous = state.get(family, {})
        record = {
            "pass_rate": float(previous.get("pass_rate", 0.0)),
            "bin_pass_rates": dict(previous.get("bin_pass_rates", {})),
            "unlocked_difficulties": list(previous.get("unlocked_difficulties", ["easy"])),
            "mastered": bool(previous.get("mastered", False)),
            "mastered_evals": int(previous.get("mastered_evals", 0)),
        }
        rollup = families.get(family)
        if isinstance(rollup, Mapping) and "success_rate" in rollup:
            record["pass_rate"] = float(rollup["success_rate"])
        for difficulty in BLOCK_SMB_MC_DIFFICULTY_BINS:
            bin_rollup = bins.get(f"{family}:{difficulty}")
            if isinstance(bin_rollup, Mapping) and "success_rate" in bin_rollup:
                record["bin_pass_rates"][difficulty] = float(bin_rollup["success_rate"])
        unlocked = record["unlocked_difficulties"]
        bin_rates = record["bin_pass_rates"]
        if (
            "medium" not in unlocked
            and float(bin_rates.get("easy", 0.0)) >= family_pass_rate_gate
        ):
            unlocked.append("medium")
        if (
            "hard" not in unlocked
            and "medium" in unlocked
            and float(bin_rates.get("medium", 0.0)) >= family_pass_rate_gate
        ):
            unlocked.append("hard")
        record["mastered"] = record["pass_rate"] >= family_pass_rate_gate
        # Graduated retention: count consecutive evaluations at mastery so
        # the practice weight can ease off gradually. A regression resets
        # the count — the family returns to full focus and, once it passes
        # again, restarts the ramp instead of dropping straight to the floor.
        record["mastered_evals"] = (
            record["mastered_evals"] + 1 if record["mastered"] else 0
        )
        updated[family] = record
    return updated


def block_smb_mastery_family_weights(
    state: Mapping[str, Mapping[str, Any]],
    *,
    family_pass_rate_gate: float,
    retention_weight: float,
    retention_grace_evals: int = 3,
) -> dict[str, float]:
    """Weight every family: focus on unmastered skills, retain mastered ones.

    Unmastered families weigh ``1.0 + (gate - pass_rate)`` so the furthest-from
    -mastery skills draw the most samples; mastered families keep a small
    positive retention weight so they are rehearsed and regressions surface at
    the next evaluation. Every family always has nonzero weight, so no gated
    family can be silently excluded from training.
    """

    weights: dict[str, float] = {}
    for family in BLOCK_SMB_MC_FAMILIES:
        record = state.get(family, {})
        if bool(record.get("mastered", False)):
            # A family that just crossed the gate is barely learned; dropping
            # it straight to the retention floor starves it and it decays —
            # the frontier thrash observed across the full-volume runs. Ease
            # from full practice down to the floor over the grace period.
            grace = max(0, int(retention_grace_evals))
            count = max(1, int(record.get("mastered_evals", grace)))
            if grace <= 0 or count >= grace:
                weights[family] = float(retention_weight)
            else:
                progress = count / float(grace)
                weights[family] = 1.0 + (float(retention_weight) - 1.0) * progress
        else:
            deficit = max(0.0, family_pass_rate_gate - float(record.get("pass_rate", 0.0)))
            weights[family] = 1.0 + deficit
    return weights


def build_mastery_monte_carlo_curriculum(
    config: BlockSMBTrainingConfig,
    state: Mapping[str, Mapping[str, Any]],
    *,
    phase: int,
) -> list[tuple[str, dict]]:
    """Sample a train curriculum focused on unmastered families.

    Families are drawn from the mastery weights and each sample's difficulty is
    drawn uniformly from that family's unlocked difficulties, so a family
    masters easy bins before it sees medium or hard ones. Deterministic per
    (monte_carlo_seed, phase).
    """

    sample_count = int(config.monte_carlo_train_samples_per_epoch)
    if sample_count <= 0:
        return []
    weights = block_smb_mastery_family_weights(
        state,
        family_pass_rate_gate=config.monte_carlo_family_pass_rate_gate,
        retention_weight=config.mastery_retention_weight,
        retention_grace_evals=config.mastery_retention_grace_evals,
    )
    families = list(BLOCK_SMB_MC_FAMILIES)
    weight_values = [weights[family] for family in families]
    rng = random.Random(int(config.monte_carlo_seed) + 800_000 + int(phase))
    scenarios: list[tuple[str, dict]] = []
    for sample_index in range(sample_count):
        family = rng.choices(families, weights=weight_values, k=1)[0]
        unlocked = list(state.get(family, {}).get("unlocked_difficulties", ["easy"]))
        difficulty = rng.choice(unlocked or ["easy"])
        sample = sample_block_smb_monte_carlo_scenario(
            distribution_id=config.monte_carlo_distribution_id,
            split="train",
            seed=int(config.monte_carlo_seed) + 800_000 + int(phase),
            sample_index=sample_index,
            family=family,
            difficulty=difficulty,
            validate_reachability=config.monte_carlo_validate_reachability,
            max_rejections=config.monte_carlo_max_rejections,
        )
        scenarios.append((sample.scenario_id, copy.deepcopy(dict(sample.scenario))))
    return scenarios


def summarize_block_smb_mastery_state(
    state: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact per-family mastery summary for logs and checkpoints."""

    return {
        "mastered_families": sorted(
            family for family, record in state.items() if record.get("mastered")
        ),
        "unmastered_families": sorted(
            family for family, record in state.items() if not record.get("mastered")
        ),
        "families": {
            family: {
                "pass_rate": float(record.get("pass_rate", 0.0)),
                "unlocked_difficulties": list(record.get("unlocked_difficulties", ["easy"])),
                "mastered": bool(record.get("mastered", False)),
                "mastered_evals": int(record.get("mastered_evals", 0)),
            }
            for family, record in state.items()
        },
    }


def summarize_block_smb_curriculum(
    curriculum: list[tuple[str, dict]],
) -> dict[str, Any]:
    fixed_names: list[str] = []
    monte_carlo_scenarios: list[Mapping[str, Any]] = []
    for scenario_name, scenario in curriculum:
        metadata = block_smb_monte_carlo_metadata(scenario)
        if metadata:
            monte_carlo_scenarios.append(scenario)
        else:
            fixed_names.append(scenario_name)
    return {
        "scenario_count": len(curriculum),
        "fixed_scenario_count": len(fixed_names),
        "fixed_scenarios": fixed_names,
        "monte_carlo_sample_count": len(monte_carlo_scenarios),
        "monte_carlo": (
            summarize_block_smb_monte_carlo_samples(monte_carlo_scenarios)
            if monte_carlo_scenarios
            else {}
        ),
    }


def block_smb_architecture_metadata(config: BlockSMBTrainingConfig) -> dict[str, Any]:
    architecture = get_architecture(config.architecture_name)
    return {
        "name": architecture.name,
        "config": dict(config.architecture_config),
        "spec": architecture.metadata(),
    }


def block_smb_architecture_specs(config: BlockSMBTrainingConfig) -> dict[str, Any]:
    metadata = block_smb_architecture_metadata(config)
    return {
        "architecture": metadata["spec"],
        "architecture_config": metadata["config"],
    }


def make_block_smb_model(config: BlockSMBTrainingConfig) -> torch.nn.Module:
    architecture = get_architecture(config.architecture_name)
    if architecture.output_contract not in POLICY_TUPLE_OUTPUT_CONTRACTS:
        raise ValueError(
            "Block SMB training requires a trainer-compatible architecture output "
            f"contract in {POLICY_TUPLE_OUTPUT_CONTRACTS!r}, got "
            f"{architecture.output_contract!r}"
        )
    model = build_architecture(
        config.architecture_name,
        BLOCK_SMB_SPEC,
        dict(config.architecture_config),
    )
    if hasattr(model, "ranked_candidate_search"):
        # Attribute flip rather than a constructor/architecture-config change:
        # the search adds no parameters, so checkpoints stay compatible in both
        # directions and the setting is recorded via the training config.
        model.ranked_candidate_search = bool(config.ranked_candidate_search)
    if hasattr(model, "deterministic_critic_slots"):
        model.deterministic_critic_slots = (
            block_smb_deterministic_critic_slots()
            if config.deterministic_critic_gates
            else None
        )
    return model


def make_target_network(model: torch.nn.Module) -> torch.nn.Module:
    target_model = copy.deepcopy(model)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    return target_model


@torch.no_grad()
def update_target_network(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
    tau: float,
) -> None:
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    for target_parameter, source_parameter in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)
    for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers()):
        target_buffer.copy_(source_buffer)
    target_model.eval()


def target_network_parameter_delta(
    model: torch.nn.Module,
    target_model: Optional[torch.nn.Module],
    device: torch.device,
) -> torch.Tensor:
    if target_model is None:
        return torch.zeros((), dtype=torch.float32, device=device)
    terms = [
        F.mse_loss(parameter.detach(), target_parameter.detach().to(parameter.device))
        for parameter, target_parameter in zip(model.parameters(), target_model.parameters())
    ]
    if not terms:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack([term.to(device) for term in terms]).mean()


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains NaN or infinite values")


def check_model_gradients(model: torch.nn.Module) -> None:
    saw_gradient = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        saw_gradient = True
        finite_or_raise(f"gradient {name}", parameter.grad)
    if not saw_gradient:
        raise FloatingPointError("no gradients were produced")


def discounted_returns(
    rewards: list[float], masks: list[float], gamma: float, device: torch.device
) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward, mask in zip(reversed(rewards), reversed(masks)):
        running = reward + gamma * running * mask
        returns.append(running)
    returns.reverse()
    values = torch.tensor(returns, dtype=torch.float32, device=device)
    if values.numel() > 1:
        values = (values - values.mean()) / values.std().clamp_min(1e-6)
    return values


def _write_block_smb_spans(
    config: BlockSMBTrainingConfig, trajectory: BlockSMBTrajectory
) -> None:
    """Append the rollout's HSP0 spans to the run's spans JSONL."""

    if not config.emit_temporal_spans or config.log_path is None or not trajectory.spans:
        return
    from retroagi.core.temporal import write_transitions_jsonl

    version = f"{config.architecture_name}@seed{config.seed}"
    for span in trajectory.spans:
        span.policy_version = version
    spans_path = Path(config.log_path).with_name("spans.jsonl")
    write_transitions_jsonl(spans_path, trajectory.spans)


def _goal_reached(env: MarioScenarioEnv) -> bool:
    # The env's credited-goal event is authoritative. Under goal_on_stomp the
    # goal rect is only a tracking proxy riding the enemy, so positional
    # overlap must never count: Mario overlaps the proxy on the very frame he
    # walks into the enemy and dies, which would label deaths as successes.
    if getattr(env, "_goal_credited", False):
        return True
    if getattr(env, "_goal_on_stomp", False):
        return False
    if env.goal is None:
        return False
    import pygame

    mario_rect = pygame.Rect(env.mario["x"], env.mario["y"], env.mario["w"], env.mario["h"])
    return bool(mario_rect.colliderect(env.goal))


def _ablation_config(
    ablation: BlockSMBAblationConfig | Mapping[str, Any] | None,
) -> BlockSMBAblationConfig:
    if ablation is None:
        return BlockSMBAblationConfig()
    if isinstance(ablation, BlockSMBAblationConfig):
        return ablation
    if isinstance(ablation, Mapping):
        return BlockSMBAblationConfig(**dict(ablation))
    raise TypeError("ablation must be a BlockSMBAblationConfig, mapping, or None")


def _zero_fusion_slots(
    src_c: torch.Tensor,
    fusion: Mapping[str, Any],
    slots: tuple[str, ...],
) -> torch.Tensor:
    masked = src_c.clone()
    saw_slot = False
    for slot in slots:
        value = fusion.get(slot)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            continue
        start, end = int(value[0]), int(value[1])
        masked[:, start:end] = 0.0
        saw_slot = True
    if not saw_slot:
        return torch.zeros_like(src_c)
    return masked


def apply_block_smb_ablations(
    batch: StageBatch,
    ablation: BlockSMBAblationConfig | Mapping[str, Any] | None,
) -> StageBatch:
    """Return a batch with configured Block SMB observation pathways disabled."""

    config = _ablation_config(ablation)
    metadata = dict(batch.metadata or {})
    metadata["ablation"] = to_plain_data(config)

    src_a = batch.src_a
    src_b = batch.src_b
    src_c = batch.src_c

    if not config.vision_enabled:
        src_a = torch.zeros_like(src_a)
        src_b = torch.zeros_like(src_b)
        fusion = metadata.get("vision_fusion")
        if isinstance(fusion, Mapping):
            src_c = _zero_fusion_slots(
                src_c,
                fusion,
                (
                    "c_position",
                    "c_semantic_probabilities",
                    "c_support_state",
                    "c_patch_tokens",
                ),
            )
        else:
            src_c = torch.zeros_like(src_c)

    if not config.hierarchy_enabled:
        src_a = torch.zeros_like(src_a)
        src_b = torch.zeros_like(src_b)

    return StageBatch(
        src_a=src_a,
        target_a=batch.target_a,
        src_b=src_b,
        target_b=batch.target_b,
        src_c=src_c,
        target_c=batch.target_c,
        metadata=metadata,
    )


def block_smb_forced_action_for_rollout(scenario: Mapping[str, Any] | None) -> int | None:
    """Return the scenario's given A-level action intent, when present.

    B-level isolation families (for example ``pipe_mount``) hand the A-level
    decision to the rollout via an ``a_level_action`` Monte Carlo parameter, so
    only the B-level primitive parameters remain to be learned. Unlike oracle
    forcing, the forced action goes through the normal primitive executor, so
    the model's own duration head chooses the hold length.
    """

    metadata = block_smb_monte_carlo_metadata(scenario) if scenario is not None else {}
    parameters = metadata.get("parameters", {}) if isinstance(metadata, Mapping) else {}
    value = parameters.get("a_level_action") if isinstance(parameters, Mapping) else None
    if value is None:
        return None
    return int(value)


def block_smb_forced_action_scope(scenario: Mapping[str, Any] | None) -> str:
    """How long the scenario's given A-level action stays in force.

    "episode" (default) forces every step, as the jump-teaching families
    use. "first_primitive" forces only the opening primitive — the
    bridge_wait family gives the WAIT decision and hands control back once
    the event-terminated wait completes, so the policy crosses on its own.
    """

    metadata = block_smb_monte_carlo_metadata(scenario) if scenario is not None else {}
    parameters = metadata.get("parameters", {}) if isinstance(metadata, Mapping) else {}
    scope = parameters.get("a_level_action_scope") if isinstance(parameters, Mapping) else None
    return str(scope) if scope else "episode"


def block_smb_oracle_actions_for_rollout(
    scenario: Mapping[str, Any] | None,
    *,
    rollout_steps: int,
) -> tuple[int, ...]:
    """Return validated Monte Carlo oracle actions for one rollout, when present."""

    if not isinstance(scenario, Mapping):
        return ()
    metadata = block_smb_monte_carlo_metadata(scenario)
    if not metadata:
        return ()
    try:
        actions = block_smb_monte_carlo_oracle_actions(
            scenario,
            max_steps=max(1, int(rollout_steps)),
        )
    except ValueError:
        return ()
    validated = []
    for action in actions:
        value = int(action)
        if 0 <= value < BLOCK_SMB_ACTION_COUNT:
            validated.append(value)
    return tuple(validated)


def _oracle_hold_frames(actions: tuple[int, ...], step_index: int) -> int | None:
    if step_index < 0 or step_index >= len(actions):
        return None
    action = int(actions[step_index])
    if not is_smb_jump_action(action):
        return None
    end = step_index
    while end < len(actions) and int(actions[end]) == action:
        end += 1
    return max(1, end - step_index)


def _oracle_duration_bin_index(
    motor_primitives: Any,
    hold_frames: int | None,
) -> int | None:
    if hold_frames is None:
        return None
    duration_values = getattr(motor_primitives, "duration_bin_values", None)
    if duration_values is None:
        return None
    if isinstance(duration_values, torch.Tensor):
        values = duration_values.detach().cpu().reshape(-1).numpy()
    else:
        values = np.asarray(duration_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return None
    return int(np.abs(values.astype(np.float32) - float(hold_frames)).argmin())


def _oracle_primitive_execution(
    action: int,
    *,
    next_action: int | None,
    hold_frames: int | None,
    motor_primitives: Any,
) -> SMBPrimitiveExecution:
    action_value = int(action)
    started = is_smb_jump_action(action_value)
    duration_bin_index = _oracle_duration_bin_index(motor_primitives, hold_frames)
    released = False
    if started and next_action is not None:
        released = int(next_action) == int(smb_jump_release_action(action_value))
    return SMBPrimitiveExecution(
        action=action_value,
        started=started,
        active=started,
        released=released,
        cancelled=False,
        duration_bin_index=duration_bin_index,
        hold_frames=hold_frames,
    )


def _action_from_model(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    deterministic: bool,
    tau: float,
    world_model_state: WorldModelState | None = None,
    critic_feedback_enabled: bool = True,
    world_model_enabled: bool = True,
    primitive_executor: SMBParameterizedPrimitiveExecutor | None = None,
    oracle_action: int | None = None,
    oracle_next_action: int | None = None,
    oracle_hold_frames: int | None = None,
    forced_action: int | None = None,
    skill_goal: torch.Tensor | None = None,
    wait_event: bool = False,
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
    WorldModelState | None,
    SMBPrimitiveExecution,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(
            episode_mask, dtype=batch.src_c.dtype, device=batch.src_c.device
        )
    (
        actions1,
        next_state_pred,
        criticism,
        actions2,
        logits_a,
        _w,
        _b,
        next_world_model_state,
    ) = model(
        batch.src_a,
        batch.src_b,
        batch.src_c,
        tau=tau,
        world_model_state=world_model_state,
        episode_mask=episode_mask,
        return_world_model_state=True,
        critic_feedback_enabled=critic_feedback_enabled,
        world_model_enabled=world_model_enabled,
        skill_goal=skill_goal,
    )
    action_logits = logits_a[:, -1, :BLOCK_SMB_ACTION_COUNT]
    finite_or_raise("action_logits", action_logits)
    distribution = torch.distributions.Categorical(logits=action_logits)
    motor_primitives = getattr(model, "last_motor_primitives", None)
    oracle_action_tensor = None
    if oracle_action is not None:
        action_value = int(oracle_action)
        if action_value < 0 or action_value >= BLOCK_SMB_ACTION_COUNT:
            raise ValueError(f"oracle_action must be in [0, {BLOCK_SMB_ACTION_COUNT})")
        oracle_action_tensor = torch.tensor(
            [action_value],
            dtype=torch.long,
            device=action_logits.device,
        )
    searched_action_id = getattr(model, "last_selected_action_id", None)
    if oracle_action_tensor is not None:
        action_tensor = oracle_action_tensor
    elif forced_action is not None:
        # B-level isolation: the scenario hands the A-level decision to the
        # rollout; the forced action goes through the normal primitive
        # executor below so the model's duration head stays in control.
        action_tensor = torch.tensor(
            [int(forced_action)],
            dtype=torch.long,
            device=action_logits.device,
        )
    elif searched_action_id is not None:
        # Ranked-candidate critic search already picked the most likely action
        # the world model predicts will progress without death; execute exactly
        # that token in both stochastic training rollouts and deterministic
        # evaluation.
        action_tensor = torch.tensor(
            [int(searched_action_id)],
            dtype=torch.long,
            device=action_logits.device,
        )
    else:
        action_tensor = (
            action_logits.argmax(dim=-1) if deterministic else distribution.sample()
        )
    execution = SMBPrimitiveExecution(action=int(action_tensor.item()))
    if oracle_action_tensor is not None:
        execution = _oracle_primitive_execution(
            int(action_tensor.item()),
            next_action=oracle_next_action,
            hold_frames=oracle_hold_frames,
            motor_primitives=motor_primitives,
        )
    elif primitive_executor is not None:
        execution = primitive_executor.execute(
            int(action_tensor.item()),
            motor_primitives=motor_primitives,
            batch=batch,
            wait_event=wait_event,
        )
        if execution.action != int(action_tensor.item()):
            action_tensor = torch.tensor(
                [execution.action],
                dtype=action_tensor.dtype,
                device=action_tensor.device,
            )
    log_prob = distribution.log_prob(action_tensor).squeeze(0)
    log_prob = log_prob + _smb_primitive_duration_log_prob(
        motor_primitives,
        execution,
        device=action_logits.device,
        dtype=log_prob.dtype,
    )
    entropy = distribution.entropy().squeeze(0)
    primitive_aux_loss = _smb_primitive_auxiliary_loss(
        motor_primitives,
        oracle_action_tensor,
        execution,
        action_count=BLOCK_SMB_ACTION_COUNT,
        device=action_logits.device,
        dtype=log_prob.dtype,
    )
    expected_hold = _smb_expected_hold_fraction(
        motor_primitives,
        execution,
        device=action_logits.device,
        dtype=log_prob.dtype,
    )
    release_logit = _smb_release_logit(
        motor_primitives,
        execution,
        device=action_logits.device,
        dtype=log_prob.dtype,
    )
    return (
        int(action_tensor.item()),
        log_prob,
        entropy,
        primitive_aux_loss,
        (actions1, actions2, next_state_pred, criticism, logits_a),
        next_world_model_state,
        execution,
        expected_hold,
        release_logit,
    )


def _smb_primitive_duration_log_prob(
    motor_primitives: Any,
    execution: SMBPrimitiveExecution,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    zero = torch.zeros((), dtype=dtype, device=device)
    if motor_primitives is None or not execution.started or execution.duration_bin_index is None:
        return zero
    logits = getattr(motor_primitives, "hold_duration_logits", None)
    if logits is None:
        return zero
    logits = logits.to(device=device, dtype=dtype)
    if logits.ndim != 3 or logits.size(0) != 1:
        return zero
    index = int(execution.duration_bin_index)
    if index < 0 or index >= logits.size(-1):
        return zero
    target = torch.tensor([index], dtype=torch.long, device=device)
    return F.log_softmax(logits[:, -1, :], dim=-1).gather(1, target.view(1, 1)).mean()


# Longest committed wait in frames: the duration menu (1-16) scaled by the
# controller's wait factor (4). Hindsight wait targets normalize against it.
_SMB_MAX_WAIT_FRAMES = 64.0


def block_smb_wait_target_frames(
    records: Sequence[Mapping[str, Any]] | list,
    start_index: int,
    mario_x: float,
    *,
    horizon: int = 64,
    minimum: int = 4,
    approach_threshold: float = 24.0,
) -> int | None:
    """Hindsight correct wait: frames until the moving platform is nearest.

    A moving bridge's motion is deterministic, so the right wait is
    computable after the fact: from the wait's start, find the recorded
    frame (within the wait ceiling) where the platform's center passed
    closest to where the agent stood. Waiting shorter or longer than that
    misses the bridge in one direction or the other. Returns None when no
    moving platform was recorded or it never came close enough to matter.
    """

    best_offset = None
    best_distance = None
    for offset in range(0, int(horizon) + 1):
        index = start_index + offset
        if index >= len(records):
            break
        platform_x = records[index].get("platform_x")
        if platform_x is None:
            continue
        distance = abs(float(platform_x) - float(mario_x))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_offset = offset
    if best_offset is None or best_distance is None:
        return None
    first = next(
        (
            abs(float(r.get("platform_x")) - float(mario_x))
            for r in records[start_index : start_index + 1]
            if r.get("platform_x") is not None
        ),
        None,
    )
    approached = first is not None and (first - best_distance) >= 8.0
    if best_distance > float(approach_threshold) and not approached:
        return None
    return max(int(minimum), min(int(horizon), best_offset))


# Longest hold-duration bin the executor exposes (frames); hindsight hold
# targets and the expected-hold fraction are both normalized against it.
_SMB_MAX_DURATION_BIN_VALUE = 16.0


def _smb_expected_hold_fraction(
    motor_primitives: Any,
    execution: SMBPrimitiveExecution,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Graph-attached expected hold duration, normalized to the longest bin.

    Emitted for every frame a jump primitive is engaged (initiation and
    in-flight), so the primitive-outcome loss can supervise the duration
    belief per frame rather than only at initiation.
    """

    if motor_primitives is None or not (execution.started or execution.active):
        return None
    logits = getattr(motor_primitives, "hold_duration_logits", None)
    if logits is None:
        return None
    logits = logits.to(device=device, dtype=dtype)
    if logits.ndim != 3 or logits.size(0) != 1:
        return None
    probabilities = F.softmax(logits[:, -1, :], dim=-1).reshape(-1)
    values = getattr(motor_primitives, "duration_bin_values", None)
    if values is not None:
        values = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
        if values.numel() != probabilities.numel():
            values = None
    if values is None:
        values = torch.arange(
            1, probabilities.numel() + 1, device=device, dtype=dtype
        )
    max_value = values.detach().abs().max().clamp_min(1.0)
    return (probabilities * values.detach()).sum() / max_value


def _smb_release_logit(
    motor_primitives: Any,
    execution: SMBPrimitiveExecution,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Graph-attached release logit while a jump primitive is engaged.

    HSP1 trains the release head as a span-supervised predictor of "the
    hindsight-correct hold is over" without giving it back any runtime
    authority over the committed executor.
    """

    if motor_primitives is None or not (execution.started or execution.active):
        return None
    logit = getattr(motor_primitives, "release_logit", None)
    if logit is None:
        return None
    logit = logit.to(device=device, dtype=dtype)
    if logit.numel() < 1:
        return None
    return logit.reshape(-1)[-1]


def _smb_primitive_auxiliary_loss(
    motor_primitives: Any,
    oracle_action_tensor: torch.Tensor | None,
    execution: SMBPrimitiveExecution,
    *,
    action_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    zero = torch.zeros((), dtype=dtype, device=device)
    if motor_primitives is None or oracle_action_tensor is None:
        return zero
    target = oracle_action_tensor.detach().to(device=device, dtype=torch.long).view(-1)
    if target.numel() != 1:
        return zero

    losses: list[torch.Tensor] = []
    combo_logits = getattr(motor_primitives, "button_combo_logits", None)
    if combo_logits is not None and combo_logits.ndim == 3:
        losses.append(F.cross_entropy(combo_logits[:, -1, :action_count], target))

    post_release_logits = getattr(motor_primitives, "post_release_logits", None)
    if post_release_logits is not None and post_release_logits.ndim == 3:
        release_target = torch.tensor(
            [int(smb_jump_release_action(int(target.item())))],
            dtype=torch.long,
            device=device,
        )
        losses.append(F.cross_entropy(post_release_logits[:, -1, :action_count], release_target))

    hold_duration_logits = getattr(motor_primitives, "hold_duration_logits", None)
    if (
        hold_duration_logits is not None
        and hold_duration_logits.ndim == 3
        and execution.started
        and execution.duration_bin_index is not None
    ):
        duration_target = torch.tensor(
            [int(execution.duration_bin_index)],
            dtype=torch.long,
            device=device,
        )
        losses.append(F.cross_entropy(hold_duration_logits[:, -1, :], duration_target))

    release_logit = getattr(motor_primitives, "release_logit", None)
    if release_logit is not None and release_logit.ndim == 2:
        release_target = torch.tensor(
            [float(execution.released)],
            dtype=dtype,
            device=device,
        )
        losses.append(F.binary_cross_entropy_with_logits(release_logit[:, -1], release_target))

    cancel_logit = getattr(motor_primitives, "cancel_logit", None)
    if cancel_logit is not None and cancel_logit.ndim == 2:
        cancel_target = torch.tensor(
            [float(execution.cancelled)],
            dtype=dtype,
            device=device,
        )
        losses.append(F.binary_cross_entropy_with_logits(cancel_logit[:, -1], cancel_target))

    if not losses:
        return zero
    return torch.stack([loss.to(device=device, dtype=dtype) for loss in losses]).mean()


def collect_trajectory(
    model: torch.nn.Module,
    stage: BlockSMBStage,
    scenario_name: str,
    *,
    rollout_steps: int,
    seed: int,
    deterministic: bool,
    device: torch.device,
    record_frames: bool = False,
    ablation: BlockSMBAblationConfig | Mapping[str, Any] | None = None,
    use_oracle_actions: bool = False,
    adaptive_duration_control: bool = True,
    skill_goal_conditioning: bool = True,
    steady_duration_primitives: bool = True,
) -> BlockSMBTrajectory:
    ablation_config = _ablation_config(ablation)
    observation = stage.reset(seed=seed)
    trajectory = BlockSMBTrajectory(scenario_name=scenario_name)
    if record_frames:
        trajectory.frames.append(np.asarray(observation).copy())
    world_model_state: WorldModelState | None = None
    # Stochastic training rollouts sample the duration bin so the duration
    # head is explored and its REINFORCE log-prob term is well-founded;
    # deterministic evaluation keeps the argmax bin.
    primitive_executor = SMBParameterizedPrimitiveExecutor(
        duration_sampling=not deterministic,
        duration_seed=seed,
        adaptive_duration=adaptive_duration_control,
        steady_primitives=steady_duration_primitives,
    )
    oracle_actions = (
        block_smb_oracle_actions_for_rollout(
            stage.scenario,
            rollout_steps=rollout_steps,
        )
        if use_oracle_actions
        else ()
    )
    forced_action = block_smb_forced_action_for_rollout(stage.scenario)
    forced_action_scope = block_smb_forced_action_scope(stage.scenario)
    skill_goal = (
        requested_block_smb_skill_goal(stage.scenario)
        if skill_goal_conditioning
        else None
    )
    if skill_goal is not None:
        skill_goal = skill_goal.to(device)
    primitive_span: list[int] = []
    primitive_span_is_jump = True
    primitive_direction = 1.0
    primitive_span_start_x = 0.0
    wait_spans: list[tuple[int, int, float]] = []
    wait_span_start: int | None = None
    wait_span_start_x = 0.0
    temporal_records: list[dict[str, Any]] = []

    def _complete_primitive_span() -> None:
        # Hindsight hold relabeling: from the finished jump's realized
        # displacement and the goal position at completion (the goal rides
        # the enemy under goal_on_stomp, so a patrolling target is measured
        # where it actually was), compute the hold that WOULD have hit the
        # target and backfill every frame with it as a regression target.
        # Bounded and self-terminating: undershoot implies a longer hold,
        # overshoot a shorter one, and an on-target jump relabels to the
        # hold that was actually used — unlike a raw signed-error push,
        # whose asymmetric magnitudes collapsed the head to the floor bin.
        nonlocal primitive_span
        if not primitive_span:
            return
        span = primitive_span
        primitive_span = []
        env = stage.env
        goal = getattr(env, "goal", None)
        if goal is None or not getattr(env, "world_width", 0):
            return
        mario_center_x = float(env.mario["x"]) + float(env.mario["w"]) / 2.0
        realized = (mario_center_x - primitive_span_start_x) * primitive_direction
        target = (float(goal.centerx) - primitive_span_start_x) * primitive_direction
        if realized < 3.0 or target <= 0.0:
            return
        held = 0
        for span_index in span:
            if trajectory.transitions[span_index].action in (
                int(SMBAction.RIGHT_JUMP),
                int(SMBAction.LEFT_JUMP),
            ):
                held += 1
            else:
                break
        if held <= 0:
            return
        if abs(target - realized) < 4.0:
            correct_hold = float(held)
        else:
            ratio = max(0.25, min(4.0, target / realized))
            correct_hold = max(
                1.0, min(float(_SMB_MAX_DURATION_BIN_VALUE), held * ratio)
            )
        target_fraction = correct_hold / float(_SMB_MAX_DURATION_BIN_VALUE)
        for offset, span_index in enumerate(span):
            span_info = trajectory.transitions[span_index].info
            if isinstance(span_info, dict):
                span_info["primitive_outcome_target"] = target_fraction
                # HSP1: frame position within the primitive plus the
                # hindsight-correct hold, so the release head can learn
                # "should I be releasing now?" from complete spans.
                span_info["primitive_frame_index"] = offset
                span_info["primitive_target_hold"] = correct_hold
                # The world model's target for committed-primitive frames is
                # the OUTCOME of the primitive — the state at completion
                # (landing) — not the next frame. A jump's value is invisible
                # one frame ahead; it lives where the arc comes down.
                span_info["primitive_outcome_batch"] = trajectory.transitions[
                    span[-1]
                ].next_batch

    for step_index in range(rollout_steps):
        batch = apply_block_smb_ablations(stage.encode_observation(observation), ablation_config)
        batch.src_a = batch.src_a.to(device)
        batch.src_b = batch.src_b.to(device)
        batch.src_c = batch.src_c.to(device)
        carried_state = world_model_state if ablation_config.recurrent_state_enabled else None
        oracle_action = (
            int(oracle_actions[step_index]) if step_index < len(oracle_actions) else None
        )
        oracle_next_action = (
            int(oracle_actions[step_index + 1]) if step_index + 1 < len(oracle_actions) else None
        )
        step_skill_goal = skill_goal
        pre_step_mario_center_x = float(stage.env.mario["x"]) + float(stage.env.mario["w"]) / 2.0
        pre_step_mario_y = float(stage.env.mario["y"])
        wait_event = False
        for plat in stage.env.platforms:
            if not plat.get("moving"):
                continue
            # The awaited event: the platform reached the end of its travel
            # nearest the agent — its closest approach — regardless of the
            # absolute distance (the agent may wait well back from the edge).
            near_end = (
                float(plat["move_min"])
                if abs(plat["move_min"] - pre_step_mario_center_x)
                <= abs(plat["move_max"] - pre_step_mario_center_x)
                else float(plat["move_max"])
            )
            if abs(float(plat["move_x"]) - near_end) < 6.0:
                wait_event = True
            break
        (
            action,
            log_prob,
            entropy,
            primitive_aux_loss,
            outputs,
            next_world_model_state,
            execution,
            expected_hold,
            release_logit,
        ) = _action_from_model(
            model,
            batch,
            deterministic=deterministic,
            tau=1.0,
            world_model_state=carried_state,
            critic_feedback_enabled=ablation_config.critic_feedback_enabled,
            world_model_enabled=ablation_config.world_model_enabled,
            primitive_executor=primitive_executor,
            oracle_action=oracle_action,
            oracle_next_action=oracle_next_action,
            oracle_hold_frames=_oracle_hold_frames(oracle_actions, step_index),
            forced_action=forced_action,
            skill_goal=step_skill_goal,
            wait_event=wait_event,
        )
        next_observation, reward, terminated, truncated, info = stage.step(action)
        info = dict(info)
        info["goal_reached"] = _goal_reached(stage.env)
        next_batch = apply_block_smb_ablations(
            stage.encode_observation(next_observation, info), ablation_config
        )
        next_batch.src_a = next_batch.src_a.to(device)
        next_batch.src_b = next_batch.src_b.to(device)
        next_batch.src_c = next_batch.src_c.to(device)
        done = bool(terminated or truncated)
        episode_mask = 0.0 if done else 1.0
        actions1, actions2, next_state_pred, criticism, logits_a = outputs
        trajectory.transitions.append(
            BlockSMBTransition(
                batch=batch,
                next_batch=next_batch,
                action=action,
                reward=float(reward),
                done=done,
                episode_mask=episode_mask,
                scenario_name=scenario_name,
                info=info,
                log_prob=log_prob,
                entropy=entropy,
                actions1=actions1,
                actions2=actions2,
                next_state_pred=next_state_pred,
                criticism=criticism,
                logits_a=logits_a,
                primitive_aux_loss=primitive_aux_loss,
                oracle_action=oracle_action,
                step_index=step_index,
                noop_allowed=block_smb_noop_allowed_for_step(
                    scenario_name,
                    stage.scenario,
                    step_index,
                ),
                expected_hold=expected_hold,
                release_logit=release_logit,
            )
        )
        # Per-frame primitive-outcome bookkeeping: track the frames of the
        # engaged horizontal jump. Completion is landing / enemy contact /
        # episode end — NOT the button release, because the arc keeps
        # carrying Mario forward after release, so the hindsight error must
        # be measured where he actually came down.
        transition_index = len(trajectory.transitions) - 1
        if execution.started and execution.action in (
            int(SMBAction.RIGHT_JUMP),
            int(SMBAction.LEFT_JUMP),
            int(SMBAction.RIGHT),
            int(SMBAction.LEFT),
        ):
            primitive_span = [transition_index]
            primitive_span_is_jump = execution.action in (
                int(SMBAction.RIGHT_JUMP),
                int(SMBAction.LEFT_JUMP),
            )
            primitive_direction = (
                1.0
                if execution.action in (int(SMBAction.RIGHT_JUMP), int(SMBAction.RIGHT))
                else -1.0
            )
            primitive_span_start_x = pre_step_mario_center_x
        elif execution.started:
            # A wait (NOOP) primitive: no displacement relabel, but waits get
            # their own hindsight coaching after the rollout — the correct
            # wait is when the recorded moving platform came nearest.
            primitive_span = []
            wait_span_start = transition_index
            wait_span_start_x = pre_step_mario_center_x
        elif execution.active and primitive_span:
            primitive_span.append(transition_index)
        if (
            forced_action is not None
            and forced_action_scope == "first_primitive"
            and (execution.released or execution.landed or execution.cancelled)
        ):
            # The given opening primitive has completed; the policy owns the
            # rest of the episode.
            forced_action = None
        if wait_span_start is not None and (
            execution.released or execution.cancelled or done
        ):
            wait_spans.append((wait_span_start, transition_index, wait_span_start_x))
            wait_span_start = None
        if (
            execution.landed
            or execution.cancelled
            or done
            or (primitive_span and not primitive_span_is_jump and execution.released)
        ):
            _complete_primitive_span()
        temporal_records.append(
            {
                "action": int(action),
                "started": bool(execution.started),
                "active": bool(execution.active),
                "released": bool(execution.released),
                "landed": bool(execution.landed),
                "cancelled": bool(execution.cancelled),
                "death": bool(info.get("death", False)),
                "goal": bool(info.get("goal_reached", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "x_before": pre_step_mario_center_x,
                "x_after": float(stage.env.mario["x"])
                + float(stage.env.mario["w"]) / 2.0,
                "y_before": pre_step_mario_y,
                "y_after": float(stage.env.mario["y"]),
                "platform_x": next(
                    (
                        float(plat["move_x"]) + plat["rect"].w / 2.0
                        for plat in stage.env.platforms
                        if plat.get("moving")
                    ),
                    None,
                ),
            }
        )
        observation = next_observation
        if record_frames:
            trajectory.frames.append(np.asarray(observation).copy())
        if done:
            world_model_state = None
            break
        world_model_state = (
            next_world_model_state.detach()
            if ablation_config.recurrent_state_enabled and next_world_model_state is not None
            else None
        )
    # Rollout budget exhausted with a jump still open (e.g., vision never
    # reported the landing): supervise with the final position — the sign of
    # the error is settled even if post-landing walking inflates it.
    _complete_primitive_span()
    if wait_span_start is not None:
        wait_spans.append((wait_span_start, len(trajectory.transitions) - 1, wait_span_start_x))
    for span_start, span_end, span_x in wait_spans:
        target_frames = block_smb_wait_target_frames(
            temporal_records, span_start, span_x
        )
        if target_frames is None:
            continue
        for offset, index in enumerate(range(span_start, span_end + 1)):
            span_info = trajectory.transitions[index].info
            if isinstance(span_info, dict):
                span_info["primitive_outcome_target"] = (
                    target_frames / _SMB_MAX_WAIT_FRAMES
                )
                span_info["primitive_frame_index"] = offset
                span_info["primitive_target_hold"] = float(target_frames)
    metadata = block_smb_monte_carlo_metadata(stage.scenario)
    family = ""
    if isinstance(metadata, Mapping):
        family = str(metadata.get("family", "") or "")
    trajectory.spans = build_block_smb_temporal_spans(
        temporal_records,
        episode_id=f"{scenario_name}#seed{seed}",
        scenario_id=scenario_name,
        seed=seed,
        source="scripted" if (use_oracle_actions and oracle_actions) else "real",
        family_goal=family,
    )
    return trajectory


def compute_imagined_rollout_losses(
    model: torch.nn.Module,
    trajectories: list[BlockSMBTrajectory],
    config: BlockSMBTrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Unroll learned dynamics from replay states and compare to real futures."""

    zero = torch.zeros((), dtype=torch.float32, device=device)
    if (
        config.imagined_rollout_horizon <= 0
        or not trajectories
        or not config.ablation.world_model_enabled
    ):
        return {
            "loss_imagined_dynamics": zero,
            "loss_imagined_reward": zero,
            "loss_imagined_rollout": zero,
            "imagined_rollout_steps": zero,
        }

    dynamics_terms = []
    reward_terms = []
    for trajectory in trajectories:
        steps = trajectory.transitions
        for start_index in range(len(steps)):
            imagined_state = steps[start_index].batch.src_c.detach().to(device)
            for offset in range(config.imagined_rollout_horizon):
                step_index = start_index + offset
                if step_index >= len(steps):
                    break
                step = steps[step_index]
                (
                    _actions1,
                    next_state_pred,
                    _criticism,
                    _actions2,
                    _logits_a,
                    _w_b,
                    _b_b,
                ) = model(
                    step.batch.src_a.detach().to(device),
                    step.batch.src_b.detach().to(device),
                    imagined_state,
                    tau=1.0,
                    critic_feedback_enabled=config.ablation.critic_feedback_enabled,
                    world_model_enabled=config.ablation.world_model_enabled,
                )
                dynamics_terms.append(
                    F.mse_loss(next_state_pred, step.next_batch.src_c.detach().to(device))
                )
                reward_pred = model.predict_reward(next_state_pred)
                reward_target = torch.full_like(
                    reward_pred,
                    float(step.reward),
                    device=device,
                )
                reward_terms.append(F.mse_loss(reward_pred, reward_target))
                imagined_state = next_state_pred
                if step.done:
                    break

    if not dynamics_terms:
        return {
            "loss_imagined_dynamics": zero,
            "loss_imagined_reward": zero,
            "loss_imagined_rollout": zero,
            "imagined_rollout_steps": zero,
        }

    loss_imagined_dynamics = torch.stack(dynamics_terms).mean()
    loss_imagined_reward = torch.stack(reward_terms).mean()
    loss_imagined_rollout = (
        loss_imagined_dynamics + config.reward_loss_weight * loss_imagined_reward
    )
    return {
        "loss_imagined_dynamics": loss_imagined_dynamics,
        "loss_imagined_reward": loss_imagined_reward,
        "loss_imagined_rollout": loss_imagined_rollout,
        "imagined_rollout_steps": torch.tensor(
            float(len(dynamics_terms)), dtype=torch.float32, device=device
        ),
    }


def measured_dynamics_instability(
    transitions: list[BlockSMBTransition], device: torch.device
) -> torch.Tensor:
    if not transitions:
        return torch.zeros((), dtype=torch.float32, device=device)
    terms = [
        F.mse_loss(
            step.next_state_pred.detach().to(device),
            step.next_batch.src_c.detach().to(device),
        )
        for step in transitions
    ]
    return torch.stack(terms).mean()


def block_smb_c_stream_dynamics_slot_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: StageBatch,
) -> dict[str, torch.Tensor]:
    spans = block_smb_c_stream_slot_spans(batch)
    losses: dict[str, torch.Tensor] = {}
    for slot_name in BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_NAMES:
        start, end = spans[slot_name]
        if end <= start:
            losses[slot_name] = prediction.new_zeros(())
            continue
        losses[slot_name] = F.mse_loss(prediction[:, start:end], target[:, start:end])
    return losses


def block_smb_dynamics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slot_losses: Mapping[str, torch.Tensor],
    *,
    world_model_slot_weights: Mapping[str, float],
) -> torch.Tensor:
    if not world_model_slot_weights:
        return F.mse_loss(prediction, target)
    weighted_loss: torch.Tensor | None = None
    total_weight = 0.0
    for slot_name in BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_NAMES:
        slot_loss = slot_losses.get(slot_name)
        if slot_loss is None:
            continue
        slot_weight = float(world_model_slot_weights.get(slot_name, 1.0))
        if slot_weight <= 0.0:
            continue
        weighted = slot_loss * slot_weight
        weighted_loss = weighted if weighted_loss is None else weighted_loss + weighted
        total_weight += slot_weight
    if weighted_loss is None or total_weight <= 0.0:
        return F.mse_loss(prediction, target)
    return weighted_loss / total_weight


def block_smb_c_stream_dynamics_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: StageBatch,
    *,
    semantic_accuracy_threshold: float,
) -> dict[str, torch.Tensor]:
    spans = block_smb_c_stream_slot_spans(batch)
    metrics: dict[str, torch.Tensor] = {}
    for slot_name in BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_NAMES:
        start, end = spans[slot_name]
        if end <= start:
            metrics[f"dynamics_{slot_name}_rmse"] = prediction.new_zeros(())
            metrics[f"dynamics_{slot_name}_mae"] = prediction.new_zeros(())
            continue
        diff = prediction[:, start:end] - target[:, start:end]
        metrics[f"dynamics_{slot_name}_rmse"] = diff.pow(2).mean().sqrt()
        metrics[f"dynamics_{slot_name}_mae"] = diff.abs().mean()
    semantics_start, semantics_end = spans["semantic_probabilities"]
    if semantics_end > semantics_start:
        predicted_semantics = prediction[:, semantics_start:semantics_end]
        target_semantics = target[:, semantics_start:semantics_end]
        semantic_accuracy = (
            (predicted_semantics.argmax(dim=1) == target_semantics.argmax(dim=1)).float().mean()
        )
        semantic_cosine = F.cosine_similarity(
            predicted_semantics.float(),
            target_semantics.float(),
            dim=1,
        ).mean()
    else:
        semantic_accuracy = prediction.new_zeros(())
        semantic_cosine = prediction.new_zeros(())
    metrics["dynamics_semantic_prediction_accuracy"] = semantic_accuracy
    metrics["dynamics_semantic_prediction_cosine"] = semantic_cosine
    metrics["dynamics_semantic_prediction_gate_met"] = (
        semantic_accuracy >= float(semantic_accuracy_threshold)
    ).to(dtype=prediction.dtype)
    return metrics


def block_smb_c_stream_slot_spans(batch: StageBatch) -> dict[str, tuple[int, int]]:
    feature_length = int(batch.src_c.shape[1])
    metadata = batch.metadata if isinstance(batch.metadata, Mapping) else {}
    fusion = metadata.get("vision_fusion", {})
    if not isinstance(fusion, Mapping):
        fusion = {}
    position = _block_smb_c_stream_span(fusion, "c_position", feature_length, default=(0, 0))
    semantics = _block_smb_c_stream_span(
        fusion,
        "c_semantic_probabilities",
        feature_length,
        default=(position[1], position[1]),
    )
    support = _block_smb_c_stream_span(
        fusion,
        "c_support_state",
        feature_length,
        default=(semantics[1], semantics[1]),
    )
    state = _block_smb_c_stream_span(
        fusion,
        "c_state",
        feature_length,
        default=(support[1], support[1]),
    )
    terminal_start = max(state[0], state[1] - 3)
    terminal_outcome = (terminal_start, state[1])
    patch_tokens = _block_smb_c_stream_span(
        fusion,
        "c_patch_tokens",
        feature_length,
        default=(state[1], feature_length),
    )
    return {
        "position": position,
        "semantic_probabilities": semantics,
        "support_state": support,
        "state": state,
        "terminal_outcome": terminal_outcome,
        "patch_tokens": patch_tokens,
    }


def _block_smb_c_stream_span(
    fusion: Mapping[str, Any],
    name: str,
    feature_length: int,
    *,
    default: tuple[int, int],
) -> tuple[int, int]:
    raw = fusion.get(name, default)
    try:
        start, end = int(raw[0]), int(raw[1])
    except (TypeError, ValueError, IndexError):
        start, end = default
    start = max(0, min(feature_length, start))
    end = max(start, min(feature_length, end))
    return start, end


def target_network_is_active(
    config: BlockSMBTrainingConfig,
    target_model: Optional[torch.nn.Module],
    instability: torch.Tensor,
) -> bool:
    if target_model is None or config.target_network_mode == "off":
        return False
    if config.target_network_mode == "on":
        return True
    return float(instability.detach().cpu()) >= config.target_network_instability_threshold


def _block_smb_transition_progress_target(step: BlockSMBTransition) -> float:
    reward_terms = step.info.get("reward_terms", {})
    progress_reward = 0.0
    if isinstance(reward_terms, Mapping):
        progress_reward = float(reward_terms.get("progress", 0.0) or 0.0)
        goal_reward = float(reward_terms.get("goal", 0.0) or 0.0)
    else:
        goal_reward = 0.0
    return 1.0 if progress_reward > 0.0 or goal_reward > 0.0 else 0.0


def _block_smb_transition_death_target(step: BlockSMBTransition) -> float:
    reward_terms = step.info.get("reward_terms", {})
    if isinstance(reward_terms, Mapping):
        if float(reward_terms.get("fall_death", 0.0) or 0.0) < 0.0:
            return 1.0
        if float(reward_terms.get("enemy_hit", 0.0) or 0.0) < 0.0:
            return 1.0
    if step.done and not bool(step.info.get("goal_reached", False)):
        return 1.0
    return 0.0


def _critic_action_outcome_loss(
    model: torch.nn.Module,
    step: BlockSMBTransition,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not (
        hasattr(model, "predict_action_progress_logit")
        and hasattr(model, "predict_action_death_logit")
    ):
        return step.next_state_pred.new_zeros(())
    progress_target = torch.tensor(
        [_block_smb_transition_progress_target(step)],
        dtype=step.next_state_pred.dtype,
        device=device,
    )
    death_target = torch.tensor(
        [_block_smb_transition_death_target(step)],
        dtype=step.next_state_pred.dtype,
        device=device,
    )
    progress_logit = model.predict_action_progress_logit(
        step.next_state_pred,
        current_state=step.batch.src_c.detach(),
    )
    death_logit = model.predict_action_death_logit(step.next_state_pred)
    return F.binary_cross_entropy_with_logits(
        progress_logit,
        progress_target,
    ) + F.binary_cross_entropy_with_logits(death_logit, death_target)


def block_smb_noop_suppression_loss(
    step: BlockSMBTransition,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Penalize NOOP probability unless the scenario's plan explicitly waits."""

    logits = step.logits_a.to(device=device)
    zero = logits.new_zeros(())
    if bool(step.noop_allowed):
        return zero
    if logits.ndim != 3 or logits.size(-1) <= BLOCK_SMB_NOOP_ACTION:
        return zero
    action_logits = logits[:, -1, :BLOCK_SMB_ACTION_COUNT]
    if action_logits.size(-1) <= 1:
        return zero
    noop_logit = action_logits[:, BLOCK_SMB_NOOP_ACTION]
    non_noop_logsumexp = torch.logsumexp(
        torch.cat(
            (
                action_logits[:, :BLOCK_SMB_NOOP_ACTION],
                action_logits[:, BLOCK_SMB_NOOP_ACTION + 1 :],
            ),
            dim=-1,
        ),
        dim=-1,
    )
    return F.softplus(noop_logit - non_noop_logsumexp).mean()


def block_smb_oracle_action_loss(
    step: BlockSMBTransition,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Supervise policy logits from scenario oracle actions when available."""

    logits = step.logits_a.to(device=device)
    zero = logits.new_zeros(())
    if step.oracle_action is None:
        return zero
    oracle_action = int(step.oracle_action)
    if oracle_action < 0 or oracle_action >= BLOCK_SMB_ACTION_COUNT:
        return zero
    if logits.ndim != 3 or logits.size(-1) <= oracle_action:
        return zero
    target = torch.tensor([oracle_action], dtype=torch.long, device=device)
    return F.cross_entropy(logits[:, -1, :BLOCK_SMB_ACTION_COUNT], target)


def compute_block_smb_losses(
    model: torch.nn.Module,
    transitions: list[BlockSMBTransition],
    config: BlockSMBTrainingConfig,
    device: torch.device,
    trajectories: Optional[list[BlockSMBTrajectory]] = None,
    target_model: Optional[torch.nn.Module] = None,
) -> dict[str, torch.Tensor]:
    if not transitions:
        raise ValueError("transitions must be non-empty")
    target_instability = measured_dynamics_instability(transitions, device)
    target_active = target_network_is_active(config, target_model, target_instability)
    target_model_for_loss = target_model if target_active else model
    masks = [step.episode_mask for step in transitions]
    if trajectories and sum(len(t.transitions) for t in trajectories) == len(masks):
        # Terminate the return scan at every trajectory boundary so returns from
        # one trajectory never leak into a preceding, budget-truncated trajectory.
        boundary_index = -1
        for trajectory in trajectories:
            boundary_index += len(trajectory.transitions)
            if boundary_index >= 0:
                masks[boundary_index] = 0.0
    returns = discounted_returns(
        [step.reward for step in transitions],
        masks,
        config.gamma,
        device,
    )
    policy_terms = []
    entropy_terms = []
    representation_terms = []
    dynamics_terms = []
    dynamics_slot_terms: dict[str, list[torch.Tensor]] = {
        slot_name: [] for slot_name in BLOCK_SMB_C_STREAM_DYNAMICS_SLOT_NAMES
    }
    dynamics_metric_terms: dict[str, list[torch.Tensor]] = {}
    reward_terms = []
    value_terms = []
    action_aux_terms = []
    oracle_action_terms = []
    noop_terms = []
    critic_terms = []
    primitive_outcome_terms = []
    release_timing_terms = []
    oracle_supervised_steps = 0
    for index, step in enumerate(transitions):
        return_target = returns[index].view(1)
        reward_target = torch.tensor([step.reward], dtype=torch.float32, device=device)
        value_pred = model.predict_value(step.batch.src_c.detach())
        reward_pred = model.predict_reward(step.next_state_pred)
        outcome_batch = (
            step.info.get("primitive_outcome_batch")
            if isinstance(step.info, Mapping)
            else None
        )
        dynamics_target_batch = outcome_batch if outcome_batch is not None else step.next_batch
        predicted_representation = model.transition_representation(step.next_state_pred)
        with torch.no_grad():
            target_representation = target_model_for_loss.transition_representation(
                dynamics_target_batch.src_c.detach()
            )
        advantage = (return_target - value_pred.detach()).detach()

        policy_terms.append(-step.log_prob * advantage.squeeze(0))
        entropy_terms.append(step.entropy)
        representation_terms.append(F.mse_loss(predicted_representation, target_representation))
        next_state_target = dynamics_target_batch.src_c.detach()
        slot_losses = block_smb_c_stream_dynamics_slot_losses(
            step.next_state_pred,
            next_state_target,
            dynamics_target_batch,
        )
        dynamics_terms.append(
            block_smb_dynamics_loss(
                step.next_state_pred,
                next_state_target,
                slot_losses,
                world_model_slot_weights=config.world_model_slot_weights,
            )
        )
        for slot_name, slot_loss in slot_losses.items():
            dynamics_slot_terms.setdefault(slot_name, []).append(slot_loss)
        dynamics_metrics = block_smb_c_stream_dynamics_metrics(
            step.next_state_pred,
            next_state_target,
            dynamics_target_batch,
            semantic_accuracy_threshold=config.semantic_prediction_accuracy_threshold,
        )
        for metric_name, metric_value in dynamics_metrics.items():
            dynamics_metric_terms.setdefault(metric_name, []).append(metric_value)
        reward_terms.append(F.mse_loss(reward_pred, reward_target))
        value_terms.append(F.mse_loss(value_pred, return_target.detach()))
        if step.primitive_aux_loss is None:
            action_aux_terms.append(step.log_prob.new_zeros(()))
        else:
            action_aux_terms.append(step.primitive_aux_loss.to(device=device))
        oracle_loss = block_smb_oracle_action_loss(step, device=device)
        oracle_action_terms.append(oracle_loss)
        if step.oracle_action is not None:
            oracle_supervised_steps += 1
        noop_terms.append(block_smb_noop_suppression_loss(step, device=device))
        critic_terms.append(
            step.criticism.pow(2).mean() + _critic_action_outcome_loss(model, step, device=device)
        )
        # Per-frame primitive-outcome supervision: regress each jump frame's
        # expected hold toward the hindsight-relabeled correct hold for that
        # jump's geometry. Short- and long-target scenarios have different
        # targets, which a context-blind duration head cannot satisfy — the
        # gradient is forced into the geometry-conditioned pathway. Applying
        # it at every in-flight frame (where a patrolling target has moved
        # since initiation) additionally teaches mid-air belief updates.
        outcome_target = (
            step.info.get("primitive_outcome_target")
            if isinstance(step.info, Mapping)
            else None
        )
        if step.expected_hold is not None and outcome_target is not None:
            primitive_outcome_terms.append(
                (step.expected_hold.to(device=device) - float(outcome_target)) ** 2
            )
        frame_index = (
            step.info.get("primitive_frame_index") if isinstance(step.info, Mapping) else None
        )
        target_hold = (
            step.info.get("primitive_target_hold") if isinstance(step.info, Mapping) else None
        )
        if step.release_logit is not None and frame_index is not None and target_hold is not None:
            should_release = 1.0 if float(frame_index) + 1.0 >= float(target_hold) else 0.0
            release_timing_terms.append(
                F.binary_cross_entropy_with_logits(
                    step.release_logit.to(device=device).reshape(()),
                    torch.tensor(should_release, dtype=torch.float32, device=device),
                )
            )
    loss_representation = torch.stack(representation_terms).mean()
    loss_dynamics = torch.stack(dynamics_terms).mean()
    loss_reward = torch.stack(reward_terms).mean()
    loss_value = torch.stack(value_terms).mean()
    loss_policy = torch.stack(policy_terms).mean()
    loss_action_aux = torch.stack(action_aux_terms).mean()
    loss_oracle_action = torch.stack(oracle_action_terms).mean()
    loss_noop = torch.stack(noop_terms).mean()
    loss_critic_feedback = torch.stack(critic_terms).mean()
    loss_primitive_outcome = (
        torch.stack(primitive_outcome_terms).mean()
        if primitive_outcome_terms
        else loss_policy.new_zeros(())
    )
    loss_release_timing = (
        torch.stack(release_timing_terms).mean()
        if release_timing_terms
        else loss_policy.new_zeros(())
    )
    entropy_bonus = torch.stack(entropy_terms).mean()
    imagined_losses = compute_imagined_rollout_losses(model, trajectories or [], config, device)
    world_model_weight = config.world_model_weight if config.ablation.world_model_enabled else 0.0
    imagined_rollout_weight = (
        config.imagined_rollout_weight if config.ablation.world_model_enabled else 0.0
    )
    loss_total = (
        config.representation_weight * loss_representation
        + world_model_weight * loss_dynamics
        + config.reward_loss_weight * loss_reward
        + config.value_loss_weight * loss_value
        + config.policy_loss_weight * loss_policy
        + config.action_aux_weight * loss_action_aux
        + config.oracle_action_loss_weight * loss_oracle_action
        + config.noop_loss_weight * loss_noop
        + config.critic_loss_weight * loss_critic_feedback
        + config.primitive_outcome_weight * loss_primitive_outcome
        + config.release_timing_weight * loss_release_timing
        + imagined_rollout_weight * imagined_losses["loss_imagined_rollout"]
        - config.entropy_weight * entropy_bonus
    )
    losses = {
        "loss_representation": loss_representation,
        "loss_dynamics": loss_dynamics,
        **{
            f"loss_dynamics_{slot_name}": (
                torch.stack(values).mean() if values else loss_dynamics.new_zeros(())
            )
            for slot_name, values in dynamics_slot_terms.items()
        },
        **{
            metric_name: torch.stack(values).mean()
            for metric_name, values in dynamics_metric_terms.items()
            if values
        },
        "loss_reward": loss_reward,
        "loss_value": loss_value,
        "loss_policy": loss_policy,
        "loss_action_aux": loss_action_aux,
        "loss_primitive_outcome": loss_primitive_outcome,
        "loss_release_timing": loss_release_timing,
        "primitive_outcome_supervised_steps": torch.tensor(
            float(len(primitive_outcome_terms)), device=device
        ),
        "loss_oracle_action": loss_oracle_action,
        "oracle_action_supervised_steps": torch.tensor(
            float(oracle_supervised_steps),
            dtype=torch.float32,
            device=device,
        ),
        "loss_noop": loss_noop,
        "loss_critic_feedback": loss_critic_feedback,
        **imagined_losses,
        "target_network_active": torch.tensor(
            float(target_active), dtype=torch.float32, device=device
        ),
        "target_network_instability": target_instability,
        "target_network_drift": target_network_parameter_delta(model, target_model, device),
        "target_network_tau": torch.tensor(
            config.target_network_tau, dtype=torch.float32, device=device
        ),
        "loss_entropy": entropy_bonus,
        "loss_total": loss_total,
        # Backward-compatible metric aliases for existing run summaries.
        "loss_actor_pass1": loss_representation,
        "loss_actor_pass2": loss_policy,
        "loss_world_model": loss_dynamics,
        "loss_critic": loss_critic_feedback,
    }
    if "dynamics_semantic_prediction_accuracy" in losses:
        losses["dynamics_semantic_prediction_gate_met"] = (
            losses["dynamics_semantic_prediction_accuracy"]
            >= float(config.semantic_prediction_accuracy_threshold)
        ).to(dtype=loss_dynamics.dtype)
    else:
        losses["dynamics_semantic_prediction_gate_met"] = loss_dynamics.new_zeros(())
    for name, value in losses.items():
        finite_or_raise(name, value)
    return losses


def train_block_smb_epoch(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    curriculum: list[tuple[str, dict]],
    config: BlockSMBTrainingConfig,
    epoch: int,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
    target_model: Optional[torch.nn.Module] = None,
    success_replay: Optional[BlockSMBSuccessReplay] = None,
) -> tuple[dict[str, float], BlockSMBReplayBuffer]:
    model.train()
    if target_model is not None:
        target_model.eval()
    replay = BlockSMBReplayBuffer()
    episode_count = (
        max(config.episodes_per_epoch, len(curriculum))
        if (config.cover_curriculum_per_epoch)
        else config.episodes_per_epoch
    )
    update_batch_size = min(max(1, int(config.update_batch_episodes)), episode_count)
    metric_totals: dict[str, float] = {}
    total_update_episodes = 0
    total_returns: list[float] = []
    all_actions: list[int] = []
    update_count = 0

    def flush_update_batch() -> None:
        nonlocal replay, total_update_episodes, update_count
        if not replay.trajectories:
            return
        losses = compute_block_smb_losses(
            model,
            replay.transitions(),
            config,
            device,
            trajectories=replay.trajectories,
            target_model=target_model,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        check_model_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if not torch.isfinite(grad_norm).item():
            raise FloatingPointError("gradient norm is NaN or infinite")
        optimizer.step()
        if target_model is not None:
            update_target_network(target_model, model, config.target_network_tau)
        batch_episodes = len(replay.trajectories)
        batch_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
        batch_losses["gradient_norm"] = float(grad_norm.detach().cpu())
        for key, value in batch_losses.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_episodes
        total_update_episodes += batch_episodes
        update_count += 1
        replay.clear()

    for episode in range(episode_count):
        scenario_name, scenario = curriculum[(epoch * episode_count + episode) % len(curriculum)]
        stage = BlockSMBStage(
            env=MarioScenarioEnv(reward_config=config.reward_config),
            scenario=scenario,
            vision=vision_factory(),
        )
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=config.rollout_steps,
                seed=config.seed + epoch * 10_000 + episode,
                deterministic=False,
                device=device,
                ablation=config.ablation,
                use_oracle_actions=config.use_oracle_actions,
                adaptive_duration_control=config.adaptive_duration_control,
                skill_goal_conditioning=config.skill_goal_conditioning,
                steady_duration_primitives=config.steady_duration_primitives,
            )
            _write_block_smb_spans(config, trajectory)
        finally:
            stage.env.close()
        replay.add(trajectory)
        if success_replay is not None:
            metadata = block_smb_monte_carlo_metadata(scenario)
            family = (
                str(metadata.get("family", "") or "")
                if isinstance(metadata, Mapping)
                else ""
            )
            success_replay.add(trajectory, family, scenario_name, scenario)
        total_returns.append(trajectory.total_return)
        all_actions.extend(step.action for step in trajectory.transitions)
        if len(replay.trajectories) >= update_batch_size:
            flush_update_batch()

    replay_metrics: dict[str, float] = {}
    if success_replay is not None:
        replay_metrics = {
            "success_replay_episodes": float(len(success_replay)),
            "success_replay_families": float(len(success_replay.families())),
        }
        rehearsals = success_replay.sample_scenarios(
            int(config.success_replay_rehearsals_per_epoch)
        )
        if rehearsals:
            rehearsal_successes = 0
            rehearsal_by_family: dict[str, list[int]] = {}
            for rehearsal_index, rehearsal in enumerate(rehearsals):
                stage = BlockSMBStage(
                    env=MarioScenarioEnv(reward_config=config.reward_config),
                    scenario=copy.deepcopy(rehearsal["scenario"]),
                    vision=vision_factory(),
                )
                try:
                    trajectory = collect_trajectory(
                        model,
                        stage,
                        rehearsal["scenario_id"],
                        rollout_steps=config.rollout_steps,
                        seed=config.seed + 900_000 + epoch * 100 + rehearsal_index,
                        deterministic=False,
                        device=device,
                        ablation=config.ablation,
                        adaptive_duration_control=config.adaptive_duration_control,
                        skill_goal_conditioning=config.skill_goal_conditioning,
                        steady_duration_primitives=config.steady_duration_primitives,
                    )
                finally:
                    stage.env.close()
                replay.add(trajectory)
                total_returns.append(trajectory.total_return)
                all_actions.extend(step.action for step in trajectory.transitions)
                family_counts = rehearsal_by_family.setdefault(
                    rehearsal["family"], [0, 0]
                )
                family_counts[0] += 1
                if trajectory.success:
                    family_counts[1] += 1
                    rehearsal_successes += 1
                    success_replay.add(
                        trajectory,
                        rehearsal["family"],
                        rehearsal["scenario_id"],
                        rehearsal["scenario"],
                    )
                if len(replay.trajectories) >= update_batch_size:
                    flush_update_batch()
            replay_metrics["success_rehearsals"] = float(len(rehearsals))
            replay_metrics["success_rehearsal_success_rate"] = (
                rehearsal_successes / len(rehearsals)
            )
            # Per-family rehearsal outcomes split "forgot a retained skill"
            # from "was never reliable at it" — the aggregate rate conflates
            # the two.
            for family, (attempts, successes) in rehearsal_by_family.items():
                replay_metrics[f"rehearsal_attempts_{family}"] = float(attempts)
                replay_metrics[f"rehearsal_rate_{family}"] = successes / attempts
    flush_update_batch()
    if total_update_episodes <= 0:
        raise ValueError("no Block SMB rollout episodes were collected")
    epoch_losses = {
        key: value / float(total_update_episodes) for key, value in metric_totals.items()
    }
    epoch_losses["mean_return"] = float(np.mean(total_returns)) if total_returns else 0.0
    epoch_losses["episodes"] = float(total_update_episodes)
    epoch_losses["optimizer_updates"] = float(update_count)
    epoch_losses.update(replay_metrics)
    action_counts = summarize_block_smb_monte_carlo_action_counts(all_actions)
    epoch_losses.update(block_smb_action_count_metric_values("train", action_counts))
    epoch_losses.update(
        block_smb_action_distribution_gate_metrics(
            "train",
            action_counts,
            min_distinct_actions=config.action_gate_min_distinct_actions,
            max_dominant_fraction=config.action_gate_max_dominant_fraction,
            required_actions=config.action_gate_required_actions,
        )
    )
    return epoch_losses, replay


def evaluate_block_smb_monte_carlo(
    model: torch.nn.Module,
    config: BlockSMBTrainingConfig,
    *,
    split: str,
    sample_count: int,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
    record_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Evaluate a policy on a held-out Monte Carlo split."""

    if sample_count <= 0 and not config.monte_carlo_parameter_sweep:
        raise ValueError("sample_count must be positive")
    if config.monte_carlo_parameter_sweep:
        sample_set = sample_block_smb_monte_carlo_parameter_sweep(
            distribution_id=config.monte_carlo_distribution_id,
            split=split,
            seed=int(config.monte_carlo_seed),
            repeats_per_difficulty=config.monte_carlo_sweep_repeats_per_difficulty,
            validate_reachability=config.monte_carlo_validate_reachability,
            max_rejections=config.monte_carlo_max_rejections,
        )
    else:
        sample_set = sample_block_smb_monte_carlo_split(
            distribution_id=config.monte_carlo_distribution_id,
            split=split,
            seed=int(config.monte_carlo_seed),
            sample_count=int(sample_count),
            family_weights=None,
            validate_reachability=config.monte_carlo_validate_reachability,
            max_rejections=config.monte_carlo_max_rejections,
        )
    model.eval()
    scenario_results: dict[str, dict[str, Any]] = {}
    family_rollups: dict[str, dict[str, Any]] = {}
    bin_rollups: dict[str, dict[str, Any]] = {}
    returns: list[float] = []
    successes: list[float] = []
    all_actions: list[int] = []
    primitive_jump_spans = 0
    primitive_jump_landings = 0
    primitive_duration_gaps: list[float] = []
    with torch.no_grad():
        for sample_index, sample in enumerate(sample_set.samples):
            scenario_returns: list[float] = []
            scenario_successes: list[float] = []
            scenario_actions: list[int] = []
            scenario_max_progress: list[float] = []
            for episode in range(config.evaluation_episodes):
                stage = BlockSMBStage(
                    env=MarioScenarioEnv(reward_config=config.reward_config),
                    scenario=copy.deepcopy(dict(sample.scenario)),
                    vision=vision_factory(),
                )
                try:
                    trajectory = collect_trajectory(
                        model,
                        stage,
                        sample.scenario_id,
                        rollout_steps=config.evaluation_max_steps,
                        seed=int(sample.sample_seed % (2**31)) + episode,
                        deterministic=True,
                        device=device,
                        record_frames=record_dir is not None,
                        ablation=config.ablation,
                        adaptive_duration_control=config.adaptive_duration_control,
                    )
                finally:
                    stage.env.close()
                actions = [step.action for step in trajectory.transitions]
                max_progress = (
                    max(
                        float(step.info.get("max_x_reached", 0.0))
                        for step in trajectory.transitions
                    )
                    if trajectory.transitions
                    else 0.0
                )
                scenario_returns.append(trajectory.total_return)
                scenario_successes.append(float(trajectory.success))
                scenario_actions.extend(actions)
                scenario_max_progress.append(max_progress)
                # HSP1 primitive metrics: landing rate and duration
                # calibration of jump spans against hindsight targets.
                for span in trajectory.spans:
                    if span.level != "motor_primitive":
                        continue
                    if span.command.get("primitive") != "jump":
                        continue
                    primitive_jump_spans += 1
                    if span.termination_reason == "success":
                        primitive_jump_landings += 1
                    start_info = trajectory.transitions[span.start_frame].info
                    target_hold = (
                        start_info.get("primitive_target_hold")
                        if isinstance(start_info, Mapping)
                        else None
                    )
                    held = span.command.get("held_frames")
                    if target_hold is not None and held is not None:
                        primitive_duration_gaps.append(
                            abs(float(held) - float(target_hold))
                            / _SMB_MAX_DURATION_BIN_VALUE
                        )
                if record_dir is not None:
                    split_record_dir = record_dir / f"monte_carlo_{split}"
                    split_record_dir.mkdir(parents=True, exist_ok=True)
                    frames = np.stack(trajectory.frames) if trajectory.frames else np.empty((0,))
                    np.savez_compressed(
                        split_record_dir / f"{sample.scenario_id}_episode{episode}.npz",
                        frames=frames,
                        actions=np.array(actions, dtype=np.int64),
                        rewards=np.array(
                            [step.reward for step in trajectory.transitions],
                            dtype=np.float32,
                        ),
                    )
            success_rate = float(np.mean(scenario_successes)) if scenario_successes else 0.0
            mean_return = float(np.mean(scenario_returns)) if scenario_returns else 0.0
            max_progress = float(max(scenario_max_progress)) if scenario_max_progress else 0.0
            action_counts = summarize_block_smb_monte_carlo_action_counts(scenario_actions)
            result = {
                "scenario_id": sample.scenario_id,
                "family": sample.family,
                "split": sample.split,
                "sample_index": sample.sample_index,
                "difficulty_bin": sample.difficulty_bin,
                "parameters": dict(sample.parameters),
                "return": mean_return,
                "success_rate": success_rate,
                "episodes": config.evaluation_episodes,
                "max_progress": max_progress,
                "action_counts": action_counts,
            }
            scenario_results[sample.scenario_id] = result
            returns.extend(scenario_returns)
            successes.extend(scenario_successes)
            all_actions.extend(scenario_actions)
            _add_monte_carlo_rollup(
                family_rollups,
                sample.family,
                result,
                scenario_actions,
            )
            _add_monte_carlo_rollup(
                bin_rollups,
                f"{sample.family}:{sample.difficulty_bin}",
                result,
                scenario_actions,
            )

    families = _finalize_monte_carlo_rollups(family_rollups)
    bins = _finalize_monte_carlo_rollups(bin_rollups)
    failure_bins = {
        bin_name: {
            "sample_count": rollup["sample_count"],
            "failure_count": rollup["failure_count"],
            "success_rate": rollup["success_rate"],
            "failures": rollup["failures"],
        }
        for bin_name, rollup in bins.items()
        if int(rollup["failure_count"]) > 0
    }
    evaluation: dict[str, Any] = {
        "schema_version": sample_set.schema_version,
        "distribution_id": sample_set.distribution_id,
        "split": sample_set.split,
        "seed": sample_set.seed,
        "sample_count": sample_set.sample_count,
        "requested_sample_count": int(sample_count),
        "parameter_sweep": bool(config.monte_carlo_parameter_sweep),
        "sweep_repeats_per_difficulty": int(config.monte_carlo_sweep_repeats_per_difficulty),
        "evaluation_episodes": config.evaluation_episodes,
        "evaluation_max_steps": config.evaluation_max_steps,
        "scenarios": scenario_results,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "coverage": sample_set.manifest()["coverage"],
        "rejected_counts": dict(sample_set.rejected_counts),
        "rejected_sample_count": int(sum(sample_set.rejected_counts.values())),
        "families": families,
        "difficulty_bins": bins,
        "failure_bins": failure_bins,
        "action_counts": summarize_block_smb_monte_carlo_action_counts(all_actions),
        "scenario_ids": [sample.scenario_id for sample in sample_set.samples],
    }
    evaluation["action_collapse"] = {
        "all_noop": block_smb_action_counts_all_noop(evaluation["action_counts"]),
        **action_distribution_stats(
            evaluation["action_counts"], action_count=BLOCK_SMB_ACTION_COUNT
        ),
    }
    evaluation["gates"] = evaluate_block_smb_monte_carlo_gates(
        evaluation,
        pass_rate_gate=config.monte_carlo_pass_rate_gate,
        family_pass_rate_gate=config.monte_carlo_family_pass_rate_gate,
    )
    evaluation["primitive_metrics"] = {
        "jump_spans": float(primitive_jump_spans),
        "landing_rate": (
            primitive_jump_landings / primitive_jump_spans
            if primitive_jump_spans
            else 0.0
        ),
        "duration_gap_mean": (
            sum(primitive_duration_gaps) / len(primitive_duration_gaps)
            if primitive_duration_gaps
            else 0.0
        ),
    }
    return evaluation


def _add_monte_carlo_rollup(
    rollups: dict[str, dict[str, Any]],
    key: str,
    result: Mapping[str, Any],
    actions: list[int],
) -> None:
    rollup = rollups.setdefault(
        key,
        {
            "returns": [],
            "success_rates": [],
            "scenario_ids": [],
            "failures": [],
            "actions": [],
        },
    )
    rollup["returns"].append(float(result.get("return", 0.0)))
    success_rate = float(result.get("success_rate", 0.0))
    rollup["success_rates"].append(success_rate)
    scenario_id = str(result.get("scenario_id", ""))
    rollup["scenario_ids"].append(scenario_id)
    rollup["actions"].extend(actions)
    if success_rate < 1.0:
        rollup["failures"].append(
            {
                "scenario_id": scenario_id,
                "success_rate": success_rate,
                "return": float(result.get("return", 0.0)),
                "max_progress": float(result.get("max_progress", 0.0)),
                "action_counts": dict(result.get("action_counts", {})),
            }
        )


def _finalize_monte_carlo_rollups(
    rollups: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key, rollup in rollups.items():
        returns = [float(value) for value in rollup.get("returns", [])]
        success_rates = [float(value) for value in rollup.get("success_rates", [])]
        failures = list(rollup.get("failures", []))
        actions = [int(action) for action in rollup.get("actions", [])]
        finalized[str(key)] = {
            "sample_count": len(success_rates),
            "scenario_ids": list(rollup.get("scenario_ids", [])),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "success_rate": float(np.mean(success_rates)) if success_rates else 0.0,
            "failure_count": len(failures),
            "failures": failures,
            "action_counts": summarize_block_smb_monte_carlo_action_counts(actions),
        }
    return finalized


def block_smb_action_counts_all_noop(action_counts: Mapping[str, Any]) -> bool:
    """Return true when a deterministic action summary contains only NOOP actions."""

    counts: dict[int, float] = {}
    for raw_action, raw_count in action_counts.items():
        try:
            action = int(raw_action)
            count = float(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0.0:
            counts[action] = counts.get(action, 0.0) + count
    total = sum(counts.values())
    return total > 0.0 and counts.get(0, 0.0) == total


def block_smb_action_count_metric_values(
    prefix: str,
    action_counts: Mapping[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for action_index in range(BLOCK_SMB_ACTION_COUNT):
        try:
            count = float(action_counts.get(str(action_index), 0.0))
        except (TypeError, ValueError):
            count = 0.0
        metrics[f"{prefix}_action_count_{action_index}"] = count
    metrics[f"{prefix}_all_noop_action_collapse"] = float(
        block_smb_action_counts_all_noop(action_counts)
    )
    stats = action_distribution_stats(action_counts, action_count=BLOCK_SMB_ACTION_COUNT)
    metrics[f"{prefix}_action_entropy"] = stats["normalized_entropy"]
    metrics[f"{prefix}_action_dominant_share"] = stats["dominant_share"]
    metrics[f"{prefix}_action_collapse"] = float(stats["collapsed"])
    return metrics


def block_smb_action_distribution_gate_metrics(
    prefix: str,
    action_counts: Mapping[str, Any],
    *,
    min_distinct_actions: int,
    max_dominant_fraction: float,
    required_actions: tuple[int, ...],
) -> dict[str, float]:
    """Summarize whether rollout actions avoid degenerate single-action collapse."""

    counts: dict[int, float] = {}
    for raw_action, raw_count in action_counts.items():
        try:
            action = int(raw_action)
            count = float(raw_count)
        except (TypeError, ValueError):
            continue
        if action < 0 or action >= BLOCK_SMB_ACTION_COUNT:
            continue
        counts[action] = max(0.0, count)
    total = float(sum(counts.values()))
    active = {action: count for action, count in counts.items() if count > 0.0}
    dominant_fraction = (max(active.values()) / total) if total > 0.0 and active else 0.0
    missing_required = sum(1 for action in required_actions if counts.get(action, 0.0) <= 0.0)
    distinct_gate_met = len(active) >= int(min_distinct_actions)
    dominant_gate_met = total > 0.0 and dominant_fraction <= float(max_dominant_fraction)
    required_gate_met = missing_required == 0
    gate_met = distinct_gate_met and dominant_gate_met and required_gate_met
    return {
        f"{prefix}_total_actions": total,
        f"{prefix}_distinct_actions": float(len(active)),
        f"{prefix}_dominant_action_fraction": float(dominant_fraction),
        f"{prefix}_missing_required_actions": float(missing_required),
        f"{prefix}_distinct_gate_met": float(distinct_gate_met),
        f"{prefix}_dominant_gate_met": float(dominant_gate_met),
        f"{prefix}_required_gate_met": float(required_gate_met),
        f"{prefix}_distribution_gate_met": float(gate_met),
    }


def evaluate_block_smb(
    model: torch.nn.Module,
    config: BlockSMBTrainingConfig,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
    record_dir: Optional[Path] = None,
) -> dict[str, Any]:
    model.eval()
    fixed = load_fixed_scenarios(config.fixed_scenarios)
    results = {}
    returns = []
    successes = []
    all_actions: list[int] = []
    with torch.no_grad():
        for scenario_index, (scenario_name, scenario) in enumerate(fixed):
            scenario_returns = []
            scenario_successes = []
            scenario_actions: list[int] = []
            for episode in range(config.evaluation_episodes):
                stage = BlockSMBStage(
                    env=MarioScenarioEnv(reward_config=config.reward_config),
                    scenario=scenario,
                    vision=vision_factory(),
                )
                try:
                    trajectory = collect_trajectory(
                        model,
                        stage,
                        scenario_name,
                        rollout_steps=config.evaluation_max_steps,
                        seed=config.seed + 1_000_000 + scenario_index * 100 + episode,
                        deterministic=True,
                        device=device,
                        record_frames=record_dir is not None,
                        ablation=config.ablation,
                        adaptive_duration_control=config.adaptive_duration_control,
                    )
                finally:
                    stage.env.close()
                actions = [step.action for step in trajectory.transitions]
                scenario_returns.append(trajectory.total_return)
                scenario_successes.append(float(trajectory.success))
                scenario_actions.extend(actions)
                all_actions.extend(actions)
                if record_dir is not None:
                    record_dir.mkdir(parents=True, exist_ok=True)
                    frames = np.stack(trajectory.frames) if trajectory.frames else np.empty((0,))
                    np.savez_compressed(
                        record_dir / f"{scenario_name}_episode{episode}.npz",
                        frames=frames,
                        actions=np.array(
                            [step.action for step in trajectory.transitions], dtype=np.int64
                        ),
                        rewards=np.array(
                            [step.reward for step in trajectory.transitions],
                            dtype=np.float32,
                        ),
                    )
            mean_return = float(np.mean(scenario_returns))
            success_rate = float(np.mean(scenario_successes))
            action_counts = summarize_block_smb_monte_carlo_action_counts(scenario_actions)
            results[scenario_name] = {
                "return": mean_return,
                "success_rate": success_rate,
                "action_counts": action_counts,
                "action_collapse": {
                    "all_noop": block_smb_action_counts_all_noop(action_counts),
                },
            }
            returns.extend(scenario_returns)
            successes.extend(scenario_successes)
    threshold_results = evaluate_fixed_success_thresholds(
        results,
        evaluation_episodes=config.evaluation_episodes,
        evaluation_max_steps=config.evaluation_max_steps,
    )
    tuning_metrics = summarize_fixed_success_metrics(results, threshold_results)
    for scenario_name, threshold_result in threshold_results.items():
        results[scenario_name]["threshold"] = threshold_result["threshold"]
        results[scenario_name]["threshold_met"] = threshold_result["threshold_met"]
        results[scenario_name]["threshold_diagnostics"] = {
            key: value
            for key, value in threshold_result.items()
            if key not in {"threshold", "threshold_met"}
        }
    evaluation: dict[str, Any] = {
        "fixed_scenarios": results,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "success_thresholds_met": (
            all(
                threshold_result["threshold_met"] for threshold_result in threshold_results.values()
            )
            if threshold_results
            else False
        ),
        "tuning_metrics": tuning_metrics,
        "action_counts": summarize_block_smb_monte_carlo_action_counts(all_actions),
    }
    evaluation["action_collapse"] = {
        "all_noop": block_smb_action_counts_all_noop(evaluation["action_counts"]),
        **action_distribution_stats(
            evaluation["action_counts"], action_count=BLOCK_SMB_ACTION_COUNT
        ),
    }
    if config.monte_carlo_validation_samples > 0 or config.monte_carlo_parameter_sweep:
        validation_record_dir = record_dir / "monte_carlo" if record_dir is not None else None
        evaluation["monte_carlo_validation"] = evaluate_block_smb_monte_carlo(
            model,
            config,
            split="validation",
            sample_count=(
                config.monte_carlo_validation_samples
                if config.monte_carlo_validation_samples > 0
                else block_smb_monte_carlo_sweep_sample_count(config)
            ),
            device=device,
            vision_factory=vision_factory,
            record_dir=validation_record_dir,
        )
    if config.monte_carlo_test_samples > 0:
        test_record_dir = record_dir / "monte_carlo" if record_dir is not None else None
        evaluation["monte_carlo_test"] = evaluate_block_smb_monte_carlo(
            model,
            config,
            split="test",
            sample_count=config.monte_carlo_test_samples,
            device=device,
            vision_factory=vision_factory,
            record_dir=test_record_dir,
        )
    return evaluation


def evaluate_block_smb_multi_seed(
    model: torch.nn.Module,
    config: BlockSMBTrainingConfig,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
    seed_count: int = DEFAULT_EVALUATION_SEED_COUNT,
) -> dict[str, Any]:
    """Evaluate across several seeds and report metric dispersion.

    Fixed-scenario evaluation seeds every episode from ``config.seed``, so a
    single run is one draw; go/no-go comparisons should look at the
    mean/std/min/max across seeds instead of a point estimate.
    """

    def _evaluate(seed: int) -> dict[str, Any]:
        evaluation = evaluate_block_smb(
            model,
            replace(config, seed=seed),
            device=device,
            vision_factory=vision_factory,
        )
        return {
            "success_rate": float(evaluation.get("success_rate", 0.0)),
            "mean_return": float(evaluation.get("mean_return", 0.0)),
            "success_thresholds_met": float(bool(evaluation.get("success_thresholds_met"))),
            "action_collapse": float(bool(evaluation.get("action_collapse", {}).get("collapsed"))),
        }

    return evaluate_over_seeds(
        _evaluate,
        base_seed=config.seed,
        seed_count=seed_count,
    )


def save_block_smb_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    config: BlockSMBTrainingConfig,
    metrics: Mapping[str, float],
    target_model: Optional[torch.nn.Module] = None,
) -> dict[str, Any]:
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    if target_model is not None:
        states["target_model"] = target_model.state_dict()
    checkpoint = build_checkpoint(
        stage=BLOCK_SMB_SPEC.name,
        model_name=BLOCK_SMB_MODEL_NAME,
        checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        config=to_plain_data(config),
        specs={
            "stage": {
                "name": BLOCK_SMB_SPEC.name,
                "seq_len_a": BLOCK_SMB_SPEC.seq_len_a,
                "seq_len_b": BLOCK_SMB_SPEC.seq_len_b,
                "seq_len_c": BLOCK_SMB_SPEC.seq_len_c,
                "ratio_bc": BLOCK_SMB_SPEC.ratio_bc,
                "vocab_size": BLOCK_SMB_SPEC.vocab_size,
                "action_count": BLOCK_SMB_SPEC.action_count,
            },
            **block_smb_architecture_specs(config),
        },
        states=states,
    )
    save_checkpoint(path, checkpoint)
    return checkpoint


def append_block_smb_log_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append one structured JSONL event for Block SMB operations."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"stage": BLOCK_SMB_SPEC.name, **dict(event)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_plain_data(record), sort_keys=True) + "\n")


def _initialize_block_smb_log(config: BlockSMBTrainingConfig) -> None:
    if config.log_path is None:
        return
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    if config.resume_path is None:
        config.log_path.write_text("", encoding="utf-8")


def _log_block_smb_event(
    training_config: BlockSMBTrainingConfig,
    event: str,
    **payload: Any,
) -> None:
    if training_config.log_path is None:
        return
    append_block_smb_log_event(training_config.log_path, {"event": event, **payload})


def _should_evaluate_epoch(config: BlockSMBTrainingConfig, completed_epoch: int) -> bool:
    return (
        completed_epoch == config.epochs or completed_epoch % config.evaluation_interval_epochs == 0
    )


def restore_block_smb_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    *,
    map_location: Any = "cpu",
    target_model: Optional[torch.nn.Module] = None,
    architecture_name: Optional[str] = None,
    architecture_config: Optional[Mapping[str, Any]] = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint["stage"] != BLOCK_SMB_SPEC.name:
        raise ValueError("checkpoint stage does not match block_smb")
    if checkpoint["model_name"] != BLOCK_SMB_MODEL_NAME:
        raise ValueError("checkpoint model does not match Block SMB trainer")
    if checkpoint["checkpoint_kind"] != BLOCK_SMB_CHECKPOINT_KIND:
        raise ValueError("checkpoint kind does not match Block SMB trainer")
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_architecture_name = checkpoint_config.get("architecture_name")
    if architecture_name is not None and checkpoint_architecture_name is not None:
        if str(checkpoint_architecture_name) != architecture_name:
            raise ValueError(
                "checkpoint architecture "
                f"{checkpoint_architecture_name!r} does not match "
                f"{architecture_name!r}"
            )
    checkpoint_architecture_config = checkpoint_config.get("architecture_config")
    if architecture_config is not None and checkpoint_architecture_config is not None:
        if dict(checkpoint_architecture_config) != dict(architecture_config):
            raise ValueError(
                "checkpoint architecture config "
                f"{checkpoint_architecture_config!r} does not match "
                f"{dict(architecture_config)!r}"
            )
    states = checkpoint["states"]
    model_state, skipped_world_model_keys = action_level_world_model_state_dict(
        model,
        states["model"],
    )
    load_result = model.load_state_dict(model_state, strict=False)
    allowed_missing_prefixes = (
        "transition_representation_head.",
        "reward_head.",
        "value_head.",
        *ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    )
    unexpected = list(load_result.unexpected_keys)
    unsupported_missing = [
        key for key in load_result.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or unsupported_missing:
        raise ValueError(
            "checkpoint model state is incompatible with Block SMB trainer; "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )
    if optimizer is not None:
        try:
            optimizer.load_state_dict(states["optimizer"])
        except ValueError:
            if unsupported_missing or (
                not load_result.missing_keys and not skipped_world_model_keys
            ):
                raise
    if target_model is not None:
        target_state = states.get("target_model", states["model"])
        target_state, _skipped_target_world_model_keys = action_level_world_model_state_dict(
            target_model,
            target_state,
        )
        target_model.load_state_dict(target_state, strict=False)
        target_model.eval()
    if restore_rng:
        if "torch_rng" in states:
            torch.set_rng_state(states["torch_rng"].cpu())
        if "python_rng" in states:
            random.setstate(states["python_rng"])
        if "numpy_rng" in states:
            np.random.set_state(states["numpy_rng"])
    return checkpoint


def train_and_evaluate_block_smb(
    config: Optional[BlockSMBTrainingConfig] = None,
    *,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
) -> dict[str, Any]:
    config = config or BlockSMBTrainingConfig()
    seed_everything(config.seed, config.deterministic)
    device = select_device(config.device)
    model = make_block_smb_model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    target_model = (
        make_target_network(model).to(device) if config.target_network_mode != "off" else None
    )
    start_epoch = 0
    global_step = 0
    _initialize_block_smb_log(config)
    if config.resume_path is not None:
        checkpoint = restore_block_smb_checkpoint(
            config.resume_path,
            model,
            optimizer,
            map_location=device,
            target_model=target_model,
            architecture_name=config.architecture_name,
            architecture_config=config.architecture_config,
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
    elif config.init_checkpoint is not None:
        # Weights-only warm start: model parameters come from the checkpoint,
        # everything else (optimizer, epochs, curriculum, mastery state) is
        # fresh so a new training regime can continue from learned skills.
        restore_block_smb_checkpoint(
            config.init_checkpoint,
            model,
            optimizer=None,
            map_location=device,
            architecture_name=config.architecture_name,
            architecture_config=config.architecture_config,
            restore_rng=False,
        )
        if target_model is not None:
            update_target_network(target_model, model, tau=1.0)
    elif target_model is not None:
        update_target_network(target_model, model, tau=1.0)
    mastery_state = initial_block_smb_mastery_state()
    mastery_phase = 0
    best_primitive_score = float("-inf")
    success_replay = BlockSMBSuccessReplay(
        max_episodes_per_family=config.success_replay_episodes_per_family,
        seed=config.seed,
    )
    if config.mastery_gated_schedule:
        curriculum = load_fixed_scenarios(config.fixed_scenarios)
        curriculum.extend(
            build_mastery_monte_carlo_curriculum(config, mastery_state, phase=mastery_phase)
        )
    else:
        curriculum = build_curriculum(config)
    vector_env = SequentialBlockSMBVectorEnv(
        curriculum,
        num_envs=config.num_envs,
        reward_config=config.reward_config,
    )
    vector_env.close()
    tracker = make_experiment_tracker(
        ExperimentTrackerConfig(
            backend=config.tracking_backend,
            log_dir=config.tracking_log_dir,
            project=config.tracking_project,
            run_name=config.tracking_run_name,
            mode=config.tracking_mode,
        ),
        default_log_dir=Path("artifacts/block_smb/tracking"),
    )
    tracker.log_config(to_plain_data(config))
    _log_block_smb_event(
        config,
        "run_started",
        config=to_plain_data(config),
        device=str(device),
        start_epoch=start_epoch,
        global_step=global_step,
        resumed_from=str(config.resume_path) if config.resume_path is not None else None,
        curriculum=[name for name, _scenario in curriculum],
        curriculum_summary=summarize_block_smb_curriculum(curriculum),
    )
    history: list[dict[str, float]] = []
    evaluations: list[dict[str, Any]] = []
    last_metrics: dict[str, float] = {}
    recent_monte_carlo_failure_bins: Mapping[str, Any] = {}
    for epoch in range(start_epoch, config.epochs):
        replay_curriculum = build_adaptive_monte_carlo_replay_curriculum(
            config,
            recent_monte_carlo_failure_bins,
            epoch=epoch,
        )
        epoch_curriculum = build_epoch_curriculum(curriculum, replay_curriculum)
        losses, _replay = train_block_smb_epoch(
            model,
            optimizer,
            epoch_curriculum,
            config,
            epoch,
            device=device,
            vision_factory=vision_factory,
            target_model=target_model,
            success_replay=success_replay,
        )
        losses["adaptive_replay_samples"] = float(len(replay_curriculum))
        global_step += int(losses["episodes"])
        completed_epoch = epoch + 1
        last_metrics = dict(losses)
        _log_block_smb_event(
            config,
            "train_epoch",
            epoch=completed_epoch,
            global_step=global_step,
            metrics=last_metrics,
            curriculum_summary=summarize_block_smb_curriculum(epoch_curriculum),
        )
        tracker.log_metrics(last_metrics, step=global_step, prefix="train")
        if _should_evaluate_epoch(config, completed_epoch):
            evaluation = evaluate_block_smb(
                model,
                config,
                device=device,
                vision_factory=vision_factory,
                record_dir=config.video_dir if config.record_videos else None,
            )
            evaluations.append(
                {
                    "epoch": completed_epoch,
                    "global_step": global_step,
                    "evaluation": evaluation,
                }
            )
            last_metrics = {
                **last_metrics,
                "eval_mean_return": float(evaluation["mean_return"]),
                "eval_success_rate": float(evaluation["success_rate"]),
                "eval_threshold_pass_rate": float(
                    evaluation["tuning_metrics"]["threshold_pass_rate"]
                ),
                "eval_tuning_score": float(evaluation["tuning_metrics"]["score"]),
            }
            last_metrics.update(
                block_smb_action_count_metric_values(
                    "eval_fixed",
                    evaluation.get("action_counts", {}),
                )
            )
            last_metrics.update(
                block_smb_action_distribution_gate_metrics(
                    "eval_fixed",
                    evaluation.get("action_counts", {}),
                    min_distinct_actions=config.action_gate_min_distinct_actions,
                    max_dominant_fraction=config.action_gate_max_dominant_fraction,
                    required_actions=config.action_gate_required_actions,
                )
            )
            last_metrics.update(block_smb_monte_carlo_eval_metrics(evaluation))
            monte_carlo_validation = evaluation.get("monte_carlo_validation", {})
            # HSP1: held-out primitive metrics select a best-primitives
            # checkpoint independent of episode return.
            if isinstance(monte_carlo_validation, Mapping):
                primitive_metrics = monte_carlo_validation.get("primitive_metrics", {})
                if isinstance(primitive_metrics, Mapping) and primitive_metrics.get(
                    "jump_spans", 0.0
                ):
                    landing_rate = float(primitive_metrics.get("landing_rate", 0.0))
                    duration_gap = float(
                        primitive_metrics.get("duration_gap_mean", 0.0)
                    )
                    primitive_score = landing_rate - duration_gap
                    last_metrics["eval_primitive_landing_rate"] = landing_rate
                    last_metrics["eval_primitive_duration_gap"] = duration_gap
                    if primitive_score > best_primitive_score and config.checkpoint_path:
                        best_primitive_score = primitive_score
                        best_path = Path(config.checkpoint_path).with_suffix(
                            ".best_primitives.pth"
                        )
                        if Path(config.checkpoint_path).exists():
                            import shutil

                            shutil.copyfile(config.checkpoint_path, best_path)
                            _log_block_smb_event(
                                config,
                                "best_primitive_checkpoint",
                                epoch=completed_epoch,
                                score=primitive_score,
                                landing_rate=landing_rate,
                                duration_gap_mean=duration_gap,
                                path=str(best_path),
                            )
            if isinstance(monte_carlo_validation, Mapping):
                failure_bins = monte_carlo_validation.get("failure_bins", {})
                if isinstance(failure_bins, Mapping):
                    recent_monte_carlo_failure_bins = failure_bins
            if config.mastery_gated_schedule and isinstance(monte_carlo_validation, Mapping):
                mastery_state = update_block_smb_mastery_state(
                    mastery_state,
                    monte_carlo_validation,
                    family_pass_rate_gate=config.monte_carlo_family_pass_rate_gate,
                )
                mastery_phase += 1
                curriculum = load_fixed_scenarios(config.fixed_scenarios)
                curriculum.extend(
                    build_mastery_monte_carlo_curriculum(
                        config,
                        mastery_state,
                        phase=mastery_phase,
                    )
                )
                mastery_summary = summarize_block_smb_mastery_state(mastery_state)
                last_metrics["eval_mastered_family_count"] = float(
                    len(mastery_summary["mastered_families"])
                )
                _log_block_smb_event(
                    config,
                    "mastery_schedule_updated",
                    epoch=completed_epoch,
                    global_step=global_step,
                    phase=mastery_phase,
                    mastery=mastery_summary,
                    family_weights=block_smb_mastery_family_weights(
                        mastery_state,
                        family_pass_rate_gate=config.monte_carlo_family_pass_rate_gate,
                        retention_weight=config.mastery_retention_weight,
                        retention_grace_evals=config.mastery_retention_grace_evals,
                    ),
                    curriculum_summary=summarize_block_smb_curriculum(curriculum),
                )
            _log_block_smb_event(
                config,
                "deterministic_evaluation",
                epoch=completed_epoch,
                global_step=global_step,
                metrics={
                    key: value for key, value in last_metrics.items() if key.startswith("eval_")
                },
                evaluation=evaluation,
            )
            tracker.log_metrics(
                {key: value for key, value in last_metrics.items() if key.startswith("eval_")},
                step=global_step,
                prefix="eval",
            )
        history.append(last_metrics)
        if config.save_checkpoints and config.checkpoint_path is not None:
            save_block_smb_checkpoint(
                config.checkpoint_path,
                model,
                optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=config,
                metrics=last_metrics,
                target_model=target_model,
            )
            _log_block_smb_event(
                config,
                "checkpoint_saved",
                epoch=completed_epoch,
                global_step=global_step,
                checkpoint_path=str(config.checkpoint_path),
                metrics=last_metrics,
            )
    if evaluations and evaluations[-1]["epoch"] == config.epochs:
        evaluation = evaluations[-1]["evaluation"]
    else:
        evaluation = evaluate_block_smb(
            model,
            config,
            device=device,
            vision_factory=vision_factory,
            record_dir=config.video_dir if config.record_videos else None,
        )
        evaluations.append(
            {
                "epoch": config.epochs,
                "global_step": global_step,
                "evaluation": evaluation,
            }
        )
    evaluation = apply_block_smb_semantic_prediction_gate(
        evaluation,
        last_metrics,
        threshold=config.semantic_prediction_accuracy_threshold,
    )
    _log_block_smb_event(
        config,
        "run_finished",
        epoch=config.epochs,
        global_step=global_step,
        metrics=last_metrics,
        evaluation=evaluation,
    )
    tracker.close()
    return {
        "history": history,
        "evaluations": evaluations,
        "metrics": last_metrics,
        "evaluation": evaluation,
        "curriculum": [name for name, _scenario in curriculum],
        "curriculum_summary": summarize_block_smb_curriculum(curriculum),
        "architecture": block_smb_architecture_metadata(config),
        "model": model,
    }


def block_smb_monte_carlo_eval_metrics(evaluation: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, prefix in (
        ("monte_carlo_validation", "eval_monte_carlo_validation"),
        ("monte_carlo_test", "eval_monte_carlo_test"),
    ):
        result = evaluation.get(key)
        if not isinstance(result, Mapping):
            continue
        gates = result.get("gates", {})
        metrics[f"{prefix}_success_rate"] = float(result.get("success_rate", 0.0))
        metrics[f"{prefix}_mean_return"] = float(result.get("mean_return", 0.0))
        metrics[f"{prefix}_gate_met"] = float(
            bool(gates.get("gate_met", False)) if isinstance(gates, Mapping) else False
        )
        metrics[f"{prefix}_family_gate_met"] = float(
            bool(gates.get("family_pass_rate_gate_met", False))
            if isinstance(gates, Mapping)
            else False
        )
        metrics.update(
            block_smb_action_count_metric_values(
                prefix,
                result.get("action_counts", {}),
            )
        )
    return metrics


def apply_block_smb_semantic_prediction_gate(
    evaluation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Gate scenario success on the learned dynamics semantic prediction metric."""

    gated = copy.deepcopy(dict(evaluation))
    semantic_accuracy = _optional_float(metrics.get("dynamics_semantic_prediction_accuracy"))
    if semantic_accuracy is None:
        gate_met = False
    else:
        gate_met = semantic_accuracy >= float(threshold)
    tuning = dict(gated.get("tuning_metrics", {}))
    tuning.update(
        {
            "semantic_prediction_accuracy": semantic_accuracy,
            "semantic_prediction_accuracy_threshold": float(threshold),
            "semantic_prediction_gate_met": bool(gate_met),
        }
    )
    gated["tuning_metrics"] = tuning
    gated["semantic_prediction_gate_met"] = bool(gate_met)
    gated["semantic_prediction_accuracy"] = semantic_accuracy
    gated["semantic_prediction_accuracy_threshold"] = float(threshold)
    gated["success_thresholds_met"] = bool(gated.get("success_thresholds_met")) and bool(gate_met)
    return gated


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
