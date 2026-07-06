"""Training utilities for the Block SMB stage."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import (
    ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    BASELINE_ARCHITECTURE_NAME,
    POLICY_TUPLE_OUTPUT_CONTRACTS,
    SUPPORTED_CONTROLLER_SCHEDULES,
    TRACKING_BACKENDS,
    ExperimentTrackerConfig,
    SMBParameterizedPrimitiveExecutor,
    SMBPrimitiveExecution,
    StageBatch,
    VisionEncoder,
    WorldModelState,
    action_level_world_model_state_dict,
    build_architecture,
    build_checkpoint,
    get_architecture,
    is_smb_jump_action,
    load_checkpoint,
    make_experiment_tracker,
    save_checkpoint,
    select_device,
    smb_jump_release_action,
    to_plain_data,
)

from .adapter import BLOCK_SMB_SPEC, SCENARIOS_DIR, BlockSMBStage
from .env import BlockSMBRewardConfig, MarioScenarioEnv
from .monte_carlo import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID,
    block_smb_monte_carlo_metadata,
    block_smb_monte_carlo_oracle_actions,
    evaluate_block_smb_monte_carlo_gates,
    sample_block_smb_monte_carlo_parameter_sweep,
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
    noop_loss_weight: float = 0.25
    critic_loss_weight: float = 0.001
    imagined_rollout_weight: float = 0.0
    imagined_rollout_horizon: int = 0
    target_network_mode: str = "off"
    target_network_tau: float = 0.01
    target_network_instability_threshold: float = 1.0
    gradient_clip_norm: float = 1.0
    hidden_dim: int = 32
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
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 200
    cover_curriculum_per_epoch: bool = True
    update_batch_episodes: int = 16
    action_gate_min_distinct_actions: int = 2
    action_gate_max_dominant_fraction: float = 0.95
    action_gate_required_actions: tuple[int, ...] = (1, 2)
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
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


@dataclass
class BlockSMBTrajectory:
    scenario_name: str
    transitions: list[BlockSMBTransition] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)

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
    return build_architecture(
        config.architecture_name,
        BLOCK_SMB_SPEC,
        dict(config.architecture_config),
    )


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


def _goal_reached(env: MarioScenarioEnv) -> bool:
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
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
    WorldModelState | None,
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
    action_tensor = (
        oracle_action_tensor
        if oracle_action_tensor is not None
        else (action_logits.argmax(dim=-1) if deterministic else distribution.sample())
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
    return (
        int(action_tensor.item()),
        log_prob,
        entropy,
        primitive_aux_loss,
        (actions1, actions2, next_state_pred, criticism, logits_a),
        next_world_model_state,
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
) -> BlockSMBTrajectory:
    ablation_config = _ablation_config(ablation)
    observation = stage.reset(seed=seed)
    trajectory = BlockSMBTrajectory(scenario_name=scenario_name)
    if record_frames:
        trajectory.frames.append(np.asarray(observation).copy())
    world_model_state: WorldModelState | None = None
    primitive_executor = SMBParameterizedPrimitiveExecutor()
    oracle_actions = (
        block_smb_oracle_actions_for_rollout(
            stage.scenario,
            rollout_steps=rollout_steps,
        )
        if use_oracle_actions
        else ()
    )

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
        (
            action,
            log_prob,
            entropy,
            primitive_aux_loss,
            outputs,
            next_world_model_state,
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
            )
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
    returns = discounted_returns(
        [step.reward for step in transitions],
        [step.episode_mask for step in transitions],
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
    oracle_supervised_steps = 0
    for index, step in enumerate(transitions):
        return_target = returns[index].view(1)
        reward_target = torch.tensor([step.reward], dtype=torch.float32, device=device)
        value_pred = model.predict_value(step.batch.src_c.detach())
        reward_pred = model.predict_reward(step.next_state_pred)
        predicted_representation = model.transition_representation(step.next_state_pred)
        with torch.no_grad():
            target_representation = target_model_for_loss.transition_representation(
                step.next_batch.src_c.detach()
            )
        advantage = (return_target - value_pred.detach()).detach()

        policy_terms.append(-step.log_prob * advantage.squeeze(0))
        entropy_terms.append(step.entropy)
        representation_terms.append(F.mse_loss(predicted_representation, target_representation))
        next_state_target = step.next_batch.src_c.detach()
        slot_losses = block_smb_c_stream_dynamics_slot_losses(
            step.next_state_pred,
            next_state_target,
            step.next_batch,
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
            step.next_batch,
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
    loss_representation = torch.stack(representation_terms).mean()
    loss_dynamics = torch.stack(dynamics_terms).mean()
    loss_reward = torch.stack(reward_terms).mean()
    loss_value = torch.stack(value_terms).mean()
    loss_policy = torch.stack(policy_terms).mean()
    loss_action_aux = torch.stack(action_aux_terms).mean()
    loss_oracle_action = torch.stack(oracle_action_terms).mean()
    loss_noop = torch.stack(noop_terms).mean()
    loss_critic_feedback = torch.stack(critic_terms).mean()
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
                use_oracle_actions=True,
            )
        finally:
            stage.env.close()
        replay.add(trajectory)
        total_returns.append(trajectory.total_return)
        all_actions.extend(step.action for step in trajectory.transitions)
        if len(replay.trajectories) >= update_batch_size:
            flush_update_batch()

    flush_update_batch()
    if total_update_episodes <= 0:
        raise ValueError("no Block SMB rollout episodes were collected")
    epoch_losses = {
        key: value / float(total_update_episodes) for key, value in metric_totals.items()
    }
    epoch_losses["mean_return"] = float(np.mean(total_returns)) if total_returns else 0.0
    epoch_losses["episodes"] = float(total_update_episodes)
    epoch_losses["optimizer_updates"] = float(update_count)
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
    }
    evaluation["gates"] = evaluate_block_smb_monte_carlo_gates(
        evaluation,
        pass_rate_gate=config.monte_carlo_pass_rate_gate,
        family_pass_rate_gate=config.monte_carlo_family_pass_rate_gate,
    )
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
    elif target_model is not None:
        update_target_network(target_model, model, tau=1.0)
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
            if isinstance(monte_carlo_validation, Mapping):
                failure_bins = monte_carlo_validation.get("failure_bins", {})
                if isinstance(failure_bins, Mapping):
                    recent_monte_carlo_failure_bins = failure_bins
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
