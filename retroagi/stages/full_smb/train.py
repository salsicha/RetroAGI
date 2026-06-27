"""Direct Full SMB policy training, evaluation, resume, and checkpointing."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    CHECKPOINT_SCHEMA_KEY,
    SMB_ACTIONS,
    SMBAction,
    TRACKING_BACKENDS,
    ExperimentTrackerConfig,
    StageBatch,
    WorldModelState,
    build_checkpoint,
    load_checkpoint,
    make_experiment_tracker,
    save_checkpoint,
    select_device,
    to_plain_data,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.block_smb.train import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
)
from retroagi.stages.full_smb.adapter import (
    DEFAULT_FULL_SMB_CONTENT,
    FULL_SMB_SPEC,
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBRewardConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.success import (
    FIXED_FULL_SMB_SUCCESS_THRESHOLDS,
    evaluate_fixed_full_smb_success_thresholds,
    evaluate_full_smb_success_threshold,
    summarize_fixed_full_smb_success_metrics,
)
from retroagi.stages.full_smb.tasks import (
    FULL_SMB_TASK_SET_NAMES,
    FullSMBTaskSpec,
    full_smb_task_catalog,
)
from retroagi.stages.full_smb.transfer import (
    FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    FULL_SMB_TRANSFER_MODEL_NAME,
    load_transferred_full_smb_policy,
    make_full_smb_policy_model,
    policy_architecture_from_checkpoint,
    transfer_block_smb_checkpoint_to_full_smb,
)
from retroagi.stages.full_smb.vision import (
    DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    FullSMBSegmentationVision,
)

FULL_SMB_POLICY_MODEL_NAME = "full_smb_policy"
FULL_SMB_POLICY_CHECKPOINT_KIND = "full_smb_policy_trainer"
FULL_SMB_ACTION_COUNT = len(SMB_ACTIONS)
FULL_SMB_TRAINING_SOURCE_SCRATCH = "scratch"
FULL_SMB_TRAINING_SOURCE_INIT_CHECKPOINT = "init_checkpoint"
FULL_SMB_TRAINING_SOURCE_RESUME_CHECKPOINT = "resume_checkpoint"
FULL_SMB_TRAINING_MODE_AUTO = "auto"
FULL_SMB_TRAINING_MODE_SCRATCH = "scratch"
FULL_SMB_TRAINING_MODE_FINE_TUNE = "fine_tune"
FULL_SMB_TRAINING_MODES = (
    FULL_SMB_TRAINING_MODE_AUTO,
    FULL_SMB_TRAINING_MODE_SCRATCH,
    FULL_SMB_TRAINING_MODE_FINE_TUNE,
)
_FULL_SMB_TRAINING_MODE_ALIASES = {
    FULL_SMB_TRAINING_MODE_AUTO: FULL_SMB_TRAINING_MODE_AUTO,
    FULL_SMB_TRAINING_MODE_SCRATCH: FULL_SMB_TRAINING_MODE_SCRATCH,
    "fine-tune": FULL_SMB_TRAINING_MODE_FINE_TUNE,
    FULL_SMB_TRAINING_MODE_FINE_TUNE: FULL_SMB_TRAINING_MODE_FINE_TUNE,
    "finetune": FULL_SMB_TRAINING_MODE_FINE_TUNE,
}
FULL_SMB_INIT_SOURCE_BLOCK_POLICY = "block_smb_policy_checkpoint"
FULL_SMB_INIT_SOURCE_TRANSFER = "full_smb_transfer_checkpoint"
FULL_SMB_PERCEPTION_FREEZE = "freeze"
FULL_SMB_PERCEPTION_FINE_TUNE = "fine_tune"
FULL_SMB_PERCEPTION_REPLACE = "replace"
FULL_SMB_PERCEPTION_MODES = (
    FULL_SMB_PERCEPTION_FREEZE,
    FULL_SMB_PERCEPTION_FINE_TUNE,
    FULL_SMB_PERCEPTION_REPLACE,
)
_FULL_SMB_PERCEPTION_ALIASES = {
    FULL_SMB_PERCEPTION_FREEZE: FULL_SMB_PERCEPTION_FREEZE,
    "frozen": FULL_SMB_PERCEPTION_FREEZE,
    "fine-tune": FULL_SMB_PERCEPTION_FINE_TUNE,
    FULL_SMB_PERCEPTION_FINE_TUNE: FULL_SMB_PERCEPTION_FINE_TUNE,
    "finetune": FULL_SMB_PERCEPTION_FINE_TUNE,
    FULL_SMB_PERCEPTION_REPLACE: FULL_SMB_PERCEPTION_REPLACE,
    "scratch": FULL_SMB_PERCEPTION_REPLACE,
}
_FULL_SMB_ROLLOUT_STORAGE_FIELDS = (
    "step_index",
    "action",
    "action_name",
    "reward",
    "done",
    "terminated",
    "truncated",
    "episode_mask",
    "boundary_reasons",
    "scenario_id",
    "task_id",
    "emulator_state_id",
    "signals",
)
_FULL_SMB_SELECTED_SIGNAL_FIELDS = (
    "position",
    "progress",
    "score",
    "coins",
    "collectibles",
    "lives",
    "screen",
    "level",
    "world",
    "stage",
    "power_state",
    "completion",
    "death",
    "timeout",
    "game_over",
    "objectives",
    "terminated",
    "truncated",
    "termination_reason",
)
_FULL_SMB_DEFAULT_MAX_ABS_LOSS = 1_000_000.0
_FULL_SMB_DEFAULT_MAX_ABS_SCALED_REWARD = 1_000.0
_FULL_SMB_DEFAULT_MAX_ABS_PREDICTION = 1_000_000.0
_FULL_SMB_DEFAULT_WORLD_MODEL_WEIGHT = 0.05
_FULL_SMB_RECORDING_SCHEMA_VERSION = 1
_FULL_SMB_VIDEO_SUFFIXES = (".avi", ".mkv", ".mov", ".mp4")
DEFAULT_FULL_SMB_RECORDING_DIR = Path("artifacts/full_smb/recordings")
DEFAULT_FULL_SMB_RECORDING_MANIFEST = Path("artifacts/full_smb/recording_manifest.npz")
_FULL_SMB_FIXED_LEVEL_TASK_NAMES = {
    "level1-1": "benchmark_1_1_start",
    "1-1": "benchmark_1_1_start",
    "level1-2": "benchmark_1_2_start",
    "1-2": "benchmark_1_2_start",
    "level2-1": "benchmark_2_1_start",
    "2-1": "benchmark_2_1_start",
}
_FULL_SMB_PLAY_RENDER_MODES = ("human", "none")


def _validate_full_smb_action_id(action: Any) -> int:
    try:
        action_id = int(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Full SMB action id {action!r}") from exc
    if action_id < 0 or action_id >= FULL_SMB_ACTION_COUNT:
        valid = ", ".join(f"{int(action)}={action.name.lower()}" for action in SMB_ACTIONS)
        raise ValueError(f"invalid Full SMB action id {action_id}; expected one of: {valid}")
    return action_id


def _empty_full_smb_recording_manifest() -> dict[str, Any]:
    return {
        "schema_version": _FULL_SMB_RECORDING_SCHEMA_VERSION,
        "enabled": False,
        "recording_dir": None,
        "recording_path": None,
        "recording_prefix": None,
        "artifact_count": 0,
        "artifacts": [],
        "manifest_path": None,
        "video_export": {"enabled": False, "artifacts": []},
    }


def _resolve_perception_mode(
    mode: Optional[str],
    *,
    freeze_vision: bool,
) -> str:
    if mode is None:
        return FULL_SMB_PERCEPTION_FREEZE if freeze_vision else FULL_SMB_PERCEPTION_FINE_TUNE
    normalized = _FULL_SMB_PERCEPTION_ALIASES.get(str(mode).strip().lower())
    if normalized is None:
        choices = ", ".join(FULL_SMB_PERCEPTION_MODES)
        raise ValueError(f"perception_mode must be one of {choices}")
    return normalized


def _resolve_training_mode(mode: Optional[str]) -> str:
    normalized = _FULL_SMB_TRAINING_MODE_ALIASES.get(str(mode or "auto").strip().lower())
    if normalized is None:
        choices = ", ".join(FULL_SMB_TRAINING_MODES)
        raise ValueError(f"training_mode must be one of {choices}")
    return normalized


def _loss_weight_metadata(config: Any) -> dict[str, float]:
    return {
        "policy": float(config.policy_loss_weight),
        "entropy": float(config.entropy_weight),
        "representation": float(config.representation_weight),
        "world_model": float(config.world_model_weight),
        "reward": float(config.reward_loss_weight),
        "value": float(config.value_loss_weight),
        "action_aux": float(config.action_aux_weight),
        "critic": float(config.critic_loss_weight),
    }


@dataclass(frozen=True)
class FullSMBTrainingConfig:
    """Configuration for direct emulator-level Full SMB policy updates."""

    seed: int = 0
    training_mode: str = FULL_SMB_TRAINING_MODE_AUTO
    architecture_name: str = BASELINE_ARCHITECTURE_NAME
    architecture_config: Mapping[str, Any] = field(default_factory=dict)
    epochs: int = 1
    episodes_per_epoch: int = 1
    max_steps_per_episode: int = 64
    updates_per_epoch: Optional[int] = None
    rollout_length: Optional[int] = None
    vector_env_count: int = 1
    learning_rate: float = 1e-4
    entropy_weight: float = 0.01
    policy_loss_weight: float = 1.0
    representation_weight: float = 0.0
    world_model_weight: float = _FULL_SMB_DEFAULT_WORLD_MODEL_WEIGHT
    reward_loss_weight: float = 0.0
    value_loss_weight: float = 0.0
    action_aux_weight: float = 0.0
    critic_loss_weight: float = 0.0
    reward_scale: float = 1.0
    reward_config: FullSMBRewardConfig | Mapping[str, float] = field(
        default_factory=FullSMBRewardConfig
    )
    gradient_clip_norm: float = 1.0
    max_abs_loss: float = _FULL_SMB_DEFAULT_MAX_ABS_LOSS
    max_abs_scaled_reward: float = _FULL_SMB_DEFAULT_MAX_ABS_SCALED_REWARD
    max_abs_prediction: float = _FULL_SMB_DEFAULT_MAX_ABS_PREDICTION
    deterministic: bool = True
    deterministic_actions: bool = False
    device: str = "auto"
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 64
    evaluation_interval_epochs: int = 1
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    init_checkpoint: Optional[Path] = None
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT
    perception_mode: Optional[str] = None
    freeze_vision: bool = True
    game_id: str = DEFAULT_FULL_SMB_CONTENT.game
    task_set: Optional[str] = None
    task_name: Optional[str] = None
    level: Optional[str] = None
    emulator_state: Optional[str] = None
    scenario: Optional[str] = None
    frame_skip: Optional[int] = None
    save_checkpoints: bool = False
    output_summary: Optional[Path] = None
    log_path: Optional[Path] = None
    recording_dir: Optional[Path] = None
    recording_path: Optional[Path] = None
    tracking_backend: str = "none"
    tracking_log_dir: Optional[Path] = None
    tracking_project: str = "retroagi"
    tracking_run_name: Optional[str] = None
    tracking_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.architecture_name:
            raise ValueError("architecture_name must be non-empty")
        if any(not str(key) for key in self.architecture_config):
            raise ValueError("architecture_config keys must be non-empty")
        if isinstance(self.reward_config, Mapping):
            object.__setattr__(
                self,
                "reward_config",
                FullSMBRewardConfig(**dict(self.reward_config)),
            )
        elif not isinstance(self.reward_config, FullSMBRewardConfig):
            raise TypeError("reward_config must be a FullSMBRewardConfig or mapping")
        rollout_length = (
            self.max_steps_per_episode if self.rollout_length is None else self.rollout_length
        )
        updates_per_epoch = (
            self.episodes_per_epoch if self.updates_per_epoch is None else self.updates_per_epoch
        )
        object.__setattr__(self, "rollout_length", int(rollout_length))
        object.__setattr__(self, "max_steps_per_episode", int(rollout_length))
        object.__setattr__(self, "updates_per_epoch", int(updates_per_epoch))
        object.__setattr__(self, "episodes_per_epoch", int(updates_per_epoch))
        for name in (
            "epochs",
            "episodes_per_epoch",
            "updates_per_epoch",
            "max_steps_per_episode",
            "rollout_length",
            "evaluation_episodes",
            "evaluation_max_steps",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.evaluation_interval_epochs <= 0:
            raise ValueError("evaluation_interval_epochs must be positive")
        if self.vector_env_count <= 0:
            raise ValueError("vector_env_count must be positive")
        for name in (
            "learning_rate",
            "reward_scale",
            "gradient_clip_norm",
            "max_abs_loss",
            "max_abs_scaled_reward",
            "max_abs_prediction",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        loss_weights = _loss_weight_metadata(self).values()
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("loss weights must be non-negative")
        object.__setattr__(self, "tracking_backend", self.tracking_backend.lower())
        if self.tracking_backend not in TRACKING_BACKENDS:
            raise ValueError(f"tracking_backend must be one of {TRACKING_BACKENDS}")
        if not self.tracking_project:
            raise ValueError("tracking_project must be non-empty")
        if not str(self.game_id).strip():
            raise ValueError("game_id must be non-empty")
        if self.task_set is not None and self.task_set not in FULL_SMB_TASK_SET_NAMES:
            raise ValueError(f"task_set must be one of {FULL_SMB_TASK_SET_NAMES}")
        for name in ("task_name", "level", "emulator_state", "scenario"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when provided")
        if self.frame_skip is not None:
            if int(self.frame_skip) <= 0:
                raise ValueError("frame_skip must be positive")
            object.__setattr__(self, "frame_skip", int(self.frame_skip))
        for path_name in (
            "checkpoint_path",
            "resume_path",
            "init_checkpoint",
            "full_smb_vision_checkpoint",
            "output_summary",
            "log_path",
            "recording_dir",
            "recording_path",
            "tracking_log_dir",
        ):
            path_value = getattr(self, path_name)
            if path_value is not None and not isinstance(path_value, Path):
                object.__setattr__(self, path_name, Path(path_value))
        if self.resume_path is not None and self.init_checkpoint is not None:
            raise ValueError("resume_path and init_checkpoint are mutually exclusive")
        training_mode = _resolve_training_mode(self.training_mode)
        object.__setattr__(self, "training_mode", training_mode)
        if training_mode == FULL_SMB_TRAINING_MODE_SCRATCH:
            if self.resume_path is not None:
                raise ValueError("training_mode='scratch' cannot be used with resume_path")
            if self.init_checkpoint is not None:
                raise ValueError("training_mode='scratch' cannot be used with init_checkpoint")
        if training_mode == FULL_SMB_TRAINING_MODE_FINE_TUNE:
            if self.resume_path is not None:
                raise ValueError("training_mode='fine_tune' cannot be used with resume_path")
            if self.init_checkpoint is None:
                raise ValueError("training_mode='fine_tune' requires init_checkpoint")
        perception_mode = _resolve_perception_mode(
            self.perception_mode,
            freeze_vision=self.freeze_vision,
        )
        object.__setattr__(self, "perception_mode", perception_mode)
        object.__setattr__(
            self,
            "freeze_vision",
            perception_mode == FULL_SMB_PERCEPTION_FREEZE,
        )
        if (
            perception_mode != FULL_SMB_PERCEPTION_REPLACE
            and self.full_smb_vision_checkpoint is None
        ):
            raise ValueError(
                "full_smb_vision_checkpoint is required unless " "perception_mode='replace'"
            )


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
    recording: Mapping[str, Any] = field(default_factory=_empty_full_smb_recording_manifest)
    fixed_task_results: Mapping[str, Any] = field(default_factory=dict)
    tuning_metrics: Mapping[str, float] = field(default_factory=dict)
    success_thresholds_met: bool = False

    def as_dict(self) -> dict[str, Any]:
        return to_plain_data(asdict(self))


@dataclass(frozen=True)
class FullSMBPlayConfig:
    """Runtime controls for playing back a saved Full SMB policy."""

    max_steps: int = 1_000
    action_repeat: int = 1
    render: bool = True
    fps: float = 30.0
    deterministic_policy: bool = True
    sampling_temperature: float = 1.0
    reset_on_done: bool = True
    pause_at_start: bool = False
    interactive_controls: bool = True
    recording_prefix: str = "play"
    inspection_overlay: bool = False
    overlay_interval_steps: int = 1
    overlay_top_actions: int = 5
    overlay_history_limit: int = 256
    human_control: bool = False
    human_default_action: int = int(SMB_ACTIONS[0])
    human_action_script: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if int(self.max_steps) < 0:
            raise ValueError("max_steps must be non-negative")
        if int(self.action_repeat) <= 0:
            raise ValueError("action_repeat must be positive")
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "action_repeat", int(self.action_repeat))
        if float(self.fps) < 0.0:
            raise ValueError("fps must be non-negative")
        if float(self.sampling_temperature) <= 0.0:
            raise ValueError("sampling_temperature must be positive")
        for name in ("overlay_interval_steps", "overlay_top_actions", "overlay_history_limit"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(getattr(self, name)))
        if not str(self.recording_prefix).strip():
            raise ValueError("recording_prefix must be non-empty")
        object.__setattr__(
            self,
            "human_default_action",
            _validate_full_smb_action_id(self.human_default_action),
        )
        object.__setattr__(
            self,
            "human_action_script",
            tuple(_validate_full_smb_action_id(action) for action in self.human_action_script),
        )


@dataclass(frozen=True)
class FullSMBPlayResult:
    """Summary produced by Full SMB policy playback."""

    steps: int
    resets: int
    completed_episodes: int
    total_return: float
    episode_returns: tuple[float, ...]
    actions: tuple[int, ...]
    action_names: tuple[str, ...]
    deterministic_policy: bool
    sampling_temperature: float
    render: bool
    fps: float
    action_repeat: int = 1
    control_mode: str = "policy"
    quit_requested: bool = False
    recording: Mapping[str, Any] = field(default_factory=_empty_full_smb_recording_manifest)
    final_signals: Mapping[str, Any] = field(default_factory=dict)
    last_reward_terms: Mapping[str, float] = field(default_factory=dict)
    last_overlay: Mapping[str, Any] = field(default_factory=dict)
    overlay_history: tuple[Mapping[str, Any], ...] = ()
    human_action_bindings: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_return(self) -> float:
        return _mean(tuple(self.episode_returns))

    def as_dict(self) -> dict[str, Any]:
        payload = to_plain_data(asdict(self))
        payload["mean_return"] = self.mean_return
        return payload


@dataclass(frozen=True)
class FullSMBPlayCommand:
    """One non-blocking terminal command for Full SMB play mode."""

    kind: str
    action: Optional[int] = None


@dataclass(frozen=True)
class FullSMBRolloutStep:
    """Serializable Full SMB transition record for replay and diagnostics."""

    step_index: int
    action: int
    action_name: str
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    episode_mask: float
    boundary_reasons: tuple[str, ...]
    scenario_id: Optional[str]
    task_id: Optional[str]
    emulator_state_id: Optional[str]
    signals: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return to_plain_data(asdict(self))


@dataclass(frozen=True)
class FullSMBRolloutStorage:
    """Compact replay record for one Full SMB training rollout."""

    rollout_id: str
    seed: int
    max_steps: int
    steps: tuple[FullSMBRolloutStep, ...] = ()

    @property
    def total_return(self) -> float:
        return float(sum(step.reward for step in self.steps))

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "seed": int(self.seed),
            "max_steps": int(self.max_steps),
            "steps": [step.as_dict() for step in self.steps],
            "total_return": self.total_return,
            "step_count": self.step_count,
        }


@dataclass(frozen=True)
class FullSMBEpisodeTrainingResult:
    """Metrics and replay storage produced by one online training rollout."""

    metrics: Mapping[str, float]
    rollout: FullSMBRolloutStorage


@dataclass(frozen=True)
class FullSMBTrainingResult:
    """Artifacts from one direct Full SMB training run."""

    checkpoint: dict[str, Any]
    history: Mapping[str, list[float]]
    evaluation: FullSMBEvaluationResult
    checkpoint_path: Optional[Path]
    evaluations: tuple[Mapping[str, Any], ...] = ()
    rollouts: tuple[FullSMBRolloutStorage, ...] = ()

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
                "metadata": to_plain_data(self.checkpoint.get("metadata", {})),
            },
            "history": to_plain_data(self.history),
            "evaluations": to_plain_data(self.evaluations),
            "rollouts": [rollout.as_dict() for rollout in self.rollouts],
            "evaluation": self.evaluation.as_dict(),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


@dataclass(frozen=True)
class FullSMBRolloutBoundary:
    """Episode-boundary decision for recurrent Full SMB rollout state."""

    reset_recurrent_state: bool
    episode_mask: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FullSMBPolicyForwardResult:
    """Forward-pass data needed for action selection and safety diagnostics."""

    logits: torch.Tensor
    next_world_model_state: WorldModelState | None
    next_state_pred: Optional[torch.Tensor] = None
    criticism: Optional[torch.Tensor] = None
    motor_primitives: Any = None

    def __iter__(self):
        yield self.logits
        yield self.next_world_model_state


StageFactory = Callable[[FullSMBSegmentationVision], FullSMBStage]


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


def train_full_smb_policy(
    config: FullSMBTrainingConfig = FullSMBTrainingConfig(),
    *,
    make_stage: Optional[StageFactory] = None,
) -> FullSMBTrainingResult:
    """Run online policy-gradient updates directly against the Full SMB stage."""

    seed_everything(config.seed, deterministic=config.deterministic)
    device = select_device(config.device)
    (
        model,
        start_epoch,
        global_step,
        architecture_name,
        architecture_config,
        checkpoint,
        vision,
        training_source,
    ) = _load_training_state(config, device)
    if vision is None:
        vision = _build_full_smb_perception(config, device)
    _restore_perception_state(vision, checkpoint)
    optimizer = _make_training_optimizer(model, vision, config)
    _restore_optimizer_state(optimizer, checkpoint, strict=True)
    if config.resume_path is not None and checkpoint is not None:
        training_source["restored_rng_state"] = _restore_resume_rng_state(checkpoint)
    history: dict[str, list[float]] = {"episode_return": [], "policy_loss": []}
    rollouts: list[FullSMBRolloutStorage] = []
    evaluations: list[dict[str, Any]] = []
    _initialize_full_smb_log(config)
    tracker = make_experiment_tracker(
        ExperimentTrackerConfig(
            backend=config.tracking_backend,
            log_dir=config.tracking_log_dir,
            project=config.tracking_project,
            run_name=config.tracking_run_name,
            mode=config.tracking_mode,
        ),
        default_log_dir=Path("artifacts/full_smb/tracking"),
    )
    try:
        tracker.log_config(to_plain_data(config))
        _log_full_smb_event(
            config,
            "run_started",
            config=to_plain_data(config),
            device=str(device),
            start_epoch=start_epoch,
            global_step=global_step,
            resumed_from=str(config.resume_path) if config.resume_path is not None else None,
            initialized_from=(
                str(config.init_checkpoint) if config.init_checkpoint is not None else None
            ),
        )
        stage = _make_stage(make_stage, vision, config)
        backend_metadata = _full_smb_backend_metadata(stage, config)
        try:
            model.train()
            _set_perception_training_mode(vision, config)
            gradient_parameters = _gradient_parameters(model, vision, config)
            final_evaluation: Optional[FullSMBEvaluationResult] = None
            for epoch in range(start_epoch, config.epochs):
                for update_index in range(config.updates_per_epoch):
                    episode_seed = config.seed + epoch * config.updates_per_epoch + update_index
                    try:
                        episode = _train_episode(
                            model,
                            optimizer,
                            stage,
                            seed=episode_seed,
                            max_steps=config.rollout_length,
                            device=device,
                            reward_scale=config.reward_scale,
                            entropy_weight=config.entropy_weight,
                            policy_loss_weight=config.policy_loss_weight,
                            world_model_weight=config.world_model_weight,
                            gradient_clip_norm=config.gradient_clip_norm,
                            max_abs_loss=config.max_abs_loss,
                            max_abs_scaled_reward=config.max_abs_scaled_reward,
                            max_abs_prediction=config.max_abs_prediction,
                            deterministic_actions=config.deterministic_actions,
                            gradient_parameters=gradient_parameters,
                            rollout_id=f"epoch{epoch + 1:04d}_update{update_index + 1:04d}",
                        )
                    except FloatingPointError as exc:
                        _log_full_smb_event(
                            config,
                            "training_stopped_early",
                            epoch=epoch + 1,
                            update=update_index + 1,
                            global_step=global_step,
                            reason=str(exc),
                        )
                        raise
                    metrics = episode.metrics
                    rollouts.append(episode.rollout)
                    global_step += int(metrics["steps"])
                    history["episode_return"].append(float(metrics["return"]))
                    history["policy_loss"].append(float(metrics["loss"]))
                    for metric_name, metric_value in metrics.items():
                        if metric_name in {"return", "loss", "steps"}:
                            continue
                        history.setdefault(metric_name, []).append(float(metric_value))
                    tracker.log_metrics(metrics, step=global_step, prefix="train")
                    _log_full_smb_event(
                        config,
                        "train_rollout",
                        epoch=epoch + 1,
                        update=update_index + 1,
                        global_step=global_step,
                        metrics=metrics,
                    )
                completed_epoch = epoch + 1
                if _should_evaluate_epoch(config, completed_epoch):
                    evaluation = evaluate_full_smb_policy(
                        model,
                        config=config,
                        make_stage=make_stage,
                        vision=vision,
                        device=device,
                        recording_prefix=f"epoch{completed_epoch:04d}",
                    )
                    eval_metrics = _full_smb_evaluation_metrics(evaluation)
                    evaluations.append(
                        {
                            "epoch": completed_epoch,
                            "global_step": global_step,
                            "metrics": eval_metrics,
                            "evaluation": evaluation.as_dict(),
                        }
                    )
                    for metric_name, metric_value in eval_metrics.items():
                        history.setdefault(metric_name, []).append(float(metric_value))
                    _log_full_smb_event(
                        config,
                        "deterministic_evaluation",
                        epoch=completed_epoch,
                        global_step=global_step,
                        metrics=eval_metrics,
                        evaluation=evaluation.as_dict(),
                    )
                    tracker.log_metrics(eval_metrics, step=global_step, prefix="eval")
                    model.train()
                    _set_perception_training_mode(vision, config)
                    if completed_epoch == config.epochs:
                        final_evaluation = evaluation
            if final_evaluation is None:
                if config.evaluation_episodes > 0 and config.evaluation_max_steps > 0:
                    final_evaluation = evaluate_full_smb_policy(
                        model,
                        config=config,
                        make_stage=make_stage,
                        vision=vision,
                        device=device,
                        recording_prefix="final",
                    )
                else:
                    final_evaluation = FullSMBEvaluationResult(
                        episodes=0,
                        max_steps_per_episode=0,
                        steps=0,
                        returns=(),
                        mean_return=0.0,
                        success_rate=0.0,
                        terminated_count=0,
                        truncated_count=0,
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
                "mean_loss_dynamics": _mean(history.get("loss_dynamics", [])),
                "mean_loss_world_model": _mean(history.get("loss_world_model", [])),
                "mean_action_entropy": _mean(history.get("mean_entropy", [])),
                "mean_gradient_norm": _mean(history.get("mean_gradient_norm", [])),
                "max_gradient_norm": _max(history.get("max_gradient_norm", [])),
                "max_abs_scaled_reward": _max(history.get("max_abs_scaled_reward", [])),
                "max_abs_prediction": _max(history.get("max_abs_prediction", [])),
                "evaluation_mean_return": final_evaluation.mean_return,
                "evaluation_success_rate": final_evaluation.success_rate,
                "periodic_evaluation_count": float(len(evaluations)),
            },
            architecture_name=architecture_name,
            architecture_config=architecture_config,
            vision=vision,
            training_source=training_source,
            rollouts=rollouts,
            evaluations=evaluations,
            backend_metadata=backend_metadata,
        )
        tracker.log_metrics(
            checkpoint["metrics"],
            step=global_step,
            prefix="final",
        )
        _log_full_smb_event(
            config,
            "run_finished",
            epoch=config.epochs,
            global_step=global_step,
            metrics=checkpoint["metrics"],
            evaluation=final_evaluation.as_dict(),
            evaluations=evaluations,
            checkpoint_path=(
                str(config.checkpoint_path)
                if config.save_checkpoints and config.checkpoint_path is not None
                else None
            ),
        )
        checkpoint_path = config.checkpoint_path
        if checkpoint_path is not None and config.save_checkpoints:
            save_checkpoint(checkpoint_path, checkpoint)
        result = FullSMBTrainingResult(
            checkpoint=checkpoint,
            history=history,
            evaluation=final_evaluation,
            checkpoint_path=checkpoint_path if config.save_checkpoints else None,
            evaluations=tuple(evaluations),
            rollouts=tuple(rollouts),
        )
        if config.output_summary is not None:
            save_full_smb_training_summary(config.output_summary, result)
        return result
    finally:
        tracker.close()


@torch.no_grad()
def evaluate_full_smb_policy(
    model: torch.nn.Module,
    *,
    config: FullSMBTrainingConfig = FullSMBTrainingConfig(),
    make_stage: Optional[StageFactory] = None,
    vision: Optional[FullSMBSegmentationVision] = None,
    device: Optional[torch.device] = None,
    recording_prefix: str = "evaluation",
) -> FullSMBEvaluationResult:
    """Evaluate a Full SMB policy with deterministic action selection."""

    resolved_device = device or select_device(config.device)
    model.eval()
    owns_vision = vision is None
    if vision is None:
        vision = _build_full_smb_perception(config, resolved_device)
    if isinstance(vision, torch.nn.Module):
        vision.eval()
    stage = _make_stage(make_stage, vision, config)
    returns: list[float] = []
    steps = 0
    terminated_count = 0
    truncated_count = 0
    recording_targets = _resolve_full_smb_recording_targets(config, recording_prefix)
    recording_artifacts: list[dict[str, Any]] = []
    fixed_task_episode_metrics: dict[str, list[dict[str, float]]] = {}
    try:
        for episode_index in range(config.evaluation_episodes):
            episode_seed = config.seed + 10_000 + episode_index
            observation = stage.reset(seed=episode_seed)
            episode_return = 0.0
            fixed_task_metrics = _start_full_smb_fixed_task_episode_metrics(
                stage,
                stage.last_info,
            )
            episode_recording = _start_full_smb_episode_recording(
                episode_index=episode_index,
                seed=episode_seed,
                observation=observation,
                info=stage.last_info,
                stage=stage,
            )
            for _step in range(config.evaluation_max_steps):
                batch = stage.encode_observation(observation)
                logits = _policy_action_logits(model, batch, device=resolved_device)
                action = int(logits.argmax(dim=-1).item())
                observation, reward, terminated, truncated, info = stage.step(action)
                _update_full_smb_fixed_task_episode_metrics(
                    fixed_task_metrics,
                    stage,
                    info,
                )
                if recording_targets["enabled"]:
                    _append_full_smb_episode_recording_step(
                        episode_recording,
                        observation=observation,
                        action=action,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        stage=stage,
                    )
                episode_return += float(reward)
                steps += 1
                if terminated:
                    terminated_count += 1
                if truncated:
                    truncated_count += 1
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
            task_name = fixed_task_metrics["task_name"]
            if task_name is not None:
                fixed_task_episode_metrics.setdefault(str(task_name), []).append(
                    _finalize_full_smb_fixed_task_episode_metrics(
                        fixed_task_metrics,
                        episode_return=float(episode_return),
                    )
                )
            if recording_targets["enabled"]:
                recording_artifacts.append(
                    _write_full_smb_episode_recording(
                        episode_recording,
                        recording_targets,
                        total_return=float(episode_return),
                    )
                )
    finally:
        stage.close()
        if owns_vision:
            del vision

    episodes = len(returns)
    successes = terminated_count
    recording_manifest = _full_smb_recording_manifest(
        recording_targets,
        recording_artifacts,
    )
    fixed_task_results = _full_smb_fixed_task_results(
        fixed_task_episode_metrics,
        evaluation_episodes=config.evaluation_episodes,
        evaluation_max_steps=config.evaluation_max_steps,
    )
    tuning_metrics = summarize_fixed_full_smb_success_metrics(
        fixed_task_results,
        {
            task_name: {
                **dict(result.get("threshold_diagnostics", {})),
                "threshold_met": bool(result.get("threshold_met", False)),
            }
            for task_name, result in fixed_task_results.items()
        },
    )
    success_thresholds_met = bool(fixed_task_results) and all(
        bool(result.get("threshold_met", False)) for result in fixed_task_results.values()
    )
    return FullSMBEvaluationResult(
        episodes=episodes,
        max_steps_per_episode=config.evaluation_max_steps,
        steps=steps,
        returns=tuple(returns),
        mean_return=_mean(returns),
        success_rate=float(successes / episodes) if episodes else 0.0,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
        recording=recording_manifest,
        fixed_task_results=fixed_task_results,
        tuning_metrics=tuning_metrics,
        success_thresholds_met=success_thresholds_met,
    )


@torch.no_grad()
def play_full_smb_policy(
    model: Optional[torch.nn.Module],
    *,
    config: FullSMBTrainingConfig = FullSMBTrainingConfig(),
    play_config: FullSMBPlayConfig = FullSMBPlayConfig(),
    make_stage: Optional[StageFactory] = None,
    vision: Optional[FullSMBSegmentationVision] = None,
    device: Optional[torch.device] = None,
) -> FullSMBPlayResult:
    """Play a saved Full SMB policy with optional rendering and recording."""

    if model is None and not play_config.human_control:
        raise ValueError("policy playback requires a model unless human_control is enabled")
    seed_everything(config.seed, deterministic=config.deterministic)
    resolved_device = device or select_device(config.device)
    if model is not None:
        model.eval()
    owns_vision = vision is None
    if vision is None:
        vision = _build_full_smb_perception(config, resolved_device)
    if isinstance(vision, torch.nn.Module):
        vision.eval()
    stage = _make_stage(make_stage, vision, config)
    recording_targets = _resolve_full_smb_recording_targets(
        config,
        play_config.recording_prefix,
    )
    recording_artifacts: list[dict[str, Any]] = []
    episode_returns: list[float] = []
    actions: list[int] = []
    action_names: list[str] = []
    total_return = 0.0
    completed_episodes = 0
    resets = 0
    quit_requested = False
    final_info: Mapping[str, Any] = {}
    world_model_state: WorldModelState | None = None
    overlay_history: list[Mapping[str, Any]] = []
    last_overlay: Mapping[str, Any] = {}
    interactive = bool(play_config.interactive_controls and _stdin_supports_play_controls())
    paused = bool(play_config.pause_at_start and interactive)
    script_index = 0
    control_mode = "human" if play_config.human_control else "policy"
    try:
        episode_index = 0
        episode_seed = config.seed
        observation = stage.reset(seed=episode_seed)
        resets += 1
        _render_full_smb_stage(stage, enabled=play_config.render)
        episode_return = 0.0
        episode_recording = _start_full_smb_episode_recording(
            episode_index=episode_index,
            seed=episode_seed,
            observation=observation,
            info=stage.last_info,
            stage=stage,
        )
        fixed_task_metrics = _start_full_smb_fixed_task_episode_metrics(stage, stage.last_info)
        episode_open = True
        while len(actions) < play_config.max_steps:
            command = _poll_full_smb_play_command(
                enabled=interactive,
                allow_actions=play_config.human_control,
            )
            if command is not None and command.kind == "quit":
                quit_requested = True
                break
            if command is not None and command.kind == "pause":
                paused = not paused
            if command is not None and command.kind == "reset":
                if recording_targets["enabled"]:
                    _finish_full_smb_play_recording_episode(
                        episode_recording,
                        recording_targets,
                        recording_artifacts,
                        episode_return=episode_return,
                    )
                episode_open = False
                episode_returns.append(float(episode_return))
                episode_index += 1
                episode_seed = config.seed + episode_index
                observation = stage.reset(seed=episode_seed)
                resets += 1
                world_model_state = None
                episode_return = 0.0
                episode_recording = _start_full_smb_episode_recording(
                    episode_index=episode_index,
                    seed=episode_seed,
                    observation=observation,
                    info=stage.last_info,
                    stage=stage,
                )
                fixed_task_metrics = _start_full_smb_fixed_task_episode_metrics(
                    stage,
                    stage.last_info,
                )
                episode_open = True
                _render_full_smb_stage(stage, enabled=play_config.render)
            if paused:
                _sleep_full_smb_play_frame(play_config.fps)
                continue

            if play_config.human_control:
                action, script_index = _select_full_smb_human_action(
                    command,
                    play_config=play_config,
                    script_index=script_index,
                )
                next_world_model_state = None
                action_probabilities = None
            else:
                assert model is not None
                batch = stage.encode_observation(observation)
                forward = _coerce_policy_forward_result(
                    _policy_action_logits_and_state(
                        model,
                        batch,
                        device=resolved_device,
                        world_model_state=world_model_state,
                    )
                )
                logits = forward.logits
                _finite_tensor_or_raise("action_logits", logits)
                action = _select_full_smb_play_action(
                    logits,
                    deterministic=play_config.deterministic_policy,
                    temperature=play_config.sampling_temperature,
                )
                action_probabilities = _full_smb_action_probabilities(
                    logits,
                    temperature=play_config.sampling_temperature,
                )
                next_world_model_state = forward.next_world_model_state
            repeated_steps = min(
                play_config.action_repeat,
                play_config.max_steps - len(actions),
            )
            for repeat_index in range(repeated_steps):
                observation, reward, terminated, truncated, info = stage.step(action)
                final_info = info
                if recording_targets["enabled"]:
                    _append_full_smb_episode_recording_step(
                        episode_recording,
                        observation=observation,
                        action=action,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        stage=stage,
                    )
                actions.append(action)
                action_names.append(SMB_ACTIONS[action].name)
                total_return += float(reward)
                episode_return += float(reward)
                _render_full_smb_stage(stage, enabled=play_config.render)
                _update_full_smb_fixed_task_episode_metrics(
                    fixed_task_metrics,
                    stage,
                    info,
                )
                boundary = _full_smb_rollout_boundary(
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                if play_config.inspection_overlay:
                    last_overlay = _full_smb_play_overlay_snapshot(
                        step_index=len(actions),
                        episode_index=episode_index,
                        action=action,
                        action_probabilities=action_probabilities,
                        probability_source=(
                            "human_selection" if play_config.human_control else "policy_softmax"
                        ),
                        reward=float(reward),
                        episode_return=float(episode_return),
                        total_return=float(total_return),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        boundary=boundary,
                        fixed_task_metrics=fixed_task_metrics,
                        play_config=play_config,
                    )
                    overlay_history.append(last_overlay)
                    if len(overlay_history) > play_config.overlay_history_limit:
                        overlay_history.pop(0)
                    if (
                        len(actions) % play_config.overlay_interval_steps == 0
                        or terminated
                        or truncated
                    ):
                        _write_full_smb_play_overlay(last_overlay)
                if boundary.reset_recurrent_state:
                    world_model_state = None
                elif next_world_model_state is not None and repeat_index == 0:
                    world_model_state = next_world_model_state.detach()
                else:
                    world_model_state = None

                if terminated or truncated:
                    completed_episodes += 1
                    episode_returns.append(float(episode_return))
                    if recording_targets["enabled"]:
                        _finish_full_smb_play_recording_episode(
                            episode_recording,
                            recording_targets,
                            recording_artifacts,
                            episode_return=episode_return,
                        )
                    episode_open = False
                    if not play_config.reset_on_done or len(actions) >= play_config.max_steps:
                        episode_recording = None
                        break
                    episode_index += 1
                    episode_seed = config.seed + episode_index
                    observation = stage.reset(seed=episode_seed)
                    resets += 1
                    episode_return = 0.0
                    episode_recording = _start_full_smb_episode_recording(
                        episode_index=episode_index,
                        seed=episode_seed,
                        observation=observation,
                        info=stage.last_info,
                        stage=stage,
                    )
                    fixed_task_metrics = _start_full_smb_fixed_task_episode_metrics(
                        stage,
                        stage.last_info,
                    )
                    episode_open = True
                    _render_full_smb_stage(stage, enabled=play_config.render)
                    break
                _sleep_full_smb_play_frame(play_config.fps)
            if play_config.action_repeat > 1 and not (terminated or truncated):
                world_model_state = None
        if episode_open:
            if episode_return or not episode_returns:
                episode_returns.append(float(episode_return))
            if recording_targets["enabled"]:
                _finish_full_smb_play_recording_episode(
                    episode_recording,
                    recording_targets,
                    recording_artifacts,
                    episode_return=episode_return,
                )
        recording_manifest = _full_smb_recording_manifest(
            recording_targets,
            recording_artifacts,
        )
        return FullSMBPlayResult(
            steps=len(actions),
            resets=resets,
            completed_episodes=completed_episodes,
            total_return=float(total_return),
            episode_returns=tuple(episode_returns),
            actions=tuple(actions),
            action_names=tuple(action_names),
            deterministic_policy=play_config.deterministic_policy,
            sampling_temperature=play_config.sampling_temperature,
            render=play_config.render,
            fps=play_config.fps,
            action_repeat=play_config.action_repeat,
            control_mode=control_mode,
            quit_requested=quit_requested,
            recording=recording_manifest,
            final_signals=_selected_full_smb_signal_fields(final_info),
            last_reward_terms=_last_full_smb_reward_terms(final_info),
            last_overlay=last_overlay,
            overlay_history=tuple(overlay_history),
            human_action_bindings=(
                _full_smb_human_action_bindings() if play_config.human_control else {}
            ),
        )
    finally:
        stage.close()
        if owns_vision:
            del vision


def _select_full_smb_play_action(
    logits: torch.Tensor,
    *,
    deterministic: bool,
    temperature: float,
) -> int:
    scaled_logits = logits / float(temperature)
    if deterministic:
        return int(scaled_logits.argmax(dim=-1).item())
    distribution = torch.distributions.Categorical(logits=scaled_logits)
    return int(distribution.sample().item())


def _full_smb_action_probabilities(
    logits: torch.Tensor,
    *,
    temperature: float,
) -> tuple[float, ...]:
    scaled_logits = logits.detach().float() / float(temperature)
    probabilities = torch.softmax(scaled_logits, dim=-1)
    if probabilities.ndim > 1:
        probabilities = probabilities[0]
    probabilities = probabilities[:FULL_SMB_ACTION_COUNT].cpu()
    return tuple(float(value) for value in probabilities.tolist())


def _full_smb_play_overlay_snapshot(
    *,
    step_index: int,
    episode_index: int,
    action: int,
    action_probabilities: Optional[tuple[float, ...]],
    probability_source: str,
    reward: float,
    episode_return: float,
    total_return: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    boundary: FullSMBRolloutBoundary,
    fixed_task_metrics: Mapping[str, Any],
    play_config: FullSMBPlayConfig,
) -> dict[str, Any]:
    probabilities, top_actions = _full_smb_play_action_probability_payload(
        action=action,
        action_probabilities=action_probabilities,
        source=probability_source,
        top_actions=play_config.overlay_top_actions,
    )
    signals = _full_smb_play_overlay_signals(info)
    threshold_status = _full_smb_play_threshold_status(
        fixed_task_metrics,
        episode_return=episode_return,
        evaluation_max_steps=play_config.max_steps,
    )
    return {
        "step": int(step_index),
        "episode": int(episode_index),
        "action_id": int(action),
        "action_name": SMB_ACTIONS[action].name,
        "action_probability_source": probability_source,
        "action_probabilities": probabilities,
        "top_action_probabilities": top_actions,
        "reward": float(reward),
        "episode_return": float(episode_return),
        "total_return": float(total_return),
        "reward_terms": _last_full_smb_reward_terms(info),
        "signals": signals,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": _full_smb_play_termination_reason(info, boundary),
        "boundary_reasons": tuple(boundary.reasons),
        "threshold_status": threshold_status,
    }


def _full_smb_play_action_probability_payload(
    *,
    action: int,
    action_probabilities: Optional[tuple[float, ...]],
    source: str,
    top_actions: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if action_probabilities is None:
        probabilities = {
            action_item.name: (1.0 if int(action_item) == int(action) else 0.0)
            for action_item in SMB_ACTIONS
        }
    else:
        probabilities = {
            action_item.name: float(action_probabilities[int(action_item)])
            for action_item in SMB_ACTIONS
            if int(action_item) < len(action_probabilities)
        }
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    action_ids_by_name = {action_item.name: int(action_item) for action_item in SMB_ACTIONS}
    top = [
        {
            "action_id": action_ids_by_name[name],
            "action_name": name,
            "probability": float(probability),
            "source": source,
        }
        for name, probability in ranked[: int(top_actions)]
    ]
    return probabilities, top


def _full_smb_play_overlay_signals(info: Mapping[str, Any]) -> dict[str, Any]:
    signals = _selected_full_smb_signal_fields(info)
    source = _full_smb_signal_source(info)
    progress = _full_smb_signal_progress(source)
    if progress is not None:
        signals["progress"] = progress
    score = _optional_float(source.get("score"))
    if score is not None:
        signals["score"] = score
    coins = _optional_float(source.get("coins"))
    if coins is None:
        collectibles = source.get("collectibles")
        if isinstance(collectibles, Mapping):
            coins = _optional_float(collectibles.get("coins"))
    if coins is not None:
        signals["coins"] = coins
    return signals


def _full_smb_play_termination_reason(
    info: Mapping[str, Any],
    boundary: FullSMBRolloutBoundary,
) -> Optional[str]:
    source = _full_smb_signal_source(info)
    for key in ("termination_reason", "done_reason", "reason"):
        value = source.get(key)
        if value is not None and value != "":
            return str(value)
    if boundary.reasons:
        return ",".join(boundary.reasons)
    return None


def _full_smb_play_threshold_status(
    fixed_task_metrics: Mapping[str, Any],
    *,
    episode_return: float,
    evaluation_max_steps: int,
) -> dict[str, Any]:
    task_name = fixed_task_metrics.get("task_name")
    if task_name is None:
        return {
            "available": False,
            "task_name": None,
            "threshold_met": False,
            "reason": "not_fixed_benchmark_task",
        }
    task_name = str(task_name)
    if task_name not in FIXED_FULL_SMB_SUCCESS_THRESHOLDS:
        return {
            "available": False,
            "task_name": task_name,
            "threshold_met": False,
            "reason": "not_fixed_benchmark_task",
        }
    observed = _finalize_full_smb_fixed_task_episode_metrics(
        fixed_task_metrics,
        episode_return=float(episode_return),
    )
    status = evaluate_full_smb_success_threshold(
        task_name,
        observed,
        evaluation_episodes=1,
        evaluation_max_steps=int(evaluation_max_steps),
    )
    return {
        "available": True,
        "task_name": task_name,
        **to_plain_data(status),
    }


def _write_full_smb_play_overlay(snapshot: Mapping[str, Any]) -> None:
    print(_format_full_smb_play_overlay(snapshot), file=sys.stderr, flush=True)


def _format_full_smb_play_overlay(snapshot: Mapping[str, Any]) -> str:
    top_actions = snapshot.get("top_action_probabilities", ())
    top_text = " ".join(
        f"{item.get('action_name')}={float(item.get('probability', 0.0)):.2f}"
        for item in top_actions
        if isinstance(item, Mapping)
    )
    signals = snapshot.get("signals", {})
    signal_text = ""
    if isinstance(signals, Mapping):
        signal_parts = []
        for name in ("progress", "score", "coins", "lives"):
            if name in signals:
                signal_parts.append(f"{name}={signals[name]}")
        signal_text = " ".join(signal_parts)
    reward_terms = snapshot.get("reward_terms", {})
    reward_text = ""
    if isinstance(reward_terms, Mapping):
        reward_text = " ".join(
            f"{name}={float(value):.3f}"
            for name, value in reward_terms.items()
            if _optional_float(value) is not None
        )
    threshold = snapshot.get("threshold_status", {})
    threshold_text = "threshold=n/a"
    if isinstance(threshold, Mapping) and threshold.get("available"):
        threshold_text = (
            f"threshold={threshold.get('task_name')}:"
            f"{'pass' if threshold.get('threshold_met') else 'pending'}"
        )
    reason = snapshot.get("termination_reason") or "-"
    return (
        "[FullSMB overlay] "
        f"step={snapshot.get('step')} episode={snapshot.get('episode')} "
        f"action={snapshot.get('action_name')} reward={float(snapshot.get('reward', 0.0)):.3f} "
        f"return={float(snapshot.get('episode_return', 0.0)):.3f} "
        f"top=[{top_text}] signals=[{signal_text}] rewards=[{reward_text}] "
        f"termination={reason} {threshold_text}"
    )


def _select_full_smb_human_action(
    command: Optional[FullSMBPlayCommand],
    *,
    play_config: FullSMBPlayConfig,
    script_index: int,
) -> tuple[int, int]:
    if command is not None and command.kind == "action" and command.action is not None:
        return command.action, script_index
    if script_index < len(play_config.human_action_script):
        return play_config.human_action_script[script_index], script_index + 1
    return play_config.human_default_action, script_index


def _full_smb_action_id_arg(value: str) -> int:
    parsed = _full_smb_human_action_id(value)
    if parsed is None:
        valid = ", ".join(f"{int(action)}={action.name.lower()}" for action in SMB_ACTIONS)
        raise argparse.ArgumentTypeError(
            f"invalid action {value!r}; expected action id/name or one of: {valid}"
        )
    return parsed


def _full_smb_human_action_id(value: str) -> Optional[int]:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = _full_smb_human_action_aliases()
    if normalized in aliases:
        return aliases[normalized]
    try:
        return _validate_full_smb_action_id(int(normalized))
    except (TypeError, ValueError):
        return None


def _full_smb_human_action_aliases() -> dict[str, int]:
    return {
        "": int(SMB_ACTIONS[0]),
        "noop": int(SMB_ACTIONS[0]),
        "none": int(SMB_ACTIONS[0]),
        "idle": int(SMB_ACTIONS[0]),
        ".": int(SMB_ACTIONS[0]),
        "right": int(SMB_ACTIONS[1]),
        "d": int(SMB_ACTIONS[1]),
        "right_jump": int(SMB_ACTIONS[2]),
        "rightjump": int(SMB_ACTIONS[2]),
        "jump_right": int(SMB_ACTIONS[2]),
        "dr": int(SMB_ACTIONS[2]),
        "rd": int(SMB_ACTIONS[2]),
        "d+": int(SMB_ACTIONS[2]),
        "right+a": int(SMB_ACTIONS[2]),
        "left": int(SMB_ACTIONS[3]),
        "l": int(SMB_ACTIONS[3]),
        "a": int(SMB_ACTIONS[3]),
        "left_jump": int(SMB_ACTIONS[4]),
        "leftjump": int(SMB_ACTIONS[4]),
        "jump_left": int(SMB_ACTIONS[4]),
        "al": int(SMB_ACTIONS[4]),
        "la": int(SMB_ACTIONS[4]),
        "a+": int(SMB_ACTIONS[4]),
        "left+a": int(SMB_ACTIONS[4]),
        "jump": int(SMB_ACTIONS[5]),
        "j": int(SMB_ACTIONS[5]),
        "w": int(SMB_ACTIONS[5]),
        "space": int(SMB_ACTIONS[5]),
    }


def _full_smb_human_action_bindings() -> dict[str, Any]:
    by_action: dict[str, list[str]] = {action.name.lower(): [] for action in SMB_ACTIONS}
    for alias, action_id in _full_smb_human_action_aliases().items():
        by_action[SMB_ACTIONS[action_id].name.lower()].append(alias or "<enter>")
    return {
        "input_mode": "line",
        "commands": {
            "pause_or_resume": ("p", "pause", "resume"),
            "reset": ("r", "reset"),
            "quit": ("q", "quit", "exit"),
        },
        "actions": {name: tuple(sorted(aliases)) for name, aliases in by_action.items()},
    }


def _finish_full_smb_play_recording_episode(
    recording: Optional[Mapping[str, Any]],
    targets: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
    *,
    episode_return: float,
) -> None:
    if recording is None:
        return
    artifacts.append(
        _write_full_smb_episode_recording(
            recording,
            targets,
            total_return=float(episode_return),
        )
    )


def _render_full_smb_stage(stage: FullSMBStage, *, enabled: bool) -> None:
    if not enabled:
        return
    render = getattr(stage.env, "render", None)
    if render is not None:
        render()


def _sleep_full_smb_play_frame(fps: float) -> None:
    if fps <= 0.0:
        return
    time.sleep(1.0 / float(fps))


def _stdin_supports_play_controls() -> bool:
    stream = sys.stdin
    return bool(stream is not None and hasattr(stream, "isatty") and stream.isatty())


def _poll_full_smb_play_command(
    *,
    enabled: bool,
    allow_actions: bool = False,
) -> Optional[FullSMBPlayCommand]:
    if not enabled:
        return None
    try:
        import select

        readable, _writable, _errors = select.select([sys.stdin], [], [], 0.0)
    except (OSError, ValueError):
        return None
    if not readable:
        return None
    text = sys.stdin.readline().strip().lower()
    if text in {"q", "quit", "exit"}:
        return FullSMBPlayCommand("quit")
    if text in {"p", "pause", "resume"}:
        return FullSMBPlayCommand("pause")
    if text in {"r", "reset"}:
        return FullSMBPlayCommand("reset")
    if allow_actions:
        action = _full_smb_human_action_id(text)
        if action is not None:
            return FullSMBPlayCommand("action", action=action)
    return None


def _last_full_smb_reward_terms(info: Mapping[str, Any]) -> dict[str, float]:
    reward_terms = info.get("reward_terms") if isinstance(info, Mapping) else None
    if not isinstance(reward_terms, Mapping):
        return {}
    terms: dict[str, float] = {}
    for name, value in reward_terms.items():
        numeric_value = _optional_float(value)
        if numeric_value is not None:
            terms[str(name)] = numeric_value
    return terms


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
        _restore_optimizer_state(optimizer, checkpoint, strict=False)
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
    vision: Optional[FullSMBSegmentationVision] = None,
    training_source: Optional[Mapping[str, Any]] = None,
    rollouts: Optional[list[FullSMBRolloutStorage] | tuple[FullSMBRolloutStorage, ...]] = None,
    evaluations: Optional[list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]] = None,
    backend_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    perception = _perception_checkpoint_metadata(config, vision)
    rollout = _rollout_metadata(config)
    rollout_storage = _rollout_storage_metadata(rollouts or ())
    evaluation = _evaluation_metadata(config, evaluations or ())
    task_curriculum = _task_curriculum_metadata(
        config,
        epoch=epoch,
        global_step=global_step,
        rollouts=rollouts or (),
    )
    backend = dict(backend_metadata or _full_smb_backend_metadata(None, config))
    loss_weights = _loss_weight_metadata(config)
    safety = _safety_metadata(config)
    recording = _recording_metadata(config)
    tracking = _tracking_metadata(config)
    source = dict(
        training_source
        or _training_source_metadata(
            FULL_SMB_TRAINING_SOURCE_SCRATCH,
            checkpoint_path=None,
            checkpoint=None,
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        )
    )
    stage_batch_contract = _stage_batch_contract_metadata()
    states: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    if _perception_optimizer_enabled(config) and vision is not None:
        states["perception"] = vision.state_dict()
    rng_state = _rng_state_metadata(config, states)
    return build_checkpoint(
        stage=FULL_SMB_SPEC.name,
        model_name=FULL_SMB_POLICY_MODEL_NAME,
        checkpoint_kind=FULL_SMB_POLICY_CHECKPOINT_KIND,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        config={
            **to_plain_data(config),
            "perception": perception,
            "rollout": rollout,
            "loss_weights": loss_weights,
            "safety": safety,
            "reward": config.reward_config.to_manifest(),
            "recording": recording,
            "tracking": tracking,
            "training_source": source,
            "stage_batch_contract": stage_batch_contract,
            "rollout_storage": rollout_storage,
            "evaluation": evaluation,
            "task_curriculum": task_curriculum,
            "backend": backend,
            "rng_state": rng_state,
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
        states=states,
        metadata={
            "perception": perception,
            "training": {
                "deterministic": config.deterministic,
                "deterministic_actions": config.deterministic_actions,
                "rollout": rollout,
                "loss_weights": loss_weights,
                "safety": safety,
                "reward": config.reward_config.to_manifest(),
                "recording": recording,
                "tracking": tracking,
                "source": source,
                "stage_batch_contract": stage_batch_contract,
                "rollout_storage": rollout_storage,
                "evaluation": evaluation,
                "task_curriculum": task_curriculum,
                "backend": backend,
                "rng_state": rng_state,
            },
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
) -> tuple[
    torch.nn.Module,
    int,
    int,
    str,
    dict[str, Any],
    Optional[Mapping[str, Any]],
    Optional[FullSMBSegmentationVision],
    dict[str, Any],
]:
    if config.resume_path is not None:
        checkpoint = load_checkpoint(config.resume_path, map_location=device)
        _validate_full_smb_policy_checkpoint(checkpoint)
        resume_contract = _validate_resume_contract(config, checkpoint)
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        model = make_full_smb_policy_model(
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        ).to(device)
        model.load_state_dict(checkpoint["states"]["model"])
        source = _training_source_metadata(
            FULL_SMB_TRAINING_SOURCE_RESUME_CHECKPOINT,
            checkpoint_path=config.resume_path,
            checkpoint=checkpoint,
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        )
        source["resume_contract"] = resume_contract
        return (
            model,
            int(checkpoint.get("epoch", 0)),
            int(checkpoint.get("global_step", 0)),
            architecture_name,
            architecture_config,
            checkpoint,
            None,
            source,
        )

    if config.init_checkpoint is not None:
        return _load_init_training_state(config, device)
    architecture_name = config.architecture_name
    architecture_config = dict(config.architecture_config)
    model = make_full_smb_policy_model(
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    ).to(device)
    vision = None
    checkpoint = None
    source = _training_source_metadata(
        FULL_SMB_TRAINING_SOURCE_SCRATCH,
        checkpoint_path=None,
        checkpoint=None,
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    )
    return model, 0, 0, architecture_name, dict(architecture_config), checkpoint, vision, source


def _load_init_training_state(
    config: FullSMBTrainingConfig,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    int,
    int,
    str,
    dict[str, Any],
    Optional[Mapping[str, Any]],
    Optional[FullSMBSegmentationVision],
    dict[str, Any],
]:
    if config.init_checkpoint is None:
        raise ValueError("init_checkpoint is required")
    init_checkpoint = load_checkpoint(config.init_checkpoint, map_location=device)
    if _matches_checkpoint_identity(
        init_checkpoint,
        stage=FULL_SMB_SPEC.name,
        model_name=FULL_SMB_TRANSFER_MODEL_NAME,
        checkpoint_kind=FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    ):
        transferred = load_transferred_full_smb_policy(
            config.init_checkpoint,
            full_smb_vision_checkpoint=_perception_checkpoint_path(config),
            device=device,
            freeze_vision=config.freeze_vision,
        )
        model = transferred.model
        vision = transferred.vision
        checkpoint = transferred.checkpoint
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        source = _training_source_metadata(
            FULL_SMB_TRAINING_SOURCE_INIT_CHECKPOINT,
            checkpoint_path=config.init_checkpoint,
            checkpoint=checkpoint,
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        )
        source["init_checkpoint_source"] = FULL_SMB_INIT_SOURCE_TRANSFER
        return model, 0, 0, architecture_name, dict(architecture_config), checkpoint, vision, source

    if _matches_checkpoint_identity(
        init_checkpoint,
        stage=BLOCK_SMB_SPEC.name,
        model_name=BLOCK_SMB_MODEL_NAME,
        checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
    ):
        transferred = transfer_block_smb_checkpoint_to_full_smb(
            config.init_checkpoint,
            output_checkpoint=None,
            full_smb_vision_checkpoint=_perception_checkpoint_path(config),
            block_vision_checkpoint=None,
            device=device,
            freeze_vision=config.freeze_vision,
        )
        model = transferred.model
        vision = transferred.vision
        checkpoint = transferred.checkpoint
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        source = _training_source_metadata(
            FULL_SMB_TRAINING_SOURCE_INIT_CHECKPOINT,
            checkpoint_path=config.init_checkpoint,
            checkpoint=transferred.source_checkpoint,
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        )
        source.update(
            {
                "init_checkpoint_source": FULL_SMB_INIT_SOURCE_BLOCK_POLICY,
                "resolved_transfer_stage": checkpoint.get("stage"),
                "resolved_transfer_model_name": checkpoint.get("model_name"),
                "resolved_transfer_checkpoint_kind": checkpoint.get("checkpoint_kind"),
                "resolved_transfer_epoch": int(checkpoint.get("epoch", 0)),
                "resolved_transfer_global_step": int(checkpoint.get("global_step", 0)),
                "full_smb_vision_checkpoint": (
                    str(transferred.full_smb_vision_path)
                    if transferred.full_smb_vision_path is not None
                    else None
                ),
            }
        )
        return model, 0, 0, architecture_name, dict(architecture_config), checkpoint, vision, source

    raise ValueError(
        "init_checkpoint must be a Block SMB policy checkpoint or a Full SMB " "transfer checkpoint"
    )


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
    policy_loss_weight: float,
    world_model_weight: float,
    gradient_clip_norm: float,
    max_abs_loss: float,
    max_abs_scaled_reward: float,
    max_abs_prediction: float,
    deterministic_actions: bool,
    gradient_parameters: tuple[torch.nn.Parameter, ...],
    rollout_id: str,
) -> FullSMBEpisodeTrainingResult:
    observation = stage.reset(seed=seed)
    total_return = 0.0
    losses: list[float] = []
    policy_losses: list[float] = []
    entropies: list[float] = []
    gradient_norms: list[float] = []
    gradient_clip_events = 0
    scaled_rewards: list[float] = []
    value_prediction_abs_values: list[float] = []
    reward_prediction_abs_values: list[float] = []
    next_state_prediction_abs_values: list[float] = []
    dynamics_losses: list[float] = []
    world_model_losses: list[float] = []
    steps: list[FullSMBRolloutStep] = []
    world_model_state: WorldModelState | None = None
    recurrent_state_resets = 1
    boundary_counts: dict[str, int] = {"manual_reset": 1}
    for _step in range(max_steps):
        batch = stage.encode_observation(observation)
        forward = _coerce_policy_forward_result(
            _policy_action_logits_and_state(
                model,
                batch,
                device=device,
                world_model_state=world_model_state,
            )
        )
        logits = forward.logits
        next_world_model_state = forward.next_world_model_state
        _finite_tensor_or_raise("action_logits", logits)
        prediction_metrics = _prediction_safety_metrics(
            model,
            batch,
            forward,
            device=device,
            max_abs_prediction=max_abs_prediction,
        )
        value_prediction_abs_values.append(prediction_metrics["value_prediction_abs_max"])
        reward_prediction_abs_values.append(prediction_metrics["reward_prediction_abs_max"])
        next_state_prediction_abs_values.append(
            prediction_metrics["next_state_prediction_abs_max"]
        )
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic_actions:
            action_tensor = logits.argmax(dim=-1)
        else:
            action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        _finite_tensor_or_raise("action_log_prob", log_prob)
        observation, reward, terminated, truncated, info = stage.step(int(action_tensor.item()))
        boundary = _full_smb_rollout_boundary(
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        loss_dynamics = torch.zeros((), dtype=log_prob.dtype, device=log_prob.device)
        if world_model_weight > 0.0 and forward.next_state_pred is not None:
            with torch.no_grad():
                next_batch = stage.encode_observation(observation, info)
                next_state_target = next_batch.src_c.to(device).detach()
            if tuple(forward.next_state_pred.shape) != tuple(next_state_target.shape):
                raise ValueError(
                    "Full SMB world-model target shape mismatch: "
                    f"prediction={tuple(forward.next_state_pred.shape)}, "
                    f"target={tuple(next_state_target.shape)}"
                )
            _finite_tensor_or_raise("next_state_target", next_state_target)
            loss_dynamics = F.mse_loss(forward.next_state_pred, next_state_target)
            _finite_tensor_or_raise("loss_dynamics", loss_dynamics)
        loss_world_model = loss_dynamics * float(world_model_weight)
        _finite_tensor_or_raise("loss_world_model", loss_world_model)
        action = int(action_tensor.item())
        identifiers = _full_smb_rollout_identifiers(stage, info)
        steps.append(
            FullSMBRolloutStep(
                step_index=_step,
                action=action,
                action_name=SMB_ACTIONS[action].name,
                reward=float(reward),
                done=bool(terminated or truncated),
                terminated=bool(terminated),
                truncated=bool(truncated),
                episode_mask=float(boundary.episode_mask),
                boundary_reasons=boundary.reasons,
                scenario_id=identifiers["scenario_id"],
                task_id=identifiers["task_id"],
                emulator_state_id=identifiers["emulator_state_id"],
                signals=_selected_full_smb_signal_fields(info),
            )
        )
        scaled_reward_value = _checked_scaled_reward(
            reward,
            reward_scale=reward_scale,
            max_abs_scaled_reward=max_abs_scaled_reward,
        )
        scaled_reward = torch.as_tensor(
            scaled_reward_value,
            dtype=log_prob.dtype,
            device=log_prob.device,
        )
        entropy = distribution.entropy().mean()
        _finite_tensor_or_raise("action_entropy", entropy)
        policy_loss = -(log_prob.mean() * scaled_reward.detach())
        loss = policy_loss_weight * policy_loss + loss_world_model - entropy_weight * entropy
        _finite_tensor_or_raise("policy_loss", policy_loss)
        _finite_tensor_or_raise("loss", loss)
        _bounded_tensor_or_raise("loss", loss.detach(), max_abs_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _check_gradients_or_raise(gradient_parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            gradient_parameters,
            gradient_clip_norm,
        )
        _finite_tensor_or_raise("gradient_norm", grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        policy_losses.append(float(policy_loss.detach().cpu().item()))
        dynamics_losses.append(float(loss_dynamics.detach().cpu().item()))
        world_model_losses.append(float(loss_world_model.detach().cpu().item()))
        entropies.append(float(entropy.detach().cpu().item()))
        gradient_norm = float(grad_norm.detach().cpu().item())
        gradient_norms.append(gradient_norm)
        if gradient_norm > gradient_clip_norm:
            gradient_clip_events += 1
        scaled_rewards.append(float(scaled_reward_value))
        total_return += float(reward)
        if boundary.reset_recurrent_state:
            recurrent_state_resets += 1
            for reason in boundary.reasons:
                boundary_counts[reason] = boundary_counts.get(reason, 0) + 1
            world_model_state = None
        elif next_world_model_state is not None:
            world_model_state = next_world_model_state.detach()
        else:
            world_model_state = None
        if terminated or truncated:
            break
    metrics = {
        "return": total_return,
        "loss": _mean(losses),
        "steps": float(len(losses)),
        "recurrent_state_resets": float(recurrent_state_resets),
        "loss_policy": _mean(policy_losses),
        "loss_dynamics": _mean(dynamics_losses),
        "loss_world_model": _mean(world_model_losses),
        "mean_entropy": _mean(entropies),
        "min_entropy": min(entropies) if entropies else 0.0,
        "max_entropy": max(entropies) if entropies else 0.0,
        "mean_gradient_norm": _mean(gradient_norms),
        "max_gradient_norm": max(gradient_norms) if gradient_norms else 0.0,
        "gradient_clip_events": float(gradient_clip_events),
        "mean_scaled_reward": _mean(scaled_rewards),
        "max_abs_scaled_reward": _max_abs(scaled_rewards),
        "mean_value_prediction_abs": _mean(value_prediction_abs_values),
        "max_value_prediction_abs": _max(value_prediction_abs_values),
        "mean_reward_prediction_abs": _mean(reward_prediction_abs_values),
        "max_reward_prediction_abs": _max(reward_prediction_abs_values),
        "mean_next_state_prediction_abs": _mean(next_state_prediction_abs_values),
        "max_next_state_prediction_abs": _max(next_state_prediction_abs_values),
        "max_abs_prediction": max(
            _max(next_state_prediction_abs_values),
            _max(value_prediction_abs_values),
            _max(reward_prediction_abs_values),
        ),
    }
    for reason, count in boundary_counts.items():
        metrics[f"boundary_{reason}"] = float(count)
    return FullSMBEpisodeTrainingResult(
        metrics=metrics,
        rollout=FullSMBRolloutStorage(
            rollout_id=rollout_id,
            seed=int(seed),
            max_steps=int(max_steps),
            steps=tuple(steps),
        ),
    )


def _policy_action_logits(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    forward = _coerce_policy_forward_result(
        _policy_action_logits_and_state(model, batch, device=device)
    )
    return forward.logits


def _policy_action_logits_and_state(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
    world_model_state: WorldModelState | None = None,
) -> FullSMBPolicyForwardResult:
    _validate_full_smb_stage_batch(batch)
    src_a = batch.src_a.to(device)
    src_b = batch.src_b.to(device)
    src_c = batch.src_c.to(device)
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(episode_mask, dtype=src_c.dtype, device=src_c.device)
    outputs = model(
        src_a,
        src_b,
        src_c,
        tau=1.0,
        world_model_state=world_model_state,
        episode_mask=episode_mask,
        return_world_model_state=True,
    )
    next_state_pred = outputs[1]
    criticism = outputs[2]
    logits_a = outputs[4]
    next_world_model_state = outputs[-1]
    motor_primitives = getattr(model, "last_motor_primitives", None)
    logits = logits_a[:, -1, :FULL_SMB_ACTION_COUNT]
    logits = _apply_full_smb_motor_primitive_bias(logits, motor_primitives)
    return FullSMBPolicyForwardResult(
        logits=logits,
        next_world_model_state=next_world_model_state,
        next_state_pred=next_state_pred,
        criticism=criticism,
        motor_primitives=motor_primitives,
    )


def _apply_full_smb_motor_primitive_bias(
    logits: torch.Tensor,
    motor_primitives: Any,
) -> torch.Tensor:
    if motor_primitives is None or logits.size(-1) < FULL_SMB_ACTION_COUNT:
        return logits
    try:
        confidence = motor_primitives.confidence[:, -1]
        replan_probability = motor_primitives.replan_probability[:, -1]
    except (AttributeError, IndexError, TypeError):
        return logits
    combo_strength = (confidence * replan_probability).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    if combo_strength.ndim != 1 or combo_strength.size(0) != logits.size(0):
        return logits

    max_boost = 12.0
    base_boost = (0.5 * combo_strength).clamp(min=0.0, max=max_boost)
    bias = torch.zeros_like(logits)
    bias = bias + _combined_full_smb_action_bias(
        logits,
        primary=int(SMBAction.RIGHT),
        jump=int(SMBAction.JUMP),
        combo=int(SMBAction.RIGHT_JUMP),
        base_boost=base_boost,
        max_boost=max_boost,
    )
    bias = bias + _combined_full_smb_action_bias(
        logits,
        primary=int(SMBAction.LEFT),
        jump=int(SMBAction.JUMP),
        combo=int(SMBAction.LEFT_JUMP),
        base_boost=base_boost,
        max_boost=max_boost,
    )
    return logits + bias


def _combined_full_smb_action_bias(
    logits: torch.Tensor,
    *,
    primary: int,
    jump: int,
    combo: int,
    base_boost: torch.Tensor,
    max_boost: float,
) -> torch.Tensor:
    del jump
    primary_support = logits[:, primary]
    combo_gap = (primary_support - logits[:, combo]).clamp(min=0.0, max=max_boost)
    active = (base_boost > 0.0).to(dtype=logits.dtype, device=logits.device)
    boost = (base_boost + combo_gap) * active
    one_hot = torch.zeros(logits.size(-1), dtype=logits.dtype, device=logits.device)
    one_hot[combo] = 1.0
    return boost.unsqueeze(-1) * one_hot.unsqueeze(0)


def _coerce_policy_forward_result(value: Any) -> FullSMBPolicyForwardResult:
    if isinstance(value, FullSMBPolicyForwardResult):
        return value
    logits, next_world_model_state = value
    return FullSMBPolicyForwardResult(
        logits=logits,
        next_world_model_state=next_world_model_state,
    )


def _prediction_safety_metrics(
    model: torch.nn.Module,
    batch: StageBatch,
    forward: FullSMBPolicyForwardResult,
    *,
    device: torch.device,
    max_abs_prediction: float,
) -> dict[str, float]:
    value_prediction_abs_max = 0.0
    reward_prediction_abs_max = 0.0
    next_state_prediction_abs_max = 0.0
    with torch.no_grad():
        if forward.next_state_pred is not None:
            _finite_tensor_or_raise("next_state_prediction", forward.next_state_pred)
            _bounded_tensor_or_raise(
                "next_state_prediction",
                forward.next_state_pred,
                max_abs_prediction,
            )
            next_state_prediction_abs_max = _tensor_abs_max(forward.next_state_pred)
        if hasattr(model, "predict_value"):
            value_pred = model.predict_value(batch.src_c.to(device).detach())
            _finite_tensor_or_raise("value_prediction", value_pred)
            _bounded_tensor_or_raise(
                "value_prediction",
                value_pred,
                max_abs_prediction,
            )
            value_prediction_abs_max = _tensor_abs_max(value_pred)
        if forward.next_state_pred is not None and hasattr(model, "predict_reward"):
            reward_pred = model.predict_reward(forward.next_state_pred.detach())
            _finite_tensor_or_raise("reward_prediction", reward_pred)
            _bounded_tensor_or_raise(
                "reward_prediction",
                reward_pred,
                max_abs_prediction,
            )
            reward_prediction_abs_max = _tensor_abs_max(reward_pred)
    return {
        "value_prediction_abs_max": value_prediction_abs_max,
        "reward_prediction_abs_max": reward_prediction_abs_max,
        "next_state_prediction_abs_max": next_state_prediction_abs_max,
    }


def _checked_scaled_reward(
    reward: float,
    *,
    reward_scale: float,
    max_abs_scaled_reward: float,
) -> float:
    scaled_reward = float(reward) * float(reward_scale)
    if not math.isfinite(scaled_reward):
        raise FloatingPointError(
            "Full SMB training stopped early: scaled reward is NaN or infinite"
        )
    if abs(scaled_reward) > max_abs_scaled_reward:
        raise FloatingPointError(
            "Full SMB training stopped early: scaled reward "
            f"{scaled_reward:.6g} exceeds max_abs_scaled_reward "
            f"{max_abs_scaled_reward:.6g}"
        )
    return scaled_reward


def _finite_tensor_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(
            f"Full SMB training stopped early: {name} contains NaN or infinite values"
        )


def _bounded_tensor_or_raise(name: str, tensor: torch.Tensor, max_abs_value: float) -> None:
    observed = _tensor_abs_max(tensor)
    if observed > max_abs_value:
        raise FloatingPointError(
            f"Full SMB training stopped early: {name} absolute value "
            f"{observed:.6g} exceeds configured bound {max_abs_value:.6g}"
        )


def _tensor_abs_max(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max().cpu().item())


def _check_gradients_or_raise(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            _finite_tensor_or_raise("gradient", parameter.grad)


def _full_smb_rollout_boundary(
    *,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> FullSMBRolloutBoundary:
    reasons: list[str] = []
    signals = info.get("full_smb_signals") if isinstance(info, Mapping) else None
    signal_info = signals if isinstance(signals, Mapping) else {}
    if terminated:
        reasons.append("terminated")
    if truncated:
        reasons.append("truncated")
    if _info_flag(signal_info, ("death",)) or _info_flag(info, ("death", "dead")):
        reasons.append("death")
    if (
        truncated
        or _info_flag(signal_info, ("timeout",))
        or _info_flag(info, ("timeout", "time_up", "TimeLimit.truncated"))
        or _reason_matches_info(info, ("timeout", "time up", "time_up", "out of time"))
    ):
        reasons.append("timeout")
    if (
        _info_flag(signal_info, ("completion",))
        or _info_flag(info, ("level_complete", "completion", "flag_get"))
        or _reason_matches_info(info, ("level_complete", "complete", "clear", "goal", "flag"))
    ):
        reasons.append("level_completion")
    if (
        _info_flag(signal_info, ("game_over",))
        or _info_flag(info, ("game_over",))
        or _reason_matches_info(info, ("game_over", "game over"))
    ):
        reasons.append("game_over")
    if _info_flag(
        info, ("manual_reset", "reset_requested", "manual_reset_requested")
    ) or _reason_matches_info(info, ("manual_reset", "manual reset", "user_reset")):
        reasons.append("manual_reset")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return FullSMBRolloutBoundary(
        reset_recurrent_state=bool(unique_reasons),
        episode_mask=0.0 if unique_reasons else 1.0,
        reasons=unique_reasons,
    )


def _info_flag(info: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                continue
        try:
            tensor = torch.as_tensor(value)
        except (RuntimeError, TypeError, ValueError):
            continue
        if tensor.numel() == 1 and bool(tensor.item()):
            return True
    return False


def _reason_matches_info(info: Mapping[str, Any], patterns: tuple[str, ...]) -> bool:
    reason = None
    if "full_smb_signals" in info and isinstance(info["full_smb_signals"], Mapping):
        reason = info["full_smb_signals"].get("termination_reason")
    if reason is None:
        for key in ("termination_reason", "done_reason", "reason"):
            if key in info and info[key] is not None:
                reason = info[key]
                break
    if reason is None:
        return False
    normalized = str(reason).strip().lower()
    return any(pattern in normalized for pattern in patterns)


def _make_stage(
    make_stage: Optional[StageFactory],
    vision: FullSMBSegmentationVision,
    config: FullSMBTrainingConfig,
) -> FullSMBStage:
    if make_stage is not None:
        return make_stage(vision)
    observation_config = FullSMBObservationConfig()
    if config.frame_skip is not None:
        observation_config = FullSMBObservationConfig(frame_skip=config.frame_skip)
    return FullSMBStage(
        env_config=FullSMBEnvConfig(
            game=config.game_id,
            state=config.emulator_state,
            scenario=config.scenario,
        ),
        vision=vision,
        observation_config=observation_config,
        reward_config=config.reward_config,
    )


def _perception_checkpoint_path(config: FullSMBTrainingConfig) -> Optional[Path]:
    if config.perception_mode == FULL_SMB_PERCEPTION_REPLACE:
        return None
    return config.full_smb_vision_checkpoint


def _build_full_smb_perception(
    config: FullSMBTrainingConfig,
    device: torch.device,
) -> FullSMBSegmentationVision:
    return FullSMBSegmentationVision(
        checkpoint=_perception_checkpoint_path(config),
        device=device,
        freeze=config.freeze_vision,
    )


def _perception_optimizer_enabled(config: FullSMBTrainingConfig) -> bool:
    return config.perception_mode in {
        FULL_SMB_PERCEPTION_FINE_TUNE,
        FULL_SMB_PERCEPTION_REPLACE,
    }


def _make_training_optimizer(
    model: torch.nn.Module,
    vision: FullSMBSegmentationVision,
    config: FullSMBTrainingConfig,
) -> optim.Optimizer:
    param_groups: list[dict[str, Any]] = [{"params": list(model.parameters()), "name": "policy"}]
    if _perception_optimizer_enabled(config):
        perception_parameters = [param for param in vision.parameters() if param.requires_grad]
        if perception_parameters:
            param_groups.append({"params": perception_parameters, "name": "perception"})
    return optim.AdamW(param_groups, lr=config.learning_rate)


def _gradient_parameters(
    model: torch.nn.Module,
    vision: FullSMBSegmentationVision,
    config: FullSMBTrainingConfig,
) -> tuple[torch.nn.Parameter, ...]:
    parameters = list(model.parameters())
    if _perception_optimizer_enabled(config):
        parameters.extend(param for param in vision.parameters() if param.requires_grad)
    return tuple(parameters)


def _set_perception_training_mode(
    vision: FullSMBSegmentationVision,
    config: FullSMBTrainingConfig,
) -> None:
    if not isinstance(vision, torch.nn.Module):
        return
    if _perception_optimizer_enabled(config):
        vision.train()
    else:
        vision.eval()


def _restore_perception_state(
    vision: FullSMBSegmentationVision,
    checkpoint: Optional[Mapping[str, Any]],
) -> None:
    if checkpoint is None:
        return
    perception_state = checkpoint.get("states", {}).get("perception")
    if perception_state is None:
        return
    load_result = vision.load_state_dict(perception_state, strict=False)
    missing = tuple(load_result.missing_keys)
    unexpected = tuple(load_result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            "checkpoint perception state is incompatible with Full SMB perception: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _restore_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint: Optional[Mapping[str, Any]],
    *,
    strict: bool,
) -> None:
    if checkpoint is None:
        return
    optimizer_state = checkpoint.get("states", {}).get("optimizer")
    if optimizer_state is None:
        return
    try:
        optimizer.load_state_dict(optimizer_state)
    except ValueError as exc:
        if strict:
            raise ValueError(
                "checkpoint optimizer state is incompatible with the selected "
                "Full SMB perception mode"
            ) from exc


def _perception_checkpoint_metadata(
    config: FullSMBTrainingConfig,
    vision: Optional[FullSMBSegmentationVision] = None,
) -> dict[str, Any]:
    requested_checkpoint = config.full_smb_vision_checkpoint
    resolved_checkpoint = None
    if vision is not None:
        resolved_checkpoint = vision.checkpoint_path
    elif config.perception_mode != FULL_SMB_PERCEPTION_REPLACE:
        resolved_checkpoint = requested_checkpoint
    return {
        "mode": config.perception_mode,
        "encoder": "full_smb_vit",
        "requested_checkpoint_path": (
            str(requested_checkpoint) if requested_checkpoint is not None else None
        ),
        "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint is not None else None,
        "checkpoint_loaded": resolved_checkpoint is not None,
        "frozen": bool(config.freeze_vision),
        "trainable": not bool(config.freeze_vision),
        "optimizer_updates_enabled": _perception_optimizer_enabled(config),
        "state_saved": bool(_perception_optimizer_enabled(config) and vision is not None),
    }


def _rollout_metadata(config: FullSMBTrainingConfig) -> dict[str, Any]:
    return {
        "rollout_length": int(config.rollout_length),
        "updates_per_epoch": int(config.updates_per_epoch),
        "epochs": int(config.epochs),
        "vector_env_count": int(config.vector_env_count),
        "active_vector_env_count": 1,
        "vectorized_training_enabled": False,
        "recurrent_state_policy": "carry_until_full_smb_boundary",
        "recurrent_state_reset_reasons": (
            "manual_reset",
            "terminated",
            "truncated",
            "death",
            "timeout",
            "level_completion",
            "game_over",
        ),
    }


def _rollout_storage_metadata(
    rollouts: list[FullSMBRolloutStorage] | tuple[FullSMBRolloutStorage, ...],
) -> dict[str, Any]:
    stored_rollouts = [rollout.as_dict() for rollout in rollouts]
    return {
        "schema_version": 1,
        "storage_kind": "compact_full_smb_rollout_replay",
        "stored_rollouts": len(stored_rollouts),
        "stored_steps": int(sum(rollout["step_count"] for rollout in stored_rollouts)),
        "fields": _FULL_SMB_ROLLOUT_STORAGE_FIELDS,
        "rollouts": stored_rollouts,
    }


def _evaluation_metadata(
    config: FullSMBTrainingConfig,
    evaluations: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    stored_evaluations = [to_plain_data(evaluation) for evaluation in evaluations]
    return {
        "schema_version": 1,
        "cadence": "periodic_deterministic",
        "enabled": bool(config.evaluation_episodes > 0 and config.evaluation_max_steps > 0),
        "interval_epochs": int(config.evaluation_interval_epochs),
        "episodes": int(config.evaluation_episodes),
        "max_steps_per_episode": int(config.evaluation_max_steps),
        "separate_from_training_rollouts": True,
        "stored_evaluations": len(stored_evaluations),
        "evaluations": stored_evaluations,
    }


def _rng_state_metadata(
    config: FullSMBTrainingConfig,
    states: Mapping[str, Any],
) -> dict[str, Any]:
    python_rng = states.get("python_rng")
    return {
        "schema_version": 1,
        "saved_state_keys": [
            key for key in ("torch_rng", "python_rng", "numpy_rng") if key in states
        ],
        "deterministic_algorithms_requested": bool(config.deterministic),
        "deterministic_actions": bool(config.deterministic_actions),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "python_random_version": (
            int(python_rng[0]) if isinstance(python_rng, tuple) and python_rng else None
        ),
    }


def _task_curriculum_metadata(
    config: FullSMBTrainingConfig,
    *,
    epoch: int,
    global_step: int,
    rollouts: list[FullSMBRolloutStorage] | tuple[FullSMBRolloutStorage, ...],
) -> dict[str, Any]:
    rollout_records = [rollout.as_dict() for rollout in rollouts]
    completed_rollouts = len(rollout_records)
    completed_steps = int(sum(record["step_count"] for record in rollout_records))
    training_complete = int(epoch) >= int(config.epochs)
    next_update_index = None if training_complete else 1
    next_epoch = None if training_complete else int(epoch) + 1
    next_episode_seed = (
        None if training_complete else int(config.seed) + int(epoch) * int(config.updates_per_epoch)
    )
    task_ids = _unique_rollout_values(rollout_records, "task_id")
    scenario_ids = _unique_rollout_values(rollout_records, "scenario_id")
    emulator_state_ids = _unique_rollout_values(rollout_records, "emulator_state_id")
    return {
        "schema_version": 1,
        "schedule_kind": "seeded_epoch_update_rollouts",
        "task_source": (
            "stage_info_fields"
            if task_ids or scenario_ids or emulator_state_ids
            else "stage_default"
        ),
        "base_seed": int(config.seed),
        "episode_seed_formula": "seed + epoch_index * updates_per_epoch + update_index",
        "completed_epoch": int(epoch),
        "target_epochs": int(config.epochs),
        "updates_per_epoch": int(config.updates_per_epoch),
        "global_step": int(global_step),
        "completed_rollouts": completed_rollouts,
        "completed_steps": completed_steps,
        "training_complete": training_complete,
        "next_epoch": next_epoch,
        "next_update_index": next_update_index,
        "next_episode_seed": next_episode_seed,
        "observed_task_ids": task_ids,
        "observed_scenario_ids": scenario_ids,
        "observed_emulator_state_ids": emulator_state_ids,
        "rollout_ids": [str(record["rollout_id"]) for record in rollout_records],
    }


def _unique_rollout_values(
    rollout_records: list[Mapping[str, Any]],
    field_name: str,
) -> list[str]:
    values: list[str] = []
    for record in rollout_records:
        for step in record.get("steps", ()):
            if not isinstance(step, Mapping):
                continue
            value = step.get(field_name)
            if value is None or value == "":
                continue
            text = str(value)
            if text not in values:
                values.append(text)
    return values


def _full_smb_backend_metadata(
    stage: Optional[FullSMBStage],
    config: FullSMBTrainingConfig,
) -> dict[str, Any]:
    env_config = getattr(stage, "env_config", None)
    content_spec = getattr(stage, "content_spec", DEFAULT_FULL_SMB_CONTENT)
    signal_config = getattr(stage, "signal_config", None)
    observation_config = getattr(stage, "observation_config", None)
    reward_config = getattr(stage, "reward_config", config.reward_config)
    backend = getattr(stage, "backend", None)
    env = getattr(stage, "env", None)
    buttons = ()
    if stage is not None:
        try:
            buttons = tuple(str(button) for button in stage.buttons)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            buttons = ()
    env_manifest = _metadata_manifest(env_config) or {"game": DEFAULT_FULL_SMB_CONTENT.game}
    return {
        "schema_version": 1,
        "provider": "stable-retro",
        "stage_spec": {
            "name": FULL_SMB_SPEC.name,
            "action_space_name": FULL_SMB_SPEC.action_space_name,
            "action_count": FULL_SMB_SPEC.action_count,
        },
        "stage_class": _class_path(stage),
        "backend_adapter_class": _class_path(backend),
        "env_class": _class_path(env),
        "env_config": env_manifest,
        "content": _metadata_manifest(content_spec),
        "signal_config": _metadata_manifest(signal_config),
        "observation": _metadata_manifest(observation_config),
        "reward": _metadata_manifest(reward_config),
        "buttons": buttons,
        "headless": True,
    }


def _metadata_manifest(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_manifest"):
        return value.to_manifest()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return to_plain_data(value)


def _class_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _safety_metadata(config: FullSMBTrainingConfig) -> dict[str, Any]:
    return {
        "finite_checks_enabled": True,
        "early_stop_on_nonfinite": True,
        "early_stop_on_exploding_loss": True,
        "gradient_clip_norm": float(config.gradient_clip_norm),
        "reward_scale": float(config.reward_scale),
        "max_abs_loss": float(config.max_abs_loss),
        "max_abs_scaled_reward": float(config.max_abs_scaled_reward),
        "max_abs_prediction": float(config.max_abs_prediction),
        "prediction_bounds": {
            "min": -float(config.max_abs_prediction),
            "max": float(config.max_abs_prediction),
            "fields": ("value_prediction", "reward_prediction"),
        },
        "tracked_metrics": (
            "mean_entropy",
            "min_entropy",
            "max_entropy",
            "mean_gradient_norm",
            "max_gradient_norm",
            "gradient_clip_events",
            "mean_scaled_reward",
            "max_abs_scaled_reward",
            "mean_value_prediction_abs",
            "max_value_prediction_abs",
            "mean_reward_prediction_abs",
            "max_reward_prediction_abs",
            "max_abs_prediction",
        ),
    }


def _full_smb_rollout_identifiers(
    stage: FullSMBStage,
    info: Mapping[str, Any],
) -> dict[str, Optional[str]]:
    env_config = getattr(stage, "env_config", None)
    scenario_id = _first_present_string(
        info,
        ("scenario_id", "scenario_name", "scenario", "task_scenario"),
    )
    task_id = _first_present_string(
        info,
        ("task_id", "task_name", "task", "full_smb_task", "task_spec"),
    )
    emulator_state_id = _first_present_string(
        info,
        (
            "emulator_state_id",
            "state_id",
            "save_state_id",
            "save_state",
            "state",
        ),
    )
    if scenario_id is None and env_config is not None:
        scenario_id = _attribute_string(env_config, "scenario")
    if emulator_state_id is None and env_config is not None:
        emulator_state_id = _attribute_string(env_config, "state")
    return {
        "scenario_id": scenario_id,
        "task_id": task_id,
        "emulator_state_id": emulator_state_id,
    }


def _first_present_string(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Optional[str]:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None or value == "":
            continue
        return str(value)
    return None


def _attribute_string(value: Any, name: str) -> Optional[str]:
    item = getattr(value, name, None)
    if item is None or item == "":
        return None
    return str(item)


def _selected_full_smb_signal_fields(info: Mapping[str, Any]) -> dict[str, Any]:
    signals = info.get("full_smb_signals")
    source = signals if isinstance(signals, Mapping) else info
    selected = {
        field: to_plain_data(source[field])
        for field in _FULL_SMB_SELECTED_SIGNAL_FIELDS
        if field in source
    }
    if "reward_terms" in info and isinstance(info["reward_terms"], Mapping):
        selected["reward_terms"] = to_plain_data(info["reward_terms"])
    return selected


def _recording_metadata(config: FullSMBTrainingConfig) -> dict[str, Any]:
    recording_path = config.recording_path
    return {
        "schema_version": _FULL_SMB_RECORDING_SCHEMA_VERSION,
        "enabled": bool(config.recording_dir is not None or recording_path is not None),
        "recording_dir": str(config.recording_dir) if config.recording_dir else None,
        "recording_path": str(recording_path) if recording_path else None,
        "episode_artifact_format": "npz_compressed",
        "episode_fields": (
            "frames",
            "actions",
            "action_names",
            "rewards",
            "terminated",
            "truncated",
            "signals_json",
            "task_ids",
            "scenario_ids",
            "emulator_state_ids",
            "episode_metadata_json",
        ),
        "frames_are_initial_plus_post_step": True,
        "optional_video_export": bool(
            recording_path is not None and recording_path.suffix.lower() in _FULL_SMB_VIDEO_SUFFIXES
        ),
    }


def _resolve_full_smb_recording_targets(
    config: FullSMBTrainingConfig,
    recording_prefix: str,
) -> dict[str, Any]:
    enabled = bool(config.recording_dir is not None or config.recording_path is not None)
    safe_prefix = _safe_full_smb_recording_prefix(recording_prefix)
    recording_path = (
        _scoped_full_smb_recording_path(config.recording_path, safe_prefix)
        if config.recording_path is not None
        else None
    )
    video_path = (
        recording_path
        if recording_path is not None and recording_path.suffix.lower() in _FULL_SMB_VIDEO_SUFFIXES
        else None
    )
    manifest_path = recording_path if video_path is None else None
    if config.recording_dir is not None:
        episode_dir = config.recording_dir
        if safe_prefix:
            episode_dir = episode_dir / safe_prefix
    elif recording_path is not None:
        episode_dir = recording_path.parent / f"{recording_path.stem}_episodes"
    else:
        episode_dir = None
    return {
        "schema_version": _FULL_SMB_RECORDING_SCHEMA_VERSION,
        "enabled": enabled,
        "recording_prefix": safe_prefix,
        "episode_dir": episode_dir,
        "manifest_path": manifest_path,
        "video_path": video_path,
        "configured_recording_dir": config.recording_dir,
        "configured_recording_path": config.recording_path,
    }


def _safe_full_smb_recording_prefix(prefix: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(prefix or "evaluation")
    ).strip("_")
    return cleaned or "evaluation"


def _scoped_full_smb_recording_path(path: Path, recording_prefix: str) -> Path:
    if recording_prefix == "evaluation":
        return path
    return path.with_name(f"{path.stem}_{recording_prefix}{path.suffix}")


def _start_full_smb_episode_recording(
    *,
    episode_index: int,
    seed: int,
    observation: np.ndarray,
    info: Mapping[str, Any],
    stage: FullSMBStage,
) -> dict[str, Any]:
    identifiers = _full_smb_rollout_identifiers(stage, info)
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "frames": [_recording_frame_array(observation)],
        "actions": [],
        "action_names": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "signals": [],
        "task_ids": [],
        "scenario_ids": [],
        "emulator_state_ids": [],
        "initial_signals": _selected_full_smb_signal_fields(info),
        "initial_identifiers": identifiers,
    }


def _append_full_smb_episode_recording_step(
    recording: dict[str, Any],
    *,
    observation: np.ndarray,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    stage: FullSMBStage,
) -> None:
    identifiers = _full_smb_rollout_identifiers(stage, info)
    recording["frames"].append(_recording_frame_array(observation))
    recording["actions"].append(int(action))
    recording["action_names"].append(SMB_ACTIONS[int(action)].name)
    recording["rewards"].append(float(reward))
    recording["terminated"].append(bool(terminated))
    recording["truncated"].append(bool(truncated))
    recording["signals"].append(_selected_full_smb_signal_fields(info))
    recording["task_ids"].append(identifiers["task_id"] or "")
    recording["scenario_ids"].append(identifiers["scenario_id"] or "")
    recording["emulator_state_ids"].append(identifiers["emulator_state_id"] or "")


def _write_full_smb_episode_recording(
    recording: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    total_return: float,
) -> dict[str, Any]:
    episode_dir = targets.get("episode_dir")
    if not isinstance(episode_dir, Path):
        raise ValueError("Full SMB recording requires an episode directory")
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_index = int(recording["episode_index"])
    prefix = str(targets.get("recording_prefix") or "evaluation")
    artifact_path = episode_dir / f"{prefix}_episode{episode_index:04d}.npz"
    frames = np.stack(recording["frames"]).astype(np.uint8, copy=False)
    actions = np.asarray(recording["actions"], dtype=np.int64)
    rewards = np.asarray(recording["rewards"], dtype=np.float32)
    terminated = np.asarray(recording["terminated"], dtype=np.bool_)
    truncated = np.asarray(recording["truncated"], dtype=np.bool_)
    signals_json = np.asarray(
        [_json_dumps_plain(signal) for signal in recording["signals"]],
        dtype=np.str_,
    )
    np.savez_compressed(
        artifact_path,
        frames=frames,
        actions=actions,
        action_names=_string_array(recording["action_names"]),
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        signals_json=signals_json,
        task_ids=_string_array(recording["task_ids"]),
        scenario_ids=_string_array(recording["scenario_ids"]),
        emulator_state_ids=_string_array(recording["emulator_state_ids"]),
        episode_metadata_json=np.asarray(
            _json_dumps_plain(
                {
                    "schema_version": _FULL_SMB_RECORDING_SCHEMA_VERSION,
                    "episode_index": episode_index,
                    "seed": int(recording["seed"]),
                    "step_count": int(actions.shape[0]),
                    "total_return": float(total_return),
                    "frames_are_initial_plus_post_step": True,
                    "initial_signals": recording["initial_signals"],
                    "initial_identifiers": recording["initial_identifiers"],
                }
            ),
            dtype=np.str_,
        ),
    )
    video = _write_full_smb_episode_video(recording, targets, frames=frames)
    unique_task_ids = _unique_nonempty_strings(recording["task_ids"])
    unique_scenario_ids = _unique_nonempty_strings(recording["scenario_ids"])
    unique_emulator_state_ids = _unique_nonempty_strings(recording["emulator_state_ids"])
    artifact = {
        "schema_version": _FULL_SMB_RECORDING_SCHEMA_VERSION,
        "episode_index": episode_index,
        "seed": int(recording["seed"]),
        "path": str(artifact_path),
        "step_count": int(actions.shape[0]),
        "frame_count": int(frames.shape[0]),
        "total_return": float(total_return),
        "task_ids": unique_task_ids,
        "scenario_ids": unique_scenario_ids,
        "emulator_state_ids": unique_emulator_state_ids,
        "fields": _recording_metadata_fields(),
        "video": video,
    }
    return artifact


def _write_full_smb_episode_video(
    recording: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    frames: np.ndarray,
) -> dict[str, Any]:
    video_path = targets.get("video_path")
    if not isinstance(video_path, Path):
        return {"enabled": False, "status": "not_requested", "path": None}
    episode_index = int(recording["episode_index"])
    episode_count_suffix = f"_episode{episode_index:04d}"
    target_path = video_path
    if episode_index > 0:
        target_path = video_path.with_name(
            f"{video_path.stem}{episode_count_suffix}{video_path.suffix}"
        )
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return {
            "enabled": True,
            "status": "skipped",
            "path": str(target_path),
            "reason": "opencv-python is not installed",
        }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (int(width), int(height)),
    )
    if not writer.isOpened():
        return {
            "enabled": True,
            "status": "skipped",
            "path": str(target_path),
            "reason": "OpenCV could not open the video writer",
        }
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return {"enabled": True, "status": "written", "path": str(target_path)}


def _full_smb_recording_manifest(
    targets: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not targets.get("enabled"):
        return _empty_full_smb_recording_manifest()
    manifest = {
        "schema_version": _FULL_SMB_RECORDING_SCHEMA_VERSION,
        "enabled": True,
        "recording_dir": (
            str(targets["configured_recording_dir"])
            if targets.get("configured_recording_dir") is not None
            else None
        ),
        "recording_path": (
            str(targets["configured_recording_path"])
            if targets.get("configured_recording_path") is not None
            else None
        ),
        "resolved_episode_dir": (
            str(targets["episode_dir"]) if targets.get("episode_dir") is not None else None
        ),
        "recording_prefix": targets.get("recording_prefix"),
        "artifact_count": len(artifacts),
        "artifacts": to_plain_data(artifacts),
        "manifest_path": (
            str(targets["manifest_path"]) if targets.get("manifest_path") is not None else None
        ),
        "video_export": {
            "enabled": targets.get("video_path") is not None,
            "artifacts": [artifact["video"] for artifact in artifacts],
        },
    }
    manifest_path = targets.get("manifest_path")
    if isinstance(manifest_path, Path):
        _write_full_smb_recording_manifest(manifest_path, manifest)
    return manifest


def _write_full_smb_recording_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_dumps_plain(manifest)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(
            path,
            manifest_json=np.asarray(payload, dtype=np.str_),
            episode_paths=_string_array(
                artifact["path"] for artifact in manifest.get("artifacts", ())
            ),
        )
        return
    path.write_text(payload + "\n", encoding="utf-8")


def _recording_frame_array(observation: np.ndarray) -> np.ndarray:
    array = np.asarray(observation)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("Full SMB recording frames must have RGB or RGBA channel layout")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if bool(array.size) and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array)


def _recording_metadata_fields() -> tuple[str, ...]:
    return (
        "frames",
        "actions",
        "action_names",
        "rewards",
        "terminated",
        "truncated",
        "signals_json",
        "task_ids",
        "scenario_ids",
        "emulator_state_ids",
        "episode_metadata_json",
    )


def _string_array(values: Any) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def _unique_nonempty_strings(values: Any) -> list[str]:
    unique: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in unique:
            continue
        unique.append(text)
    return unique


def _json_dumps_plain(value: Any) -> str:
    return json.dumps(to_plain_data(value), sort_keys=True)


def _tracking_metadata(config: FullSMBTrainingConfig) -> dict[str, Any]:
    return {
        "backend": config.tracking_backend,
        "log_dir": str(config.tracking_log_dir) if config.tracking_log_dir else None,
        "project": config.tracking_project,
        "run_name": config.tracking_run_name,
        "mode": config.tracking_mode,
        "structured_log_path": str(config.log_path) if config.log_path else None,
    }


def _training_source_metadata(
    mode: str,
    *,
    checkpoint_path: Optional[Path],
    checkpoint: Optional[Mapping[str, Any]],
    architecture_name: str,
    architecture_config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_stage = None
    checkpoint_model_name = None
    checkpoint_kind = None
    checkpoint_epoch = None
    checkpoint_global_step = None
    if checkpoint is not None:
        checkpoint_stage = checkpoint.get("stage")
        checkpoint_model_name = checkpoint.get("model_name")
        checkpoint_kind = checkpoint.get("checkpoint_kind")
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_global_step = int(checkpoint.get("global_step", 0))
    provenance = _source_checkpoint_provenance(
        mode,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_model_name": checkpoint_model_name,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "checkpoint_schema_version": provenance["checkpoint_schema_version"],
        "source_checkpoint_provenance": provenance,
        "uses_shared_architecture_factory": True,
        "architecture_name": architecture_name,
        "architecture_config": dict(architecture_config),
    }


def _source_checkpoint_provenance(
    mode: str,
    *,
    checkpoint_path: Optional[Path],
    checkpoint: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, Mapping) else {}
    config = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
    return {
        "schema_version": 1,
        "mode": mode,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_schema_version": (
            checkpoint.get(CHECKPOINT_SCHEMA_KEY) if isinstance(checkpoint, Mapping) else None
        ),
        "stage": checkpoint.get("stage") if isinstance(checkpoint, Mapping) else None,
        "model_name": checkpoint.get("model_name") if isinstance(checkpoint, Mapping) else None,
        "checkpoint_kind": (
            checkpoint.get("checkpoint_kind") if isinstance(checkpoint, Mapping) else None
        ),
        "epoch": int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, Mapping) else None,
        "global_step": (
            int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, Mapping) else None
        ),
        "metrics": checkpoint.get("metrics", {}) if isinstance(checkpoint, Mapping) else {},
        "architecture": (
            checkpoint.get("architecture", {}) if isinstance(checkpoint, Mapping) else {}
        ),
        "code_revision": (metadata.get("code_revision") if isinstance(metadata, Mapping) else None),
        "upstream_training_source": (
            config.get("training_source") if isinstance(config, Mapping) else None
        ),
    }


def _validate_resume_contract(
    config: FullSMBTrainingConfig,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Full SMB resume checkpoint config must be a mapping")
    checked_fields: dict[str, Any] = {}
    for field_name in (
        "seed",
        "updates_per_epoch",
        "rollout_length",
        "max_steps_per_episode",
        "deterministic",
        "deterministic_actions",
    ):
        if field_name not in checkpoint_config:
            continue
        expected = checkpoint_config[field_name]
        actual = getattr(config, field_name)
        if actual != expected:
            raise ValueError(
                "Full SMB resume task schedule mismatch: "
                f"{field_name}={actual!r} does not match checkpoint value {expected!r}"
            )
        checked_fields[field_name] = actual

    previous_rollout = checkpoint_config.get("rollout", {})
    if isinstance(previous_rollout, Mapping):
        current_rollout = _rollout_metadata(config)
        for field_name in ("recurrent_state_policy", "recurrent_state_reset_reasons"):
            if field_name not in previous_rollout:
                continue
            expected = previous_rollout[field_name]
            actual = current_rollout[field_name]
            expected_value = tuple(expected) if isinstance(expected, (list, tuple)) else expected
            actual_value = tuple(actual) if isinstance(actual, (list, tuple)) else actual
            if actual_value != expected_value:
                raise ValueError(
                    "Full SMB resume recurrent-state contract mismatch: "
                    f"{field_name}={actual!r} does not match checkpoint value {expected!r}"
                )
            checked_fields[f"rollout.{field_name}"] = to_plain_data(actual)

    previous_tracking = checkpoint_config.get("tracking", {})
    if isinstance(previous_tracking, Mapping):
        current_tracking = _tracking_metadata(config)
        for field_name in ("backend", "log_dir", "project", "run_name", "mode"):
            if field_name not in previous_tracking:
                continue
            expected = previous_tracking[field_name]
            actual = current_tracking[field_name]
            if actual != expected:
                raise ValueError(
                    "Full SMB resume tracking destination mismatch: "
                    f"{field_name}={actual!r} does not match checkpoint value {expected!r}"
                )
            checked_fields[f"tracking.{field_name}"] = actual

    rng_state = checkpoint_config.get("rng_state", {})
    saved_rng_keys = []
    if isinstance(rng_state, Mapping):
        saved_rng_keys = list(rng_state.get("saved_state_keys", ()))
    missing_rng_keys = [
        key
        for key in ("torch_rng", "python_rng", "numpy_rng")
        if key not in checkpoint.get("states", {})
    ]
    if missing_rng_keys:
        raise ValueError(
            "Full SMB resume checkpoint is missing RNG state keys: " + ", ".join(missing_rng_keys)
        )
    return {
        "schema_version": 1,
        "validated": True,
        "start_epoch": int(checkpoint.get("epoch", 0)),
        "start_global_step": int(checkpoint.get("global_step", 0)),
        "checked_fields": checked_fields,
        "saved_rng_state_keys": saved_rng_keys,
        "required_rng_state_keys": ("torch_rng", "python_rng", "numpy_rng"),
    }


def _restore_resume_rng_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    states = checkpoint.get("states", {})
    restored: list[str] = []
    torch_rng = states.get("torch_rng")
    if torch_rng is not None:
        torch.set_rng_state(torch.as_tensor(torch_rng, dtype=torch.uint8).cpu())
        restored.append("torch_rng")
    python_rng = states.get("python_rng")
    if python_rng is not None:
        random.setstate(python_rng)
        restored.append("python_rng")
    numpy_rng = states.get("numpy_rng")
    if numpy_rng is not None:
        np.random.set_state(numpy_rng)
        restored.append("numpy_rng")
    return {
        "schema_version": 1,
        "restored": True,
        "restored_state_keys": restored,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
    }


def _matches_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    *,
    stage: str,
    model_name: str,
    checkpoint_kind: str,
) -> bool:
    return (
        checkpoint.get("stage") == stage
        and checkpoint.get("model_name") == model_name
        and checkpoint.get("checkpoint_kind") == checkpoint_kind
    )


def _stage_batch_contract_metadata() -> dict[str, Any]:
    return {
        "source": "FullSMBStage.encode_observation",
        "src_a": {"rank": 2, "sequence_length": FULL_SMB_SPEC.seq_len_a, "dtype": "long"},
        "src_b": {"rank": 2, "sequence_length": FULL_SMB_SPEC.seq_len_b, "dtype": "long"},
        "src_c": {
            "rank": 2,
            "feature_length": FULL_SMB_SPEC.seq_len_c,
            "dtype": "floating",
        },
    }


def _validate_full_smb_stage_batch(batch: StageBatch) -> None:
    if batch.src_a.ndim != 2 or batch.src_a.shape[1] != FULL_SMB_SPEC.seq_len_a:
        raise ValueError(
            "Full SMB stage batch src_a must have shape "
            f"[B, {FULL_SMB_SPEC.seq_len_a}], got {tuple(batch.src_a.shape)}"
        )
    if batch.src_b.ndim != 2 or batch.src_b.shape[1] != FULL_SMB_SPEC.seq_len_b:
        raise ValueError(
            "Full SMB stage batch src_b must have shape "
            f"[B, {FULL_SMB_SPEC.seq_len_b}], got {tuple(batch.src_b.shape)}"
        )
    if batch.src_c.ndim != 2 or batch.src_c.shape[1] != FULL_SMB_SPEC.seq_len_c:
        raise ValueError(
            "Full SMB stage batch src_c must have shape "
            f"[B, {FULL_SMB_SPEC.seq_len_c}], got {tuple(batch.src_c.shape)}"
        )
    batch_size = batch.src_a.shape[0]
    if batch.src_b.shape[0] != batch_size or batch.src_c.shape[0] != batch_size:
        raise ValueError("Full SMB stage batch streams must share the same batch size")
    if batch.src_a.dtype != torch.long or batch.src_b.dtype != torch.long:
        raise ValueError("Full SMB stage batch A/B streams must use torch.long tokens")
    if not torch.is_floating_point(batch.src_c):
        raise ValueError("Full SMB stage batch C stream must use floating point features")


def append_full_smb_log_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append one structured JSONL event for Full SMB training operations."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"stage": FULL_SMB_SPEC.name, **dict(event)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_plain_data(record), sort_keys=True) + "\n")


def _initialize_full_smb_log(config: FullSMBTrainingConfig) -> None:
    if config.log_path is None:
        return
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    if config.resume_path is None:
        config.log_path.write_text("", encoding="utf-8")


def _log_full_smb_event(
    training_config: FullSMBTrainingConfig,
    event: str,
    **payload: Any,
) -> None:
    if training_config.log_path is None:
        return
    append_full_smb_log_event(training_config.log_path, {"event": event, **payload})


def _should_evaluate_epoch(config: FullSMBTrainingConfig, completed_epoch: int) -> bool:
    if config.evaluation_episodes <= 0 or config.evaluation_max_steps <= 0:
        return False
    return (
        completed_epoch == config.epochs or completed_epoch % config.evaluation_interval_epochs == 0
    )


def _full_smb_evaluation_metrics(evaluation: FullSMBEvaluationResult) -> dict[str, float]:
    metrics = {
        "eval_mean_return": float(evaluation.mean_return),
        "eval_success_rate": float(evaluation.success_rate),
        "eval_steps": float(evaluation.steps),
        "eval_terminated_count": float(evaluation.terminated_count),
        "eval_truncated_count": float(evaluation.truncated_count),
    }
    for metric_name, metric_value in evaluation.tuning_metrics.items():
        metrics[f"eval_{metric_name}"] = float(metric_value)
    metrics["eval_success_thresholds_met"] = float(evaluation.success_thresholds_met)
    return metrics


def _start_full_smb_fixed_task_episode_metrics(
    stage: FullSMBStage,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "task_name": _fixed_full_smb_task_name(stage, info),
        "progress": [],
        "scores": [],
        "coins": [],
        "completion": False,
        "death": False,
    }
    _update_full_smb_fixed_task_episode_metrics(metrics, stage, info)
    return metrics


def _update_full_smb_fixed_task_episode_metrics(
    metrics: dict[str, Any],
    stage: FullSMBStage,
    info: Mapping[str, Any],
) -> None:
    if metrics["task_name"] is None:
        metrics["task_name"] = _fixed_full_smb_task_name(stage, info)
    source = _full_smb_signal_source(info)
    progress = _full_smb_signal_progress(source)
    if progress is not None:
        metrics["progress"].append(progress)
    score = _optional_float(source.get("score"))
    if score is not None:
        metrics["scores"].append(score)
    coins = _optional_float(source.get("coins"))
    if coins is None:
        collectibles = source.get("collectibles")
        if isinstance(collectibles, Mapping):
            coins = _optional_float(collectibles.get("coins"))
    if coins is not None:
        metrics["coins"].append(coins)
    metrics["completion"] = bool(metrics["completion"] or source.get("completion", False))
    metrics["death"] = bool(
        metrics["death"] or source.get("death", False) or source.get("game_over", False)
    )


def _finalize_full_smb_fixed_task_episode_metrics(
    metrics: Mapping[str, Any],
    *,
    episode_return: float,
) -> dict[str, float]:
    progress_values = tuple(float(value) for value in metrics.get("progress", ()))
    scores = tuple(float(value) for value in metrics.get("scores", ()))
    coins = tuple(float(value) for value in metrics.get("coins", ()))
    death = bool(metrics.get("death", False))
    completion = bool(metrics.get("completion", False))
    return {
        "max_progress": _max(progress_values),
        "mean_progress": _mean(progress_values),
        "completion_rate": 1.0 if completion else 0.0,
        "survival_rate": 0.0 if death else 1.0,
        "mean_score": scores[-1] if scores else 0.0,
        "mean_coins": coins[-1] if coins else 0.0,
        "death_count": 1.0 if death else 0.0,
        "mean_return": float(episode_return),
    }


def _full_smb_fixed_task_results(
    episode_metrics: Mapping[str, list[dict[str, float]]],
    *,
    evaluation_episodes: int,
    evaluation_max_steps: int,
) -> dict[str, Any]:
    raw_results = {
        task_name: _aggregate_full_smb_fixed_task_episode_metrics(episodes)
        for task_name, episodes in episode_metrics.items()
        if task_name in FIXED_FULL_SMB_SUCCESS_THRESHOLDS
    }
    threshold_results = evaluate_fixed_full_smb_success_thresholds(
        raw_results,
        evaluation_episodes=evaluation_episodes,
        evaluation_max_steps=evaluation_max_steps,
    )
    results: dict[str, Any] = {}
    for task_name, raw_result in raw_results.items():
        threshold_result = dict(threshold_results.get(task_name, {}))
        threshold = threshold_result.pop("threshold", None)
        threshold_met = bool(threshold_result.pop("threshold_met", False))
        results[task_name] = {
            **raw_result,
            "threshold": threshold,
            "threshold_met": threshold_met,
            "threshold_diagnostics": threshold_result,
        }
    return results


def _aggregate_full_smb_fixed_task_episode_metrics(
    episodes: list[dict[str, float]],
) -> dict[str, float]:
    progress = tuple(float(episode.get("max_progress", 0.0)) for episode in episodes)
    scores = tuple(float(episode.get("mean_score", 0.0)) for episode in episodes)
    coins = tuple(float(episode.get("mean_coins", 0.0)) for episode in episodes)
    returns = tuple(float(episode.get("mean_return", 0.0)) for episode in episodes)
    completion = tuple(float(episode.get("completion_rate", 0.0)) for episode in episodes)
    survival = tuple(float(episode.get("survival_rate", 0.0)) for episode in episodes)
    deaths = tuple(float(episode.get("death_count", 0.0)) for episode in episodes)
    return {
        "episodes": float(len(episodes)),
        "max_progress": _max(progress),
        "mean_progress": _mean(progress),
        "completion_rate": _mean(completion),
        "survival_rate": _mean(survival),
        "mean_score": _mean(scores),
        "mean_coins": _mean(coins),
        "death_count": float(sum(deaths)),
        "mean_return": _mean(returns),
    }


def _fixed_full_smb_task_name(stage: FullSMBStage, info: Mapping[str, Any]) -> Optional[str]:
    identifiers = _full_smb_rollout_identifiers(stage, info)
    for field_name in ("task_id", "scenario_id", "emulator_state_id"):
        value = identifiers.get(field_name)
        if value in FIXED_FULL_SMB_SUCCESS_THRESHOLDS:
            return value
    source = _full_smb_signal_source(info)
    for value in (source.get("level"), identifiers.get("emulator_state_id")):
        task_name = _fixed_full_smb_task_name_from_level(value)
        if task_name is not None:
            return task_name
    return None


def _fixed_full_smb_task_name_from_level(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = _full_smb_level_selector_key(value)
    return _FULL_SMB_FIXED_LEVEL_TASK_NAMES.get(key)


def _full_smb_level_selector_key(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "")


def _full_smb_state_from_level(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = _full_smb_level_selector_key(text)
    suffix = key[len("level") :] if key.startswith("level") else key
    if suffix and suffix[0].isdigit():
        return f"Level{suffix}"
    return text


def _full_smb_task_matches_level(task: FullSMBTaskSpec, level: str) -> bool:
    key = _full_smb_level_selector_key(level)
    state = task.start.state
    state_key = _full_smb_level_selector_key(state) if state is not None else ""
    canonical_state = _full_smb_state_from_level(level)
    canonical_key = (
        _full_smb_level_selector_key(canonical_state) if canonical_state is not None else ""
    )
    return state_key in {key, canonical_key}


def _select_full_smb_task(
    *,
    task_set: Optional[str],
    task_name: Optional[str],
    level: Optional[str],
) -> Optional[FullSMBTaskSpec]:
    catalog = full_smb_task_catalog()
    if task_name is not None:
        try:
            task = catalog.task(task_name)
        except KeyError as exc:
            raise ValueError(f"unknown Full SMB task {task_name!r}") from exc
        if task_set is not None and task.task_set != task_set:
            raise ValueError(
                f"Full SMB task {task_name!r} belongs to task set "
                f"{task.task_set!r}, not {task_set!r}"
            )
        if level is not None and not _full_smb_task_matches_level(task, level):
            raise ValueError(f"Full SMB task {task_name!r} does not start at level {level!r}")
        return task
    if level is not None:
        if task_set is None:
            task_name_for_level = _fixed_full_smb_task_name_from_level(level)
            if task_name_for_level is not None:
                return catalog.task(task_name_for_level)
        candidates = catalog.tasks_for_set(task_set) if task_set is not None else catalog.tasks
        for task in candidates:
            if _full_smb_task_matches_level(task, level):
                return task
        if task_set is not None:
            raise ValueError(f"Full SMB task set {task_set!r} has no level {level!r}")
        return None
    if task_set is not None:
        tasks = catalog.tasks_for_set(task_set)
        if not tasks:
            raise ValueError(f"Full SMB task set {task_set!r} has no tasks")
        return tasks[0]
    return None


def _resolve_full_smb_task_selection(args: argparse.Namespace) -> dict[str, Optional[str]]:
    task_set = getattr(args, "task_set", None)
    task_name = getattr(args, "task_name", None)
    level = getattr(args, "level", None)
    emulator_state = getattr(args, "emulator_state", None)
    scenario = getattr(args, "scenario", None)
    task = _select_full_smb_task(
        task_set=task_set,
        task_name=task_name,
        level=level,
    )
    if task is not None:
        task_set = task.task_set if task_set is None else task_set
        task_name = task.name
        level = task.start.state if level is None else level
        emulator_state = task.start.state if emulator_state is None else emulator_state
    elif level is not None and emulator_state is None:
        emulator_state = _full_smb_state_from_level(level)
    return {
        "task_set": task_set,
        "task_name": task_name,
        "level": level,
        "emulator_state": emulator_state,
        "scenario": scenario,
    }


def _full_smb_signal_source(info: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = info.get("full_smb_signals")
    return signals if isinstance(signals, Mapping) else info


def _full_smb_signal_progress(source: Mapping[str, Any]) -> Optional[float]:
    progress = _optional_float(source.get("progress"))
    if progress is not None:
        return progress
    position = source.get("position")
    if position is None:
        return None
    try:
        values = tuple(position)
    except TypeError:
        return None
    return _optional_float(values[0]) if values else None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _max(values: list[float] | tuple[float, ...]) -> float:
    return float(max(values)) if values else 0.0


def _max_abs(values: list[float] | tuple[float, ...]) -> float:
    return float(max((abs(value) for value in values), default=0.0))


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


def _add_training_update_args(
    parser: argparse.ArgumentParser,
    *,
    include_start_mode: bool,
) -> None:
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--episodes-per-epoch", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=64)
    parser.add_argument("--updates-per-epoch", type=int)
    parser.add_argument("--rollout-length", "--rollout-steps", dest="rollout_length", type=int)
    if include_start_mode:
        parser.add_argument(
            "--mode",
            "--training-mode",
            dest="training_mode",
            default=FULL_SMB_TRAINING_MODE_AUTO,
            choices=(
                FULL_SMB_TRAINING_MODE_AUTO,
                FULL_SMB_TRAINING_MODE_SCRATCH,
                "fine-tune",
                FULL_SMB_TRAINING_MODE_FINE_TUNE,
                "finetune",
            ),
            help=(
                "training start mode: auto infers from --resume/--init-checkpoint, "
                "scratch starts a new policy, fine_tune requires --init-checkpoint"
            ),
        )
    parser.add_argument("--vector-env-count", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--representation-weight", type=float, default=0.0)
    parser.add_argument(
        "--world-model-weight",
        type=float,
        default=_FULL_SMB_DEFAULT_WORLD_MODEL_WEIGHT,
    )
    parser.add_argument("--reward-loss-weight", type=float, default=0.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--action-aux-weight", type=float, default=0.0)
    parser.add_argument("--critic-loss-weight", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-abs-loss",
        type=float,
        default=_FULL_SMB_DEFAULT_MAX_ABS_LOSS,
    )
    parser.add_argument(
        "--max-abs-scaled-reward",
        type=float,
        default=_FULL_SMB_DEFAULT_MAX_ABS_SCALED_REWARD,
    )
    parser.add_argument(
        "--max-abs-prediction",
        type=float,
        default=_FULL_SMB_DEFAULT_MAX_ABS_PREDICTION,
    )
    parser.add_argument("--deterministic-actions", action="store_true")
    _add_recording_args(parser, use_defaults=False)


def _add_recording_args(parser: argparse.ArgumentParser, *, use_defaults: bool) -> None:
    parser.add_argument(
        "--record-dir",
        "--recording-dir",
        dest="recording_dir",
        type=Path,
        default=DEFAULT_FULL_SMB_RECORDING_DIR if use_defaults else None,
    )
    parser.add_argument(
        "--recording-path",
        "--record-output",
        type=Path,
        default=DEFAULT_FULL_SMB_RECORDING_MANIFEST if use_defaults else None,
    )


def _add_full_smb_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game-id", default=DEFAULT_FULL_SMB_CONTENT.game)
    parser.add_argument("--task-set", choices=FULL_SMB_TASK_SET_NAMES)
    parser.add_argument("--task", dest="task_name")
    parser.add_argument("--level")
    parser.add_argument("--state", dest="emulator_state")
    parser.add_argument("--scenario")
    parser.add_argument("--frame-skip", type=int)


def _add_play_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--human",
        action="store_true",
        help="use manual human actions instead of loading a policy checkpoint",
    )
    parser.add_argument("--steps", "--max-steps", dest="play_max_steps", type=int, default=1_000)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.set_defaults(play_render=True)
    parser.add_argument("--render", action="store_true", dest="play_render")
    parser.add_argument("--no-render", action="store_false", dest="play_render")
    parser.add_argument("--render-mode", choices=_FULL_SMB_PLAY_RENDER_MODES)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.set_defaults(policy_deterministic=True)
    policy_mode = parser.add_mutually_exclusive_group()
    policy_mode.add_argument(
        "--deterministic-policy",
        action="store_true",
        dest="policy_deterministic",
        help="take the highest-probability policy action at each decision point",
    )
    policy_mode.add_argument(
        "--sample",
        "--sampling-policy",
        action="store_false",
        dest="policy_deterministic",
        help="sample actions from the policy distribution instead of taking argmax",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pause-at-start", action="store_true")
    parser.set_defaults(interactive_controls=True, reset_on_done=True)
    parser.add_argument(
        "--no-interactive-controls",
        action="store_false",
        dest="interactive_controls",
        help="disable stdin controls: p=pause/resume, r=reset, q=quit",
    )
    parser.add_argument("--reset-on-done", action="store_true", dest="reset_on_done")
    parser.add_argument("--no-reset-on-done", action="store_false", dest="reset_on_done")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--recording-prefix", default="play")
    parser.set_defaults(inspection_overlay=False)
    parser.add_argument(
        "--inspection-overlay",
        "--overlay",
        action="store_true",
        dest="inspection_overlay",
        help="emit policy inspection overlay lines to stderr during play",
    )
    parser.add_argument(
        "--no-inspection-overlay",
        "--no-overlay",
        action="store_false",
        dest="inspection_overlay",
    )
    parser.add_argument("--overlay-interval-steps", type=int, default=1)
    parser.add_argument("--overlay-top-actions", type=int, default=5)
    parser.add_argument(
        "--human-default-action",
        type=_full_smb_action_id_arg,
        default=int(SMB_ACTIONS[0]),
        help="fallback action for human mode when no input/script action is available",
    )
    parser.add_argument(
        "--human-action",
        dest="human_actions",
        action="append",
        type=_full_smb_action_id_arg,
        default=[],
        help="scripted human action for non-interactive debugging; may be repeated",
    )
    _add_recording_args(parser, use_defaults=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_common_args(train)
    _add_training_update_args(train, include_start_mode=True)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--init-checkpoint", type=Path)
    train.add_argument("--save-checkpoints", action="store_true")

    resume = subparsers.add_parser("resume")
    _add_common_args(resume)
    _add_training_update_args(resume, include_start_mode=False)
    resume.add_argument("--checkpoint", required=True, type=Path, dest="resume_checkpoint")
    resume.add_argument(
        "--save-checkpoint",
        "--output-checkpoint",
        type=Path,
        dest="checkpoint",
        help="optional output checkpoint path; defaults to overwriting --checkpoint",
    )

    evaluate = subparsers.add_parser("evaluate")
    _add_common_args(evaluate)
    evaluate.add_argument("--policy-checkpoint", "--checkpoint", type=Path, required=True)

    record = subparsers.add_parser("record")
    _add_common_args(record)
    record.add_argument("--policy-checkpoint", "--checkpoint", type=Path, required=True)
    _add_recording_args(record, use_defaults=True)

    play = subparsers.add_parser("play")
    _add_common_args(play)
    play.add_argument("--policy-checkpoint", "--checkpoint", type=Path)
    _add_play_args(play)

    args = parser.parse_args(argv)
    if args.command == "play" and not args.human and args.policy_checkpoint is None:
        parser.error("play requires --checkpoint unless --human is set")
    config = _config_from_args(args)
    if args.command in {"train", "resume"}:
        result = train_full_smb_policy(config)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    elif args.command in {"evaluate", "record"}:
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
    elif args.command == "play":
        model = None
        architecture_name = config.architecture_name
        architecture_config = dict(config.architecture_config)
        if args.policy_checkpoint is not None and not args.human:
            model, _optimizer, checkpoint = load_full_smb_policy_checkpoint(
                args.policy_checkpoint,
                device=select_device(config.device),
            )
            architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        play_config = _play_config_from_args(args)
        play_training_config = FullSMBTrainingConfig(
            **{
                **to_plain_data(config),
                "architecture_name": architecture_name,
                "architecture_config": architecture_config,
            }
        )
        result = play_full_smb_policy(
            model,
            config=play_training_config,
            play_config=play_config,
        )
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
    parser.add_argument(
        "--perception-mode",
        choices=(
            FULL_SMB_PERCEPTION_FREEZE,
            "fine-tune",
            FULL_SMB_PERCEPTION_FINE_TUNE,
            FULL_SMB_PERCEPTION_REPLACE,
        ),
    )
    parser.add_argument(
        "--fine-tune-vision",
        action="store_true",
        help="legacy alias for --perception-mode fine_tune",
    )
    _add_reward_args(parser)
    parser.set_defaults(deterministic=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        dest="deterministic",
        help="enable deterministic Torch algorithms",
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_false",
        dest="deterministic",
        help="disable deterministic Torch algorithms",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=1)
    parser.add_argument("--evaluation-max-steps", type=int, default=64)
    parser.add_argument("--evaluation-interval-epochs", type=int, default=1)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--tracking-backend", choices=TRACKING_BACKENDS, default="none")
    parser.add_argument("--tracking-log-dir", type=Path)
    parser.add_argument("--tracking-project", default="retroagi")
    parser.add_argument("--tracking-run-name")
    parser.add_argument("--tracking-mode")
    _add_full_smb_runtime_args(parser)


def _add_reward_args(parser: argparse.ArgumentParser) -> None:
    for term_name in FullSMBRewardConfig().term_names:
        parser.add_argument(
            f"--reward-{term_name.replace('_', '-')}",
            dest=f"reward_{term_name}",
            type=float,
        )


def _reward_config_from_args(args: argparse.Namespace) -> FullSMBRewardConfig:
    values = FullSMBRewardConfig().as_dict()
    for term_name in FullSMBRewardConfig().term_names:
        override = getattr(args, f"reward_{term_name}", None)
        if override is not None:
            values[term_name] = float(override)
    return FullSMBRewardConfig(**values)


def _play_config_from_args(args: argparse.Namespace) -> FullSMBPlayConfig:
    render = bool(args.play_render)
    if args.render_mode is not None:
        render = args.render_mode != "none"
    return FullSMBPlayConfig(
        max_steps=args.play_max_steps,
        action_repeat=args.action_repeat,
        render=render,
        fps=args.fps,
        deterministic_policy=bool(args.policy_deterministic),
        sampling_temperature=args.temperature,
        reset_on_done=args.reset_on_done,
        pause_at_start=args.pause_at_start,
        interactive_controls=args.interactive_controls,
        recording_prefix=args.recording_prefix,
        inspection_overlay=args.inspection_overlay,
        overlay_interval_steps=args.overlay_interval_steps,
        overlay_top_actions=args.overlay_top_actions,
        human_control=bool(args.human),
        human_default_action=args.human_default_action,
        human_action_script=tuple(args.human_actions or ()),
    )


def _config_from_args(args: argparse.Namespace) -> FullSMBTrainingConfig:
    architecture_config = dict(args.architecture_config or ())
    perception_mode = getattr(args, "perception_mode", None)
    if args.fine_tune_vision:
        if (
            perception_mode is not None
            and _resolve_perception_mode(
                perception_mode,
                freeze_vision=False,
            )
            != FULL_SMB_PERCEPTION_FINE_TUNE
        ):
            raise ValueError("--fine-tune-vision conflicts with --perception-mode")
        perception_mode = FULL_SMB_PERCEPTION_FINE_TUNE
    is_resume_command = getattr(args, "command", None) == "resume"
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    resume_path = resume_checkpoint if is_resume_command else getattr(args, "resume", None)
    checkpoint_path = getattr(args, "checkpoint", None)
    if is_resume_command and checkpoint_path is None:
        checkpoint_path = resume_checkpoint
    recording_dir = getattr(args, "recording_dir", None)
    recording_path = getattr(args, "recording_path", None)
    if getattr(args, "record", False) and recording_dir is None and recording_path is None:
        recording_dir = DEFAULT_FULL_SMB_RECORDING_DIR
        recording_path = DEFAULT_FULL_SMB_RECORDING_MANIFEST
    task_selection = _resolve_full_smb_task_selection(args)
    return FullSMBTrainingConfig(
        seed=args.seed,
        training_mode=getattr(args, "training_mode", FULL_SMB_TRAINING_MODE_AUTO),
        architecture_name=args.architecture,
        architecture_config=architecture_config,
        epochs=getattr(args, "epochs", 0),
        episodes_per_epoch=getattr(args, "episodes_per_epoch", 0),
        max_steps_per_episode=getattr(args, "max_steps_per_episode", 0),
        updates_per_epoch=getattr(args, "updates_per_epoch", None),
        rollout_length=getattr(args, "rollout_length", None),
        vector_env_count=getattr(args, "vector_env_count", 1),
        learning_rate=getattr(args, "learning_rate", 1e-4),
        entropy_weight=getattr(args, "entropy_weight", 0.01),
        policy_loss_weight=getattr(args, "policy_loss_weight", 1.0),
        representation_weight=getattr(args, "representation_weight", 0.0),
        world_model_weight=getattr(
            args,
            "world_model_weight",
            _FULL_SMB_DEFAULT_WORLD_MODEL_WEIGHT,
        ),
        reward_loss_weight=getattr(args, "reward_loss_weight", 0.0),
        value_loss_weight=getattr(args, "value_loss_weight", 0.0),
        action_aux_weight=getattr(args, "action_aux_weight", 0.0),
        critic_loss_weight=getattr(args, "critic_loss_weight", 0.0),
        reward_scale=getattr(args, "reward_scale", 1.0),
        reward_config=_reward_config_from_args(args),
        gradient_clip_norm=getattr(args, "gradient_clip_norm", 1.0),
        max_abs_loss=getattr(args, "max_abs_loss", _FULL_SMB_DEFAULT_MAX_ABS_LOSS),
        max_abs_scaled_reward=getattr(
            args,
            "max_abs_scaled_reward",
            _FULL_SMB_DEFAULT_MAX_ABS_SCALED_REWARD,
        ),
        max_abs_prediction=getattr(
            args,
            "max_abs_prediction",
            _FULL_SMB_DEFAULT_MAX_ABS_PREDICTION,
        ),
        deterministic=True if args.deterministic is None else bool(args.deterministic),
        deterministic_actions=getattr(args, "deterministic_actions", False),
        device=args.device,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps,
        evaluation_interval_epochs=args.evaluation_interval_epochs,
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
        init_checkpoint=getattr(args, "init_checkpoint", None),
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        perception_mode=perception_mode,
        freeze_vision=not args.fine_tune_vision,
        game_id=getattr(args, "game_id", DEFAULT_FULL_SMB_CONTENT.game),
        task_set=task_selection["task_set"],
        task_name=task_selection["task_name"],
        level=task_selection["level"],
        emulator_state=task_selection["emulator_state"],
        scenario=task_selection["scenario"],
        frame_skip=getattr(args, "frame_skip", None),
        save_checkpoints=(
            is_resume_command
            or getattr(args, "save_checkpoints", False)
            or getattr(args, "checkpoint", None) is not None
        ),
        output_summary=args.output_summary,
        log_path=args.log_path,
        recording_dir=recording_dir,
        recording_path=recording_path,
        tracking_backend=args.tracking_backend,
        tracking_log_dir=args.tracking_log_dir,
        tracking_project=args.tracking_project,
        tracking_run_name=args.tracking_run_name,
        tracking_mode=args.tracking_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
