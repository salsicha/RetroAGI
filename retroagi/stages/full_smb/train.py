"""Direct Full SMB policy training, evaluation, resume, and checkpointing."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.optim as optim

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    SMB_ACTIONS,
    TRACKING_BACKENDS,
    ExperimentTrackerConfig,
    StageBatch,
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
    FULL_SMB_SPEC,
    FullSMBRewardConfig,
    FullSMBStage,
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
    world_model_weight: float = 0.0
    reward_loss_weight: float = 0.0
    value_loss_weight: float = 0.0
    action_aux_weight: float = 0.0
    critic_loss_weight: float = 0.0
    reward_scale: float = 1.0
    reward_config: FullSMBRewardConfig | Mapping[str, float] = field(
        default_factory=FullSMBRewardConfig
    )
    gradient_clip_norm: float = 1.0
    deterministic: bool = True
    deterministic_actions: bool = False
    device: str = "auto"
    evaluation_episodes: int = 1
    evaluation_max_steps: int = 64
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    init_checkpoint: Optional[Path] = None
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT
    perception_mode: Optional[str] = None
    freeze_vision: bool = True
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
        if self.vector_env_count <= 0:
            raise ValueError("vector_env_count must be positive")
        for name in ("learning_rate", "reward_scale", "gradient_clip_norm"):
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullSMBTrainingResult:
    """Artifacts from one direct Full SMB training run."""

    checkpoint: dict[str, Any]
    history: Mapping[str, list[float]]
    evaluation: FullSMBEvaluationResult
    checkpoint_path: Optional[Path]

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
            "evaluation": self.evaluation.as_dict(),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


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
    history: dict[str, list[float]] = {"episode_return": [], "policy_loss": []}
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
        try:
            model.train()
            _set_perception_training_mode(vision, config)
            gradient_parameters = _gradient_parameters(model, vision, config)
            for epoch in range(start_epoch, config.epochs):
                for update_index in range(config.updates_per_epoch):
                    episode_seed = config.seed + epoch * config.updates_per_epoch + update_index
                    metrics = _train_episode(
                        model,
                        optimizer,
                        stage,
                        seed=episode_seed,
                        max_steps=config.rollout_length,
                        device=device,
                        reward_scale=config.reward_scale,
                        entropy_weight=config.entropy_weight,
                        policy_loss_weight=config.policy_loss_weight,
                        gradient_clip_norm=config.gradient_clip_norm,
                        deterministic_actions=config.deterministic_actions,
                        gradient_parameters=gradient_parameters,
                    )
                    global_step += int(metrics["steps"])
                    history["episode_return"].append(float(metrics["return"]))
                    history["policy_loss"].append(float(metrics["loss"]))
                    tracker.log_metrics(metrics, step=global_step, prefix="train")
                    _log_full_smb_event(
                        config,
                        "train_rollout",
                        epoch=epoch + 1,
                        update=update_index + 1,
                        global_step=global_step,
                        metrics=metrics,
                    )
            evaluation = evaluate_full_smb_policy(
                model,
                config=config,
                make_stage=make_stage,
                vision=vision,
                device=device,
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
                "evaluation_mean_return": evaluation.mean_return,
                "evaluation_success_rate": evaluation.success_rate,
            },
            architecture_name=architecture_name,
            architecture_config=architecture_config,
            vision=vision,
            training_source=training_source,
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
            evaluation=evaluation.as_dict(),
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
            evaluation=evaluation,
            checkpoint_path=checkpoint_path if config.save_checkpoints else None,
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
    try:
        for episode_index in range(config.evaluation_episodes):
            observation = stage.reset(seed=config.seed + 10_000 + episode_index)
            episode_return = 0.0
            for _step in range(config.evaluation_max_steps):
                batch = stage.encode_observation(observation)
                logits = _policy_action_logits(model, batch, device=resolved_device)
                action = int(logits.argmax(dim=-1).item())
                observation, reward, terminated, truncated, _info = stage.step(action)
                episode_return += float(reward)
                steps += 1
                if terminated:
                    terminated_count += 1
                if truncated:
                    truncated_count += 1
                if terminated or truncated:
                    break
            returns.append(float(episode_return))
    finally:
        stage.close()
        if owns_vision:
            del vision

    episodes = len(returns)
    successes = terminated_count
    return FullSMBEvaluationResult(
        episodes=episodes,
        max_steps_per_episode=config.evaluation_max_steps,
        steps=steps,
        returns=tuple(returns),
        mean_return=_mean(returns),
        success_rate=float(successes / episodes) if episodes else 0.0,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
    )


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
) -> dict[str, Any]:
    perception = _perception_checkpoint_metadata(config, vision)
    rollout = _rollout_metadata(config)
    loss_weights = _loss_weight_metadata(config)
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
            "reward": config.reward_config.to_manifest(),
            "recording": recording,
            "tracking": tracking,
            "training_source": source,
            "stage_batch_contract": stage_batch_contract,
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
                "reward": config.reward_config.to_manifest(),
                "recording": recording,
                "tracking": tracking,
                "source": source,
                "stage_batch_contract": stage_batch_contract,
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
    gradient_clip_norm: float,
    deterministic_actions: bool,
    gradient_parameters: tuple[torch.nn.Parameter, ...],
) -> dict[str, float]:
    observation = stage.reset(seed=seed)
    total_return = 0.0
    losses: list[float] = []
    for _step in range(max_steps):
        batch = stage.encode_observation(observation)
        logits = _policy_action_logits(model, batch, device=device)
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic_actions:
            action_tensor = logits.argmax(dim=-1)
        else:
            action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        observation, reward, terminated, truncated, _info = stage.step(int(action_tensor.item()))
        scaled_reward = torch.as_tensor(
            float(reward) * reward_scale,
            dtype=log_prob.dtype,
            device=log_prob.device,
        )
        entropy = distribution.entropy().mean()
        policy_loss = -(log_prob.mean() * scaled_reward.detach())
        loss = policy_loss_weight * policy_loss - entropy_weight * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gradient_parameters, gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        total_return += float(reward)
        if terminated or truncated:
            break
    return {"return": total_return, "loss": _mean(losses), "steps": float(len(losses))}


def _policy_action_logits(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    _validate_full_smb_stage_batch(batch)
    src_a = batch.src_a.to(device)
    src_b = batch.src_b.to(device)
    src_c = batch.src_c.to(device)
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(episode_mask, dtype=src_c.dtype, device=src_c.device)
    outputs = model(src_a, src_b, src_c, tau=1.0, episode_mask=episode_mask)
    logits_a = outputs[4]
    return logits_a[:, -1, :FULL_SMB_ACTION_COUNT]


def _make_stage(
    make_stage: Optional[StageFactory],
    vision: FullSMBSegmentationVision,
    config: FullSMBTrainingConfig,
) -> FullSMBStage:
    if make_stage is not None:
        return make_stage(vision)
    return FullSMBStage(vision=vision, reward_config=config.reward_config)


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
    }


def _recording_metadata(config: FullSMBTrainingConfig) -> dict[str, Any]:
    return {
        "recording_dir": str(config.recording_dir) if config.recording_dir else None,
        "recording_path": str(config.recording_path) if config.recording_path else None,
    }


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
    return {
        "mode": mode,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_model_name": checkpoint_model_name,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "uses_shared_architecture_factory": True,
        "architecture_name": architecture_name,
        "architecture_config": dict(architecture_config),
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_common_args(train)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--episodes-per-epoch", type=int, default=1)
    train.add_argument("--max-steps-per-episode", type=int, default=64)
    train.add_argument("--updates-per-epoch", type=int)
    train.add_argument("--rollout-length", "--rollout-steps", dest="rollout_length", type=int)
    train.add_argument("--vector-env-count", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--entropy-weight", type=float, default=0.01)
    train.add_argument("--policy-loss-weight", type=float, default=1.0)
    train.add_argument("--representation-weight", type=float, default=0.0)
    train.add_argument("--world-model-weight", type=float, default=0.0)
    train.add_argument("--reward-loss-weight", type=float, default=0.0)
    train.add_argument("--value-loss-weight", type=float, default=0.0)
    train.add_argument("--action-aux-weight", type=float, default=0.0)
    train.add_argument("--critic-loss-weight", type=float, default=0.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--deterministic-actions", action="store_true")
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--init-checkpoint", type=Path)
    train.add_argument("--save-checkpoints", action="store_true")
    train.add_argument("--recording-dir", type=Path)
    train.add_argument("--recording-path", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    _add_common_args(evaluate)
    evaluate.add_argument("--policy-checkpoint", type=Path, required=True)

    args = parser.parse_args(argv)
    config = _config_from_args(args)
    if args.command == "train":
        result = train_full_smb_policy(config)
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    elif args.command == "evaluate":
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
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--tracking-backend", choices=TRACKING_BACKENDS, default="none")
    parser.add_argument("--tracking-log-dir", type=Path)
    parser.add_argument("--tracking-project", default="retroagi")
    parser.add_argument("--tracking-run-name")
    parser.add_argument("--tracking-mode")


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
    return FullSMBTrainingConfig(
        seed=args.seed,
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
        world_model_weight=getattr(args, "world_model_weight", 0.0),
        reward_loss_weight=getattr(args, "reward_loss_weight", 0.0),
        value_loss_weight=getattr(args, "value_loss_weight", 0.0),
        action_aux_weight=getattr(args, "action_aux_weight", 0.0),
        critic_loss_weight=getattr(args, "critic_loss_weight", 0.0),
        reward_scale=getattr(args, "reward_scale", 1.0),
        reward_config=_reward_config_from_args(args),
        gradient_clip_norm=getattr(args, "gradient_clip_norm", 1.0),
        deterministic=True if args.deterministic is None else bool(args.deterministic),
        deterministic_actions=getattr(args, "deterministic_actions", False),
        device=args.device,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_max_steps=args.evaluation_max_steps,
        checkpoint_path=getattr(args, "checkpoint", None),
        resume_path=getattr(args, "resume", None),
        init_checkpoint=getattr(args, "init_checkpoint", None),
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        perception_mode=perception_mode,
        freeze_vision=not args.fine_tune_vision,
        save_checkpoints=getattr(args, "save_checkpoints", False)
        or getattr(args, "checkpoint", None) is not None,
        output_summary=args.output_summary,
        log_path=args.log_path,
        recording_dir=getattr(args, "recording_dir", None),
        recording_path=getattr(args, "recording_path", None),
        tracking_backend=args.tracking_backend,
        tracking_log_dir=args.tracking_log_dir,
        tracking_project=args.tracking_project,
        tracking_run_name=args.tracking_run_name,
        tracking_mode=args.tracking_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
