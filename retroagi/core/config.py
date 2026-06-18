"""Typed experiment configuration shared by stages and trainers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional


CheckpointMode = Literal["min", "max"]
ConfigPrimitive = str | int | float | bool | None
ConfigValue = ConfigPrimitive | tuple[ConfigPrimitive, ...] | Mapping[str, ConfigPrimitive]


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def to_plain_data(value: Any) -> Any:
    """Convert dataclass configuration values into JSON-friendly primitives."""
    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment identity, seeding, and rollout sampling settings."""

    stage: str
    seed: int = 0
    rollout_steps: int = 1
    num_envs: int = 1
    scenario: Optional[str] = None
    deterministic: bool = True
    metadata: Mapping[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must be non-empty")
        _require_positive("rollout_steps", self.rollout_steps)
        _require_positive("num_envs", self.num_envs)


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture settings independent from a specific trainer."""

    name: str
    hidden_dim: int = 64
    depth: int = 2
    heads: int = 4
    dropout: float = 0.0
    patch_size: Optional[int] = None
    vocab_size: Optional[int] = None
    metadata: Mapping[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        _require_positive("hidden_dim", self.hidden_dim)
        _require_positive("depth", self.depth)
        _require_positive("heads", self.heads)
        _require_non_negative("dropout", self.dropout)
        if self.dropout >= 1:
            raise ValueError("dropout must be less than 1")
        if self.patch_size is not None:
            _require_positive("patch_size", self.patch_size)
        if self.vocab_size is not None:
            _require_positive("vocab_size", self.vocab_size)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and training-loop settings."""

    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    seed: int = 0
    samples_per_epoch: Optional[int] = None
    gradient_clip_norm: Optional[float] = None
    device: str = "auto"
    metadata: Mapping[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive("epochs", self.epochs)
        _require_positive("batch_size", self.batch_size)
        _require_positive("learning_rate", self.learning_rate)
        _require_non_negative("weight_decay", self.weight_decay)
        if self.samples_per_epoch is not None:
            _require_positive("samples_per_epoch", self.samples_per_epoch)
        if self.gradient_clip_norm is not None:
            _require_positive("gradient_clip_norm", self.gradient_clip_norm)
        if not self.device:
            raise ValueError("device must be non-empty")


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation cadence, sample counts, seeds, and metric names."""

    interval_epochs: int = 1
    samples: Optional[int] = None
    episodes: Optional[int] = None
    seed: int = 1_000_000
    deterministic: bool = True
    metrics: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive("interval_epochs", self.interval_epochs)
        if self.samples is not None:
            _require_positive("samples", self.samples)
        if self.episodes is not None:
            _require_positive("episodes", self.episodes)
        for metric in self.metrics:
            if not metric:
                raise ValueError("metrics must not contain empty names")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint locations and best-checkpoint selection settings."""

    output_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    save_interval_epochs: int = 1
    best_metric: Optional[str] = None
    best_mode: CheckpointMode = "max"
    metadata: Mapping[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive("save_interval_epochs", self.save_interval_epochs)
        if self.best_mode not in ("min", "max"):
            raise ValueError("best_mode must be 'min' or 'max'")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete typed configuration bundle for one run."""

    environment: EnvironmentConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    name: Optional[str] = None
    metadata: Mapping[str, ConfigValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)
