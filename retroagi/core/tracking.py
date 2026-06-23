"""Optional experiment tracking integrations."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from .config import to_plain_data

TRACKING_BACKENDS = ("none", "tensorboard", "wandb")


class ExperimentTracker(Protocol):
    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the resolved run configuration."""

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        prefix: Optional[str] = None,
    ) -> None:
        """Record scalar metrics at one step."""

    def close(self) -> None:
        """Flush and close any tracker resources."""


@dataclass(frozen=True)
class ExperimentTrackerConfig:
    backend: str = "none"
    log_dir: Optional[Path] = None
    project: str = "retroagi"
    run_name: Optional[str] = None
    mode: Optional[str] = None

    def __post_init__(self) -> None:
        backend = self.backend.lower()
        if backend not in TRACKING_BACKENDS:
            raise ValueError(f"tracking backend must be one of {TRACKING_BACKENDS}")
        object.__setattr__(self, "backend", backend)
        if self.log_dir is not None and not isinstance(self.log_dir, Path):
            object.__setattr__(self, "log_dir", Path(self.log_dir))
        if not self.project:
            raise ValueError("tracking project must be non-empty")


class NullExperimentTracker:
    def log_config(self, config: Mapping[str, Any]) -> None:
        del config

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        prefix: Optional[str] = None,
    ) -> None:
        del metrics, step, prefix

    def close(self) -> None:
        return None


class TensorBoardExperimentTracker:
    def __init__(self, config: ExperimentTrackerConfig, *, default_log_dir: Path):
        tensorboard = _import_optional(
            "torch.utils.tensorboard",
            "TensorBoard tracking requires tensorboard. Install with retroagi[tracking].",
        )
        log_dir = config.log_dir or default_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = tensorboard.SummaryWriter(log_dir=str(log_dir))

    def log_config(self, config: Mapping[str, Any]) -> None:
        self.writer.add_text(
            "config",
            json.dumps(to_plain_data(config), indent=2, sort_keys=True),
            global_step=0,
        )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        prefix: Optional[str] = None,
    ) -> None:
        for name, value in flatten_numeric_metrics(metrics).items():
            metric_name = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(metric_name, value, global_step=step)

    def close(self) -> None:
        self.writer.close()


class WandbExperimentTracker:
    def __init__(self, config: ExperimentTrackerConfig, *, default_log_dir: Path):
        wandb = _import_optional(
            "wandb",
            "Weights & Biases tracking requires wandb. Install with retroagi[tracking].",
        )
        log_dir = config.log_dir or default_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=config.project,
            name=config.run_name,
            dir=str(log_dir),
            mode=config.mode,
        )

    def log_config(self, config: Mapping[str, Any]) -> None:
        config_values = to_plain_data(config)
        update = getattr(self.run.config, "update", None)
        if update is not None:
            update(config_values, allow_val_change=True)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        prefix: Optional[str] = None,
    ) -> None:
        values = flatten_numeric_metrics(metrics)
        if prefix:
            values = {f"{prefix}/{name}": value for name, value in values.items()}
        self.run.log(values, step=step)

    def close(self) -> None:
        self.run.finish()


def make_experiment_tracker(
    config: ExperimentTrackerConfig,
    *,
    default_log_dir: Path = Path("artifacts/tracking"),
) -> ExperimentTracker:
    if config.backend == "none":
        return NullExperimentTracker()
    if config.backend == "tensorboard":
        return TensorBoardExperimentTracker(config, default_log_dir=default_log_dir)
    if config.backend == "wandb":
        return WandbExperimentTracker(config, default_log_dir=default_log_dir)
    raise ValueError(f"unsupported tracking backend {config.backend!r}")


def flatten_numeric_metrics(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
        elif isinstance(value, Mapping):
            flat.update(flatten_numeric_metrics(value, prefix=name))
    return flat


def _import_optional(module_name: str, message: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(message) from exc
