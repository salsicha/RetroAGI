"""Versioned checkpoint schema shared across curriculum stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .config import to_plain_data


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_KEY = "checkpoint_schema_version"

StateDict = Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointPayload:
    """Serializable checkpoint envelope used by every stage."""

    stage: str
    model_name: str
    checkpoint_kind: str
    states: Mapping[str, StateDict]
    epoch: int = 0
    global_step: int = 0
    metrics: Mapping[str, float] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    specs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema version {self.schema_version}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        if not self.stage:
            raise ValueError("stage must be non-empty")
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if not self.checkpoint_kind:
            raise ValueError("checkpoint_kind must be non-empty")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        if not self.states:
            raise ValueError("states must contain at least one named state dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            CHECKPOINT_SCHEMA_KEY: self.schema_version,
            "stage": self.stage,
            "model_name": self.model_name,
            "checkpoint_kind": self.checkpoint_kind,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": to_plain_data(self.metrics),
            "config": to_plain_data(self.config),
            "specs": to_plain_data(self.specs),
            "states": dict(self.states),
            "metadata": to_plain_data(self.metadata),
        }


def build_checkpoint(
    *,
    stage: str,
    model_name: str,
    checkpoint_kind: str,
    states: Mapping[str, StateDict],
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    config: Optional[Mapping[str, Any]] = None,
    specs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a validated versioned checkpoint dictionary."""
    payload = CheckpointPayload(
        stage=stage,
        model_name=model_name,
        checkpoint_kind=checkpoint_kind,
        states=states,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics or {},
        config=config or {},
        specs=specs or {},
        metadata=metadata or {},
    )
    return payload.to_dict()


def is_versioned_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    return CHECKPOINT_SCHEMA_KEY in checkpoint


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized checkpoint dict or raise a clear schema error."""
    if CHECKPOINT_SCHEMA_KEY not in checkpoint:
        raise ValueError("checkpoint is missing checkpoint_schema_version")
    version = int(checkpoint[CHECKPOINT_SCHEMA_KEY])
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema version {version}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    required = ("stage", "model_name", "checkpoint_kind", "states")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint schema missing required keys: {', '.join(missing)}")

    payload = CheckpointPayload(
        stage=str(checkpoint["stage"]),
        model_name=str(checkpoint["model_name"]),
        checkpoint_kind=str(checkpoint["checkpoint_kind"]),
        states=checkpoint["states"],
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        metrics=checkpoint.get("metrics", {}),
        config=checkpoint.get("config", {}),
        specs=checkpoint.get("specs", {}),
        metadata=checkpoint.get("metadata", {}),
        schema_version=version,
    )
    return payload.to_dict()


def save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """Validate and write a versioned checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(validate_checkpoint(checkpoint), path)


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load and validate a versioned checkpoint."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    return validate_checkpoint(checkpoint)
