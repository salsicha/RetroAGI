"""Versioned checkpoint schema shared across curriculum stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import platform
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
from typing import Any, Mapping, Optional

import torch

from .config import to_plain_data


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_KEY = "checkpoint_schema_version"

StateDict = Mapping[str, Any]


def collect_code_revision_metadata(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """Return best-effort source revision metadata for checkpoint traceability."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return dict(_collect_code_revision_metadata(str(root)))


@lru_cache(maxsize=None)
def _collect_code_revision_metadata(repo_root: str) -> tuple[tuple[str, Any], ...]:
    root = Path(repo_root)
    revision = _git_output(root, "rev-parse", "HEAD")
    short_revision = _git_output(root, "rev-parse", "--short", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    status = _git_output(root, "status", "--porcelain")
    return tuple(
        {
            "system": "git",
            "root": str(root),
            "revision": revision,
            "short_revision": short_revision,
            "branch": branch,
            "dirty": bool(status) if status is not None else None,
        }.items()
    )


def _git_output(repo_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def collect_runtime_environment_metadata() -> dict[str, Any]:
    """Return runtime and hardware metadata that affects reproducibility."""

    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(mps_backend and mps_backend.is_available()),
            "mps_built": bool(mps_backend and mps_backend.is_built()),
        },
    }


def checkpoint_trace_metadata(
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Merge caller metadata with standard checkpoint traceability fields."""

    merged = dict(to_plain_data(metadata or {}))
    merged.setdefault("code_revision", collect_code_revision_metadata())
    merged.setdefault("runtime_environment", collect_runtime_environment_metadata())
    return merged


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
        metadata=checkpoint_trace_metadata(metadata),
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


def checkpoint_summary_path(path: Path) -> Path:
    """Return the JSON sidecar path for a checkpoint file."""

    return Path(path).with_suffix(".json")


def checkpoint_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly sidecar summary without tensor state payloads."""

    metadata = to_plain_data(checkpoint.get("metadata", {}))
    states = checkpoint.get("states", {})
    state_keys = sorted(states) if isinstance(states, Mapping) else []
    return {
        CHECKPOINT_SCHEMA_KEY: checkpoint.get(CHECKPOINT_SCHEMA_KEY),
        "stage": checkpoint.get("stage"),
        "model_name": checkpoint.get("model_name"),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "metrics": to_plain_data(checkpoint.get("metrics", {})),
        "config": to_plain_data(checkpoint.get("config", {})),
        "specs": to_plain_data(checkpoint.get("specs", {})),
        "metadata": metadata,
        "code_revision": metadata.get("code_revision", {}),
        "environment": metadata.get("runtime_environment", {}),
        "state_keys": state_keys,
    }


def write_checkpoint_summary(path: Path, checkpoint: Mapping[str, Any]) -> Path:
    """Write the checkpoint sidecar summary and return its path."""

    summary_path = checkpoint_summary_path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(checkpoint_summary(checkpoint), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """Validate and write a versioned checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_checkpoint(checkpoint)
    normalized["metadata"] = checkpoint_trace_metadata(normalized.get("metadata", {}))
    torch.save(normalized, path)
    write_checkpoint_summary(path, normalized)


def load_checkpoint(path: Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load and validate a versioned checkpoint."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    return validate_checkpoint(checkpoint)
