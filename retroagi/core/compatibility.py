"""Startup compatibility checks for stages, models, actions, and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, Optional

from .actions import SMB_ACTIONS
from .checkpoint import validate_checkpoint
from .config import ModelConfig
from .interfaces import StageSpec, VisionSpec


class CompatibilityError(ValueError):
    """Raised when a stage, model, action, or checkpoint contract is incompatible."""


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _fail(context: str, message: str) -> None:
    raise CompatibilityError(f"{context}: {message}")


def validate_stage_spec(stage: StageSpec, *, context: str = "stage") -> None:
    """Validate stage dimensions and action vocabulary assumptions."""
    if not stage.name:
        _fail(context, "name must be non-empty")
    if not stage.observation_kind:
        _fail(context, "observation_kind must be non-empty")
    if not stage.action_kind:
        _fail(context, "action_kind must be non-empty")
    for field_name in ("seq_len_a", "ratio_ab", "ratio_bc", "vocab_size"):
        if getattr(stage, field_name) <= 0:
            _fail(context, f"{field_name} must be positive")

    if "SMBAction" in stage.action_kind or "SMB" in stage.action_kind:
        expected_ids = list(range(len(SMB_ACTIONS)))
        actual_ids = [int(action) for action in SMB_ACTIONS]
        if actual_ids != expected_ids:
            _fail(context, f"SMB action IDs must be contiguous from zero; got {actual_ids}")
        if stage.vocab_size < len(SMB_ACTIONS):
            _fail(
                context,
                f"vocab_size {stage.vocab_size} cannot represent {len(SMB_ACTIONS)} SMB actions",
            )


def validate_model_vision_compatibility(
    model: ModelConfig,
    vision: VisionSpec,
    *,
    context: str = "model",
) -> None:
    """Validate model config against a concrete vision encoder contract."""
    if model.name != vision.name:
        _fail(context, f"model name {model.name!r} does not match vision spec {vision.name!r}")
    if model.hidden_dim != vision.token_dim:
        _fail(
            context,
            f"hidden_dim {model.hidden_dim} does not match vision token_dim {vision.token_dim}",
        )
    if model.vocab_size is not None and model.vocab_size < vision.num_classes:
        _fail(
            context,
            f"vocab_size {model.vocab_size} cannot represent {vision.num_classes} classes",
        )


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    stage: Optional[StageSpec] = None,
    model: Optional[ModelConfig] = None,
    vision: Optional[VisionSpec] = None,
    checkpoint_kind: Optional[str] = None,
    required_states: Iterable[str] = (),
    context: str = "checkpoint",
) -> dict[str, Any]:
    """Validate schema-v1 checkpoint identity, specs, and named state sections."""
    normalized = validate_checkpoint(checkpoint)

    if stage is not None and normalized["stage"] != stage.name:
        _fail(context, f"stage {normalized['stage']!r} does not match expected {stage.name!r}")
    if model is not None and normalized["model_name"] != model.name:
        _fail(
            context,
            f"model_name {normalized['model_name']!r} does not match expected {model.name!r}",
        )
    if checkpoint_kind is not None and normalized["checkpoint_kind"] != checkpoint_kind:
        _fail(
            context,
            "checkpoint_kind "
            f"{normalized['checkpoint_kind']!r} does not match expected {checkpoint_kind!r}",
        )

    states = normalized["states"]
    missing_states = [name for name in required_states if name not in states]
    if missing_states:
        _fail(context, f"missing required state sections: {', '.join(missing_states)}")

    if vision is not None:
        checkpoint_vision = normalized.get("specs", {}).get("vision")
        if checkpoint_vision is None:
            _fail(context, "missing specs.vision")
        expected = _plain(vision)
        actual = _plain(checkpoint_vision)
        for key in ("name", "semantic_classes", "token_dim", "position_dim"):
            if actual.get(key) != expected.get(key):
                _fail(
                    context,
                    f"vision spec {key} {actual.get(key)!r} does not match expected "
                    f"{expected.get(key)!r}",
                )

    return normalized
