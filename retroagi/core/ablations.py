"""Stage-agnostic architecture ablation variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import SUPPORTED_CONTROLLER_SCHEDULES

ABLATION_ALIASES = {
    "vision": "vision_enabled",
    "vision_enabled": "vision_enabled",
    "visual": "vision_enabled",
    "hierarchy": "hierarchy_enabled",
    "hierarchy_enabled": "hierarchy_enabled",
    "world_model": "world_model_enabled",
    "world-model": "world_model_enabled",
    "dynamics": "world_model_enabled",
    "critic": "critic_feedback_enabled",
    "critic_feedback": "critic_feedback_enabled",
    "critic-feedback": "critic_feedback_enabled",
    "recurrent": "recurrent_state_enabled",
    "recurrent_state": "recurrent_state_enabled",
    "recurrent-state": "recurrent_state_enabled",
    "checkpoint_transfer": "checkpoint_transfer_enabled",
    "checkpoint-transfer": "checkpoint_transfer_enabled",
    "transfer": "checkpoint_transfer_enabled",
    "target_network": "target_network_mode",
    "target-network": "target_network_mode",
    "controller_schedule": "controller_schedule",
    "controller-schedule": "controller_schedule",
    "schedule": "controller_schedule",
    "auxiliary": "auxiliary_objectives_enabled",
    "auxiliary_objectives": "auxiliary_objectives_enabled",
    "auxiliary-objectives": "auxiliary_objectives_enabled",
}
BOOLEAN_ABLATION_FIELDS = {
    "vision_enabled",
    "hierarchy_enabled",
    "world_model_enabled",
    "critic_feedback_enabled",
    "recurrent_state_enabled",
    "checkpoint_transfer_enabled",
    "auxiliary_objectives_enabled",
}
TARGET_NETWORK_MODES = ("off", "on", "auto")


@dataclass(frozen=True)
class ArchitectureAblationConfig:
    """Normalized switches for architecture-variant sweeps."""

    vision_enabled: bool | None = None
    hierarchy_enabled: bool | None = None
    world_model_enabled: bool | None = None
    critic_feedback_enabled: bool | None = None
    recurrent_state_enabled: bool | None = None
    checkpoint_transfer_enabled: bool | None = None
    target_network_mode: str | None = None
    controller_schedule: str | None = None
    auxiliary_objectives_enabled: bool | None = None

    @classmethod
    def from_items(
        cls, items: Sequence[tuple[str, Any]] | None = None
    ) -> "ArchitectureAblationConfig":
        values: dict[str, Any] = {}
        for name, value in items or ():
            values[name] = value
        return cls(**values)

    def __post_init__(self) -> None:
        for name in BOOLEAN_ABLATION_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool or None")
        if (
            self.target_network_mode is not None
            and self.target_network_mode not in TARGET_NETWORK_MODES
        ):
            raise ValueError(f"target_network_mode must be one of {TARGET_NETWORK_MODES}")
        if (
            self.controller_schedule is not None
            and self.controller_schedule not in SUPPORTED_CONTROLLER_SCHEDULES
        ):
            raise ValueError(
                "controller_schedule must be one of " f"{SUPPORTED_CONTROLLER_SCHEDULES}"
            )

    def configured_values(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items() if value is not None}

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(self.configured_values().items()))


@dataclass(frozen=True)
class ArchitectureVariant:
    """Resolved architecture config plus per-stage ablation arguments."""

    ablation: ArchitectureAblationConfig
    architecture_config: Mapping[str, Any]
    stage_args: Mapping[str, tuple[str, ...]]
    forward_kwargs: Mapping[str, Any]

    @property
    def ablation_items(self) -> tuple[tuple[str, Any], ...]:
        return self.ablation.items()

    def args_for_stage(self, stage: str) -> list[str]:
        return list(self.stage_args.get(stage, ()))

    def metadata(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation.configured_values(),
            "architecture_config": dict(self.architecture_config),
            "stage_args": {stage: list(args) for stage, args in self.stage_args.items()},
            "forward_kwargs": dict(self.forward_kwargs),
        }


def parse_architecture_ablation_item(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("ablation must use KEY=VALUE syntax")
    raw_key, raw_value = value.split("=", 1)
    try:
        key = ABLATION_ALIASES[raw_key.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(ABLATION_ALIASES))
        raise argparse.ArgumentTypeError(
            f"unknown architecture ablation {raw_key!r}; expected one of: {choices}"
        ) from exc
    parsed = _parse_ablation_value(key, raw_value.strip())
    return key, parsed


def build_architecture_variant(
    architecture_config: Mapping[str, Any] | None = None,
    ablation_items: Sequence[tuple[str, Any]] | None = None,
) -> ArchitectureVariant:
    ablation = ArchitectureAblationConfig.from_items(ablation_items)
    resolved_architecture_config = dict(architecture_config or {})
    if ablation.controller_schedule is not None:
        resolved_architecture_config["controller_schedule"] = ablation.controller_schedule
    return ArchitectureVariant(
        ablation=ablation,
        architecture_config=resolved_architecture_config,
        stage_args={
            "synthetic-1d": tuple(_synthetic_stage_args(ablation)),
            "block-smb": tuple(_block_smb_stage_args(ablation)),
        },
        forward_kwargs=_forward_kwargs(ablation),
    )


def _parse_ablation_value(key: str, value: str) -> Any:
    if key in BOOLEAN_ABLATION_FIELDS:
        return _parse_bool(value)
    normalized = value.lower()
    if key == "target_network_mode":
        if normalized in {"true", "yes", "enable", "enabled"}:
            return "on"
        if normalized in {"false", "no", "disable", "disabled"}:
            return "off"
        if normalized not in TARGET_NETWORK_MODES:
            raise argparse.ArgumentTypeError(
                f"target_network must be one of {TARGET_NETWORK_MODES}"
            )
        return normalized
    if key == "controller_schedule":
        if normalized not in SUPPORTED_CONTROLLER_SCHEDULES:
            raise argparse.ArgumentTypeError(
                "controller_schedule must be one of " f"{SUPPORTED_CONTROLLER_SCHEDULES}"
            )
        return normalized
    raise argparse.ArgumentTypeError(f"unsupported ablation field {key!r}")


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    raise argparse.ArgumentTypeError("boolean ablations must use true/false or on/off")


def _synthetic_stage_args(ablation: ArchitectureAblationConfig) -> list[str]:
    args = []
    if ablation.auxiliary_objectives_enabled is False:
        args.extend(["--critic-loss-weight", "0"])
    return args


def _block_smb_stage_args(ablation: ArchitectureAblationConfig) -> list[str]:
    args = []
    args.extend(_bool_flag("--enable-vision", "--disable-vision", ablation.vision_enabled))
    args.extend(
        _bool_flag("--enable-world-model", "--disable-world-model", ablation.world_model_enabled)
    )
    args.extend(
        _bool_flag(
            "--enable-critic-feedback",
            "--disable-critic-feedback",
            ablation.critic_feedback_enabled,
        )
    )
    args.extend(_bool_flag("--enable-hierarchy", "--disable-hierarchy", ablation.hierarchy_enabled))
    args.extend(
        _bool_flag(
            "--enable-recurrent-state",
            "--disable-recurrent-state",
            ablation.recurrent_state_enabled,
        )
    )
    args.extend(
        _bool_flag(
            "--enable-checkpoint-transfer",
            "--disable-checkpoint-transfer",
            ablation.checkpoint_transfer_enabled,
        )
    )
    if ablation.target_network_mode is not None:
        args.extend(["--target-network-mode", ablation.target_network_mode])
    if ablation.auxiliary_objectives_enabled is False:
        args.extend(
            [
                "--representation-weight",
                "0",
                "--reward-loss-weight",
                "0",
                "--value-loss-weight",
                "0",
                "--action-aux-weight",
                "0",
                "--critic-loss-weight",
                "0",
            ]
        )
    return args


def _forward_kwargs(ablation: ArchitectureAblationConfig) -> dict[str, Any]:
    values = {}
    if ablation.critic_feedback_enabled is not None:
        values["critic_feedback_enabled"] = ablation.critic_feedback_enabled
    if ablation.world_model_enabled is not None:
        values["world_model_enabled"] = ablation.world_model_enabled
    return values


def _bool_flag(enable: str, disable: str, value: bool | None) -> list[str]:
    if value is None:
        return []
    return [enable if value else disable]
