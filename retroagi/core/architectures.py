"""Architecture specs and registry for stage-agnostic model construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch.nn as nn

from .interfaces import StageSpec
from .models import (
    SUPPORTED_CONTROLLER_SCHEDULES,
    AgentWorldModelCritic,
)

BASELINE_ARCHITECTURE_NAME = "agent_world_model_critic"
BASELINE_OUTPUT_CONTRACT = "agent_world_model_critic.forward.v1"
BASELINE_CHECKPOINT_POLICY = "strict_stage_spec_and_model_state"
POLICY_TUPLE_OUTPUT_CONTRACTS = (BASELINE_OUTPUT_CONTRACT,)


class ArchitectureFactory(Protocol):
    """Construct one architecture instance for a compatible stage."""

    def __call__(
        self,
        stage: StageSpec,
        config: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        """Build a model instance for ``stage`` using optional overrides."""


@dataclass(frozen=True)
class ArchitectureSpec:
    """Registry metadata for one model-family concept."""

    name: str
    factory: ArchitectureFactory
    supported_stage_names: tuple[str, ...]
    checkpoint_model_name: str
    checkpoint_compatibility_policy: str
    output_contract: str
    configurable_hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("architecture name must be non-empty")
        if not self.supported_stage_names:
            raise ValueError("supported_stage_names must not be empty")
        if any(not stage_name for stage_name in self.supported_stage_names):
            raise ValueError("supported_stage_names must not contain empty names")
        if not self.checkpoint_model_name:
            raise ValueError("checkpoint_model_name must be non-empty")
        if not self.checkpoint_compatibility_policy:
            raise ValueError("checkpoint_compatibility_policy must be non-empty")
        if not self.output_contract:
            raise ValueError("output_contract must be non-empty")
        if any(not str(name) for name in self.configurable_hyperparameters):
            raise ValueError("configurable_hyperparameters keys must be non-empty")

    def supports_stage(self, stage: StageSpec) -> bool:
        return stage.name in self.supported_stage_names

    def build(
        self,
        stage: StageSpec,
        config: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        if not self.supports_stage(stage):
            raise ValueError(f"architecture {self.name!r} does not support stage {stage.name!r}")
        return self.factory(stage, config)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly metadata that excludes the callable factory."""

        return {
            "name": self.name,
            "supported_stage_names": list(self.supported_stage_names),
            "checkpoint_model_name": self.checkpoint_model_name,
            "checkpoint_compatibility_policy": self.checkpoint_compatibility_policy,
            "configurable_hyperparameters": dict(self.configurable_hyperparameters),
            "output_contract": self.output_contract,
        }


class ArchitectureRegistry:
    """In-memory registry of architecture concepts available to experiments."""

    def __init__(self) -> None:
        self._specs: dict[str, ArchitectureSpec] = {}

    def register(self, spec: ArchitectureSpec, *, replace: bool = False) -> None:
        if spec.name in self._specs and not replace:
            raise ValueError(f"architecture {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ArchitectureSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise KeyError(f"unknown architecture {name!r}; available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def build(
        self,
        name: str,
        stage: StageSpec,
        config: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        return self.get(name).build(stage, config)

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {name: self._specs[name].metadata() for name in self.names()}


def make_agent_world_model_critic(
    stage: StageSpec,
    config: Mapping[str, Any] | None = None,
) -> AgentWorldModelCritic:
    values = dict(config or {})
    unknown_keys = set(values) - {"hidden_dim", "controller_schedule"}
    if unknown_keys:
        raise ValueError(f"unknown architecture config keys: {sorted(unknown_keys)}")
    hidden_dim = int(values.get("hidden_dim", 64))
    controller_schedule = str(values.get("controller_schedule", "constant"))
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if controller_schedule not in SUPPORTED_CONTROLLER_SCHEDULES:
        raise ValueError(
            "controller_schedule must be one of "
            f"{SUPPORTED_CONTROLLER_SCHEDULES}, got {controller_schedule!r}"
        )
    return AgentWorldModelCritic(
        stage.vocab_size,
        stage.seq_len_a,
        stage.seq_len_c,
        stage.ratio_bc,
        d_model=hidden_dim,
        controller_schedule=controller_schedule,
    )


BASELINE_ARCHITECTURE_SPEC = ArchitectureSpec(
    name=BASELINE_ARCHITECTURE_NAME,
    factory=make_agent_world_model_critic,
    supported_stage_names=("synthetic_1d", "block_smb", "full_smb", "pong_block"),
    checkpoint_model_name=BASELINE_ARCHITECTURE_NAME,
    checkpoint_compatibility_policy=BASELINE_CHECKPOINT_POLICY,
    configurable_hyperparameters={
        "hidden_dim": 64,
        "controller_schedule": "constant",
    },
    output_contract=BASELINE_OUTPUT_CONTRACT,
)

ARCHITECTURE_REGISTRY = ArchitectureRegistry()
ARCHITECTURE_REGISTRY.register(BASELINE_ARCHITECTURE_SPEC)


def register_architecture(spec: ArchitectureSpec, *, replace: bool = False) -> None:
    ARCHITECTURE_REGISTRY.register(spec, replace=replace)


def get_architecture(name: str) -> ArchitectureSpec:
    return ARCHITECTURE_REGISTRY.get(name)


def architecture_names() -> tuple[str, ...]:
    return ARCHITECTURE_REGISTRY.names()


def build_architecture(
    name: str,
    stage: StageSpec,
    config: Mapping[str, Any] | None = None,
) -> nn.Module:
    return ARCHITECTURE_REGISTRY.build(name, stage, config)
