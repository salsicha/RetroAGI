"""Game-aware promotion plan contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .stage_resolution import resolve_game_stage

COMPARISON_OPERATORS = (">=", "<=", ">", "<", "==")
METRIC_GATE_OPERATORS = ("present", "finite", *COMPARISON_OPERATORS)


@dataclass(frozen=True)
class PromotionRuntimeGateSpec:
    """Runtime budget gate owned by a game promotion rung."""

    budget_key: str = "runtime_seconds"
    reason: str = "runtime exceeded promotion budget"

    def __post_init__(self) -> None:
        if not self.budget_key:
            raise ValueError("runtime gate budget_key must be non-empty")
        if not self.reason:
            raise ValueError("runtime gate reason must be non-empty")

    def to_manifest(self) -> dict[str, str]:
        return {
            "kind": "runtime",
            "budget_key": self.budget_key,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PromotionMetricGateSpec:
    """Metric presence, finite-value, or threshold gate."""

    metric: str
    operator: str = "present"
    threshold: float | None = None
    threshold_key: str | None = None
    reason: str = "metric gate failed"
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric gate metric must be non-empty")
        if self.operator not in METRIC_GATE_OPERATORS:
            raise ValueError(
                f"metric gate {self.metric!r} operator must be one of " f"{METRIC_GATE_OPERATORS}"
            )
        if not self.reason:
            raise ValueError(f"metric gate {self.metric!r} reason must be non-empty")
        has_static_threshold = self.threshold is not None
        has_budget_threshold = self.threshold_key is not None
        if self.operator in COMPARISON_OPERATORS:
            if has_static_threshold == has_budget_threshold:
                raise ValueError(
                    f"metric gate {self.metric!r} must define exactly one of "
                    "threshold or threshold_key"
                )
            if self.threshold_key is not None and not self.threshold_key:
                raise ValueError(f"metric gate {self.metric!r} threshold_key is empty")
        elif has_static_threshold or has_budget_threshold:
            raise ValueError(
                f"metric gate {self.metric!r} operator {self.operator!r} "
                "must not define a threshold"
            )
        if self.name is not None and not self.name:
            raise ValueError(f"metric gate {self.metric!r} name must be non-empty")

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "metric",
            "metric": self.metric,
            "operator": self.operator,
            "reason": self.reason,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.threshold_key is not None:
            payload["threshold_key"] = self.threshold_key
        return payload


@dataclass(frozen=True)
class PromotionArtifactGateSpec:
    """Artifact existence gate owned by a game promotion rung."""

    field: str
    reason: str = "required artifact path does not exist"
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("artifact gate field must be non-empty")
        if not self.reason:
            raise ValueError(f"artifact gate {self.field!r} reason must be non-empty")
        if self.name is not None and not self.name:
            raise ValueError(f"artifact gate {self.field!r} name must be non-empty")

    def to_manifest(self) -> dict[str, str]:
        payload = {
            "kind": "artifact",
            "field": self.field,
            "reason": self.reason,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class GamePromotionGateSpec:
    """Game-owned automatic gates for one architecture promotion rung."""

    rung_name: str
    runtime: PromotionRuntimeGateSpec | None = field(default_factory=PromotionRuntimeGateSpec)
    metric_gates: tuple[PromotionMetricGateSpec, ...] = ()
    artifact_gates: tuple[PromotionArtifactGateSpec, ...] = ()
    failure_reason: str = "game promotion gates failed"

    def __post_init__(self) -> None:
        if not self.rung_name:
            raise ValueError("game promotion gate rung_name must be non-empty")
        if not self.failure_reason:
            raise ValueError(
                f"game promotion gate {self.rung_name!r} failure_reason must be non-empty"
            )
        metric_names = [gate.name or gate.metric for gate in self.metric_gates]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError(
                f"game promotion gate {self.rung_name!r} metric gate names must be unique"
            )
        artifact_names = [gate.name or gate.field for gate in self.artifact_gates]
        if len(set(artifact_names)) != len(artifact_names):
            raise ValueError(
                f"game promotion gate {self.rung_name!r} artifact gate names must be unique"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "rung_name": self.rung_name,
            "runtime": self.runtime.to_manifest() if self.runtime is not None else None,
            "metrics": [gate.to_manifest() for gate in self.metric_gates],
            "artifacts": [gate.to_manifest() for gate in self.artifact_gates],
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class GamePromotionPhase:
    """One game-aware phase mapped onto architecture promotion rungs."""

    name: str
    description: str
    stage_name: str | None
    architecture_rungs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("game promotion phase name must be non-empty")
        if not self.description:
            raise ValueError(f"game promotion phase {self.name!r} must define description")
        if not self.architecture_rungs:
            raise ValueError(f"game promotion phase {self.name!r} must define architecture_rungs")


@dataclass(frozen=True)
class GamePromotionPlan:
    """The ordered composition of architecture and game promotion."""

    game_name: str
    phases: tuple[GamePromotionPhase, ...]

    def __post_init__(self) -> None:
        if not self.game_name:
            raise ValueError("game promotion plan game_name must be non-empty")
        if not self.phases:
            raise ValueError(f"game promotion plan {self.game_name!r} must define phases")
        names = [phase.name for phase in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f"game promotion plan {self.game_name!r} phase names must be unique")

    def phase(self, name: str) -> GamePromotionPhase:
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(f"unknown game promotion phase {name!r} for {self.game_name!r}")

    def to_manifest(
        self,
        rung_statuses: dict[str, str],
        gate_specs: Mapping[str, GamePromotionGateSpec] | None = None,
    ) -> dict[str, Any]:
        gate_specs = gate_specs or {}
        return {
            "game": self.game_name,
            "phases": [
                {
                    "name": phase.name,
                    "stage": phase.stage_name,
                    "architecture_rungs": phase.architecture_rungs,
                    "description": phase.description,
                    "rung_statuses": {
                        rung: rung_statuses.get(rung, "unknown")
                        for rung in phase.architecture_rungs
                    },
                    "rung_gates": {
                        rung: gate_specs[rung].to_manifest()
                        for rung in phase.architecture_rungs
                        if rung in gate_specs
                    },
                }
                for phase in self.phases
            ],
        }


GAME_PROMOTION_PHASES = (
    GamePromotionPhase(
        name="architecture-smoke",
        stage_name=None,
        architecture_rungs=("interface-smoke",),
        description="Validate model construction and gradients against game-neutral tensors.",
    ),
    GamePromotionPhase(
        name="game-synthetic",
        stage_name="synthetic",
        architecture_rungs=("synthetic-concept",),
        description="Train on the game's cheapest synthetic architecture concept task.",
    ),
    GamePromotionPhase(
        name="game-block",
        stage_name="block",
        architecture_rungs=("block-smb-smoke",),
        description="Train in the game's simplified simulator with exact labels.",
    ),
    GamePromotionPhase(
        name="game-full-smoke",
        stage_name="full",
        architecture_rungs=(
            "full-smb-asset-mock-perception",
            "full-smb-transfer-smoke",
        ),
        description="Verify full-fidelity observations and actions after perception bootstrap.",
    ),
    GamePromotionPhase(
        name="game-transfer",
        stage_name="full",
        architecture_rungs=("full-smb-transfer-smoke",),
        description="Transfer block-trained policy state into full fidelity.",
    ),
    GamePromotionPhase(
        name="game-full-comparison",
        stage_name="full",
        architecture_rungs=("full-smb-transfer-vs-scratch",),
        description="Compare transferred, scratch, and known-good policies.",
    ),
    GamePromotionPhase(
        name="game-full-training",
        stage_name="full",
        architecture_rungs=("full-smb-fine-tuning",),
        description="Continue or train policy models directly in the emulator.",
    ),
)


def build_game_promotion_plan(plugin) -> GamePromotionPlan:
    """Build the default game-aware promotion plan for one plugin."""

    phases: list[GamePromotionPhase] = []
    for phase in GAME_PROMOTION_PHASES:
        if phase.stage_name is not None:
            resolve_game_stage(plugin.game, phase.stage_name)
        phases.append(phase)
    return GamePromotionPlan(game_name=plugin.name, phases=tuple(phases))
