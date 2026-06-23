"""Game-aware promotion plan contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stage_resolution import resolve_game_stage


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
            raise ValueError(
                f"game promotion phase {self.name!r} must define architecture_rungs"
            )


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
            raise ValueError(
                f"game promotion plan {self.game_name!r} phase names must be unique"
            )

    def phase(self, name: str) -> GamePromotionPhase:
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(f"unknown game promotion phase {name!r} for {self.game_name!r}")

    def to_manifest(self, rung_statuses: dict[str, str]) -> dict[str, Any]:
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
