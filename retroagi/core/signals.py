"""Game-neutral signal contracts for stage adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class GameSignals:
    """Common game-progress signals extracted from backend-native info."""

    position: tuple[float, float] | None = None
    progress: float | None = None
    score: int | None = None
    health: float | None = None
    lives: int | None = None
    inventory: Mapping[str, Any] = field(default_factory=dict)
    collectibles: Mapping[str, int] = field(default_factory=dict)
    completion: bool = False
    death: bool = False
    timeout: bool = False
    terminated: bool = False
    truncated: bool = False
    objectives: Mapping[str, Any] = field(default_factory=dict)
    termination_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "progress": self.progress,
            "score": self.score,
            "health": self.health,
            "lives": self.lives,
            "inventory": dict(self.inventory),
            "collectibles": dict(self.collectibles),
            "completion": self.completion,
            "death": self.death,
            "timeout": self.timeout,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "objectives": dict(self.objectives),
            "termination_reason": self.termination_reason,
        }


class GameSignalExtractor(Protocol):
    """Extract game-neutral signals from a stage backend info mapping."""

    game_name: str

    def extract(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> GameSignals:
        """Return normalized game signals for one backend transition."""
