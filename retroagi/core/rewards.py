"""Game-owned reward configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RewardTermSpec:
    """One tunable reward term declared by a game profile."""

    name: str
    default: float
    direction: str
    signal: str
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward term name must be non-empty")
        if self.direction not in {"positive", "negative", "neutral"}:
            raise ValueError(
                f"reward term {self.name!r} direction must be positive, negative, or neutral"
            )
        if not self.signal:
            raise ValueError(f"reward term {self.name!r} must declare a signal")
        if not self.description:
            raise ValueError(f"reward term {self.name!r} must declare a description")
        _validate_directional_value(self.name, float(self.default), self.direction)


@dataclass(frozen=True)
class RewardConfigSchema:
    """Per-game reward-term schema used by environments and trainers."""

    game_name: str
    terms: tuple[RewardTermSpec, ...]

    def __post_init__(self) -> None:
        if not self.game_name:
            raise ValueError("reward schema game_name must be non-empty")
        if not self.terms:
            raise ValueError(f"reward schema {self.game_name!r} must declare terms")
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            raise ValueError(f"reward schema {self.game_name!r} term names must be unique")

    @property
    def term_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)

    def term(self, name: str) -> RewardTermSpec:
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(f"unknown reward term {name!r} for game {self.game_name!r}")

    def defaults(self) -> dict[str, float]:
        return {term.name: float(term.default) for term in self.terms}

    def validate(self, values: Mapping[str, float] | None = None) -> dict[str, float]:
        """Return defaults overlaid with ``values`` after schema validation."""

        resolved = self.defaults()
        if values:
            unknown = sorted(set(values).difference(resolved))
            if unknown:
                raise ValueError(
                    f"reward schema {self.game_name!r} does not define terms: {unknown}"
                )
            for name, value in values.items():
                term = self.term(name)
                numeric = float(value)
                _validate_directional_value(name, numeric, term.direction)
                resolved[name] = numeric
        return resolved


def _validate_directional_value(name: str, value: float, direction: str) -> None:
    if direction == "positive" and value < 0:
        raise ValueError(f"positive reward term {name!r} must be non-negative")
    if direction == "negative" and value > 0:
        raise ValueError(f"negative reward term {name!r} must be non-positive")
