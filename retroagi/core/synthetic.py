"""Game-owned synthetic low-fidelity data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class SyntheticSplitSpec:
    """One deterministic synthetic-data split."""

    name: str
    size: int
    seed: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("synthetic split name must be non-empty")
        if self.size <= 0:
            raise ValueError(f"synthetic split {self.name!r} size must be positive")
        if self.seed < 0:
            raise ValueError(f"synthetic split {self.name!r} seed must be non-negative")


@dataclass(frozen=True)
class SyntheticDataSpec:
    """Cheap game-concept data available before pixels or emulator frames."""

    game_name: str
    name: str
    stage_name: str
    observation_kind: str
    target_kind: str
    generator: str
    splits: tuple[SyntheticSplitSpec, ...]
    shape_contract: Mapping[str, Any]
    description: str = ""
    task_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.game_name:
            raise ValueError("synthetic data spec game_name must be non-empty")
        if not self.name:
            raise ValueError("synthetic data spec name must be non-empty")
        if not self.stage_name:
            raise ValueError(f"synthetic data spec {self.name!r} must declare stage_name")
        if not self.observation_kind:
            raise ValueError(f"synthetic data spec {self.name!r} must declare observation_kind")
        if not self.target_kind:
            raise ValueError(f"synthetic data spec {self.name!r} must declare target_kind")
        if not self.generator:
            raise ValueError(f"synthetic data spec {self.name!r} must declare generator")
        if not self.splits:
            raise ValueError(f"synthetic data spec {self.name!r} must declare splits")
        split_names = [split.name for split in self.splits]
        if len(set(split_names)) != len(split_names):
            raise ValueError(f"synthetic data spec {self.name!r} split names must be unique")
        split_seeds = [split.seed for split in self.splits]
        if len(set(split_seeds)) != len(split_seeds):
            raise ValueError(f"synthetic data spec {self.name!r} split seeds must be unique")
        if not self.shape_contract:
            raise ValueError(f"synthetic data spec {self.name!r} must declare shape_contract")

    def split(self, name: str) -> SyntheticSplitSpec:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(f"unknown synthetic split {name!r} for {self.name!r}")

    def split_sizes(self) -> dict[str, int]:
        return {split.name: split.size for split in self.splits}

    def split_seeds(self) -> dict[str, int]:
        return {split.name: split.seed for split in self.splits}


class SyntheticDataGenerator(Protocol):
    """Generate a declared low-fidelity synthetic split."""

    name: str

    def generate(
        self,
        spec: SyntheticDataSpec,
        split: SyntheticSplitSpec,
    ) -> Any:
        """Return backend-native synthetic data for one split."""
