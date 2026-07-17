"""Game task, curriculum, and success-threshold contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSuccessThreshold:
    """Deterministic success threshold for one game task."""

    min_success_rate: float
    min_mean_return: float
    min_episodes: int
    max_steps: int
    rationale: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be in [0, 1]")
        if self.min_episodes <= 0:
            raise ValueError("min_episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.rationale:
            raise ValueError("success threshold rationale must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_success_rate": self.min_success_rate,
            "min_mean_return": self.min_mean_return,
            "min_episodes": self.min_episodes,
            "max_steps": self.max_steps,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class GameTaskSpec:
    """One fixed, generated, or curriculum task owned by a game profile."""

    name: str
    stage_name: str
    task_type: str
    source: str
    reset_seed: int
    curriculum_stage: int
    success_threshold: TaskSuccessThreshold | None = None
    generation_seed: int | None = None
    generation_config: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must be non-empty")
        if not self.stage_name:
            raise ValueError(f"task {self.name!r} must declare stage_name")
        if self.task_type not in {"fixed", "procedural", "curriculum"}:
            raise ValueError(
                f"task {self.name!r} task_type must be fixed, procedural, or curriculum"
            )
        if not self.source:
            raise ValueError(f"task {self.name!r} must declare source")
        if self.reset_seed < 0:
            raise ValueError(f"task {self.name!r} reset_seed must be non-negative")
        if self.curriculum_stage <= 0:
            raise ValueError(f"task {self.name!r} curriculum_stage must be positive")
        if self.task_type == "fixed" and self.success_threshold is None:
            raise ValueError(f"fixed task {self.name!r} must define success_threshold")
        if self.task_type != "fixed" and self.generation_seed is None:
            raise ValueError(f"{self.task_type} task {self.name!r} must define generation_seed")


@dataclass(frozen=True)
class GameTaskSchema:
    """Per-game task catalog for fixed tasks, generated tasks, and curricula."""

    game_name: str
    tasks: tuple[GameTaskSpec, ...]

    def __post_init__(self) -> None:
        if not self.game_name:
            raise ValueError("task schema game_name must be non-empty")
        if not self.tasks:
            raise ValueError(f"task schema {self.game_name!r} must declare tasks")
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError(f"task schema {self.game_name!r} task names must be unique")

    def task(self, name: str) -> GameTaskSpec:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"unknown task {name!r} for game {self.game_name!r}")

    def tasks_by_type(self, task_type: str) -> tuple[GameTaskSpec, ...]:
        return tuple(task for task in self.tasks if task.task_type == task_type)

    @property
    def fixed_tasks(self) -> tuple[GameTaskSpec, ...]:
        return self.tasks_by_type("fixed")

    @property
    def curriculum(self) -> tuple[GameTaskSpec, ...]:
        return tuple(sorted(self.tasks, key=lambda task: task.curriculum_stage))

    def reset_seeds(self) -> dict[str, int]:
        return {task.name: task.reset_seed for task in self.tasks}
