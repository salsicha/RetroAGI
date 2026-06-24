"""Full SMB train/evaluation task-set definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from retroagi.stages.full_smb.adapter import FULL_SMB_GAME, FullSMBEnvConfig

FULL_SMB_TASK_SET_NAMES = (
    "smoke",
    "fixed_benchmark",
    "curriculum",
    "heldout_generalization",
)
FULL_SMB_TASK_SPLITS = ("train", "eval", "heldout")
FULL_SMB_START_MODES = ("level_start", "save_state_artifact")


@dataclass(frozen=True)
class FullSMBTaskStart:
    """How a Full SMB task initializes the emulator."""

    mode: str
    state: Optional[str] = None
    save_state_path: Optional[Path] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.mode not in FULL_SMB_START_MODES:
            raise ValueError(f"Full SMB task start mode must be one of {FULL_SMB_START_MODES}")
        if self.mode == "level_start" and not self.state:
            raise ValueError("level_start tasks must define a stable-retro state")
        if self.mode == "save_state_artifact" and self.save_state_path is None:
            raise ValueError("save_state_artifact tasks must define save_state_path")
        if self.save_state_path is not None and self.save_state_path.is_absolute():
            raise ValueError("Full SMB save-state artifact paths must be repository-relative")

    def to_env_config(self, *, game: str = FULL_SMB_GAME) -> FullSMBEnvConfig:
        return FullSMBEnvConfig(game=game, state=self.state)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "state": self.state,
            "save_state_path": str(self.save_state_path) if self.save_state_path else None,
            "description": self.description,
        }


@dataclass(frozen=True)
class FullSMBTaskSpec:
    """One Full SMB train/eval task definition."""

    name: str
    task_set: str
    split: str
    start: FullSMBTaskStart
    reset_seed: int
    episodes: int
    max_steps: int
    curriculum_stage: int
    goal: str
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Full SMB task name must be non-empty")
        if self.task_set not in FULL_SMB_TASK_SET_NAMES:
            raise ValueError(f"Full SMB task_set must be one of {FULL_SMB_TASK_SET_NAMES}")
        if self.split not in FULL_SMB_TASK_SPLITS:
            raise ValueError(f"Full SMB task split must be one of {FULL_SMB_TASK_SPLITS}")
        if self.reset_seed < 0:
            raise ValueError("Full SMB task reset_seed must be non-negative")
        if self.episodes <= 0:
            raise ValueError("Full SMB task episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("Full SMB task max_steps must be positive")
        if self.curriculum_stage <= 0:
            raise ValueError("Full SMB task curriculum_stage must be positive")
        if not self.goal:
            raise ValueError("Full SMB task goal must be non-empty")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Full SMB task tags must be unique")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_set": self.task_set,
            "split": self.split,
            "start": self.start.to_manifest(),
            "reset_seed": self.reset_seed,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "curriculum_stage": self.curriculum_stage,
            "goal": self.goal,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FullSMBTaskCatalog:
    """Named Full SMB task sets for train/eval orchestration."""

    tasks: tuple[FullSMBTaskSpec, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("Full SMB task catalog must declare tasks")
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("Full SMB task names must be unique")
        missing_sets = sorted(set(FULL_SMB_TASK_SET_NAMES).difference(self.task_sets()))
        if missing_sets:
            raise ValueError(f"Full SMB task catalog is missing task sets: {missing_sets}")

    def task(self, name: str) -> FullSMBTaskSpec:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"unknown Full SMB task {name!r}")

    def tasks_for_set(self, task_set: str) -> tuple[FullSMBTaskSpec, ...]:
        if task_set not in FULL_SMB_TASK_SET_NAMES:
            raise ValueError(f"unknown Full SMB task set {task_set!r}")
        return tuple(task for task in self.tasks if task.task_set == task_set)

    def tasks_for_split(self, split: str) -> tuple[FullSMBTaskSpec, ...]:
        if split not in FULL_SMB_TASK_SPLITS:
            raise ValueError(f"unknown Full SMB task split {split!r}")
        return tuple(task for task in self.tasks if task.split == split)

    def task_sets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(task.task_set for task in self.tasks))

    @property
    def curriculum(self) -> tuple[FullSMBTaskSpec, ...]:
        return tuple(
            sorted(
                self.tasks_for_set("curriculum"),
                key=lambda task: (task.curriculum_stage, task.name),
            )
        )

    @property
    def save_state_artifact_paths(self) -> tuple[Path, ...]:
        paths = [
            task.start.save_state_path
            for task in self.tasks
            if task.start.save_state_path is not None
        ]
        return tuple(path for path in paths if path is not None)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "task_sets": {
                task_set: [task.to_manifest() for task in self.tasks_for_set(task_set)]
                for task_set in FULL_SMB_TASK_SET_NAMES
            },
            "save_state_artifact_paths": [str(path) for path in self.save_state_artifact_paths],
        }


def _level_start(state: str, description: str) -> FullSMBTaskStart:
    return FullSMBTaskStart(
        mode="level_start",
        state=state,
        description=description,
    )


def _save_state(path: str, base_state: str, description: str) -> FullSMBTaskStart:
    return FullSMBTaskStart(
        mode="save_state_artifact",
        state=base_state,
        save_state_path=Path(path),
        description=description,
    )


FULL_SMB_TASK_CATALOG = FullSMBTaskCatalog(
    tasks=(
        FullSMBTaskSpec(
            name="smoke_1_1_spawn",
            task_set="smoke",
            split="eval",
            start=_level_start("Level1-1", "Stable-retro level 1-1 start state."),
            reset_seed=210_001,
            episodes=1,
            max_steps=128,
            curriculum_stage=1,
            goal="Verify reset, stepping, signals, and short-horizon survival.",
            tags=("smoke", "headless", "level_start"),
        ),
        FullSMBTaskSpec(
            name="benchmark_1_1_start",
            task_set="fixed_benchmark",
            split="eval",
            start=_level_start("Level1-1", "Canonical level 1-1 start."),
            reset_seed=220_101,
            episodes=3,
            max_steps=2400,
            curriculum_stage=1,
            goal="Measure early-game progress, survival, score, and completion from 1-1.",
            tags=("benchmark", "level_start", "overworld"),
        ),
        FullSMBTaskSpec(
            name="benchmark_1_2_start",
            task_set="fixed_benchmark",
            split="eval",
            start=_level_start("Level1-2", "Canonical level 1-2 start."),
            reset_seed=220_102,
            episodes=3,
            max_steps=2400,
            curriculum_stage=2,
            goal="Measure transfer under underground visuals and tighter terrain.",
            tags=("benchmark", "level_start", "underground"),
        ),
        FullSMBTaskSpec(
            name="benchmark_2_1_start",
            task_set="fixed_benchmark",
            split="eval",
            start=_level_start("Level2-1", "Canonical level 2-1 start."),
            reset_seed=220_201,
            episodes=3,
            max_steps=2600,
            curriculum_stage=3,
            goal="Measure progress on a later overworld level with denser hazards.",
            tags=("benchmark", "level_start", "overworld"),
        ),
        FullSMBTaskSpec(
            name="curriculum_1_1_opening",
            task_set="curriculum",
            split="train",
            start=_level_start("Level1-1", "First training rung from the level start."),
            reset_seed=230_101,
            episodes=4,
            max_steps=600,
            curriculum_stage=1,
            goal="Learn stable rightward movement, jumping, and survival from spawn.",
            tags=("curriculum", "level_start", "opening"),
        ),
        FullSMBTaskSpec(
            name="curriculum_1_1_midpipe",
            task_set="curriculum",
            split="train",
            start=_save_state(
                "local/full_smb/states/curriculum/1_1_midpipe.state",
                "Level1-1",
                "Local save-state near the first pipe in level 1-1.",
            ),
            reset_seed=230_102,
            episodes=4,
            max_steps=900,
            curriculum_stage=2,
            goal="Train obstacle timing after the opening segment is stable.",
            tags=("curriculum", "save_state", "obstacle"),
        ),
        FullSMBTaskSpec(
            name="curriculum_1_1_flagpole",
            task_set="curriculum",
            split="train",
            start=_save_state(
                "local/full_smb/states/curriculum/1_1_flagpole_approach.state",
                "Level1-1",
                "Local save-state before the level 1-1 flagpole approach.",
            ),
            reset_seed=230_103,
            episodes=4,
            max_steps=900,
            curriculum_stage=3,
            goal="Train level-completion behavior after progress and survival work.",
            tags=("curriculum", "save_state", "completion"),
        ),
        FullSMBTaskSpec(
            name="curriculum_1_2_underworld",
            task_set="curriculum",
            split="train",
            start=_level_start("Level1-2", "Underground level-start curriculum rung."),
            reset_seed=230_201,
            episodes=4,
            max_steps=1200,
            curriculum_stage=4,
            goal="Adapt the transferred policy to underground visuals and spacing.",
            tags=("curriculum", "level_start", "underground"),
        ),
        FullSMBTaskSpec(
            name="heldout_2_2_water",
            task_set="heldout_generalization",
            split="heldout",
            start=_level_start("Level2-2", "Held-out water level start."),
            reset_seed=240_202,
            episodes=3,
            max_steps=2600,
            curriculum_stage=1,
            goal="Measure generalization to water physics and non-overworld visuals.",
            tags=("heldout", "level_start", "water"),
        ),
        FullSMBTaskSpec(
            name="heldout_3_1_bridge",
            task_set="heldout_generalization",
            split="heldout",
            start=_level_start("Level3-1", "Held-out bridge/athletic level start."),
            reset_seed=240_301,
            episodes=3,
            max_steps=2600,
            curriculum_stage=2,
            goal="Measure generalization to elevated terrain and bridge-like pacing.",
            tags=("heldout", "level_start", "athletic"),
        ),
        FullSMBTaskSpec(
            name="heldout_8_1_long",
            task_set="heldout_generalization",
            split="heldout",
            start=_level_start("Level8-1", "Held-out late-game level start."),
            reset_seed=240_801,
            episodes=3,
            max_steps=3200,
            curriculum_stage=3,
            goal="Measure late-game generalization without tuning on this task.",
            tags=("heldout", "level_start", "late_game"),
        ),
    )
)


def full_smb_task_catalog() -> FullSMBTaskCatalog:
    return FULL_SMB_TASK_CATALOG
