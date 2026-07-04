"""Deterministic local save-state artifact recipes for Full SMB."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from retroagi.core import SMBAction, coerce_smb_action, to_plain_data
from retroagi.stages.full_smb.adapter import (
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBStage,
)

FULL_SMB_SAVE_STATE_SCHEMA_VERSION = 1
FULL_SMB_SAVE_STATE_KIND = "retroagi.full_smb.save_state"
FULL_SMB_SAVE_STATE_PLAN_KIND = "retroagi.full_smb.save_state_plan"
FULL_SMB_SAVE_STATE_LOCAL_ROOT = Path("local/full_smb/states")
FULL_SMB_SAVE_STATE_CATEGORIES = (
    "starting_position",
    "level_section",
    "death_retry",
    "benchmark",
)


@dataclass(frozen=True)
class FullSMBSaveStateStep:
    """One deterministic action segment used while creating a local save state."""

    action: SMBAction | int | str
    frames: int
    description: str = ""

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("Full SMB save-state script frames must be positive")
        object.__setattr__(self, "action", _coerce_action(self.action))

    def to_manifest(self) -> dict[str, Any]:
        action = coerce_smb_action(self.action)
        return {
            "action": action.name.lower(),
            "action_id": int(action),
            "frames": int(self.frames),
            "description": self.description,
        }


@dataclass(frozen=True)
class FullSMBSaveStateArtifactSpec:
    """Local-only save-state artifact recipe.

    The repo commits this recipe and the expected output path, not emulator state
    bytes. Running the recipe on a local, legally imported ROM creates the file.
    """

    name: str
    category: str
    path: Path
    source_state: str
    reset_seed: int
    description: str
    task_names: tuple[str, ...] = ()
    action_script: tuple[FullSMBSaveStateStep, ...] = ()
    terminal_ok: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Full SMB save-state artifact name must be non-empty")
        if self.category not in FULL_SMB_SAVE_STATE_CATEGORIES:
            raise ValueError(
                "Full SMB save-state category must be one of " f"{FULL_SMB_SAVE_STATE_CATEGORIES}"
            )
        if self.path.is_absolute():
            raise ValueError("Full SMB save-state artifact paths must be relative")
        if not str(self.path).startswith(str(FULL_SMB_SAVE_STATE_LOCAL_ROOT) + "/"):
            raise ValueError(
                "Full SMB save-state artifacts must live under " f"{FULL_SMB_SAVE_STATE_LOCAL_ROOT}"
            )
        if not self.source_state:
            raise ValueError("Full SMB save-state source_state must be non-empty")
        if self.reset_seed < 0:
            raise ValueError("Full SMB save-state reset_seed must be non-negative")
        if not self.description:
            raise ValueError("Full SMB save-state description must be non-empty")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("Full SMB save-state task_names must be unique")
        object.__setattr__(
            self,
            "action_script",
            tuple(_coerce_step(step) for step in self.action_script),
        )

    @property
    def frame_count(self) -> int:
        return sum(step.frames for step in self.action_script)

    def to_env_config(self) -> FullSMBEnvConfig:
        return FullSMBEnvConfig(state=self.source_state)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "path": str(self.path),
            "source_state": self.source_state,
            "reset_seed": int(self.reset_seed),
            "description": self.description,
            "task_names": list(self.task_names),
            "action_script": [step.to_manifest() for step in self.action_script],
            "frame_count": self.frame_count,
            "terminal_ok": bool(self.terminal_ok),
            "local_only": True,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FullSMBSaveStatePlan:
    """Versioned local save-state plan for Full SMB tasks."""

    artifacts: tuple[FullSMBSaveStateArtifactSpec, ...]
    local_root: Path = FULL_SMB_SAVE_STATE_LOCAL_ROOT
    schema_version: int = FULL_SMB_SAVE_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.artifacts:
            raise ValueError("Full SMB save-state plan must declare artifacts")
        names = [artifact.name for artifact in self.artifacts]
        if len(set(names)) != len(names):
            raise ValueError("Full SMB save-state artifact names must be unique")
        paths = [artifact.path for artifact in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("Full SMB save-state artifact paths must be unique")
        missing = sorted(set(FULL_SMB_SAVE_STATE_CATEGORIES).difference(self.categories()))
        if missing:
            raise ValueError(f"Full SMB save-state plan is missing categories: {missing}")

    def artifact(self, name: str) -> FullSMBSaveStateArtifactSpec:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise KeyError(f"unknown Full SMB save-state artifact {name!r}")

    def artifacts_for_category(self, category: str) -> tuple[FullSMBSaveStateArtifactSpec, ...]:
        if category not in FULL_SMB_SAVE_STATE_CATEGORIES:
            raise ValueError(f"unknown Full SMB save-state category {category!r}")
        return tuple(artifact for artifact in self.artifacts if artifact.category == category)

    def artifacts_for_task(self, task_name: str) -> tuple[FullSMBSaveStateArtifactSpec, ...]:
        return tuple(artifact for artifact in self.artifacts if task_name in artifact.task_names)

    def categories(self) -> tuple[str, ...]:
        declared = {artifact.category for artifact in self.artifacts}
        return tuple(
            category for category in FULL_SMB_SAVE_STATE_CATEGORIES if category in declared
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(artifact.path for artifact in self.artifacts)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": FULL_SMB_SAVE_STATE_PLAN_KIND,
            "schema_version": int(self.schema_version),
            "local_root": str(self.local_root),
            "copyrighted_content_committed": False,
            "artifacts": [artifact.to_manifest() for artifact in self.artifacts],
            "categories": {
                category: [artifact.name for artifact in self.artifacts_for_category(category)]
                for category in FULL_SMB_SAVE_STATE_CATEGORIES
            },
        }


@dataclass(frozen=True)
class FullSMBSaveStateArtifactResult:
    """Result from creating one local Full SMB save-state file."""

    name: str
    path: Path
    steps_executed: int
    reward_total: float
    terminated: bool
    truncated: bool
    observation_sha256: str
    bytes_written: int
    final_info: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "steps_executed": int(self.steps_executed),
            "reward_total": float(self.reward_total),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "observation_sha256": self.observation_sha256,
            "bytes_written": int(self.bytes_written),
            "final_info": dict(self.final_info),
        }


StageFactory = Callable[[FullSMBSaveStateArtifactSpec], Any]


def full_smb_save_state_plan() -> FullSMBSaveStatePlan:
    """Return the canonical local save-state artifact plan."""

    return FULL_SMB_SAVE_STATE_PLAN


def write_full_smb_save_state_plan(
    path: Path,
    plan: Optional[FullSMBSaveStatePlan] = None,
) -> dict[str, Any]:
    """Write the save-state recipe manifest without generating emulator bytes."""

    selected_plan = plan or full_smb_save_state_plan()
    manifest = selected_plan.to_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_data(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def create_full_smb_save_state_artifact(
    spec: FullSMBSaveStateArtifactSpec,
    *,
    repository_root: Path | str = Path("."),
    stage_factory: StageFactory | None = None,
    overwrite: bool = False,
) -> FullSMBSaveStateArtifactResult:
    """Create one local save-state artifact from its deterministic recipe."""

    root = Path(repository_root)
    output_path = root / spec.path
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Full SMB save-state artifact already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )
    stage = stage_factory(spec) if stage_factory is not None else _make_stage(spec)
    observation: np.ndarray | None = None
    info: Mapping[str, Any] = {}
    reward_total = 0.0
    steps_executed = 0
    terminated = False
    truncated = False
    try:
        observation = np.asarray(stage.reset(seed=spec.reset_seed))
        info = dict(getattr(stage, "last_info", {}))
        for segment in spec.action_script:
            for _ in range(segment.frames):
                observation, reward, terminated, truncated, info = stage.step(segment.action)
                observation = np.asarray(observation)
                reward_total += float(reward)
                steps_executed += 1
                if terminated or truncated:
                    if not spec.terminal_ok:
                        raise RuntimeError(
                            f"save-state artifact {spec.name!r} ended before its "
                            "script completed"
                        )
                    break
            if terminated or truncated:
                break
        snapshot = stage.save_emulator_state()
    finally:
        close = getattr(stage, "close", None)
        if close is not None:
            close()

    if observation is None:
        raise RuntimeError(f"save-state artifact {spec.name!r} did not reset")
    payload = {
        "kind": FULL_SMB_SAVE_STATE_KIND,
        "schema_version": FULL_SMB_SAVE_STATE_SCHEMA_VERSION,
        "spec": spec.to_manifest(),
        "state": snapshot,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return FullSMBSaveStateArtifactResult(
        name=spec.name,
        path=spec.path,
        steps_executed=steps_executed,
        reward_total=reward_total,
        terminated=terminated,
        truncated=truncated,
        observation_sha256=_observation_hash(observation),
        bytes_written=output_path.stat().st_size,
        final_info=dict(info),
    )


def create_full_smb_save_state_artifacts(
    *,
    artifact_names: Iterable[str] = (),
    plan: Optional[FullSMBSaveStatePlan] = None,
    repository_root: Path | str = Path("."),
    stage_factory: StageFactory | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create selected local save-state artifacts and return a manifest."""

    selected_plan = plan or full_smb_save_state_plan()
    names = tuple(artifact_names)
    artifacts = (
        tuple(selected_plan.artifact(name) for name in names) if names else selected_plan.artifacts
    )
    results = [
        create_full_smb_save_state_artifact(
            artifact,
            repository_root=repository_root,
            stage_factory=stage_factory,
            overwrite=overwrite,
        )
        for artifact in artifacts
    ]
    return {
        "kind": "retroagi.full_smb.save_state_creation",
        "schema_version": FULL_SMB_SAVE_STATE_SCHEMA_VERSION,
        "plan": selected_plan.to_manifest(),
        "results": [result.to_manifest() for result in results],
    }


def load_full_smb_save_state_payload(path: Path) -> Mapping[str, Any]:
    """Load a local save-state payload created by this module.

    Only load artifacts produced locally from trusted ROM content. Pickle is
    intentionally local-only and must not be used for downloaded artifacts.
    """

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Full SMB save-state payload must be a mapping")
    if payload.get("kind") != FULL_SMB_SAVE_STATE_KIND:
        raise ValueError("Full SMB save-state payload has the wrong kind")
    if payload.get("schema_version") != FULL_SMB_SAVE_STATE_SCHEMA_VERSION:
        raise ValueError("Full SMB save-state payload has an unsupported schema")
    return payload


def _make_stage(spec: FullSMBSaveStateArtifactSpec) -> FullSMBStage:
    return FullSMBStage(
        env_config=spec.to_env_config(),
        observation_config=FullSMBObservationConfig(
            frame_skip=1,
            frame_stack=2,
            resize_shape=None,
        ),
        vision=_NoopVision(),
    )


class _NoopVision:
    def encode(self, _observation: Any) -> Any:
        raise RuntimeError("Full SMB save-state generation does not encode vision")


def _coerce_action(action: SMBAction | int | str) -> SMBAction:
    if isinstance(action, str):
        normalized = action.upper()
        try:
            return SMBAction[normalized]
        except KeyError as exc:
            valid = ", ".join(item.name.lower() for item in SMBAction)
            raise ValueError(
                "invalid Full SMB save-state action " f"{action!r}; expected one of: {valid}"
            ) from exc
    return coerce_smb_action(action)


def _coerce_step(step: FullSMBSaveStateStep | Mapping[str, Any]) -> FullSMBSaveStateStep:
    if isinstance(step, FullSMBSaveStateStep):
        return step
    return FullSMBSaveStateStep(
        action=step["action"],
        frames=int(step["frames"]),
        description=str(step.get("description", "")),
    )


def _step(action: SMBAction, frames: int, description: str) -> FullSMBSaveStateStep:
    return FullSMBSaveStateStep(action=action, frames=frames, description=description)


def _observation_hash(observation: Any) -> str:
    array = np.asarray(observation)
    return hashlib.sha256(array.tobytes()).hexdigest()


FULL_SMB_SAVE_STATE_PLAN = FullSMBSaveStatePlan(
    artifacts=(
        FullSMBSaveStateArtifactSpec(
            name="start_1_1_spawn",
            category="starting_position",
            path=Path("local/full_smb/states/starts/1_1_spawn.state"),
            source_state="Level1-1",
            reset_seed=310_101,
            task_names=("smoke_1_1_spawn",),
            description="Initial reset state for level 1-1 smoke checks.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="benchmark_1_1_start",
            category="benchmark",
            path=Path("local/full_smb/states/benchmark/1_1_start.state"),
            source_state="Level1-1",
            reset_seed=320_101,
            task_names=("benchmark_1_1_start",),
            description="Fixed benchmark reset state for level 1-1.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="benchmark_1_2_start",
            category="benchmark",
            path=Path("local/full_smb/states/benchmark/1_2_start.state"),
            source_state="Level1-2",
            reset_seed=320_102,
            task_names=("benchmark_1_2_start",),
            description="Fixed benchmark reset state for level 1-2.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="benchmark_2_1_start",
            category="benchmark",
            path=Path("local/full_smb/states/benchmark/2_1_start.state"),
            source_state="Level2-1",
            reset_seed=320_201,
            task_names=("benchmark_2_1_start",),
            description="Fixed benchmark reset state for level 2-1.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="section_1_1_midpipe",
            category="level_section",
            path=Path("local/full_smb/states/curriculum/1_1_midpipe.state"),
            source_state="Level1-1",
            reset_seed=330_102,
            task_names=("curriculum_1_1_midpipe",),
            action_script=(
                _step(
                    SMBAction.RIGHT,
                    180,
                    "Hold right from spawn toward the first pipe.",
                ),
                _step(
                    SMBAction.RIGHT_JUMP,
                    24,
                    "Hop over the first low obstacle while preserving motion.",
                ),
                _step(SMBAction.RIGHT, 96, "Stabilize near the first pipe section."),
            ),
            description="Curriculum state near the first level 1-1 pipe section.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="section_1_1_first_enemy_approach",
            category="level_section",
            path=Path("local/full_smb/states/curriculum/1_1_first_enemy_approach.state"),
            source_state="Level1-1",
            reset_seed=330_101,
            action_script=(
                _step(
                    SMBAction.RIGHT,
                    132,
                    "Run right from spawn to just before the first enemy timing window.",
                ),
            ),
            description="Curriculum state before the first enemy in level 1-1.",
            metadata={"expected_use": "obstacle-window jump-duration sweeps"},
        ),
        FullSMBSaveStateArtifactSpec(
            name="section_1_1_flagpole_approach",
            category="level_section",
            path=Path("local/full_smb/states/curriculum/1_1_flagpole_approach.state"),
            source_state="Level1-1",
            reset_seed=330_103,
            task_names=("curriculum_1_1_flagpole",),
            action_script=(
                _step(SMBAction.RIGHT, 420, "Run through the opening terrain."),
                _step(SMBAction.RIGHT_JUMP, 40, "Clear mid-level hazards."),
                _step(SMBAction.RIGHT, 720, "Advance toward the flagpole approach."),
            ),
            description="Curriculum state before the level 1-1 flagpole approach.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="section_1_2_underworld_opening",
            category="level_section",
            path=Path("local/full_smb/states/curriculum/1_2_underworld_opening.state"),
            source_state="Level1-2",
            reset_seed=330_201,
            task_names=("curriculum_1_2_underworld",),
            description="Curriculum reset state for the opening of level 1-2.",
        ),
        FullSMBSaveStateArtifactSpec(
            name="death_retry_1_1_first_gap",
            category="death_retry",
            path=Path("local/full_smb/states/death_retry/1_1_first_gap.state"),
            source_state="Level1-1",
            reset_seed=340_101,
            action_script=(
                _step(SMBAction.RIGHT, 620, "Run right toward the first risky gap."),
                _step(
                    SMBAction.NOOP,
                    120,
                    "Release controls and allow a death/retry state.",
                ),
            ),
            terminal_ok=True,
            description="Local death/retry artifact for validating reset handling.",
            metadata={"expected_use": "episode-boundary and retry tests"},
        ),
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retroagi.stages.full_smb.save_states",
        description="Write or create local-only deterministic Full SMB save states.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write the save-state recipe manifest")
    plan.add_argument(
        "--output",
        type=Path,
        default=FULL_SMB_SAVE_STATE_LOCAL_ROOT / "save_state_plan.json",
    )

    create = subparsers.add_parser("create", help="create local save-state files")
    create.add_argument(
        "--only",
        action="append",
        default=[],
        help="artifact name to create; repeat to create multiple names",
    )
    create.add_argument(
        "--output-manifest",
        type=Path,
        default=FULL_SMB_SAVE_STATE_LOCAL_ROOT / "save_state_manifest.json",
    )
    create.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        manifest = write_full_smb_save_state_plan(args.output)
        print(json.dumps(to_plain_data(manifest), indent=2, sort_keys=True))
        return 0
    if args.command == "create":
        manifest = create_full_smb_save_state_artifacts(
            artifact_names=args.only,
            overwrite=bool(args.overwrite),
        )
        output_manifest = Path(args.output_manifest)
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(
            json.dumps(to_plain_data(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(to_plain_data(manifest), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported save-state command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
