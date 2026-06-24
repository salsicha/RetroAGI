"""Artifact layout for preserved Full SMB runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

FULL_SMB_ARTIFACT_LAYOUT_SCHEMA_VERSION = 1
DEFAULT_FULL_SMB_ARTIFACT_ROOT = Path("artifacts/full_smb")
DEFAULT_FULL_SMB_ARTIFACT_RUN_NAME = "latest"


@dataclass(frozen=True)
class FullSMBArtifactLayout:
    """Canonical workspace paths for one preserved Full SMB run."""

    run_name: str = DEFAULT_FULL_SMB_ARTIFACT_RUN_NAME
    root: Path = DEFAULT_FULL_SMB_ARTIFACT_ROOT

    def __post_init__(self) -> None:
        run_name = str(self.run_name).strip()
        if not run_name:
            raise ValueError("run_name must be non-empty")
        run_path = Path(run_name)
        if run_path.is_absolute() or len(run_path.parts) != 1 or run_name in {".", ".."}:
            raise ValueError("run_name must be a single relative path segment")
        if any(part in {"", ".", ".."} for part in run_path.parts):
            raise ValueError("run_name must not contain empty, current, or parent segments")
        root = Path(self.root)
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "root", root)

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    @property
    def summaries_dir(self) -> Path:
        return self.run_dir / "summaries"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def recordings_dir(self) -> Path:
        return self.run_dir / "recordings"

    @property
    def videos_dir(self) -> Path:
        return self.run_dir / "videos"

    @property
    def evaluations_dir(self) -> Path:
        return self.run_dir / "evaluations"

    @property
    def comparisons_dir(self) -> Path:
        return self.run_dir / "comparisons"

    @property
    def tracking_dir(self) -> Path:
        return self.run_dir / "tracking"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def directories(self) -> dict[str, Path]:
        return {
            "run": self.run_dir,
            "summaries": self.summaries_dir,
            "logs": self.logs_dir,
            "recordings": self.recordings_dir,
            "videos": self.videos_dir,
            "evaluations": self.evaluations_dir,
            "comparisons": self.comparisons_dir,
            "tracking": self.tracking_dir,
            "checkpoints": self.checkpoints_dir,
        }

    def files(self) -> dict[str, Path]:
        return {
            "content_metadata": self.run_dir / "content.json",
            "layout_manifest": self.run_dir / "artifact_layout.json",
            "throughput_benchmark": self.summaries_dir / "throughput_benchmark.json",
            "train_summary": self.summaries_dir / "train_summary.json",
            "resume_summary": self.summaries_dir / "resume_summary.json",
            "recording_summary": self.summaries_dir / "recording_summary.json",
            "play_summary": self.summaries_dir / "play_summary.json",
            "train_log": self.logs_dir / "train.jsonl",
            "evaluation_report": self.evaluations_dir / "evaluation.json",
            "fixed_task_report": self.evaluations_dir / "fixed_task_thresholds.json",
            "recording_manifest": self.recordings_dir / "recording_manifest.npz",
            "play_manifest": self.recordings_dir / "play_manifest.npz",
            "evaluation_video": self.videos_dir / "evaluation.mp4",
            "play_video": self.videos_dir / "play.mp4",
            "comparison_report": self.comparisons_dir / "policy_suite_comparison.json",
            "legacy_transfer_comparison_report": self.comparisons_dir / "transfer_vs_scratch.json",
            "policy_checkpoint": self.checkpoints_dir / "policy.pth",
            "transferred_policy_checkpoint": self.checkpoints_dir / "transferred_policy.pth",
        }

    def ensure_directories(self) -> None:
        for directory in self.directories().values():
            directory.mkdir(parents=True, exist_ok=True)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": FULL_SMB_ARTIFACT_LAYOUT_SCHEMA_VERSION,
            "run_name": self.run_name,
            "root": str(self.root),
            "run_dir": str(self.run_dir),
            "directories": {name: str(path) for name, path in self.directories().items()},
            "files": {name: str(path) for name, path in self.files().items()},
        }


def full_smb_artifact_layout(
    run_name: str = DEFAULT_FULL_SMB_ARTIFACT_RUN_NAME,
    *,
    root: Path = DEFAULT_FULL_SMB_ARTIFACT_ROOT,
) -> FullSMBArtifactLayout:
    return FullSMBArtifactLayout(run_name=run_name, root=root)
