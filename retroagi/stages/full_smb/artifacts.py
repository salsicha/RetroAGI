"""Artifact layout for preserved Full SMB runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

FULL_SMB_ARTIFACT_LAYOUT_SCHEMA_VERSION = 1
FULL_SMB_DOCUMENTED_BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_FULL_SMB_ARTIFACT_ROOT = Path("artifacts/full_smb")
DEFAULT_FULL_SMB_ARTIFACT_RUN_NAME = "latest"
DEFAULT_FULL_SMB_DOCUMENTED_BENCHMARK_RUN_NAME = "documented_benchmark_seed0"

FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_COMMANDS = (
    "check_env",
    "throughput_benchmark",
    "transfer",
    "train",
    "resume",
    "evaluate",
    "record",
    "play",
    "compare",
)

FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_ARTIFACTS = (
    "content_metadata",
    "environment_check",
    "layout_manifest",
    "throughput_benchmark",
    "transferred_policy_checkpoint",
    "policy_checkpoint",
    "resumed_policy_checkpoint",
    "train_summary",
    "resume_summary",
    "evaluation_report",
    "recording_summary",
    "recording_manifest",
    "play_summary",
    "play_manifest",
    "comparison_report",
)


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
            "environment_check": self.run_dir / "env_check.json",
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
            "resumed_policy_checkpoint": self.checkpoints_dir / "resumed_policy.pth",
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


def full_smb_documented_benchmark_manifest(
    run_name: str = DEFAULT_FULL_SMB_DOCUMENTED_BENCHMARK_RUN_NAME,
    *,
    seed: int = 0,
    root: Path = DEFAULT_FULL_SMB_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Return the canonical manifest for a local Full SMB benchmark run."""

    seed = int(seed)
    layout = full_smb_artifact_layout(run_name, root=root)
    files = layout.files()
    return {
        "schema_version": FULL_SMB_DOCUMENTED_BENCHMARK_SCHEMA_VERSION,
        "run_name": layout.run_name,
        "stage": "full_smb",
        "game": "SuperMarioBros-Nes",
        "seed": seed,
        "status": "documented_local_benchmark",
        "local_only_content": True,
        "policy_artifact_status": "produced_locally_after_running_commands",
        "rom_bytes_committed": False,
        "checkpoint_bytes_committed": False,
        "layout_manifest": layout.to_manifest(),
        "required_commands": {
            "check_env": (
                f"retroagi check-env --game smb --stage full --seed {seed} --steps 4 "
                f"--frame-skip 2 --output {files['environment_check']}"
            ),
            "throughput_benchmark": (
                "python -m retroagi.stages.full_smb.benchmark --steps 1000 "
                f"--warmup-steps 100 --seed {seed} --frame-skip 2 --device cpu --output "
                f"{files['throughput_benchmark']}"
            ),
            "transfer": (
                "retroagi transfer --game smb --stage full "
                "--block-policy-checkpoint data/block_smb/policy.pth "
                "--block-vision-checkpoint data/block_vit/block_vit.pth "
                "--full-smb-vision-checkpoint data/vit/full_smb_vit.pth "
                f"--output-checkpoint {files['transferred_policy_checkpoint']}"
            ),
            "train": (
                "retroagi train --game smb --stage full --mode fine-tune "
                f"--seed {seed} "
                f"--init-checkpoint {files['transferred_policy_checkpoint']} "
                "--full-smb-vision-checkpoint data/vit/full_smb_vit.pth "
                "--perception-mode freeze --task-set curriculum --epochs 1 "
                "--updates-per-epoch 1 --rollout-steps 64 --evaluation-episodes 1 "
                "--evaluation-max-steps 64 --evaluation-interval-epochs 1 "
                f"--checkpoint {files['policy_checkpoint']} "
                f"--recording-dir {layout.recordings_dir} "
                f"--recording-path {files['recording_manifest']} "
                f"--log-path {files['train_log']} "
                f"--tracking-log-dir {layout.tracking_dir} "
                f"--output-summary {files['train_summary']}"
            ),
            "resume": (
                "retroagi resume --game smb --stage full "
                f"--seed {seed} "
                f"--checkpoint {files['policy_checkpoint']} "
                f"--save-checkpoint {files['resumed_policy_checkpoint']} "
                "--task-set curriculum --epochs 2 --updates-per-epoch 1 "
                "--rollout-steps 64 --evaluation-episodes 1 "
                "--evaluation-max-steps 64 --evaluation-interval-epochs 1 "
                "--tracking-backend none "
                f"--tracking-log-dir {layout.tracking_dir} "
                f"--output-summary {files['resume_summary']}"
            ),
            "evaluate": (
                "retroagi evaluate --game smb --stage full "
                f"--seed {seed} "
                f"--checkpoint {files['policy_checkpoint']} --task-set fixed_benchmark "
                "--evaluation-episodes 3 --evaluation-max-steps 2400 "
                f"--output-summary {files['evaluation_report']}"
            ),
            "record": (
                "retroagi record --game smb --stage full "
                f"--seed {seed} "
                f"--checkpoint {files['policy_checkpoint']} --task-set fixed_benchmark "
                "--evaluation-episodes 3 --evaluation-max-steps 2400 "
                f"--record-dir {layout.recordings_dir} "
                f"--recording-path {files['recording_manifest']} "
                f"--output-summary {files['recording_summary']}"
            ),
            "play": (
                "retroagi play --game smb --stage full "
                f"--seed {seed} "
                f"--checkpoint {files['policy_checkpoint']} --task-set fixed_benchmark "
                "--level 1-1 --steps 1000 --frame-skip 4 --action-repeat 2 "
                "--render-mode human --deterministic-policy --inspection-overlay "
                f"--fps 30 --record --record-dir {layout.recordings_dir} "
                f"--record-output {files['play_manifest']} "
                f"--output-summary {files['play_summary']}"
            ),
            "compare": (
                "retroagi compare --game smb --stage full "
                f"--transfer-checkpoint {files['transferred_policy_checkpoint']} "
                f"--scratch-trained-checkpoint {layout.checkpoints_dir / 'scratch_policy.pth'} "
                f"--fine-tuned-checkpoint {files['policy_checkpoint']} "
                f"--known-good-checkpoint {files['policy_checkpoint']} "
                "--full-smb-vision-checkpoint data/vit/full_smb_vit.pth "
                f"--task-set fixed_benchmark --seed {seed} --seed {seed + 1} "
                f"--output {files['comparison_report']}"
            ),
        },
        "required_artifacts": {
            name: str(files[name]) for name in FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_ARTIFACTS
        },
        "qualification_gates": {
            "environment_check_passed": "env_check.json reports successful backend/content checks",
            "policy_loads": "retroagi evaluate loads the checkpoint without compatibility errors",
            "fixed_benchmark_thresholds_met": (
                "evaluation_report.success_thresholds_met is true and every "
                "fixed_task_results entry has threshold_met=true"
            ),
            "recording_written": "recording_summary.recording.enabled is true and manifests exist",
            "play_written": "play_summary records policy mode, render settings, overlay data, and output",
            "comparison_written": "comparison_report includes action_agreement for named policies",
        },
    }


def validate_full_smb_documented_benchmark_manifest(manifest: dict[str, Any]) -> None:
    """Validate the committed Full SMB benchmark-run documentation manifest."""

    if not isinstance(manifest, dict):
        raise ValueError("Full SMB documented benchmark manifest must be a JSON object")
    if manifest.get("schema_version") != FULL_SMB_DOCUMENTED_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Full SMB documented benchmark manifest schema_version mismatch")
    if manifest.get("stage") != "full_smb":
        raise ValueError("Full SMB documented benchmark manifest must target full_smb")
    if manifest.get("game") != "SuperMarioBros-Nes":
        raise ValueError("Full SMB documented benchmark manifest must target SuperMarioBros-Nes")
    if manifest.get("local_only_content") is not True:
        raise ValueError("Full SMB documented benchmark manifest must mark content local-only")
    if manifest.get("rom_bytes_committed") is not False:
        raise ValueError("Full SMB documented benchmark manifest must not commit ROM bytes")
    if manifest.get("checkpoint_bytes_committed") is not False:
        raise ValueError("Full SMB documented benchmark manifest must not commit checkpoint bytes")

    commands = manifest.get("required_commands")
    if not isinstance(commands, dict):
        raise ValueError("Full SMB documented benchmark manifest must define required_commands")
    missing_commands = [
        name for name in FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_COMMANDS if name not in commands
    ]
    if missing_commands:
        raise ValueError(
            "Full SMB documented benchmark manifest missing commands: "
            + ", ".join(missing_commands)
        )
    for name in FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_COMMANDS:
        command = commands[name]
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"Full SMB documented benchmark command {name!r} must be non-empty")

    artifacts = manifest.get("required_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Full SMB documented benchmark manifest must define required_artifacts")
    missing_artifacts = [
        name for name in FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_ARTIFACTS if name not in artifacts
    ]
    if missing_artifacts:
        raise ValueError(
            "Full SMB documented benchmark manifest missing artifacts: "
            + ", ".join(missing_artifacts)
        )
    for name in FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_ARTIFACTS:
        value = artifacts[name]
        if not isinstance(value, str) or not value.startswith("artifacts/full_smb/"):
            raise ValueError(
                f"Full SMB documented benchmark artifact {name!r} must be under artifacts/full_smb"
            )

    gates = manifest.get("qualification_gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError("Full SMB documented benchmark manifest must define qualification_gates")
