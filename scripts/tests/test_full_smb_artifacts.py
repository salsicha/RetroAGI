"""Tests for Full SMB run artifact layout helpers."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retroagi.stages.full_smb import (
    FULL_SMB_ARTIFACT_LAYOUT_SCHEMA_VERSION,
    FullSMBArtifactLayout,
    full_smb_artifact_layout,
)


class TestFullSMBArtifacts(unittest.TestCase):
    def test_layout_defines_preserved_run_paths(self):
        layout = FullSMBArtifactLayout("baseline_seed0")

        self.assertEqual(layout.run_dir, Path("artifacts/full_smb/baseline_seed0"))
        self.assertEqual(
            layout.directories(),
            {
                "run": Path("artifacts/full_smb/baseline_seed0"),
                "summaries": Path("artifacts/full_smb/baseline_seed0/summaries"),
                "logs": Path("artifacts/full_smb/baseline_seed0/logs"),
                "recordings": Path("artifacts/full_smb/baseline_seed0/recordings"),
                "videos": Path("artifacts/full_smb/baseline_seed0/videos"),
                "evaluations": Path("artifacts/full_smb/baseline_seed0/evaluations"),
                "comparisons": Path("artifacts/full_smb/baseline_seed0/comparisons"),
                "tracking": Path("artifacts/full_smb/baseline_seed0/tracking"),
                "checkpoints": Path("artifacts/full_smb/baseline_seed0/checkpoints"),
            },
        )
        files = layout.files()
        self.assertEqual(
            files["train_summary"],
            Path("artifacts/full_smb/baseline_seed0/summaries/train_summary.json"),
        )
        self.assertEqual(
            files["throughput_benchmark"],
            Path("artifacts/full_smb/baseline_seed0/summaries/throughput_benchmark.json"),
        )
        self.assertEqual(
            files["train_log"],
            Path("artifacts/full_smb/baseline_seed0/logs/train.jsonl"),
        )
        self.assertEqual(
            files["recording_manifest"],
            Path("artifacts/full_smb/baseline_seed0/recordings/recording_manifest.npz"),
        )
        self.assertEqual(
            files["evaluation_video"],
            Path("artifacts/full_smb/baseline_seed0/videos/evaluation.mp4"),
        )
        self.assertEqual(
            files["comparison_report"],
            Path("artifacts/full_smb/baseline_seed0/comparisons/policy_suite_comparison.json"),
        )
        self.assertEqual(
            files["policy_checkpoint"],
            Path("artifacts/full_smb/baseline_seed0/checkpoints/policy.pth"),
        )

    def test_layout_manifest_is_serializable(self):
        layout = full_smb_artifact_layout("smoke_run", root=Path("artifacts/custom_full_smb"))
        manifest = layout.to_manifest()

        self.assertEqual(manifest["schema_version"], FULL_SMB_ARTIFACT_LAYOUT_SCHEMA_VERSION)
        self.assertEqual(manifest["run_name"], "smoke_run")
        self.assertEqual(manifest["root"], "artifacts/custom_full_smb")
        self.assertEqual(
            manifest["directories"]["tracking"], "artifacts/custom_full_smb/smoke_run/tracking"
        )
        self.assertEqual(
            manifest["files"]["fixed_task_report"],
            "artifacts/custom_full_smb/smoke_run/evaluations/fixed_task_thresholds.json",
        )

    def test_layout_can_create_directories_under_custom_root(self):
        with TemporaryDirectory() as tmpdir:
            layout = FullSMBArtifactLayout("run_a", root=Path(tmpdir) / "full_smb")
            layout.ensure_directories()

            for directory in layout.directories().values():
                self.assertTrue(directory.is_dir())

    def test_layout_rejects_ambiguous_run_names(self):
        for run_name in ("", ".", "..", "nested/run", "/absolute"):
            with self.subTest(run_name=run_name):
                with self.assertRaisesRegex(ValueError, "run_name"):
                    FullSMBArtifactLayout(run_name)


if __name__ == "__main__":
    unittest.main()
