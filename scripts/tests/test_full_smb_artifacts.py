"""Tests for Full SMB run artifact layout helpers."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retroagi.stages.full_smb import (
    DEFAULT_FULL_SMB_DOCUMENTED_BENCHMARK_RUN_NAME,
    FULL_SMB_ARTIFACT_LAYOUT_SCHEMA_VERSION,
    FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_ARTIFACTS,
    FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_COMMANDS,
    FullSMBArtifactLayout,
    full_smb_artifact_layout,
    full_smb_documented_benchmark_manifest,
    validate_full_smb_documented_benchmark_manifest,
)

DOCUMENTED_BENCHMARK_MANIFEST = Path(
    "artifacts/full_smb/documented_benchmark_seed0/benchmark_manifest.json"
)
DOCUMENTED_BENCHMARK_RUNBOOK = Path("artifacts/full_smb/documented_benchmark_seed0/RUN.md")


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
            files["environment_check"],
            Path("artifacts/full_smb/baseline_seed0/env_check.json"),
        )
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
        self.assertEqual(
            files["resumed_policy_checkpoint"],
            Path("artifacts/full_smb/baseline_seed0/checkpoints/resumed_policy.pth"),
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

    def test_documented_benchmark_manifest_declares_local_replay_contract(self):
        manifest = full_smb_documented_benchmark_manifest()

        validate_full_smb_documented_benchmark_manifest(manifest)
        self.assertEqual(manifest["run_name"], DEFAULT_FULL_SMB_DOCUMENTED_BENCHMARK_RUN_NAME)
        self.assertEqual(
            set(manifest["required_commands"]), set(FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_COMMANDS)
        )
        self.assertEqual(
            set(manifest["required_artifacts"]),
            set(FULL_SMB_DOCUMENTED_BENCHMARK_REQUIRED_ARTIFACTS),
        )
        self.assertFalse(manifest["rom_bytes_committed"])
        self.assertFalse(manifest["checkpoint_bytes_committed"])
        self.assertIn(
            "retroagi evaluate --game smb --stage full", manifest["required_commands"]["evaluate"]
        )
        self.assertIn(
            "retroagi record --game smb --stage full", manifest["required_commands"]["record"]
        )
        self.assertIn(
            "retroagi play --game smb --stage full", manifest["required_commands"]["play"]
        )
        self.assertEqual(
            manifest["required_artifacts"]["policy_checkpoint"],
            "artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth",
        )

    def test_committed_documented_benchmark_artifact_is_valid(self):
        manifest = json.loads(DOCUMENTED_BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
        canonical = full_smb_documented_benchmark_manifest()

        validate_full_smb_documented_benchmark_manifest(manifest)
        self.assertEqual(manifest, canonical)

        runbook = DOCUMENTED_BENCHMARK_RUNBOOK.read_text(encoding="utf-8")
        for term in (
            "Documented Full SMB Benchmark Run",
            "ROM bytes committed: `false`",
            "Checkpoint bytes committed: `false`",
            "retroagi evaluate --game smb --stage full",
            "retroagi record --game smb --stage full",
            "retroagi play --game smb --stage full",
            "success_thresholds_met: true",
            "action_agreement",
        ):
            self.assertIn(term, runbook)

    def test_documented_benchmark_manifest_rejects_missing_commands(self):
        manifest = full_smb_documented_benchmark_manifest()
        del manifest["required_commands"]["play"]

        with self.assertRaisesRegex(ValueError, "missing commands"):
            validate_full_smb_documented_benchmark_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
