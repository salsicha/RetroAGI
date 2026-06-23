"""Tests for the progressive-resolution promotion pipeline."""

import io
import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retroagi import promotion
from retroagi.core import BASELINE_ARCHITECTURE_NAME


def _fake_experiment_manifest(args: Namespace, *, passed: bool = True):
    stage = args.stage[0]
    metric_name = "controller_mse" if stage == "synthetic-1d" else "eval_success_rate"
    metric_value = 0.5 if stage == "synthetic-1d" else 0.0
    return {
        "architecture": {
            "name": args.architecture_name,
            "config": dict(args.architecture_config or ()),
        },
        "seed": args.seed,
        "device": args.device,
        "artifacts_dir": str(args.artifacts_dir),
        "stages": [
            {
                "stage": stage,
                "command": ["retroagi", "train", "--stage", stage],
                "summary_path": str(args.artifacts_dir / "run_summary.json"),
                "checkpoint_path": str(args.artifacts_dir / "checkpoint.pth"),
                "log_path": None,
                "exit_code": 0 if passed else 1,
                "config": {},
                "metrics": {metric_name: metric_value},
                "gates": [{"metric": metric_name, "passed": passed}],
                "passed": passed,
            }
        ],
        "gates": [],
        "passed": passed,
    }


class TestPromotionPipeline(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = promotion.main(argv)
        return exit_code, json.loads(stream.getvalue())

    def test_interface_smoke_runs_each_baseline_stage(self):
        with TemporaryDirectory() as tmpdir:
            exit_code, manifest = self.run_main(
                [
                    "--rung",
                    "interface-smoke",
                    "--output",
                    str(Path(tmpdir) / "promotion.json"),
                    "--artifacts-dir",
                    str(Path(tmpdir) / "artifacts"),
                    "--device",
                    "cpu",
                    "--seed",
                    "3",
                    "--architecture-config",
                    "hidden_dim=8",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(manifest["architecture"]["config"], {"hidden_dim": 8})
        self.assertEqual(manifest["rungs"][0]["name"], "interface-smoke")
        self.assertEqual(manifest["rungs"][0]["status"], "passed")
        self.assertEqual(
            [stage["stage"] for stage in manifest["rungs"][0]["stages"]],
            ["synthetic_1d", "block_smb", "full_smb"],
        )
        self.assertTrue(all(stage["passed"] for stage in manifest["rungs"][0]["stages"]))

    def test_default_pipeline_records_supported_and_future_rungs(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "promotion.json"

            def fake_run_experiment(args):
                return _fake_experiment_manifest(args)

            with patch("retroagi.experiments.run_experiment", side_effect=fake_run_experiment):
                exit_code, manifest = self.run_main(
                    [
                        "--output",
                        str(output),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                        "--architecture-config",
                        "hidden_dim=8",
                    ]
                )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, manifest)
        self.assertTrue(manifest["passed"])
        rung_statuses = {rung["name"]: rung["status"] for rung in manifest["rungs"]}
        self.assertEqual(rung_statuses["interface-smoke"], "passed")
        self.assertEqual(rung_statuses["synthetic-concept"], "passed")
        self.assertEqual(rung_statuses["block-smb-smoke"], "passed")
        self.assertEqual(rung_statuses["full-smb-fine-tuning"], "skipped")
        self.assertIn("reason", manifest["rungs"][-1])

    def test_failed_experiment_rung_fails_promotion(self):
        with TemporaryDirectory() as tmpdir:
            with patch(
                "retroagi.experiments.run_experiment",
                side_effect=lambda args: _fake_experiment_manifest(args, passed=False),
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "synthetic-concept",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertFalse(manifest["passed"])
        self.assertEqual(manifest["rungs"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
