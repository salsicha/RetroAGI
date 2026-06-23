"""Tests for architecture sweep comparison reports."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from retroagi import reports
from retroagi.core import BASELINE_ARCHITECTURE_NAME


def _write_manifest(path: Path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _experiment_manifest(*, hidden_dim, controller_mse, success_rate):
    architecture = {
        "name": BASELINE_ARCHITECTURE_NAME,
        "config": {"hidden_dim": hidden_dim},
    }
    return {
        "architecture": architecture,
        "seed": 7,
        "device": "cpu",
        "artifacts_dir": "artifacts/experiment",
        "passed": True,
        "stages": [
            {
                "stage": "synthetic-1d",
                "summary_path": "artifacts/synthetic/summary.json",
                "checkpoint_path": "artifacts/synthetic/checkpoint.pth",
                "log_path": None,
                "command": ["retroagi", "train", "--stage", "synthetic-1d"],
                "metrics": {"controller_mse": controller_mse, "ignored": "not numeric"},
                "gates": [{"metric": "controller_mse", "passed": True}],
                "passed": True,
            },
            {
                "stage": "block-smb",
                "summary_path": "artifacts/block/summary.json",
                "checkpoint_path": "artifacts/block/checkpoint.pth",
                "log_path": "artifacts/block/events.jsonl",
                "command": ["retroagi", "train", "--stage", "block-smb"],
                "metrics": {"eval_success_rate": success_rate},
                "gates": [{"metric": "eval_success_rate", "passed": True}],
                "passed": True,
            },
        ],
    }


def _promotion_manifest():
    return {
        "architecture": {
            "name": BASELINE_ARCHITECTURE_NAME,
            "config": {"hidden_dim": 8},
        },
        "seed": 3,
        "device": "cpu",
        "artifacts_dir": "artifacts/promotion",
        "passed": True,
        "rungs": [
            {
                "name": "synthetic-concept",
                "status": "passed",
                "passed": True,
                "runtime_seconds": 1.5,
                "automatic_gates": [{"name": "runtime-seconds", "passed": True}],
                "experiment_manifest_path": "artifacts/promotion/synthetic/manifest.json",
                "experiment": {
                    "architecture": {
                        "name": BASELINE_ARCHITECTURE_NAME,
                        "config": {"hidden_dim": 8},
                    },
                    "seed": 3,
                    "device": "cpu",
                    "passed": True,
                    "stages": [
                        {
                            "stage": "synthetic-1d",
                            "summary_path": "artifacts/promotion/summary.json",
                            "checkpoint_path": "artifacts/promotion/checkpoint.pth",
                            "log_path": None,
                            "metrics": {"controller_mse": 0.25},
                            "gates": [{"metric": "controller_mse", "passed": True}],
                            "passed": True,
                        }
                    ],
                },
            }
        ],
    }


class TestArchitectureReports(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = reports.main(argv)
        return exit_code, json.loads(stream.getvalue())

    def test_experiment_report_flattens_rows_and_deltas(self):
        with TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            variant_path = Path(tmpdir) / "variant.json"
            output_path = Path(tmpdir) / "report.json"
            _write_manifest(
                baseline_path,
                _experiment_manifest(hidden_dim=8, controller_mse=0.5, success_rate=0.25),
            )
            _write_manifest(
                variant_path,
                _experiment_manifest(hidden_dim=12, controller_mse=0.4, success_rate=0.75),
            )

            exit_code, report = self.run_main(
                [
                    "--input",
                    str(baseline_path),
                    "--input",
                    str(variant_path),
                    "--output",
                    str(output_path),
                    "--baseline-config",
                    "hidden_dim=8",
                ]
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, report)
        self.assertEqual(report["summary"]["run_count"], 2)
        self.assertEqual(report["summary"]["row_count"], 4)
        self.assertEqual(report["baseline"]["architecture"]["config"], {"hidden_dim": 8})
        variant_synthetic = next(
            row
            for row in report["rows"]
            if row["architecture"]["config"] == {"hidden_dim": 12}
            and row["stage"] == "synthetic-1d"
        )
        self.assertEqual(
            variant_synthetic["artifacts"]["checkpoint_path"], "artifacts/synthetic/checkpoint.pth"
        )
        self.assertEqual(variant_synthetic["gates"][0]["metric"], "controller_mse")
        self.assertAlmostEqual(
            variant_synthetic["regression_deltas"]["controller_mse"]["delta"],
            -0.1,
        )

    def test_promotion_report_includes_rung_and_nested_stage_rows(self):
        with TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "promotion.json"
            output_path = Path(tmpdir) / "report.json"
            _write_manifest(manifest_path, _promotion_manifest())

            exit_code, report = self.run_main(
                ["--input", str(manifest_path), "--output", str(output_path)]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["runs"][0]["manifest_type"], "promotion")
        self.assertEqual([row["kind"] for row in report["rows"]], ["rung", "stage"])
        rung, stage = report["rows"]
        self.assertEqual(rung["runtime_seconds"], 1.5)
        self.assertEqual(rung["gates"][0]["name"], "runtime-seconds")
        self.assertEqual(stage["comparison_key"], "synthetic-concept:synthetic-1d")
        self.assertEqual(stage["automatic_gates"][0]["name"], "runtime-seconds")
        self.assertEqual(stage["metrics"]["controller_mse"], 0.25)


if __name__ == "__main__":
    unittest.main()
