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


def _experiment_manifest(*, hidden_dim, controller_mse, success_rate, game_name="smb"):
    architecture = {
        "name": BASELINE_ARCHITECTURE_NAME,
        "config": {"hidden_dim": hidden_dim},
    }
    return {
        "architecture": architecture,
        "game": {
            "name": game_name,
            "family": game_name,
            "backend": {"name": "stable-retro" if game_name == "smb" else "gymnasium-pong"},
            "stage_ladder": [
                {"name": "synthetic"},
                {"name": "block"},
                {"name": "full"},
            ],
        },
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
        "game": {
            "name": "smb",
            "family": "platformer",
            "backend": {"name": "stable-retro"},
            "stage_ladder": [
                {"name": "synthetic"},
                {"name": "block"},
                {"name": "full"},
            ],
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
        self.assertEqual(report["summary"]["game_count"], 1)
        self.assertEqual(report["summary"]["game_row_counts"], {"smb": 4})
        self.assertEqual(report["baseline"]["architecture"]["config"], {"hidden_dim": 8})
        self.assertEqual(report["runs"][0]["game"]["name"], "smb")
        variant_synthetic = next(
            row
            for row in report["rows"]
            if row["architecture"]["config"] == {"hidden_dim": 12}
            and row["stage"] == "synthetic-1d"
        )
        self.assertEqual(variant_synthetic["game"]["name"], "smb")
        self.assertEqual(variant_synthetic["game_key"], "smb")
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

    def test_report_scopes_regression_deltas_by_game(self):
        with TemporaryDirectory() as tmpdir:
            smb_baseline = Path(tmpdir) / "smb_baseline.json"
            smb_variant = Path(tmpdir) / "smb_variant.json"
            pong_baseline = Path(tmpdir) / "pong_baseline.json"
            pong_variant = Path(tmpdir) / "pong_variant.json"
            output_path = Path(tmpdir) / "report.json"
            _write_manifest(
                smb_baseline,
                _experiment_manifest(
                    hidden_dim=8,
                    controller_mse=1.0,
                    success_rate=0.25,
                    game_name="smb",
                ),
            )
            _write_manifest(
                smb_variant,
                _experiment_manifest(
                    hidden_dim=12,
                    controller_mse=0.9,
                    success_rate=0.35,
                    game_name="smb",
                ),
            )
            _write_manifest(
                pong_baseline,
                _experiment_manifest(
                    hidden_dim=8,
                    controller_mse=5.0,
                    success_rate=0.0,
                    game_name="pong",
                ),
            )
            _write_manifest(
                pong_variant,
                _experiment_manifest(
                    hidden_dim=12,
                    controller_mse=4.0,
                    success_rate=0.0,
                    game_name="pong",
                ),
            )

            exit_code, report = self.run_main(
                [
                    "--input",
                    str(smb_baseline),
                    "--input",
                    str(smb_variant),
                    "--input",
                    str(pong_baseline),
                    "--input",
                    str(pong_variant),
                    "--output",
                    str(output_path),
                    "--baseline-config",
                    "hidden_dim=8",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["summary"]["game_count"], 2)
        self.assertEqual(report["summary"]["game_row_counts"], {"pong": 4, "smb": 4})
        pong_synthetic = next(
            row
            for row in report["rows"]
            if row["game_key"] == "pong"
            and row["architecture"]["config"] == {"hidden_dim": 12}
            and row["stage"] == "synthetic-1d"
        )
        self.assertEqual(
            pong_synthetic["regression_deltas"]["controller_mse"]["baseline"],
            5.0,
        )
        self.assertEqual(
            pong_synthetic["regression_deltas"]["controller_mse"]["delta"],
            -1.0,
        )


if __name__ == "__main__":
    unittest.main()
