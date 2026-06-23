"""Tests for the stage-agnostic experiment runner."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retroagi import experiments
from retroagi.core import BASELINE_ARCHITECTURE_NAME


def _write_summary(argv, *, metrics):
    output_path = Path(argv[argv.index("--output") + 1])
    checkpoint_path = Path(argv[argv.index("--checkpoint") + 1])
    summary = {
        "config": {
            "seed": int(argv[argv.index("--seed") + 1]),
            "device": argv[argv.index("--device") + 1],
            "checkpoint_path": str(checkpoint_path),
            "architecture_name": BASELINE_ARCHITECTURE_NAME,
            "architecture_config": {"hidden_dim": 8},
        },
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary), encoding="utf-8")
    return 0


class TestExperimentRunner(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = experiments.main(argv)
        return exit_code, json.loads(stream.getvalue())

    def test_metric_gate_parses_stage_specific_threshold(self):
        gate = experiments._metric_gate("synthetic:controller_mse<=1.5")

        self.assertEqual(gate.stage, "synthetic-1d")
        self.assertEqual(gate.metric, "controller_mse")
        self.assertEqual(gate.operator, "<=")
        self.assertEqual(gate.threshold, 1.5)
        self.assertTrue(gate.evaluate({"controller_mse": 1.0})["passed"])
        self.assertFalse(gate.evaluate({"controller_mse": 2.0})["passed"])

    def test_experiment_manifest_records_stage_outputs_and_gates(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.json"
            artifacts_dir = Path(tmpdir) / "artifacts"

            def synthetic_main(argv):
                return _write_summary(argv, metrics={"controller_mse": 0.4})

            def block_main(argv):
                return _write_summary(argv, metrics={"eval_success_rate": 0.5})

            with patch("retroagi.stages.synthetic_1d.cli.main", side_effect=synthetic_main):
                with patch("retroagi.stages.block_smb.cli.main", side_effect=block_main):
                    exit_code, manifest = self.run_main(
                        [
                            "--stage",
                            "synthetic",
                            "--stage",
                            "block-smb",
                            "--output",
                            str(output),
                            "--artifacts-dir",
                            str(artifacts_dir),
                            "--seed",
                            "7",
                            "--device",
                            "cpu",
                            "--architecture",
                            "baseline",
                            "--architecture-config",
                            "hidden_dim=8",
                            "--gate",
                            "synthetic-1d:controller_mse<=1.0",
                            "--gate",
                            "block-smb:eval_success_rate>=0.5",
                        ]
                    )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, manifest)
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(manifest["architecture"]["config"], {"hidden_dim": 8})
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(
            [stage["stage"] for stage in manifest["stages"]], ["synthetic-1d", "block-smb"]
        )
        synthetic, block = manifest["stages"]
        self.assertEqual(synthetic["metrics"]["controller_mse"], 0.4)
        self.assertEqual(block["metrics"]["eval_success_rate"], 0.5)
        self.assertEqual(
            synthetic["checkpoint_path"],
            str(Path(tmpdir) / "artifacts/synthetic_1d/checkpoint.pth"),
        )
        self.assertEqual(block["log_path"], str(Path(tmpdir) / "artifacts/block_smb/events.jsonl"))
        self.assertIn("retroagi", synthetic["command"])
        self.assertIn("--architecture-config", block["command"])
        self.assertTrue(
            all(gate["passed"] for stage in manifest["stages"] for gate in stage["gates"])
        )

    def test_experiment_runner_applies_architecture_ablations(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.json"
            artifacts_dir = Path(tmpdir) / "artifacts"
            calls = {}

            def synthetic_main(argv):
                calls["synthetic"] = list(argv)
                return _write_summary(argv, metrics={"controller_mse": 0.4})

            def block_main(argv):
                calls["block"] = list(argv)
                return _write_summary(argv, metrics={"eval_success_rate": 0.5})

            with patch("retroagi.stages.synthetic_1d.cli.main", side_effect=synthetic_main):
                with patch("retroagi.stages.block_smb.cli.main", side_effect=block_main):
                    exit_code, manifest = self.run_main(
                        [
                            "--stage",
                            "synthetic",
                            "--stage",
                            "block-smb",
                            "--output",
                            str(output),
                            "--artifacts-dir",
                            str(artifacts_dir),
                            "--architecture-config",
                            "hidden_dim=8",
                            "--ablation",
                            "controller_schedule=linear",
                            "--ablation",
                            "world_model=off",
                            "--ablation",
                            "critic_feedback=off",
                            "--ablation",
                            "auxiliary_objectives=off",
                            "--ablation",
                            "target_network=off",
                            "--ablation",
                            "checkpoint_transfer=on",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            manifest["architecture"]["config"],
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )
        self.assertFalse(manifest["architecture_variant"]["ablation"]["world_model_enabled"])
        self.assertIn("--critic-loss-weight", calls["synthetic"])
        self.assertIn("--disable-world-model", calls["block"])
        self.assertIn("--disable-critic-feedback", calls["block"])
        self.assertIn("--target-network-mode", calls["block"])
        self.assertIn("--enable-checkpoint-transfer", calls["block"])
        self.assertNotIn("--disable-checkpoint-transfer", calls["block"])

    def test_failed_gate_returns_nonzero_exit(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "manifest.json"
            with patch(
                "retroagi.stages.synthetic_1d.cli.main",
                side_effect=lambda argv: _write_summary(argv, metrics={"controller_mse": 2.0}),
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--stage",
                        "synthetic-1d",
                        "--output",
                        str(output),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--gate",
                        "controller_mse<=1.0",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertFalse(manifest["passed"])
        self.assertFalse(manifest["stages"][0]["gates"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
