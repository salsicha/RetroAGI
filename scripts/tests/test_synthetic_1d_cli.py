"""Tests for the Synthetic 1D command line entry point."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retroagi.core import BASELINE_ARCHITECTURE_NAME
from retroagi.stages.synthetic_1d import cli


def fake_history():
    return {
        "loss_actor_pass1": [1.0],
        "loss_actor_pass2": [0.8],
        "loss_world_model": [0.6],
        "loss_primitive_labels": [0.5],
        "loss_primitive_outcome": [0.4],
        "loss_critic": [0.2],
        "loss_total": [2.4],
        "controller_mse": [0.4],
        "controller_mae": [0.3],
        "controller_rmse": [0.632],
        "error_B": [0.2],
        "accuracy_A": [75.0],
    }


class TestSynthetic1DCLI(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = cli.main(argv)
        return exit_code, json.loads(stream.getvalue())

    def test_train_command_builds_config_and_writes_summary(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.json"
            with patch(
                "retroagi.stages.synthetic_1d.cli.train_and_evaluate",
                return_value=fake_history(),
            ) as train:
                exit_code, payload = self.run_main(
                    [
                        "train",
                        "--seed",
                        "7",
                        "--epochs",
                        "2",
                        "--batch-size",
                        "4",
                        "--learning-rate",
                        "0.01",
                        "--critic-loss-weight",
                        "0.1",
                        "--primitive-loss-weight",
                        "0.2",
                        "--primitive-outcome-loss-weight",
                        "0.3",
                        "--primitive-outcome-horizon",
                        "6",
                        "--tau-start",
                        "4.0",
                        "--tau-end",
                        "0.2",
                        "--device",
                        "cpu",
                        "--checkpoint",
                        "data/synthetic_1d/policy.pth",
                        "--output",
                        str(output),
                        "--train-samples",
                        "8",
                        "--validation-samples",
                        "4",
                        "--test-samples",
                        "3",
                        "--train-seed",
                        "101",
                        "--validation-seed",
                        "202",
                        "--test-seed",
                        "303",
                        "--architecture",
                        "baseline",
                        "--architecture-config",
                        "hidden_dim=12",
                        "--architecture-config",
                        "controller_schedule=linear",
                    ]
                )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.critic_loss_weight, 0.1)
        self.assertEqual(config.primitive_loss_weight, 0.2)
        self.assertEqual(config.primitive_outcome_loss_weight, 0.3)
        self.assertEqual(config.primitive_outcome_horizon, 6)
        self.assertEqual(config.tau_start, 4.0)
        self.assertEqual(config.tau_end, 0.2)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.checkpoint_path, Path("data/synthetic_1d/policy.pth"))
        self.assertTrue(config.save_checkpoints)
        self.assertEqual(config.split_sizes.train, 8)
        self.assertEqual(config.split_sizes.validation, 4)
        self.assertEqual(config.split_sizes.test, 3)
        self.assertEqual(config.resolved_split_seeds.train, 101)
        self.assertEqual(config.resolved_split_seeds.validation, 202)
        self.assertEqual(config.resolved_split_seeds.test, 303)
        self.assertEqual(config.architecture_name, BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            config.architecture_config,
            {"hidden_dim": 12, "controller_schedule": "linear"},
        )
        self.assertEqual(payload["config"]["seed"], 7)
        self.assertEqual(payload["config"]["checkpoint_path"], "data/synthetic_1d/policy.pth")
        self.assertEqual(payload["metrics"]["controller_mse"], 0.4)
        self.assertEqual(payload["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(written, payload)

    def test_resume_command_merges_checkpoint_config_and_overrides(self):
        checkpoint = {
            "epoch": 1,
            "global_step": 2,
            "config": {
                "seed": 5,
                "epochs": 1,
                "architecture_name": BASELINE_ARCHITECTURE_NAME,
                "architecture_config": {
                    "hidden_dim": 16,
                    "controller_schedule": "constant",
                },
                "split_sizes": {"train": 8, "validation": 4, "test": 3},
                "split_seeds": {"train": 101, "validation": 202, "test": 303},
                "checkpoint_path": "data/synthetic_1d/old.pth",
                "save_checkpoints": True,
                "device": "cpu",
            },
        }
        with patch("retroagi.stages.synthetic_1d.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.synthetic_1d.cli.train_and_evaluate",
                return_value=fake_history(),
            ) as train:
                exit_code, payload = self.run_main(
                    [
                        "train",
                        "--resume",
                        "data/synthetic_1d/old.pth",
                        "--checkpoint",
                        "data/synthetic_1d/new.pth",
                        "--epochs",
                        "3",
                        "--hidden-dim",
                        "12",
                        "--architecture-config",
                        "controller_schedule=linear",
                    ]
                )

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        self.assertEqual(config.seed, 5)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.resume_path, Path("data/synthetic_1d/old.pth"))
        self.assertEqual(config.checkpoint_path, Path("data/synthetic_1d/new.pth"))
        self.assertTrue(config.save_checkpoints)
        self.assertEqual(config.split_sizes.train, 8)
        self.assertEqual(config.resolved_split_seeds.validation, 202)
        self.assertEqual(
            config.architecture_config,
            {"hidden_dim": 12, "controller_schedule": "linear"},
        )
        self.assertEqual(payload["config"]["resume_path"], "data/synthetic_1d/old.pth")
        self.assertEqual(payload["config"]["checkpoint_path"], "data/synthetic_1d/new.pth")

    def test_architecture_config_requires_key_value_syntax(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.run_main(["train", "--architecture-config", "hidden_dim"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
