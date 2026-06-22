"""Tests for the Block SMB command line entry point."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from retroagi.stages.block_smb import cli


def fake_result():
    return {
        "history": [{"loss_total": 1.0}],
        "metrics": {"eval_mean_return": 2.0, "eval_success_rate": 0.5},
        "evaluation": {
            "mean_return": 2.0,
            "success_rate": 0.5,
            "fixed_scenarios": {"level_1_flat.json": {"return": 2.0}},
        },
        "curriculum": ["level_1_flat.json"],
        "model": object(),
    }


class TestBlockSMBCLI(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = cli.main(argv)
        return exit_code, json.loads(stream.getvalue())

    def test_train_command_builds_training_resume_and_record_config(self):
        with patch(
            "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
            return_value=fake_result(),
        ) as train:
            exit_code, payload = self.run_main(
                [
                    "train",
                    "--epochs",
                    "3",
                    "--episodes-per-epoch",
                    "4",
                    "--checkpoint",
                    "data/block_smb/policy.pth",
                    "--resume",
                    "data/block_smb/old_policy.pth",
                    "--record",
                    "--record-dir",
                    "artifacts/block_smb/videos",
                    "--fixed-scenario",
                    "level_1_flat.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.episodes_per_epoch, 4)
        self.assertEqual(config.checkpoint_path, Path("data/block_smb/policy.pth"))
        self.assertEqual(config.resume_path, Path("data/block_smb/old_policy.pth"))
        self.assertTrue(config.save_checkpoints)
        self.assertTrue(config.record_videos)
        self.assertEqual(config.video_dir, Path("artifacts/block_smb/videos"))
        self.assertEqual(config.fixed_scenarios, ("level_1_flat.json",))
        self.assertNotIn("model", payload)
        self.assertEqual(payload["config"]["epochs"], 3)

    def test_train_command_loads_frozen_vision_checkpoint_and_writes_summary(self):
        loaded_model = object()

        def fake_train(_config, *, vision_factory):
            self.assertIs(vision_factory(), loaded_model)
            return fake_result()

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run_summary.json"
            with patch(
                "retroagi.stages.block_smb.cli.load_block_vit_checkpoint",
                return_value=SimpleNamespace(
                    model=loaded_model,
                    path=Path("data/block_vit/block_vit.pth"),
                    frozen=True,
                ),
            ) as load_vision:
                with patch(
                    "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
                    side_effect=fake_train,
                ):
                    exit_code, payload = self.run_main(
                        [
                            "train",
                            "--device",
                            "cpu",
                            "--vision-checkpoint",
                            "data/block_vit/block_vit.pth",
                            "--output",
                            str(output),
                        ]
                    )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(load_vision.call_args.args[0], Path("data/block_vit/block_vit.pth"))
        self.assertEqual(str(load_vision.call_args.kwargs["device"]), "cpu")
        self.assertTrue(load_vision.call_args.kwargs["freeze"])
        self.assertEqual(payload["vision"]["checkpoint_path"], "data/block_vit/block_vit.pth")
        self.assertTrue(payload["vision"]["frozen"])
        self.assertEqual(written["vision"], payload["vision"])

    def test_evaluate_command_reuses_checkpoint_config_without_training(self):
        checkpoint = {
            "epoch": 7,
            "config": {
                "seed": 11,
                "epochs": 7,
                "hidden_dim": 16,
                "fixed_scenarios": ["level_2_gap.json"],
                "checkpoint_path": "data/block_smb/policy.pth",
                "save_checkpoints": True,
                "record_videos": True,
                "video_dir": "old/videos",
            },
        }
        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
                return_value=fake_result(),
            ) as train:
                exit_code, payload = self.run_main(
                    [
                        "evaluate",
                        "--checkpoint",
                        "data/block_smb/policy.pth",
                        "--evaluation-episodes",
                        "2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.epochs, 7)
        self.assertEqual(config.hidden_dim, 16)
        self.assertEqual(config.fixed_scenarios, ("level_2_gap.json",))
        self.assertEqual(config.resume_path, Path("data/block_smb/policy.pth"))
        self.assertEqual(config.evaluation_episodes, 2)
        self.assertFalse(config.save_checkpoints)
        self.assertFalse(config.record_videos)
        self.assertIsNone(config.video_dir)
        self.assertEqual(payload["config"]["resume_path"], "data/block_smb/policy.pth")

    def test_record_command_enables_recording_with_checkpoint_config(self):
        checkpoint = {
            "epoch": 2,
            "config": {
                "epochs": 2,
                "hidden_dim": 8,
                "fixed_scenarios": ["level_3_stairs.json"],
            },
        }
        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
                return_value=fake_result(),
            ) as train:
                exit_code, payload = self.run_main(
                    [
                        "record",
                        "--checkpoint",
                        "data/block_smb/policy.pth",
                        "--record-dir",
                        "artifacts/records",
                    ]
                )

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        self.assertTrue(config.record_videos)
        self.assertEqual(config.video_dir, Path("artifacts/records"))
        self.assertEqual(payload["config"]["video_dir"], "artifacts/records")


if __name__ == "__main__":
    unittest.main()
