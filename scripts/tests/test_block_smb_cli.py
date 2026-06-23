"""Tests for the Block SMB command line entry point."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from retroagi.stages.block_smb import cli


class FreshVision:
    def __init__(self):
        self.device = None
        self.frozen = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def requires_grad_(self, value):
        self.frozen = not value
        return self

    def eval(self):
        self.eval_called = True
        return self


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
                    "--controller-schedule",
                    "linear",
                    "--imagined-rollout-horizon",
                    "2",
                    "--imagined-rollout-weight",
                    "0.2",
                    "--target-network-mode",
                    "auto",
                    "--target-network-tau",
                    "0.25",
                    "--target-network-instability-threshold",
                    "0.4",
                    "--evaluation-interval-epochs",
                    "2",
                    "--log-path",
                    "artifacts/block_smb/train.jsonl",
                    "--checkpoint",
                    "data/block_smb/policy.pth",
                    "--resume",
                    "data/block_smb/old_policy.pth",
                    "--record",
                    "--record-dir",
                    "artifacts/block_smb/videos",
                    "--fixed-scenario",
                    "level_1_flat.json",
                    "--policy-loss-weight",
                    "0.8",
                    "--representation-weight",
                    "0.07",
                    "--reward-loss-weight",
                    "0.03",
                    "--value-loss-weight",
                    "0.4",
                    "--reward-progress-per-pixel",
                    "0.08",
                    "--reward-goal",
                    "75",
                    "--reward-frame-penalty",
                    "-0.02",
                    "--disable-vision",
                    "--disable-world-model",
                    "--disable-critic-feedback",
                    "--disable-hierarchy",
                    "--disable-recurrent-state",
                    "--disable-checkpoint-transfer",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.episodes_per_epoch, 4)
        self.assertEqual(config.controller_schedule, "linear")
        self.assertEqual(config.imagined_rollout_horizon, 2)
        self.assertEqual(config.imagined_rollout_weight, 0.2)
        self.assertEqual(config.target_network_mode, "auto")
        self.assertEqual(config.target_network_tau, 0.25)
        self.assertEqual(config.target_network_instability_threshold, 0.4)
        self.assertEqual(config.evaluation_interval_epochs, 2)
        self.assertEqual(config.log_path, Path("artifacts/block_smb/train.jsonl"))
        self.assertEqual(config.checkpoint_path, Path("data/block_smb/policy.pth"))
        self.assertEqual(config.resume_path, Path("data/block_smb/old_policy.pth"))
        self.assertTrue(config.save_checkpoints)
        self.assertTrue(config.record_videos)
        self.assertEqual(config.video_dir, Path("artifacts/block_smb/videos"))
        self.assertEqual(config.fixed_scenarios, ("level_1_flat.json",))
        self.assertEqual(config.policy_loss_weight, 0.8)
        self.assertEqual(config.representation_weight, 0.07)
        self.assertEqual(config.reward_loss_weight, 0.03)
        self.assertEqual(config.value_loss_weight, 0.4)
        self.assertEqual(config.reward_config.progress_per_pixel, 0.08)
        self.assertEqual(config.reward_config.goal, 75.0)
        self.assertEqual(config.reward_config.frame_penalty, -0.02)
        self.assertFalse(config.ablation.vision_enabled)
        self.assertFalse(config.ablation.world_model_enabled)
        self.assertFalse(config.ablation.critic_feedback_enabled)
        self.assertFalse(config.ablation.hierarchy_enabled)
        self.assertFalse(config.ablation.recurrent_state_enabled)
        self.assertFalse(config.ablation.checkpoint_transfer_enabled)
        self.assertNotIn("model", payload)
        self.assertEqual(payload["config"]["epochs"], 3)
        self.assertEqual(payload["config"]["controller_schedule"], "linear")
        self.assertEqual(payload["config"]["imagined_rollout_horizon"], 2)
        self.assertEqual(payload["config"]["imagined_rollout_weight"], 0.2)
        self.assertEqual(payload["config"]["target_network_mode"], "auto")
        self.assertEqual(payload["config"]["target_network_tau"], 0.25)
        self.assertEqual(
            payload["config"]["target_network_instability_threshold"], 0.4
        )
        self.assertEqual(payload["config"]["evaluation_interval_epochs"], 2)
        self.assertEqual(
            payload["config"]["log_path"], "artifacts/block_smb/train.jsonl"
        )
        self.assertEqual(payload["config"]["reward_config"]["goal"], 75.0)
        self.assertFalse(payload["config"]["ablation"]["vision_enabled"])
        self.assertFalse(payload["config"]["ablation"]["world_model_enabled"])
        self.assertFalse(payload["vision"]["checkpoint_transfer"])

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
        self.assertTrue(payload["vision"]["checkpoint_transfer"])
        self.assertEqual(written["vision"], payload["vision"])

    def test_train_command_can_disable_checkpoint_transfer(self):
        fresh_vision = FreshVision()

        def fake_train(_config, *, vision_factory):
            self.assertIs(vision_factory(), fresh_vision)
            self.assertIs(vision_factory(), fresh_vision)
            return fake_result()

        with patch(
            "retroagi.stages.block_smb.cli.BlockVisionTransformer",
            return_value=fresh_vision,
        ) as fresh_factory:
            with patch(
                "retroagi.stages.block_smb.cli.load_block_vit_checkpoint"
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
                            "--disable-checkpoint-transfer",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        fresh_factory.assert_called_once_with()
        load_vision.assert_not_called()
        self.assertEqual(str(fresh_vision.device), "cpu")
        self.assertTrue(fresh_vision.frozen)
        self.assertTrue(fresh_vision.eval_called)
        self.assertIsNone(payload["vision"]["checkpoint_path"])
        self.assertTrue(payload["vision"]["frozen"])
        self.assertFalse(payload["vision"]["checkpoint_transfer"])

    def test_evaluate_command_reuses_checkpoint_config_without_training(self):
        checkpoint = {
            "epoch": 7,
            "config": {
                "seed": 11,
                "epochs": 7,
                "hidden_dim": 16,
                "fixed_scenarios": ["level_2_gap.json"],
                "ablation": {
                    "vision_enabled": False,
                    "world_model_enabled": False,
                    "critic_feedback_enabled": False,
                    "hierarchy_enabled": False,
                    "recurrent_state_enabled": False,
                    "checkpoint_transfer_enabled": False,
                },
                "reward_config": {
                    "progress_per_pixel": 0.06,
                    "coin": 9.0,
                    "enemy_stomp": 4.0,
                    "goal": 60.0,
                    "fall_death": -12.0,
                    "enemy_hit": -12.0,
                    "frame_penalty": -0.02,
                },
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
                        "--reward-goal",
                        "80",
                        "--enable-vision",
                        "--enable-critic-feedback",
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
        self.assertEqual(config.reward_config.progress_per_pixel, 0.06)
        self.assertEqual(config.reward_config.goal, 80.0)
        self.assertEqual(config.reward_config.frame_penalty, -0.02)
        self.assertTrue(config.ablation.vision_enabled)
        self.assertTrue(config.ablation.critic_feedback_enabled)
        self.assertFalse(config.ablation.world_model_enabled)
        self.assertFalse(config.ablation.hierarchy_enabled)
        self.assertFalse(config.ablation.recurrent_state_enabled)
        self.assertFalse(config.ablation.checkpoint_transfer_enabled)
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

    def test_diagnose_vision_command_reports_perception_metrics(self):
        loaded_model = object()
        fake_metrics = {
            "samples": 2.0,
            "accuracy": 0.75,
            "foreground_accuracy": 0.5,
            "mean_iou": 0.25,
            "position_rmse": 0.1,
            "position_within_tolerance": 0.0,
            "bottleneck": True,
            "bottleneck_reasons": ["mean_iou"],
            "thresholds": {},
            "per_class_iou": {},
        }

        with patch(
            "retroagi.stages.block_smb.cli.load_block_vit_checkpoint",
            return_value=SimpleNamespace(
                model=loaded_model,
                path=Path("data/block_vit/block_vit.pth"),
                frozen=True,
            ),
        ) as load_vision:
            with patch(
                "retroagi.stages.block_smb.cli._collect_vision_diagnostic_frames",
                return_value=torch.zeros(2, 240, 256, 3, dtype=torch.uint8),
            ) as collect:
                with patch(
                    "retroagi.stages.block_smb.cli.evaluate_block_vit_perception",
                    return_value=fake_metrics,
                ) as evaluate:
                    exit_code, payload = self.run_main(
                        [
                            "diagnose-vision",
                            "--vision-checkpoint",
                            "data/block_vit/block_vit.pth",
                            "--device",
                            "cpu",
                            "--samples",
                            "2",
                            "--rollout-steps",
                            "4",
                            "--batch-size",
                            "2",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(load_vision.call_args.args[0], Path("data/block_vit/block_vit.pth"))
        self.assertEqual(str(load_vision.call_args.kwargs["device"]), "cpu")
        collect.assert_called_once_with(samples=2, seed=7, rollout_steps=4)
        evaluate.assert_called_once()
        self.assertIs(evaluate.call_args.args[0], loaded_model)
        self.assertEqual(evaluate.call_args.kwargs["batch_size"], 2)
        self.assertEqual(payload["vision"]["checkpoint_path"], "data/block_vit/block_vit.pth")
        self.assertTrue(payload["perception"]["bottleneck"])
        self.assertEqual(payload["perception"]["bottleneck_reasons"], ["mean_iou"])


if __name__ == "__main__":
    unittest.main()
