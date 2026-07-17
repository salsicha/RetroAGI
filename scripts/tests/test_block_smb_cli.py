"""Tests for the Block SMB command line entry point."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from retroagi.core import BASELINE_ARCHITECTURE_NAME
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
        "curriculum_summary": {"fixed_scenario_count": 1, "monte_carlo_sample_count": 0},
        "architecture": {
            "name": BASELINE_ARCHITECTURE_NAME,
            "config": {"hidden_dim": 32, "controller_schedule": "constant"},
            "spec": {"name": BASELINE_ARCHITECTURE_NAME},
        },
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
                    "--architecture",
                    "baseline",
                    "--architecture-config",
                    "hidden_dim=12",
                    "--architecture-config",
                    "controller_schedule=linear",
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
                    "--tracking-backend",
                    "wandb",
                    "--tracking-log-dir",
                    "artifacts/block_smb/wandb",
                    "--tracking-project",
                    "retroagi-test",
                    "--tracking-run-name",
                    "smoke",
                    "--tracking-mode",
                    "offline",
                    "--checkpoint",
                    "data/block_smb/policy.pth",
                    "--resume",
                    "data/block_smb/old_policy.pth",
                    "--record",
                    "--record-dir",
                    "artifacts/block_smb/videos",
                    "--fixed-scenario",
                    "level_1_flat.json",
                    "--generated-scenarios",
                    "2",
                    "--monte-carlo-distribution",
                    "block_smb_mc_v1",
                    "--monte-carlo-train-samples-per-epoch",
                    "12",
                    "--monte-carlo-seed",
                    "60001",
                    "--monte-carlo-family-weight",
                    "flat_run=2",
                    "--monte-carlo-family-weight",
                    "single_gap=1",
                    "--monte-carlo-max-rejections",
                    "5",
                    "--monte-carlo-failure-replay-samples-per-epoch",
                    "3",
                    "--skip-monte-carlo-reachability-validation",
                    "--policy-loss-weight",
                    "0.8",
                    "--action-aux-weight",
                    "0.12",
                    "--noop-loss-weight",
                    "0.35",
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
                    "--reward-gap-jump",
                    "-7",
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
        self.assertEqual(config.architecture_name, BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            config.architecture_config,
            {"hidden_dim": 12, "controller_schedule": "linear"},
        )
        self.assertEqual(config.hidden_dim, 12)
        self.assertEqual(config.imagined_rollout_horizon, 2)
        self.assertEqual(config.imagined_rollout_weight, 0.2)
        self.assertEqual(config.target_network_mode, "auto")
        self.assertEqual(config.target_network_tau, 0.25)
        self.assertEqual(config.target_network_instability_threshold, 0.4)
        self.assertEqual(config.evaluation_interval_epochs, 2)
        self.assertEqual(config.log_path, Path("artifacts/block_smb/train.jsonl"))
        self.assertEqual(config.tracking_backend, "wandb")
        self.assertEqual(config.tracking_log_dir, Path("artifacts/block_smb/wandb"))
        self.assertEqual(config.tracking_project, "retroagi-test")
        self.assertEqual(config.tracking_run_name, "smoke")
        self.assertEqual(config.tracking_mode, "offline")
        self.assertEqual(config.checkpoint_path, Path("data/block_smb/policy.pth"))
        self.assertEqual(config.resume_path, Path("data/block_smb/old_policy.pth"))
        self.assertTrue(config.save_checkpoints)
        self.assertTrue(config.record_videos)
        self.assertEqual(config.video_dir, Path("artifacts/block_smb/videos"))
        self.assertEqual(config.fixed_scenarios, ("level_1_flat.json",))
        self.assertEqual(config.generated_scenarios, 2)
        self.assertEqual(config.monte_carlo_distribution_id, "block_smb_mc_v1")
        self.assertEqual(config.monte_carlo_train_samples_per_epoch, 12)
        self.assertEqual(config.monte_carlo_seed, 60001)
        self.assertEqual(config.monte_carlo_family_weights, {"flat_run": 2.0, "single_gap": 1.0})
        self.assertEqual(config.monte_carlo_max_rejections, 5)
        self.assertEqual(config.monte_carlo_failure_replay_samples_per_epoch, 3)
        self.assertFalse(config.monte_carlo_validate_reachability)
        self.assertEqual(config.policy_loss_weight, 0.8)
        self.assertEqual(config.action_aux_weight, 0.12)
        self.assertEqual(config.noop_loss_weight, 0.35)
        self.assertEqual(config.representation_weight, 0.07)
        self.assertEqual(config.reward_loss_weight, 0.03)
        self.assertEqual(config.value_loss_weight, 0.4)
        self.assertEqual(config.reward_config.progress_per_pixel, 0.08)
        self.assertEqual(config.reward_config.goal, 75.0)
        self.assertEqual(config.reward_config.gap_jump, -7.0)
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
        self.assertEqual(payload["config"]["architecture_name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            payload["config"]["architecture_config"],
            {"hidden_dim": 12, "controller_schedule": "linear"},
        )
        self.assertEqual(payload["config"]["hidden_dim"], 12)
        self.assertEqual(payload["config"]["imagined_rollout_horizon"], 2)
        self.assertEqual(payload["config"]["imagined_rollout_weight"], 0.2)
        self.assertEqual(payload["config"]["target_network_mode"], "auto")
        self.assertEqual(payload["config"]["target_network_tau"], 0.25)
        self.assertEqual(payload["config"]["target_network_instability_threshold"], 0.4)
        self.assertEqual(payload["config"]["evaluation_interval_epochs"], 2)
        self.assertEqual(payload["config"]["log_path"], "artifacts/block_smb/train.jsonl")
        self.assertEqual(payload["config"]["tracking_backend"], "wandb")
        self.assertEqual(payload["config"]["tracking_log_dir"], "artifacts/block_smb/wandb")
        self.assertEqual(payload["config"]["tracking_project"], "retroagi-test")
        self.assertEqual(payload["config"]["tracking_run_name"], "smoke")
        self.assertEqual(payload["config"]["tracking_mode"], "offline")
        self.assertEqual(payload["config"]["noop_loss_weight"], 0.35)
        self.assertEqual(payload["config"]["monte_carlo_train_samples_per_epoch"], 12)
        self.assertEqual(payload["config"]["monte_carlo_seed"], 60001)
        self.assertEqual(
            payload["config"]["monte_carlo_family_weights"],
            {"flat_run": 2.0, "single_gap": 1.0},
        )
        self.assertEqual(payload["config"]["monte_carlo_failure_replay_samples_per_epoch"], 3)
        self.assertFalse(payload["config"]["monte_carlo_validate_reachability"])
        self.assertEqual(payload["config"]["reward_config"]["goal"], 75.0)
        self.assertEqual(payload["config"]["reward_config"]["gap_jump"], -7.0)
        self.assertFalse(payload["config"]["ablation"]["vision_enabled"])
        self.assertFalse(payload["config"]["ablation"]["world_model_enabled"])
        self.assertFalse(payload["vision"]["checkpoint_transfer"])
        self.assertEqual(payload["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            payload["architecture"]["config"],
            {"hidden_dim": 32, "controller_schedule": "constant"},
        )

    def test_train_command_loads_frozen_vision_checkpoint_and_writes_summary(self):
        loaded_model = object()

        def fake_train(config, *, vision_factory):
            self.assertEqual(config.vision_checkpoint_path, Path("data/block_vit/block_vit.pth"))
            self.assertEqual(
                config.monte_carlo_train_samples_per_epoch,
                cli.DEFAULT_BLOCK_SMB_MC_TRAIN_SAMPLES,
            )
            self.assertEqual(
                config.monte_carlo_validation_samples,
                cli.DEFAULT_BLOCK_SMB_MC_VALIDATION_SAMPLES,
            )
            self.assertEqual(
                config.monte_carlo_test_samples,
                cli.DEFAULT_BLOCK_SMB_MC_TEST_SAMPLES,
            )
            self.assertEqual(
                config.monte_carlo_family_weights,
                cli.default_block_smb_failure_focus_monte_carlo_family_weights(),
            )
            self.assertEqual(
                config.monte_carlo_failure_replay_samples_per_epoch,
                cli.DEFAULT_BLOCK_SMB_MC_FAILURE_REPLAY_SAMPLES,
            )
            self.assertEqual(
                config.monte_carlo_pass_rate_gate,
                cli.DEFAULT_BLOCK_SMB_MC_PASS_RATE_GATE,
            )
            self.assertEqual(
                config.monte_carlo_family_pass_rate_gate,
                cli.DEFAULT_BLOCK_SMB_MC_FAMILY_PASS_RATE_GATE,
            )
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
        self.assertEqual(payload["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(written["vision"], payload["vision"])
        self.assertEqual(written["architecture"], payload["architecture"])
        self.assertEqual(written["curriculum_summary"], payload["curriculum_summary"])

    def test_train_parameter_sweep_keeps_coverage_smoke_defaults(self):
        def fake_train(config, *, vision_factory):  # noqa: ARG001
            self.assertTrue(config.monte_carlo_parameter_sweep)
            self.assertEqual(config.monte_carlo_train_samples_per_epoch, 0)
            self.assertEqual(config.monte_carlo_validation_samples, 0)
            self.assertEqual(config.monte_carlo_test_samples, 0)
            self.assertEqual(config.monte_carlo_family_weights, {})
            self.assertEqual(config.monte_carlo_failure_replay_samples_per_epoch, 0)
            self.assertEqual(
                config.monte_carlo_pass_rate_gate,
                cli.DEFAULT_BLOCK_SMB_MC_PASS_RATE_GATE,
            )
            self.assertEqual(
                config.monte_carlo_family_pass_rate_gate,
                cli.DEFAULT_BLOCK_SMB_MC_FAMILY_PASS_RATE_GATE,
            )
            return fake_result()

        with patch(
            "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
            side_effect=fake_train,
        ):
            exit_code, payload = self.run_main(
                [
                    "train",
                    "--monte-carlo-parameter-sweep",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["config"]["monte_carlo_train_samples_per_epoch"], 0)
        self.assertEqual(payload["config"]["monte_carlo_validation_samples"], 0)
        self.assertEqual(payload["config"]["monte_carlo_test_samples"], 0)
        self.assertEqual(payload["config"]["monte_carlo_family_weights"], {})
        self.assertEqual(payload["config"]["monte_carlo_failure_replay_samples_per_epoch"], 0)

    def test_evaluate_command_reuses_checkpoint_vision_path(self):
        loaded_model = object()
        checkpoint = {
            "epoch": 1,
            "config": {
                "seed": 5,
                "epochs": 1,
                "hidden_dim": 8,
                "architecture_config": {
                    "hidden_dim": 8,
                    "controller_schedule": "constant",
                },
                "vision_checkpoint_path": "data/pipeline/block_vit.pth",
            },
        }

        def fake_train(config, *, vision_factory):
            self.assertEqual(config.vision_checkpoint_path, Path("data/pipeline/block_vit.pth"))
            self.assertIs(vision_factory(), loaded_model)
            return fake_result()

        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.block_smb.cli.load_block_vit_checkpoint",
                return_value=SimpleNamespace(
                    model=loaded_model,
                    path=Path("data/pipeline/block_vit.pth"),
                    frozen=True,
                ),
            ) as load_vision:
                with patch(
                    "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
                    side_effect=fake_train,
                ):
                    exit_code, payload = self.run_main(
                        [
                            "evaluate",
                            "--checkpoint",
                            "data/block_smb/policy.pth",
                            "--device",
                            "cpu",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(load_vision.call_args.args[0], Path("data/pipeline/block_vit.pth"))
        self.assertEqual(payload["vision"]["checkpoint_path"], "data/pipeline/block_vit.pth")

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
            with patch("retroagi.stages.block_smb.cli.load_block_vit_checkpoint") as load_vision:
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
                "architecture_name": BASELINE_ARCHITECTURE_NAME,
                "architecture_config": {
                    "hidden_dim": 16,
                    "controller_schedule": "constant",
                },
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
                        "--hidden-dim",
                        "18",
                        "--architecture-config",
                        "controller_schedule=linear",
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
        self.assertEqual(config.hidden_dim, 18)
        self.assertEqual(config.controller_schedule, "linear")
        self.assertEqual(
            config.architecture_config,
            {"hidden_dim": 18, "controller_schedule": "linear"},
        )
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
        self.assertEqual(
            payload["config"]["architecture_config"],
            {"hidden_dim": 18, "controller_schedule": "linear"},
        )

    def test_evaluate_command_does_not_resume_training_from_unfinished_checkpoint(self):
        checkpoint = {
            "epoch": 4,
            "config": {
                "seed": 11,
                "epochs": 10,
                "architecture_name": BASELINE_ARCHITECTURE_NAME,
                "architecture_config": {
                    "hidden_dim": 16,
                    "controller_schedule": "constant",
                },
                "hidden_dim": 16,
                "fixed_scenarios": ["level_2_gap.json"],
            },
        }
        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.block_smb.cli.train_and_evaluate_block_smb",
                return_value=fake_result(),
            ) as train:
                exit_code, _payload = self.run_main(
                    [
                        "evaluate",
                        "--checkpoint",
                        "data/block_smb/policy.pth",
                    ]
                )

        self.assertEqual(exit_code, 0)
        config = train.call_args.args[0]
        # Pinning epochs to the checkpoint's completed epoch count keeps the
        # trainer's range(start_epoch, epochs) empty: evaluation must report
        # the checkpoint's weights, not silently train 6 more epochs.
        self.assertEqual(config.epochs, 4)

    def test_evaluate_command_routes_multi_seed_evaluation(self):
        checkpoint = {
            "epoch": 3,
            "config": {
                "seed": 9,
                "epochs": 3,
                "architecture_name": BASELINE_ARCHITECTURE_NAME,
                "architecture_config": {
                    "hidden_dim": 16,
                    "controller_schedule": "constant",
                },
                "hidden_dim": 16,
                "fixed_scenarios": ["level_1_flat.json"],
            },
        }
        multi_seed_result = {
            "seeds": [9, 1009],
            "seed_count": 2,
            "per_seed": [],
            "aggregate": {"success_rate": {"mean": 0.5}},
        }
        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch("retroagi.stages.block_smb.cli.restore_block_smb_checkpoint"):
                with patch(
                    "retroagi.stages.block_smb.cli.evaluate_block_smb_multi_seed",
                    return_value=multi_seed_result,
                ) as evaluate:
                    exit_code, payload = self.run_main(
                        [
                            "evaluate",
                            "--checkpoint",
                            "data/block_smb/policy.pth",
                            "--evaluation-seeds",
                            "2",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(evaluate.call_args.kwargs["seed_count"], 2)
        self.assertEqual(payload["multi_seed_evaluation"], multi_seed_result)

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

    def test_evaluate_monte_carlo_command_restores_checkpoint_and_reports_gates(self):
        class FakeModel:
            def to(self, _device):
                return self

        checkpoint = {
            "epoch": 3,
            "global_step": 9,
            "config": {
                "seed": 5,
                "epochs": 3,
                "hidden_dim": 8,
                "architecture_config": {
                    "hidden_dim": 8,
                    "controller_schedule": "constant",
                },
                "monte_carlo_validation_samples": 4,
                "monte_carlo_test_samples": 6,
            },
        }
        fake_evaluation = {
            "split": "test",
            "sample_count": 5,
            "success_rate": 0.0,
            "gates": {"gate_met": False},
        }
        stderr = io.StringIO()
        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.block_smb.cli.make_block_smb_model",
                return_value=FakeModel(),
            ) as make_model:
                with patch(
                    "retroagi.stages.block_smb.cli.restore_block_smb_checkpoint",
                    return_value=checkpoint,
                ) as restore:
                    with patch(
                        "retroagi.stages.block_smb.cli.evaluate_block_smb_monte_carlo",
                        return_value=fake_evaluation,
                    ) as evaluate:
                        with redirect_stderr(stderr):
                            exit_code, payload = self.run_main(
                                [
                                    "evaluate-monte-carlo",
                                    "--checkpoint",
                                    "data/block_smb/policy.pth",
                                    "--split",
                                    "test",
                                    "--samples",
                                    "5",
                                    "--device",
                                    "cpu",
                                    "--monte-carlo-parameter-sweep",
                                    "--monte-carlo-sweep-repeats-per-difficulty",
                                    "2",
                                    "--monte-carlo-pass-rate-gate",
                                    "0.8",
                                    "--monte-carlo-family-pass-rate-gate",
                                    "0.7",
                                    "--record-dir",
                                    "artifacts/block_smb/mc_eval",
                                ]
                            )

        self.assertEqual(exit_code, 0)
        make_model.assert_called_once()
        restore.assert_called_once()
        config = evaluate.call_args.args[1]
        # An explicit --samples overrides the parameter sweep so the requested
        # sample count is honored.
        self.assertFalse(config.monte_carlo_parameter_sweep)
        self.assertIn("overrides monte_carlo_parameter_sweep", stderr.getvalue())
        self.assertEqual(config.monte_carlo_sweep_repeats_per_difficulty, 2)
        self.assertEqual(config.monte_carlo_pass_rate_gate, 0.8)
        self.assertEqual(config.monte_carlo_family_pass_rate_gate, 0.7)
        self.assertEqual(evaluate.call_args.kwargs["split"], "test")
        self.assertEqual(evaluate.call_args.kwargs["sample_count"], 5)
        self.assertEqual(
            evaluate.call_args.kwargs["record_dir"],
            Path("artifacts/block_smb/mc_eval"),
        )
        self.assertEqual(payload["evaluation"], fake_evaluation)
        self.assertEqual(payload["checkpoint"]["path"], "data/block_smb/policy.pth")

    def test_evaluate_monte_carlo_command_keeps_sweep_without_explicit_samples(self):
        class FakeModel:
            def to(self, _device):
                return self

        checkpoint = {
            "epoch": 3,
            "global_step": 9,
            "config": {
                "seed": 5,
                "epochs": 3,
                "hidden_dim": 8,
                "architecture_config": {
                    "hidden_dim": 8,
                    "controller_schedule": "constant",
                },
            },
        }
        fake_evaluation = {
            "split": "test",
            "sample_count": 72,
            "success_rate": 0.0,
            "gates": {"gate_met": False},
        }
        with patch("retroagi.stages.block_smb.cli.load_checkpoint", return_value=checkpoint):
            with patch(
                "retroagi.stages.block_smb.cli.make_block_smb_model",
                return_value=FakeModel(),
            ):
                with patch(
                    "retroagi.stages.block_smb.cli.restore_block_smb_checkpoint",
                    return_value=checkpoint,
                ):
                    with patch(
                        "retroagi.stages.block_smb.cli.evaluate_block_smb_monte_carlo",
                        return_value=fake_evaluation,
                    ) as evaluate:
                        exit_code, _payload = self.run_main(
                            [
                                "evaluate-monte-carlo",
                                "--checkpoint",
                                "data/block_smb/policy.pth",
                                "--split",
                                "test",
                                "--device",
                                "cpu",
                                "--monte-carlo-parameter-sweep",
                                "--monte-carlo-sweep-repeats-per-difficulty",
                                "2",
                            ]
                        )

        self.assertEqual(exit_code, 0)
        config = evaluate.call_args.args[1]
        self.assertTrue(config.monte_carlo_parameter_sweep)
        self.assertEqual(
            evaluate.call_args.kwargs["sample_count"],
            cli.block_smb_monte_carlo_sweep_sample_count(config),
        )

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

    def test_architecture_config_requires_key_value_syntax(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.run_main(["train", "--architecture-config", "hidden_dim"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
