"""Tests for the project-level RetroAGI command line entry point."""

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from retroagi import cli


class TestRetroAGICLI(unittest.TestCase):
    def test_experiment_command_forwards_runner_arguments(self):
        with patch("retroagi.experiments.main", return_value=0) as experiment_main:
            exit_code = cli.main(
                [
                    "experiment",
                    "--stage",
                    "synthetic",
                    "--output",
                    "artifacts/experiments/latest/manifest.json",
                    "--architecture",
                    "baseline",
                ]
            )

        self.assertEqual(exit_code, 0)
        experiment_main.assert_called_once_with(
            [
                "--stage",
                "synthetic",
                "--output",
                "artifacts/experiments/latest/manifest.json",
                "--architecture",
                "baseline",
            ]
        )

    def test_promote_command_forwards_pipeline_arguments(self):
        with patch("retroagi.promotion.main", return_value=0) as promotion_main:
            exit_code = cli.main(
                [
                    "promote",
                    "--rung",
                    "interface-smoke",
                    "--output",
                    "artifacts/promotions/latest/manifest.json",
                    "--architecture",
                    "baseline",
                ]
            )

        self.assertEqual(exit_code, 0)
        promotion_main.assert_called_once_with(
            [
                "--rung",
                "interface-smoke",
                "--output",
                "artifacts/promotions/latest/manifest.json",
                "--architecture",
                "baseline",
            ]
        )

    def test_report_command_forwards_report_arguments(self):
        with patch("retroagi.reports.main", return_value=0) as reports_main:
            exit_code = cli.main(
                [
                    "report",
                    "--input",
                    "artifacts/promotions/baseline.json",
                    "--output",
                    "artifacts/promotions/report.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        reports_main.assert_called_once_with(
            [
                "--input",
                "artifacts/promotions/baseline.json",
                "--output",
                "artifacts/promotions/report.json",
            ]
        )

    def test_synthetic_1d_train_forwards_stage_arguments(self):
        with patch("retroagi.stages.synthetic_1d.cli.main", return_value=0) as synthetic_main:
            exit_code = cli.main(
                [
                    "train",
                    "--stage",
                    "synthetic-1d",
                    "--epochs",
                    "3",
                    "--architecture",
                    "baseline",
                    "--architecture-config",
                    "hidden_dim=12",
                    "--checkpoint",
                    "data/synthetic_1d/policy.pth",
                ]
            )

        self.assertEqual(exit_code, 0)
        synthetic_main.assert_called_once_with(
            [
                "train",
                "--epochs",
                "3",
                "--architecture",
                "baseline",
                "--architecture-config",
                "hidden_dim=12",
                "--checkpoint",
                "data/synthetic_1d/policy.pth",
            ]
        )

    def test_synthetic_1d_resume_rewrites_to_train_resume(self):
        with patch("retroagi.stages.synthetic_1d.cli.main", return_value=0) as synthetic_main:
            exit_code = cli.main(
                [
                    "resume",
                    "--stage",
                    "synthetic",
                    "--checkpoint",
                    "data/synthetic_1d/old.pth",
                    "--save-checkpoint",
                    "data/synthetic_1d/new.pth",
                    "--epochs",
                    "10",
                ]
            )

        self.assertEqual(exit_code, 0)
        synthetic_main.assert_called_once_with(
            [
                "train",
                "--resume",
                "data/synthetic_1d/old.pth",
                "--checkpoint",
                "data/synthetic_1d/new.pth",
                "--epochs",
                "10",
            ]
        )

    def test_block_smb_train_forwards_stage_arguments(self):
        with patch("retroagi.stages.block_smb.cli.main", return_value=0) as block_main:
            exit_code = cli.main(
                [
                    "train",
                    "--stage",
                    "block-smb",
                    "--epochs",
                    "3",
                    "--architecture",
                    "baseline",
                    "--architecture-config",
                    "hidden_dim=12",
                    "--checkpoint",
                    "data/block_smb/policy.pth",
                ]
            )

        self.assertEqual(exit_code, 0)
        block_main.assert_called_once_with(
            [
                "train",
                "--epochs",
                "3",
                "--architecture",
                "baseline",
                "--architecture-config",
                "hidden_dim=12",
                "--checkpoint",
                "data/block_smb/policy.pth",
            ]
        )

    def test_game_and_stage_are_independent_top_level_options(self):
        with patch("retroagi.stages.block_smb.cli.main", return_value=0) as block_main:
            block_exit = cli.main(
                [
                    "train",
                    "--game",
                    "smb",
                    "--stage",
                    "block",
                    "--epochs",
                    "2",
                ]
            )

        self.assertEqual(block_exit, 0)
        block_main.assert_called_once_with(["train", "--epochs", "2"])

        with patch("retroagi.stages.full_smb.run.main") as run_full_smb:
            full_exit = cli.main(
                [
                    "evaluate",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--steps",
                    "3",
                    "--seed",
                    "9",
                ]
            )

        self.assertEqual(full_exit, 0)
        run_full_smb.assert_called_once_with(
            num_steps=3,
            seed=9,
            render=False,
            encode_observations=False,
        )

    def test_unknown_game_fails_before_stage_dispatch(self):
        stream = io.StringIO()
        with redirect_stderr(stream):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["train", "--game", "missing", "--stage", "block"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unknown game 'missing'", stream.getvalue())

    def test_block_smb_resume_rewrites_to_train_resume(self):
        with patch("retroagi.stages.block_smb.cli.main", return_value=0) as block_main:
            exit_code = cli.main(
                [
                    "resume",
                    "--env",
                    "block_smb",
                    "--checkpoint",
                    "data/block_smb/old_policy.pth",
                    "--save-checkpoint",
                    "data/block_smb/new_policy.pth",
                    "--epochs",
                    "10",
                ]
            )

        self.assertEqual(exit_code, 0)
        block_main.assert_called_once_with(
            [
                "train",
                "--resume",
                "data/block_smb/old_policy.pth",
                "--checkpoint",
                "data/block_smb/new_policy.pth",
                "--epochs",
                "10",
            ]
        )

    def test_full_smb_evaluate_without_policy_dispatches_to_smoke_runner(self):
        with patch("retroagi.stages.full_smb.run.main") as run_full_smb:
            exit_code = cli.main(
                [
                    "evaluate",
                    "--stage",
                    "full",
                    "--steps",
                    "5",
                    "--seed",
                    "42",
                    "--encode-observations",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_full_smb.assert_called_once_with(
            num_steps=5,
            seed=42,
            render=False,
            encode_observations=True,
        )

    def test_full_smb_train_resume_and_policy_evaluate_forward_arguments(self):
        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            train_exit = cli.main(
                [
                    "train",
                    "--stage",
                    "full-smb",
                    "--epochs",
                    "2",
                    "--mode",
                    "scratch",
                    "--perception-mode",
                    "fine_tune",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                ]
            )

        self.assertEqual(train_exit, 0)
        train_main.assert_called_once_with(
            [
                "train",
                "--epochs",
                "2",
                "--mode",
                "scratch",
                "--perception-mode",
                "fine_tune",
                "--checkpoint",
                "data/full_smb/policy.pth",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            resume_exit = cli.main(
                [
                    "resume",
                    "--stage",
                    "full",
                    "--checkpoint",
                    "data/full_smb/old.pth",
                    "--save-checkpoint",
                    "data/full_smb/new.pth",
                    "--epochs",
                    "3",
                ]
            )

        self.assertEqual(resume_exit, 0)
        train_main.assert_called_once_with(
            [
                "resume",
                "--checkpoint",
                "data/full_smb/old.pth",
                "--save-checkpoint",
                "data/full_smb/new.pth",
                "--epochs",
                "3",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            resume_in_place_exit = cli.main(
                [
                    "resume",
                    "--stage",
                    "full-smb",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                ]
            )

        self.assertEqual(resume_in_place_exit, 0)
        train_main.assert_called_once_with(
            [
                "resume",
                "--checkpoint",
                "data/full_smb/policy.pth",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            evaluate_exit = cli.main(
                [
                    "evaluate",
                    "--stage",
                    "full-smb",
                    "--policy-checkpoint",
                    "data/full_smb/policy.pth",
                    "--evaluation-episodes",
                    "2",
                ]
            )

        self.assertEqual(evaluate_exit, 0)
        train_main.assert_called_once_with(
            [
                "evaluate",
                "--policy-checkpoint",
                "data/full_smb/policy.pth",
                "--evaluation-episodes",
                "2",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            evaluate_checkpoint_exit = cli.main(
                [
                    "evaluate",
                    "--stage",
                    "full-smb",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                    "--evaluation-episodes",
                    "3",
                ]
            )

        self.assertEqual(evaluate_checkpoint_exit, 0)
        train_main.assert_called_once_with(
            [
                "evaluate",
                "--checkpoint",
                "data/full_smb/policy.pth",
                "--evaluation-episodes",
                "3",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            record_exit = cli.main(
                [
                    "record",
                    "--stage",
                    "full-smb",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                    "--record-dir",
                    "artifacts/full_smb/recordings",
                    "--evaluation-episodes",
                    "2",
                ]
            )

        self.assertEqual(record_exit, 0)
        train_main.assert_called_once_with(
            [
                "record",
                "--checkpoint",
                "data/full_smb/policy.pth",
                "--record-dir",
                "artifacts/full_smb/recordings",
                "--evaluation-episodes",
                "2",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            play_exit = cli.main(
                [
                    "play",
                    "--stage",
                    "full-smb",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                    "--steps",
                    "5",
                    "--task-set",
                    "fixed_benchmark",
                    "--level",
                    "1-1",
                    "--frame-skip",
                    "4",
                    "--action-repeat",
                    "2",
                    "--render-mode",
                    "none",
                    "--no-render",
                    "--sampling-policy",
                    "--overlay",
                    "--overlay-top-actions",
                    "2",
                    "--record-output",
                    "artifacts/full_smb/play_manifest.npz",
                ]
            )

        self.assertEqual(play_exit, 0)
        train_main.assert_called_once_with(
            [
                "play",
                "--checkpoint",
                "data/full_smb/policy.pth",
                "--steps",
                "5",
                "--task-set",
                "fixed_benchmark",
                "--level",
                "1-1",
                "--frame-skip",
                "4",
                "--action-repeat",
                "2",
                "--render-mode",
                "none",
                "--no-render",
                "--sampling-policy",
                "--overlay",
                "--overlay-top-actions",
                "2",
                "--record-output",
                "artifacts/full_smb/play_manifest.npz",
            ]
        )

        with patch("retroagi.stages.full_smb.train.main", return_value=0) as train_main:
            human_play_exit = cli.main(
                [
                    "play",
                    "--stage",
                    "full-smb",
                    "--human",
                    "--steps",
                    "5",
                    "--state",
                    "Level1-1",
                ]
            )

        self.assertEqual(human_play_exit, 0)
        train_main.assert_called_once_with(
            [
                "play",
                "--human",
                "--steps",
                "5",
                "--state",
                "Level1-1",
            ]
        )

    def test_full_smb_transfer_and_compare_forward_arguments(self):
        with patch("retroagi.stages.full_smb.transfer.main") as transfer_main:
            transfer_exit = cli.main(
                [
                    "transfer",
                    "--stage",
                    "full-smb",
                    "--block-policy-checkpoint",
                    "data/block_smb/policy.pth",
                    "--output-checkpoint",
                    "data/full_smb/transferred_policy.pth",
                ]
            )

        self.assertEqual(transfer_exit, 0)
        transfer_main.assert_called_once_with(
            [
                "--block-policy-checkpoint",
                "data/block_smb/policy.pth",
                "--output-checkpoint",
                "data/full_smb/transferred_policy.pth",
            ]
        )

        with patch("retroagi.stages.full_smb.compare.main") as compare_main:
            compare_exit = cli.main(
                [
                    "compare",
                    "--env",
                    "full_smb",
                    "--transfer-checkpoint",
                    "data/full_smb/transferred_policy.pth",
                    "--scratch-trained-checkpoint",
                    "data/full_smb/scratch_policy.pth",
                    "--fine-tuned-checkpoint",
                    "data/full_smb/fine_tuned_policy.pth",
                    "--known-good-checkpoint",
                    "data/full_smb/known_good_policy.pth",
                    "--task-set",
                    "fixed_benchmark",
                    "--seed",
                    "0",
                    "--seed",
                    "1",
                    "--output",
                    "artifacts/full_smb/policy_suite_comparison.json",
                ]
            )

        self.assertEqual(compare_exit, 0)
        compare_main.assert_called_once_with(
            [
                "--transfer-checkpoint",
                "data/full_smb/transferred_policy.pth",
                "--scratch-trained-checkpoint",
                "data/full_smb/scratch_policy.pth",
                "--fine-tuned-checkpoint",
                "data/full_smb/fine_tuned_policy.pth",
                "--known-good-checkpoint",
                "data/full_smb/known_good_policy.pth",
                "--task-set",
                "fixed_benchmark",
                "--seed",
                "0",
                "--seed",
                "1",
                "--output",
                "artifacts/full_smb/policy_suite_comparison.json",
            ]
        )

    def test_full_smb_check_env_forwards_arguments(self):
        with patch("retroagi.stages.full_smb.capabilities.main", return_value=0) as check_main:
            exit_code = cli.main(
                [
                    "check-env",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--seed",
                    "17",
                    "--frame-skip",
                    "3",
                    "--output",
                    "artifacts/full_smb/env_check.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        check_main.assert_called_once_with(
            [
                "--seed",
                "17",
                "--frame-skip",
                "3",
                "--output",
                "artifacts/full_smb/env_check.json",
            ]
        )

    def test_full_smb_diagnose_vision_forwards_arguments(self):
        with patch("retroagi.stages.full_smb.diagnostics.main", return_value=0) as diagnostics_main:
            exit_code = cli.main(
                [
                    "diagnose-vision",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--vision-checkpoint",
                    "data/vit/full_smb_vit.pth",
                    "--samples",
                    "4",
                    "--rollout-steps",
                    "8",
                    "--output",
                    "artifacts/full_smb/perception_diagnostic.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        diagnostics_main.assert_called_once_with(
            [
                "--vision-checkpoint",
                "data/vit/full_smb_vit.pth",
                "--samples",
                "4",
                "--rollout-steps",
                "8",
                "--output",
                "artifacts/full_smb/perception_diagnostic.json",
            ]
        )

    def test_full_smb_diagnose_actions_forwards_arguments(self):
        with patch(
            "retroagi.stages.full_smb.action_diagnostics.main",
            return_value=0,
        ) as diagnostics_main:
            exit_code = cli.main(
                [
                    "diagnose-actions",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                    "--samples",
                    "4",
                    "--recording",
                    "artifacts/full_smb/recordings",
                    "--output",
                    "artifacts/full_smb/action_diagnostic.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        diagnostics_main.assert_called_once_with(
            [
                "--checkpoint",
                "data/full_smb/policy.pth",
                "--samples",
                "4",
                "--recording",
                "artifacts/full_smb/recordings",
                "--output",
                "artifacts/full_smb/action_diagnostic.json",
            ]
        )

    def test_block_smb_diagnose_actions_forwards_arguments(self):
        with patch(
            "retroagi.stages.block_smb.cli.main",
            return_value=0,
        ) as diagnostics_main:
            exit_code = cli.main(
                [
                    "diagnose-actions",
                    "--game",
                    "smb",
                    "--stage",
                    "block",
                    "--checkpoint",
                    "data/block_smb/policy.pth",
                    "--scenario",
                    "level_2_gap.json",
                    "--output",
                    "artifacts/block_smb/action_probe.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        diagnostics_main.assert_called_once_with(
            [
                "diagnose-actions",
                "--checkpoint",
                "data/block_smb/policy.pth",
                "--scenario",
                "level_2_gap.json",
                "--output",
                "artifacts/block_smb/action_probe.json",
            ]
        )

    def test_full_smb_gate_forwards_arguments(self):
        with patch(
            "retroagi.stages.full_smb.curriculum_gates.main",
            return_value=0,
        ) as gate_main:
            exit_code = cli.main(
                [
                    "gate",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                    "--episodes",
                    "2",
                    "--min-gate-pass-rate",
                    "0.75",
                    "--output",
                    "artifacts/full_smb/curriculum_gates.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        gate_main.assert_called_once_with(
            [
                "--checkpoint",
                "data/full_smb/policy.pth",
                "--episodes",
                "2",
                "--min-gate-pass-rate",
                "0.75",
                "--output",
                "artifacts/full_smb/curriculum_gates.json",
            ]
        )

    def test_full_smb_architecture_benchmark_forwards_arguments(self):
        with patch(
            "retroagi.stages.full_smb.architecture_benchmark.main",
            return_value=0,
        ) as benchmark_main:
            exit_code = cli.main(
                [
                    "benchmark-architecture",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--architecture-config",
                    "hidden_dim=8",
                    "--iterations",
                    "2",
                    "--output",
                    "artifacts/full_smb/architecture_benchmark.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        benchmark_main.assert_called_once_with(
            [
                "--architecture-config",
                "hidden_dim=8",
                "--iterations",
                "2",
                "--output",
                "artifacts/full_smb/architecture_benchmark.json",
            ]
        )

    def test_full_smb_imitate_forwards_arguments(self):
        with patch("retroagi.stages.full_smb.imitation.main", return_value=0) as imitation_main:
            exit_code = cli.main(
                [
                    "imitate",
                    "--game",
                    "smb",
                    "--stage",
                    "full",
                    "--checkpoint",
                    "data/full_smb/source.pth",
                    "--output-checkpoint",
                    "data/full_smb/warm_start.pth",
                    "--steps",
                    "64",
                    "--output-summary",
                    "artifacts/full_smb/warm_start.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        imitation_main.assert_called_once_with(
            [
                "--checkpoint",
                "data/full_smb/source.pth",
                "--output-checkpoint",
                "data/full_smb/warm_start.pth",
                "--steps",
                "64",
                "--output-summary",
                "artifacts/full_smb/warm_start.json",
            ]
        )


if __name__ == "__main__":
    unittest.main()
