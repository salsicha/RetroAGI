"""Tests for the project-level RetroAGI command line entry point."""

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from retroagi import cli


class TestRetroAGICLI(unittest.TestCase):
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

    def test_full_smb_evaluate_dispatches_to_smoke_runner(self):
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
                    "--output",
                    "artifacts/full_smb/transfer_vs_scratch.json",
                ]
            )

        self.assertEqual(compare_exit, 0)
        compare_main.assert_called_once_with(
            [
                "--transfer-checkpoint",
                "data/full_smb/transferred_policy.pth",
                "--output",
                "artifacts/full_smb/transfer_vs_scratch.json",
            ]
        )

    def test_full_smb_train_reports_unsupported_command(self):
        stream = io.StringIO()
        with redirect_stderr(stream):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["train", "--stage", "full-smb"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("direct training/resume is not implemented", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
