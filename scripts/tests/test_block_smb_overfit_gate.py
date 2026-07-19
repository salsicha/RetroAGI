"""Tests for the Block SMB tiny-overfit sanity gate."""

import unittest

import pytest

from retroagi.core import SMBAction
from retroagi.stages.block_smb.overfit_gate import (
    _jump_boundary_error,
    run_block_smb_overfit_gate,
)


class TestOverfitGate(unittest.TestCase):
    @pytest.mark.timeout(300)
    def test_default_gate_memorizes_tiny_scenario_set(self):
        # The full gate: a fresh default-architecture policy must reach
        # near-perfect teacher-action accuracy on two scenarios. If this test
        # starts failing after an architecture or supervision change, that
        # change broke the policy's ability to learn a conditional policy —
        # do not start a full curriculum run until it passes again.
        summary = run_block_smb_overfit_gate(seed=0)

        self.assertTrue(summary["passed"], summary)
        self.assertGreaterEqual(summary["teacher_action_accuracy"], 0.95)
        self.assertGreater(
            summary["teacher_action_accuracy"],
            summary["initial_teacher_action_accuracy"],
        )
        self.assertEqual(summary["gate"], "block_smb_tiny_overfit")
        self.assertLessEqual(summary["epochs_trained"], summary["max_epochs"])
        self.assertIn("action_diagnostics", summary)
        self.assertEqual(
            summary["action_diagnostics"]["misclassified_count"],
            round((1.0 - summary["teacher_action_accuracy"]) * summary["example_count"]),
        )

    def test_gate_flags_undertrained_policy(self):
        summary = run_block_smb_overfit_gate(seed=0, epochs=1)

        self.assertFalse(summary["passed"])
        self.assertIn("do not start a full curriculum run", summary["message"])
        self.assertEqual(summary["epochs_trained"], 1)
        self.assertIn("jump_boundary", summary["action_diagnostics"])

    def test_gate_validates_arguments(self):
        with self.assertRaisesRegex(ValueError, "accuracy_threshold"):
            run_block_smb_overfit_gate(accuracy_threshold=0.0)
        with self.assertRaisesRegex(ValueError, "epochs"):
            run_block_smb_overfit_gate(epochs=0)
        with self.assertRaisesRegex(ValueError, "check_interval_epochs"):
            run_block_smb_overfit_gate(check_interval_epochs=0)

    def test_jump_boundary_errors_distinguish_early_late_and_missed_actions(self):
        right = int(SMBAction.RIGHT)
        jump = int(SMBAction.RIGHT_JUMP)
        actions = [right, right, jump, jump, jump, right, right]

        self.assertEqual(_jump_boundary_error(actions, 1, jump), ("early_jump", 1))
        self.assertEqual(_jump_boundary_error(actions, 5, jump), ("late_jump", 1))
        self.assertEqual(_jump_boundary_error(actions, 3, right), ("missed_jump", 0))
        self.assertEqual(
            _jump_boundary_error([right] * 4, 2, jump),
            ("spurious_jump", None),
        )


if __name__ == "__main__":
    unittest.main()
