"""Tests for Full SMB action-contract diagnostics."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from retroagi.core import MotorPrimitiveOutput, SMB_ACTIONS, SMBAction
from retroagi.stages.full_smb import (
    FullSMBObservationConfig,
    FullSMBStage,
    compare_full_smb_action_distribution,
    run_full_smb_action_contract_diagnostic,
    scripted_block_smb_action_reference,
    summarize_full_smb_action_recordings,
)
from scripts.tests.test_full_smb_adapter import StaticFullSMBVision
from scripts.tests.test_full_smb_transfer import TinyFullSMBEnv


class StaticFullSMBPolicy(torch.nn.Module):
    def forward(
        self,
        src_A,
        src_B,
        src_C,
        tau=1.0,
        *,
        world_model_state=None,
        episode_mask=None,
        return_world_model_state=False,
        critic_feedback_enabled=True,
        world_model_enabled=True,
    ):
        del tau, world_model_state, episode_mask, critic_feedback_enabled, world_model_enabled
        batch = src_A.size(0)
        logits = torch.full(
            (batch, src_A.size(1), len(SMB_ACTIONS)),
            -12.0,
            dtype=torch.float32,
            device=src_A.device,
        )
        logits[..., int(SMBAction.RIGHT)] = 4.0
        logits[..., int(SMBAction.RIGHT_JUMP)] = -1.0
        controller = torch.ones((batch, src_B.size(1)), dtype=torch.float32, device=src_A.device)
        self.last_motor_primitives = MotorPrimitiveOutput(
            button_combo_logits=logits[:, :1].repeat(1, src_B.size(1), 1),
            hold_duration=controller,
            release_logit=torch.zeros_like(controller),
            cancel_logit=torch.zeros_like(controller),
            confidence=controller,
            interrupt_logit=controller,
            replan_probability=controller,
        )
        outputs = (
            None,
            src_C.detach(),
            torch.zeros_like(src_C),
            None,
            logits,
            controller,
            torch.zeros_like(controller),
        )
        if return_world_model_state:
            return (*outputs, None)
        return outputs


class TestFullSMBActionDiagnostics(unittest.TestCase):
    def test_action_contract_diagnostic_reports_motor_bias_and_combo_actions(self):
        stage = FullSMBStage(
            env=TinyFullSMBEnv(),
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(16, 20),
            ),
        )
        try:
            result = run_full_smb_action_contract_diagnostic(
                StaticFullSMBPolicy(),
                stage,
                device=torch.device("cpu"),
                samples=4,
                seed=7,
                sample_repeats=2,
            )
        finally:
            stage.close()

        deterministic = result["canonical_policy"]["deterministic"]
        self.assertEqual(deterministic["counts"]["RIGHT_JUMP"], 4)
        self.assertEqual(deterministic["counts"]["RIGHT"], 0)
        self.assertGreater(result["motor_primitives"]["mean_right_jump_bias"], 0.0)
        self.assertTrue(result["flags"]["overactive_right_jump_when_stalled"])
        self.assertFalse(result["flags"]["missing_right_jump_when_stalled"])
        self.assertIn("RIGHT_JUMP", result["transfer_action_comparison"]["fraction_delta"])
        # Without recordings the comparison must fall back to the canonical
        # deterministic rollout counts instead of an empty recording summary.
        self.assertEqual(result["transfer_action_comparison"]["observed_total"], 4)
        self.assertGreater(
            result["transfer_action_comparison"]["fraction_delta"]["RIGHT_JUMP"], 0.0
        )

    def test_recording_summary_flags_missing_right_jump_under_progress_gate(self):
        with TemporaryDirectory() as tmpdir:
            recording = Path(tmpdir) / "episode.npz"
            signals = [
                json.dumps({"progress": progress})
                for progress in (12.0, 18.0, 20.0)
            ]
            np.savez_compressed(
                recording,
                actions=np.asarray([int(SMBAction.RIGHT)] * 3, dtype=np.int64),
                signals_json=np.asarray(signals),
            )

            summary = summarize_full_smb_action_recordings(
                (recording,),
                progress_gate=64.0,
            )

        self.assertEqual(summary["artifact_count"], 1)
        self.assertEqual(summary["action_counts"]["RIGHT"], 3)
        self.assertEqual(summary["action_counts"]["RIGHT_JUMP"], 0)
        self.assertTrue(summary["missing_right_jump_when_stalled"])

    def test_block_reference_comparison_reports_right_jump_delta(self):
        reference = scripted_block_smb_action_reference(max_steps=20)
        comparison = compare_full_smb_action_distribution(
            reference["action_counts"],
            {"RIGHT": 20, "RIGHT_JUMP": 0},
        )

        self.assertGreater(reference["action_counts"]["RIGHT_JUMP"], 0)
        self.assertLess(comparison["right_jump_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
