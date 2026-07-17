"""Tests for Block SMB action-logit probes."""

import unittest

import torch

from retroagi.core import SMB_ACTIONS, MotorPrimitiveOutput, SMBAction
from retroagi.stages.block_smb import run_block_smb_action_probe
from scripts.tests.test_block_smb_training import static_vision_factory


class StaticBlockSMBPolicy(torch.nn.Module):
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
            -6.0,
            dtype=torch.float32,
            device=src_A.device,
        )
        logits[..., int(SMBAction.RIGHT)] = 4.0
        logits[..., int(SMBAction.RIGHT_JUMP)] = 1.0
        controller = torch.ones(
            (batch, src_B.size(1)),
            dtype=torch.float32,
            device=src_A.device,
        )
        self.last_motor_primitives = MotorPrimitiveOutput(
            button_combo_logits=logits.repeat_interleave(
                src_B.size(1) // src_A.size(1),
                dim=1,
            ),
            hold_duration=controller,
            release_logit=torch.zeros_like(controller),
            cancel_logit=torch.zeros_like(controller),
            confidence=controller,
            interrupt_logit=controller,
            replan_probability=controller,
        )
        outputs = (
            src_C,
            src_C + 0.25,
            torch.ones(batch, src_A.size(1), 4, dtype=torch.float32, device=src_A.device),
            src_C,
            logits,
            controller,
            torch.zeros_like(controller),
        )
        if return_world_model_state:
            return (*outputs, None)
        return outputs


class TestBlockSMBActionDiagnostics(unittest.TestCase):
    def test_probe_reports_logits_margins_motion_and_critic_norm(self):
        result = run_block_smb_action_probe(
            StaticBlockSMBPolicy(),
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
            scenarios=("level_2_gap.json", "level_3_stairs.json"),
            seed=5,
            max_steps=16,
            points_per_scenario=1,
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["summary"]["sample_count"], 2)
        self.assertFalse(result["missing_probe_points"])
        for sample in result["samples"]:
            self.assertEqual(sample["expected_action_name"], SMBAction.RIGHT_JUMP.name)
            self.assertEqual(sample["raw_action_name"], SMBAction.RIGHT.name)
            self.assertEqual(sample["motor_biased_action_name"], SMBAction.RIGHT_JUMP.name)
            self.assertGreater(sample["margins"]["raw_right_minus_right_jump"], 0.0)
            self.assertLess(
                sample["margins"]["motor_biased_right_minus_right_jump"],
                0.0,
            )
            self.assertGreater(sample["motor_bias"][SMBAction.RIGHT_JUMP.name], 0.0)
            self.assertEqual(sample["predicted_motion"]["available"], 1.0)
            self.assertGreater(sample["predicted_motion"]["absolute_last"], 0.0)
            self.assertGreater(sample["critic"]["norm"], 0.0)
            self.assertIn("support_right_dx", sample["state"])


if __name__ == "__main__":
    unittest.main()
