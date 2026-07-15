"""Tests for Full SMB short curriculum gates."""

import unittest

import torch

from retroagi.core import SMB_ACTIONS, SMBAction
from retroagi.stages.full_smb import (
    FullSMBCurriculumGateThreshold,
    FullSMBObservationConfig,
    FullSMBStage,
    default_full_smb_curriculum_gates,
    evaluate_full_smb_curriculum_gate_threshold,
    run_full_smb_curriculum_gate_evaluation,
)
from scripts.tests.test_full_smb_adapter import StaticFullSMBVision
from scripts.tests.test_full_smb_transfer import TinyFullSMBEnv


class RightJumpPolicy(torch.nn.Module):
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
        logits = torch.full(
            (src_A.size(0), src_A.size(1), len(SMB_ACTIONS)),
            -12.0,
            dtype=torch.float32,
            device=src_A.device,
        )
        logits[..., int(SMBAction.RIGHT_JUMP)] = 4.0
        controller = torch.zeros((src_A.size(0), src_B.size(1)), device=src_A.device)
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


def tiny_stage():
    return FullSMBStage(
        env=TinyFullSMBEnv(),
        vision=StaticFullSMBVision(),
        observation_config=FullSMBObservationConfig(
            frame_skip=1,
            frame_stack=2,
            resize_shape=(16, 20),
        ),
    )


class TestFullSMBCurriculumGates(unittest.TestCase):
    def test_default_gates_cover_level_1_1_opening_hazards(self):
        gates = default_full_smb_curriculum_gates()

        self.assertEqual(
            [gate.name for gate in gates],
            ["opening_movement", "first_pipe", "first_enemy", "first_gap_or_stair"],
        )
        self.assertTrue(all(gate.max_steps <= 1000 for gate in gates))
        self.assertGreater(gates[-1].min_progress, gates[0].min_progress)

    def test_gate_evaluation_reports_progress_survival_and_actions(self):
        gate = FullSMBCurriculumGateThreshold(
            name="unit_opening",
            min_progress=3.0,
            max_steps=4,
            episodes=2,
            min_right_jump_fraction=1.0,
            rationale="unit",
        )
        stage = tiny_stage()
        try:
            result = run_full_smb_curriculum_gate_evaluation(
                RightJumpPolicy(),
                stage,
                device=torch.device("cpu"),
                gates=(gate,),
                seed=0,
            )
        finally:
            stage.close()

        gate_result = result["gates"]["unit_opening"]
        self.assertTrue(gate_result["threshold_met"])
        self.assertEqual(gate_result["episodes"], 2)
        self.assertGreaterEqual(gate_result["max_progress"], 3.0)
        self.assertEqual(gate_result["action_counts"]["RIGHT_JUMP"], 8)
        self.assertEqual(gate_result["action_fractions"]["RIGHT_JUMP"], 1.0)
        self.assertTrue(result["summary"]["full_benchmark_allowed"])

    def test_gate_aggregate_reports_observed_max_episode_steps(self):
        gate = FullSMBCurriculumGateThreshold(
            name="unit_budget",
            min_progress=1.0,
            max_steps=10,
            rationale="unit",
        )
        stage = tiny_stage()
        try:
            result = run_full_smb_curriculum_gate_evaluation(
                RightJumpPolicy(),
                stage,
                device=torch.device("cpu"),
                gates=(gate,),
                seed=0,
            )
        finally:
            stage.close()

        gate_result = result["gates"]["unit_budget"]
        # TinyFullSMBEnv terminates after 8 steps, below the 10-step budget;
        # the aggregate must report the observed maximum, not the budget itself.
        self.assertEqual(gate_result["max_steps_per_episode"], 8)
        self.assertTrue(gate_result["threshold_diagnostics"]["within_step_budget"])

    def test_gate_evaluation_blocks_full_benchmark_when_pass_rate_is_low(self):
        passing_gate = FullSMBCurriculumGateThreshold(
            name="passing",
            min_progress=2.0,
            max_steps=4,
            rationale="unit",
        )
        blocking_gate = FullSMBCurriculumGateThreshold(
            name="blocking",
            min_progress=2000.0,
            max_steps=4,
            rationale="unit",
        )
        stage = tiny_stage()
        try:
            result = run_full_smb_curriculum_gate_evaluation(
                RightJumpPolicy(),
                stage,
                device=torch.device("cpu"),
                gates=(passing_gate, blocking_gate),
                seed=0,
                min_gate_pass_rate=1.0,
            )
        finally:
            stage.close()

        self.assertTrue(result["gates"]["passing"]["threshold_met"])
        self.assertFalse(result["gates"]["blocking"]["threshold_met"])
        self.assertEqual(result["summary"]["gate_pass_rate"], 0.5)
        self.assertEqual(result["summary"]["blocking_gates"], ["blocking"])
        self.assertFalse(result["summary"]["full_benchmark_allowed"])
        self.assertTrue(result["summary"]["full_benchmark_blocked"])

    def test_threshold_diagnostics_explain_gate_failure(self):
        gate = FullSMBCurriculumGateThreshold(
            name="unit",
            min_progress=10.0,
            max_steps=8,
            min_right_jump_fraction=0.25,
            rationale="unit",
        )
        diagnostics = evaluate_full_smb_curriculum_gate_threshold(
            gate,
            {
                "episodes": 1,
                "max_steps_per_episode": 8,
                "episode_pass_rate": 0.0,
                "max_progress": 4.0,
                "survival_rate": 1.0,
                "mean_score": 0.0,
                "action_fractions": {"RIGHT_JUMP": 0.0},
            },
        )

        self.assertFalse(diagnostics["threshold_met"])
        self.assertFalse(diagnostics["meets_progress"])
        self.assertFalse(diagnostics["meets_right_jump_fraction"])
        self.assertEqual(diagnostics["observed"]["right_jump_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
