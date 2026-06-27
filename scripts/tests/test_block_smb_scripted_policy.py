"""Tests for the scripted known-good Block SMB policy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from retroagi.core import load_checkpoint
from retroagi.stages.block_smb import (
    SCRIPTED_BLOCK_SMB_CHECKPOINT_KIND,
    SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES,
    SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS,
    SCRIPTED_BLOCK_SMB_POLICY_NAME,
    SCRIPTED_BLOCK_SMB_SEED,
    BlockSMBScriptedPolicy,
    evaluate_scripted_block_smb_policy,
    load_scripted_block_smb_checkpoint,
    save_scripted_block_smb_checkpoint,
    fixed_scenario_action_scripts,
)


class TestBlockSMBScriptedPolicy(unittest.TestCase):
    def test_scripted_policy_meets_thresholds_and_round_trips_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            record_dir = tmp / "evaluation"
            checkpoint_path = tmp / "policy.pth"
            policy = BlockSMBScriptedPolicy()
            evaluation = evaluate_scripted_block_smb_policy(
                policy,
                seed=SCRIPTED_BLOCK_SMB_SEED,
                evaluation_episodes=SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES,
                evaluation_max_steps=SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS,
                record_dir=record_dir,
            )
            save_scripted_block_smb_checkpoint(
                checkpoint_path,
                policy=policy,
                seed=SCRIPTED_BLOCK_SMB_SEED,
                evaluation=evaluation,
                evaluation_episodes=SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES,
                evaluation_max_steps=SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS,
                record_dir=record_dir,
            )
            restored_policy = load_scripted_block_smb_checkpoint(checkpoint_path)
            restored_evaluation = evaluate_scripted_block_smb_policy(
                restored_policy,
                seed=SCRIPTED_BLOCK_SMB_SEED,
                evaluation_episodes=SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES,
                evaluation_max_steps=SCRIPTED_BLOCK_SMB_EVALUATION_MAX_STEPS,
            )
            checkpoint = load_checkpoint(checkpoint_path)

            self.assertEqual(checkpoint["model_name"], SCRIPTED_BLOCK_SMB_POLICY_NAME)
            self.assertEqual(checkpoint["checkpoint_kind"], SCRIPTED_BLOCK_SMB_CHECKPOINT_KIND)
            self.assertTrue(evaluation["success_thresholds_met"])
            self.assertEqual(restored_evaluation["fixed_scenarios"], evaluation["fixed_scenarios"])
            self.assertGreaterEqual(evaluation["mean_return"], 55.0)
            for scenario_name, result in evaluation["fixed_scenarios"].items():
                self.assertEqual(result["success_rate"], 1.0)
                self.assertTrue(result["threshold_met"])
                for episode in range(SCRIPTED_BLOCK_SMB_EVALUATION_EPISODES):
                    recording = record_dir / f"{scenario_name}_episode{episode}.npz"
                    self.assertTrue(recording.exists())
                    data = np.load(recording)
                    self.assertEqual(data["frames"].shape[-3:], (240, 256, 3))
                    self.assertEqual(data["actions"].shape, data["rewards"].shape)

    def test_scripted_policy_covers_directional_and_wait_actions(self):
        scripts = fixed_scenario_action_scripts()

        action_counts = {
            action: sum(actions.count(action) for actions in scripts.values())
            for action in range(6)
        }

        self.assertGreater(action_counts[0], 0)
        self.assertGreater(action_counts[3], 0)
        self.assertGreater(action_counts[4], 0)
        self.assertIn("level_10_left_retreat.json", scripts)
        self.assertIn("level_11_left_jump_recovery.json", scripts)
        self.assertIn("level_12_wait_bridge.json", scripts)


if __name__ == "__main__":
    unittest.main()
