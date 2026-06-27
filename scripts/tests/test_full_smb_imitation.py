"""Tests for Full SMB real-emulator imitation warm starts."""

import unittest

import torch

from retroagi.stages.full_smb import (
    FullSMBObservationConfig,
    FullSMBStage,
    collect_full_smb_imitation_dataset,
    full_smb_opening_imitation_script,
    train_full_smb_imitation_warm_start,
)
from retroagi.stages.full_smb.transfer import make_full_smb_policy_model
from scripts.tests.test_full_smb_adapter import StaticFullSMBVision
from scripts.tests.test_full_smb_transfer import TinyFullSMBEnv


class TestFullSMBImitationWarmStart(unittest.TestCase):
    def test_opening_imitation_script_mixes_right_and_right_jump(self):
        script = full_smb_opening_imitation_script(320)

        self.assertEqual(len(script), 320)
        self.assertEqual(script.count(1), 280)
        self.assertEqual(script.count(2), 40)

    def test_collect_and_train_imitation_warm_start_updates_policy_head(self):
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
            dataset = collect_full_smb_imitation_dataset(
                stage,
                full_smb_opening_imitation_script(12),
                seed=5,
            )
        finally:
            stage.close()

        model = make_full_smb_policy_model(
            architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
        )
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith("agent.fc_out_A") or name.startswith("agent.fc_controller_params")
        }
        metrics, _optimizer = train_full_smb_imitation_warm_start(
            model,
            dataset,
            device=torch.device("cpu"),
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=11,
        )

        changed = [
            name
            for name, value in before.items()
            if not torch.equal(value, model.state_dict()[name])
        ]
        self.assertTrue(changed)
        self.assertGreater(dataset["metrics"]["samples"], 0.0)
        self.assertGreaterEqual(metrics["mean_action_accuracy"], 0.0)
        self.assertIn("agent.fc_out_A", metrics["trainable_prefixes"])


if __name__ == "__main__":
    unittest.main()
