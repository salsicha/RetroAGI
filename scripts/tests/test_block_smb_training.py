"""Tests for Block SMB trainer plumbing."""

import unittest

import torch

from retroagi.core.interfaces import VisionOutput, VisionSpec
from retroagi.core.smb_pixel_types import PIXEL_TYPES
from retroagi.stages.block_smb.env import MarioScenarioEnv


class StaticBlockVision:
    spec = VisionSpec(
        name="static_block_trainer",
        semantic_classes=PIXEL_TYPES,
        token_dim=4,
    )

    def encode(self, observation):
        logits = torch.full((1, self.spec.num_classes, 2, 16), -8.0)
        logits[:, 1, :, 1] = 8.0
        logits[:, 2, :, :] = torch.maximum(logits[:, 2, :, :], torch.tensor(1.0))
        return VisionOutput(
            position=torch.tensor([[0.1, 0.8]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.zeros(1, 240, self.spec.token_dim),
            support_logits=torch.tensor([[-4.0, 4.0, -4.0]]),
            metadata={"semantic_classes": PIXEL_TYPES},
        )


def static_vision_factory():
    return StaticBlockVision()


class TestBlockSMBMasterySchedule(unittest.TestCase):
    def test_goal_distance_shaping_rewards_rising_toward_elevated_goal(self):
        scenario = {
            "world_width": 256,
            "mario": [90, 200],
            "platforms": [[0, 220, 256, 20], [120, 168, 30, 52]],
            "coins": [],
            "goal": [127, 148, 16, 20],
            "reward_goal_distance_shaping": 2.0,
        }
        # A standing NES jump needs a hold of at least 22 frames to clear the
        # 52 px block while drifting the 20 px to its edge; 24 is on the menu.
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario), seed=0)
            total = 0.0
            for action in [2] * 24 + [1] * 20:
                _obs, _reward, terminated, truncated, info = env.step(action)
                total += info["reward_terms"]["goal_distance"]
                if terminated or truncated:
                    break
            self.assertGreater(total, 0.0)
            self.assertGreater(info["reward_terms"]["goal"], 0.0)
        finally:
            env.close()
        # Without the opt-in key the term stays exactly zero.
        control = {k: v for k, v in scenario.items() if k != "reward_goal_distance_shaping"}
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=control, seed=0)
            total = 0.0
            for action in [2] * 24 + [1] * 20:
                _obs, _reward, terminated, truncated, info = env.step(action)
                total += info["reward_terms"]["goal_distance"]
                if terminated or truncated:
                    break
            self.assertEqual(total, 0.0)
        finally:
            env.close()

    def test_energy_regulator_charges_held_jump_frames_only_when_opted_in(self):
        scenario = {
            "world_width": 256,
            "mario": [20, 200],
            "platforms": [[0, 220, 256, 20]],
            "coins": [],
            "goal": [230, 200, 16, 20],
            "reward_energy_jump": -0.15,
        }
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario), seed=0)
            energy = 0.0
            for action in [2] * 10 + [1] * 5:
                _obs, _reward, terminated, truncated, info = env.step(action)
                energy += info["reward_terms"]["energy"]
                if terminated or truncated:
                    break
            # Ten held jump frames at -0.15 each; walking frames are free.
            self.assertAlmostEqual(energy, -1.5, places=6)
        finally:
            env.close()
        control = {k: v for k, v in scenario.items() if k != "reward_energy_jump"}
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=control, seed=0)
            energy = 0.0
            for action in [2] * 10:
                _obs, _reward, terminated, truncated, info = env.step(action)
                energy += info["reward_terms"]["energy"]
                if terminated or truncated:
                    break
            self.assertEqual(energy, 0.0)
        finally:
            env.close()
        # Success-conditioned energy: an attempt that ends in death refunds the
        # accumulated charge, so giving up never beats trying.
        doomed = {
            "world_width": 256,
            "mario": [20, 200],
            "platforms": [[0, 220, 100, 20]],
            "coins": [],
            "goal": [230, 200, 16, 20],
            "reward_energy_jump": -0.15,
        }
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=doomed, seed=0)
            energy = 0.0
            for action in [2] * 6 + [1] * 80:
                _obs, _reward, terminated, truncated, info = env.step(action)
                energy += info["reward_terms"]["energy"]
                if terminated or truncated:
                    break
            self.assertTrue(info["death"])
            self.assertAlmostEqual(energy, 0.0, places=6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
