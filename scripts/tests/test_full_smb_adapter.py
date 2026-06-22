"""Tests for the Full SMB stable-retro stage adapter."""

import unittest

import numpy as np
import torch

from retroagi.core import (
    SMBAction,
    StageBatch,
    VisionOutput,
    VisionSpec,
    full_smb_action,
)
from retroagi.stages.full_smb import FULL_SMB_SPEC, FullSMBStage


class StaticFullSMBVision:
    spec = VisionSpec(
        name="static_full_smb",
        semantic_classes=("background", "floor", "box", "enemy", "brick", "mario"),
        token_dim=6,
    )

    def encode(self, observation):
        logits = torch.full((1, self.spec.num_classes, 15, 16), -10.0)
        logits[:, 0] = 8.0
        logits[:, self.spec.semantic_classes.index("mario"), 7, 5] = 12.0
        return VisionOutput(
            position=torch.tensor([[0.33, 0.66]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.ones(1, 240, self.spec.token_dim),
            metadata={"source": "static"},
        )


class GymnasiumRetroEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self):
        self.reset_seed = None
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        self.reset_seed = seed
        return (
            np.zeros((224, 256, 4), dtype=np.uint8),
            {"state_vec": np.array([0.0, 0.25], dtype=np.float32)},
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            np.ones((224, 256, 3), dtype=np.float32),
            3.5,
            True,
            False,
            {"score": 100, "state_vec": np.array([0.5, 1.0], dtype=np.float32)},
        )

    def close(self):
        self.closed = True


class LegacyRetroEnv(GymnasiumRetroEnv):
    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            np.zeros((224, 256, 3), dtype=np.uint8),
            -1.0,
            True,
            {"TimeLimit.truncated": True},
        )


class TestFullSMBStage(unittest.TestCase):
    def test_stage_maps_shared_actions_and_projects_observations(self):
        env = GymnasiumRetroEnv()
        stage = FullSMBStage(env=env, vision=StaticFullSMBVision())
        try:
            observation = stage.reset(seed=123)
            self.assertEqual(env.reset_seed, 123)
            self.assertEqual(observation.dtype, np.uint8)
            self.assertEqual(observation.shape, (224, 256, 3))

            next_observation, reward, terminated, truncated, info = stage.step(
                SMBAction.RIGHT_JUMP
            )
            np.testing.assert_array_equal(
                env.actions[-1],
                full_smb_action(SMBAction.RIGHT_JUMP, env.buttons),
            )
            self.assertEqual(next_observation.dtype, np.uint8)
            self.assertEqual(reward, 3.5)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["action"]["shared_name"], "RIGHT_JUMP")

            batch = stage.encode_observation(next_observation, info)
            self.assertIsInstance(batch, StageBatch)
            self.assertEqual(stage.spec, FULL_SMB_SPEC)
            self.assertEqual(batch.src_a.shape, (1, FULL_SMB_SPEC.seq_len_a))
            self.assertEqual(batch.src_b.shape, (1, FULL_SMB_SPEC.seq_len_b))
            self.assertEqual(batch.src_c.shape, (1, FULL_SMB_SPEC.seq_len_c))
            self.assertEqual(batch.metadata["episode"]["mask"].item(), 0.0)
            self.assertEqual(batch.metadata["vision_fusion"]["c_state"], (8, 10))
        finally:
            stage.close()
        self.assertTrue(env.closed)

    def test_legacy_done_api_preserves_timeout_as_truncation(self):
        env = LegacyRetroEnv()
        stage = FullSMBStage(env=env, vision=StaticFullSMBVision())
        try:
            stage.reset()
            _observation, reward, terminated, truncated, _info = stage.step(
                SMBAction.NOOP
            )
        finally:
            stage.close()

        self.assertEqual(reward, -1.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
