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
from retroagi.stages.full_smb import (
    FULL_SMB_SPEC,
    FullSMBStage,
    extract_full_smb_signals,
)


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
            {"x_pos": 12, "y_pos": 42, "score": 0, "coins": 0, "lives": 3},
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            np.ones((224, 256, 3), dtype=np.float32),
            3.5,
            True,
            False,
            {
                "xscrollHi": 1,
                "xscrollLo": 44,
                "screen_x": 6,
                "ypos": 180,
                "score": 100,
                "coins": 7,
                "lives": 2,
                "level_complete": True,
                "termination_reason": "level_complete",
            },
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
    def test_signal_extractor_accepts_nested_position_and_death_reason(self):
        signals = extract_full_smb_signals(
            {
                "position": {"player_x": 144, "player_y": 96},
                "Score": 1234,
                "coin_count": 12,
                "lives_left": 1,
                "done_reason": "death",
            },
            terminated=True,
            truncated=False,
        )

        self.assertEqual(signals.position, (144.0, 96.0))
        self.assertEqual(signals.score, 1234)
        self.assertEqual(signals.coins, 12)
        self.assertEqual(signals.lives, 1)
        self.assertFalse(signals.completion)
        self.assertTrue(signals.death)
        self.assertTrue(signals.terminated)
        self.assertFalse(signals.truncated)

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
            self.assertEqual(
                info["full_smb_signals"],
                {
                    "position": (306.0, 180.0),
                    "score": 100,
                    "coins": 7,
                    "lives": 2,
                    "completion": True,
                    "death": False,
                    "terminated": True,
                    "truncated": False,
                    "termination_reason": "level_complete",
                },
            )
            np.testing.assert_allclose(
                info["state_vec"],
                np.array(
                    [
                        306.0 / 4096.0,
                        180.0 / 240.0,
                        100.0 / 999_999.0,
                        7.0 / 99.0,
                        2.0 / 99.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            )

            batch = stage.encode_observation(next_observation, info)
            self.assertIsInstance(batch, StageBatch)
            self.assertEqual(stage.spec, FULL_SMB_SPEC)
            self.assertEqual(batch.src_a.shape, (1, FULL_SMB_SPEC.seq_len_a))
            self.assertEqual(batch.src_b.shape, (1, FULL_SMB_SPEC.seq_len_b))
            self.assertEqual(batch.src_c.shape, (1, FULL_SMB_SPEC.seq_len_c))
            self.assertEqual(batch.metadata["episode"]["mask"].item(), 0.0)
            self.assertEqual(batch.metadata["vision_fusion"]["c_state"], (8, 17))
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
        self.assertFalse(_info["full_smb_signals"]["terminated"])
        self.assertTrue(_info["full_smb_signals"]["truncated"])


if __name__ == "__main__":
    unittest.main()
