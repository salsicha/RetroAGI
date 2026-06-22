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
    FullSMBObservationConfig,
    FullSMBStage,
    extract_full_smb_signals,
)


class StaticFullSMBVision:
    spec = VisionSpec(
        name="static_full_smb",
        semantic_classes=("background", "floor", "box", "enemy", "brick", "mario"),
        token_dim=6,
    )

    def __init__(self):
        self.observations = []

    def encode(self, observation):
        self.observations.append(torch.as_tensor(observation).detach().clone())
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


class FrameSkipRetroEnv(GymnasiumRetroEnv):
    def __init__(self):
        super().__init__()
        self.step_count = 0

    def reset(self, seed=None):
        self.reset_seed = seed
        return (
            np.zeros((16, 20, 3), dtype=np.uint8),
            {"x_pos": 0, "y_pos": 0, "score": 0, "coins": 0, "lives": 3},
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.step_count += 1
        return (
            np.full((16, 20, 3), self.step_count * 40, dtype=np.uint8),
            float(self.step_count),
            False,
            False,
            {
                "x_pos": self.step_count * 10,
                "y_pos": 100,
                "score": self.step_count,
                "coins": 0,
                "lives": 3,
            },
        )


class EmulatorStateProxy:
    def __init__(self, env):
        self.env = env

    def get_state(self):
        return {"step_count": self.env.step_count}

    def set_state(self, state):
        self.env.step_count = int(state["step_count"])


class SaveStateRetroEnv(FrameSkipRetroEnv):
    def __init__(self):
        super().__init__()
        self.em = EmulatorStateProxy(self)


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

    def test_frame_skip_resize_stack_and_continuing_episode_mask(self):
        env = FrameSkipRetroEnv()
        vision = StaticFullSMBVision()
        stage = FullSMBStage(
            env=env,
            vision=vision,
            observation_config=FullSMBObservationConfig(
                frame_skip=3,
                frame_stack=3,
                resize_shape=(12, 16),
            ),
        )
        try:
            stage.reset(seed=5)
            observation, reward, terminated, truncated, info = stage.step(
                SMBAction.RIGHT
            )
            batch = stage.encode_observation(observation, info)
        finally:
            stage.close()

        self.assertEqual(env.step_count, 3)
        self.assertEqual(reward, 6.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action"]["frames_executed"], 3)
        self.assertEqual(info["action"]["frame_rewards"], [1.0, 2.0, 3.0])
        self.assertEqual(batch.metadata["episode"]["mask"].item(), 1.0)

        observation_metadata = batch.metadata["observation"]
        self.assertEqual(
            observation_metadata["frame_stack"].shape,
            (1, 3, 3, 12, 16),
        )
        self.assertTrue(observation_metadata["frame_mask"].all().item())
        self.assertEqual(observation_metadata["frame_stack_size"], 3)
        self.assertEqual(observation_metadata["frame_skip"], 3)
        self.assertEqual(observation_metadata["resize_shape"], (12, 16))
        self.assertEqual(tuple(vision.observations[-1].shape), (12, 16, 3))
        self.assertLessEqual(float(vision.observations[-1].max()), 1.0)

    def test_emulator_state_round_trips_backend_and_adapter_state(self):
        env = SaveStateRetroEnv()
        stage = FullSMBStage(
            env=env,
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(8, 8),
            ),
        )
        try:
            stage.reset(seed=7)
            observation_1, _reward_1, _terminated_1, _truncated_1, _info_1 = (
                stage.step(SMBAction.RIGHT)
            )
            saved = stage.save_emulator_state()

            observation_2, reward_2, terminated_2, truncated_2, info_2 = stage.step(
                SMBAction.LEFT
            )
            batch_2 = stage.encode_observation(observation_2, info_2)

            restored_observation = stage.load_emulator_state(saved)
            self.assertEqual(env.step_count, 1)
            np.testing.assert_array_equal(restored_observation, observation_1)
            self.assertEqual(
                stage.last_info["full_smb_signals"],
                saved.last_info["full_smb_signals"],
            )
            np.testing.assert_allclose(
                stage.last_info["state_vec"],
                saved.last_info["state_vec"],
            )

            (
                replay_observation,
                replay_reward,
                replay_terminated,
                replay_truncated,
                replay_info,
            ) = stage.step(SMBAction.LEFT)
            replay_batch = stage.encode_observation(replay_observation, replay_info)
        finally:
            stage.close()

        np.testing.assert_array_equal(replay_observation, observation_2)
        self.assertEqual(replay_reward, reward_2)
        self.assertEqual(replay_terminated, terminated_2)
        self.assertEqual(replay_truncated, truncated_2)
        self.assertEqual(
            replay_info["full_smb_signals"],
            info_2["full_smb_signals"],
        )
        torch.testing.assert_close(
            replay_batch.metadata["observation"]["frame_stack"],
            batch_2.metadata["observation"]["frame_stack"],
        )
        self.assertTrue(
            torch.equal(
                replay_batch.metadata["observation"]["frame_mask"],
                batch_2.metadata["observation"]["frame_mask"],
            )
        )

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
