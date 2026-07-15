"""Tests for the Full SMB stable-retro stage adapter."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from retroagi.core import (
    GameSignals,
    SMBAction,
    StageBatch,
    VisionOutput,
    VisionSpec,
    full_smb_action,
)
from retroagi.stages.full_smb import (
    DEFAULT_FULL_SMB_CONTENT,
    DEFAULT_FULL_SMB_REWARD_CONFIG,
    FULL_SMB_REWARD_SCHEMA,
    FULL_SMB_SPEC,
    FullSMBContentSpec,
    FullSMBObservationConfig,
    FullSMBRewardConfig,
    FullSMBSignalExtractor,
    FullSMBSmokeConfig,
    FullSMBStage,
    extract_full_smb_signals,
    make_stable_retro_env,
    run_deterministic_reset_smoke,
    run_headless_random_agent_smoke,
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


class RewardSignalRetroEnv(GymnasiumRetroEnv):
    def reset(self, seed=None):
        self.reset_seed = seed
        return (
            np.zeros((224, 256, 3), dtype=np.uint8),
            {"x_pos": 10, "y_pos": 42, "score": 100, "coins": 1, "lives": 3},
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            np.ones((224, 256, 3), dtype=np.uint8),
            4.0,
            True,
            False,
            {
                "x_pos": 96,
                "y_pos": 120,
                "score": 250,
                "coins": 3,
                "lives": 3,
                "level_complete": True,
                "enemy_stomp": 2,
                "damage_taken": 1,
                "termination_reason": "level_complete",
            },
        )


class ZeroBackendProgressRetroEnv(GymnasiumRetroEnv):
    def reset(self, seed=None):
        self.reset_seed = seed
        return (
            np.zeros((224, 256, 3), dtype=np.uint8),
            {
                "xscrollHi": 0,
                "xscrollLo": 10,
                "score": 0,
                "coins": 0,
                "lives": 3,
            },
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            np.ones((224, 256, 3), dtype=np.uint8),
            0.0,
            False,
            False,
            {
                "xscrollHi": 0,
                "xscrollLo": 30,
                "score": 0,
                "coins": 0,
                "lives": 3,
            },
        )


class RamPositionRetroEnv(GymnasiumRetroEnv):
    def __init__(self):
        super().__init__()
        self.ram = np.zeros(2048, dtype=np.uint8)
        self.ram[0x6D] = 0
        self.ram[0x86] = 40
        self.ram[0xCE] = 176

    def get_ram(self):
        return self.ram.copy()

    def reset(self, seed=None):
        self.reset_seed = seed
        self.ram[0x6D] = 0
        self.ram[0x86] = 40
        self.ram[0xCE] = 176
        return (
            np.zeros((224, 256, 3), dtype=np.uint8),
            {"xscrollHi": 0, "xscrollLo": 0, "score": 0, "coins": 0, "lives": 3},
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.ram[0x6D] = 1
        self.ram[0x86] = 48
        self.ram[0xCE] = 160
        return (
            np.ones((224, 256, 3), dtype=np.uint8),
            0.0,
            False,
            False,
            {"xscrollHi": 0, "xscrollLo": 200, "score": 0, "coins": 0, "lives": 3},
        )


class PreprocessingRetroEnv(GymnasiumRetroEnv):
    def reset(self, seed=None):
        self.reset_seed = seed
        return (
            self._observation(0),
            {
                "xscrollHi": 1,
                "xscrollLo": 20,
                "screen_x": 2,
                "screen_y": 3,
                "y_pos": 90,
                "score": 0,
                "coins": 0,
                "lives": 3,
            },
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            self._observation(17),
            1.0,
            False,
            False,
            {
                "xscrollHi": 1,
                "xscrollLo": 40,
                "screen_x": 4,
                "screen_y": 5,
                "y_pos": 96,
                "score": 10,
                "coins": 1,
                "lives": 3,
            },
        )

    @staticmethod
    def _observation(offset):
        y = np.arange(10, dtype=np.int16).reshape(10, 1)
        x = np.arange(12, dtype=np.int16).reshape(1, 12)
        red = np.broadcast_to((y * 23 + offset) % 256, (10, 12))
        green = np.broadcast_to((x * 19 + offset) % 256, (10, 12))
        blue = ((y + x) * 11 + offset) % 256
        return np.stack(
            (
                red,
                green,
                blue,
            ),
            axis=-1,
        ).astype(np.uint8)


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


class LifeLossRetroEnv(GymnasiumRetroEnv):
    def reset(self, seed=None):
        self.reset_seed = seed
        return (
            np.zeros((16, 20, 3), dtype=np.uint8),
            {"xscrollHi": 0, "xscrollLo": 200, "score": 0, "coins": 0, "lives": 2},
        )

    def step(self, action):
        self.actions.append(np.asarray(action))
        return (
            np.ones((16, 20, 3), dtype=np.uint8),
            0.0,
            False,
            False,
            {"xscrollHi": 0, "xscrollLo": 0, "score": 0, "coins": 0, "lives": 1},
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


class SeededHeadlessRetroEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self, episode_length=8):
        self.episode_length = episode_length
        self.seed_value = 0
        self.step_count = 0
        self.actions = []
        self.render_calls = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed_value = 0 if seed is None else int(seed)
        self.step_count = 0
        return self._observation(0), self._info(terminated=False)

    def step(self, action):
        button_vector = np.asarray(action, dtype=np.int8)
        self.actions.append(button_vector.copy())
        self.step_count += 1
        action_value = int(np.flatnonzero(button_vector).sum())
        terminated = self.step_count >= self.episode_length
        reward = float(self.step_count + action_value)
        return (
            self._observation(action_value),
            reward,
            terminated,
            False,
            self._info(terminated=terminated),
        )

    def render(self):
        self.render_calls += 1

    def close(self):
        self.closed = True

    def _observation(self, action_value):
        height, width = 16, 20
        y = np.arange(height, dtype=np.uint16).reshape(height, 1)
        x = np.arange(width, dtype=np.uint16).reshape(1, width)
        base = (self.seed_value * 17 + self.step_count * 31 + action_value * 13 + x + y) % 256
        return np.stack(
            (
                base,
                (base * 2) % 256,
                (base * 3) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)

    def _info(self, *, terminated):
        return {
            "x_pos": self.seed_value + self.step_count * 4,
            "y_pos": 96 + self.step_count,
            "score": self.step_count * 100,
            "coins": self.step_count % 10,
            "lives": 3,
            "level_complete": bool(terminated),
            "termination_reason": "level_complete" if terminated else None,
        }


class TestFullSMBStage(unittest.TestCase):
    def make_seeded_headless_stage(self, *, episode_length=8):
        return FullSMBStage(
            env=SeededHeadlessRetroEnv(episode_length=episode_length),
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(8, 8),
            ),
        )

    def assert_reward_total_matches_terms(self, reward, info):
        terms = info["reward_terms"]
        summed_terms = sum(float(value) for name, value in terms.items() if name != "total")
        self.assertAlmostEqual(float(reward), float(terms["total"]))
        self.assertAlmostEqual(float(reward), summed_terms)
        self.assertAlmostEqual(float(info["reward_total"]), float(reward))

    def test_content_spec_documents_supported_local_rom_setup(self):
        spec = DEFAULT_FULL_SMB_CONTENT
        manifest = spec.to_manifest()

        self.assertIsInstance(spec, FullSMBContentSpec)
        self.assertEqual(manifest["game"], "SuperMarioBros-Nes")
        self.assertEqual(manifest["stable_retro_import_dir"], "local/full_smb/roms")
        self.assertEqual(
            manifest["stable_retro_import_command"],
            "python -m retro.import local/full_smb/roms",
        )
        self.assertEqual(manifest["checksum_algorithm"], "sha256")
        self.assertEqual(
            manifest["checksum_record_path"],
            "local/full_smb/checksums/SuperMarioBros-Nes.sha256",
        )
        self.assertIn("legally obtained", manifest["legal_notice"])
        self.assertIn("Do not commit", manifest["legal_notice"])

    def test_reward_config_declares_adapter_owned_full_smb_terms(self):
        config = DEFAULT_FULL_SMB_REWARD_CONFIG
        manifest = config.to_manifest()

        self.assertIsInstance(config, FullSMBRewardConfig)
        self.assertEqual(FULL_SMB_REWARD_SCHEMA.game_name, "full_smb_adapter")
        self.assertEqual(manifest["owner"], "full_smb_adapter")
        self.assertEqual(manifest["schema"], "full_smb_adapter")
        self.assertEqual(manifest["separated_from"], "BlockSMBRewardConfig")
        self.assertTrue(manifest["defaults_preserve_backend_reward"])
        self.assertEqual(
            tuple(manifest["terms"]),
            (
                "emulator_progress",
                "completion",
                "survival",
                "score",
                "coin",
                "enemy",
                "damage",
                "death",
                "frame_penalty",
            ),
        )
        self.assertEqual(manifest["terms"]["emulator_progress"], 1.0)
        for name in (
            "completion",
            "survival",
            "score",
            "coin",
            "enemy",
            "damage",
            "death",
            "frame_penalty",
        ):
            self.assertEqual(manifest["terms"][name], 0.0)
        self.assertEqual(
            manifest["term_signals"]["damage"]["direction"],
            "negative",
        )
        self.assertEqual(
            manifest["term_signals"]["coin"]["signal"],
            "full_smb_signals.coins",
        )

    def test_reward_config_validates_term_signs(self):
        config = FullSMBRewardConfig(
            emulator_progress=0.5,
            completion=25.0,
            survival=0.01,
            score=0.001,
            coin=1.0,
            enemy=2.0,
            damage=-4.0,
            death=-20.0,
            frame_penalty=-0.001,
        )

        self.assertEqual(config.as_dict()["death"], -20.0)
        with self.assertRaisesRegex(ValueError, "positive reward term"):
            FullSMBRewardConfig(emulator_progress=-1.0)
        with self.assertRaisesRegex(ValueError, "negative reward term"):
            FullSMBRewardConfig(death=1.0)

    def test_stage_reports_resolved_full_smb_reward_config(self):
        reward_config = FullSMBRewardConfig(
            emulator_progress=0.75,
            completion=10.0,
            death=-5.0,
            frame_penalty=-0.01,
        )
        env = GymnasiumRetroEnv()
        stage = FullSMBStage(
            env=env,
            vision=StaticFullSMBVision(),
            reward_config=reward_config,
        )
        try:
            reset_observation = stage.reset(seed=123)
            observation, reward, _terminated, _truncated, info = stage.step(SMBAction.RIGHT)
        finally:
            stage.close()

        self.assertEqual(stage.reward_config, reward_config)
        self.assertEqual(
            stage.last_info["reward_config"]["terms"],
            reward_config.as_dict(),
        )
        self.assertEqual(
            info["reward_config"]["terms"]["frame_penalty"],
            -0.01,
        )
        self.assertAlmostEqual(reward, 12.615)
        self.assertEqual(info["reward_terms"]["emulator_progress"], 2.625)
        self.assertEqual(info["reward_terms"]["completion"], 10.0)
        self.assertEqual(info["reward_terms"]["frame_penalty"], -0.01)
        self.assert_reward_total_matches_terms(reward, info)
        self.assertEqual(
            info["reward_config"]["separated_from"],
            "BlockSMBRewardConfig",
        )
        self.assertEqual(reset_observation.shape, observation.shape)

    def test_custom_reward_config_reports_terms_that_sum_to_scalar_reward(self):
        reward_config = FullSMBRewardConfig(
            emulator_progress=0.5,
            completion=10.0,
            survival=0.25,
            score=0.01,
            coin=3.0,
            enemy=2.5,
            damage=-4.0,
            death=-20.0,
            frame_penalty=-0.1,
        )
        stage = FullSMBStage(
            env=RewardSignalRetroEnv(),
            vision=StaticFullSMBVision(),
            reward_config=reward_config,
        )
        try:
            stage.reset(seed=8)
            _observation, reward, terminated, truncated, info = stage.step(SMBAction.RIGHT)
        finally:
            stage.close()

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["full_smb_signals"]["objectives"]["enemy"], 2.0)
        self.assertEqual(info["full_smb_signals"]["objectives"]["damage"], 1.0)
        expected_terms = {
            "emulator_progress": 2.0,
            "completion": 10.0,
            "survival": 0.25,
            "score": 1.5,
            "coin": 6.0,
            "enemy": 5.0,
            "damage": -4.0,
            "death": 0.0,
            "frame_penalty": -0.1,
            "total": 20.65,
        }
        for name, expected in expected_terms.items():
            self.assertAlmostEqual(info["reward_terms"][name], expected)
        self.assertAlmostEqual(reward, 20.65)
        self.assert_reward_total_matches_terms(reward, info)

    def test_zero_backend_reward_uses_signal_progress_delta(self):
        reward_config = FullSMBRewardConfig(emulator_progress=0.5)
        stage = FullSMBStage(
            env=ZeroBackendProgressRetroEnv(),
            vision=StaticFullSMBVision(),
            reward_config=reward_config,
        )
        try:
            stage.reset(seed=9)
            _observation, reward, _terminated, _truncated, info = stage.step(SMBAction.RIGHT)
        finally:
            stage.close()

        self.assertEqual(info["full_smb_signals"]["position"], (30.0, 0.0))
        self.assertEqual(info["full_smb_signals"]["progress"], 30.0)
        self.assertEqual(info["reward_terms"]["emulator_progress"], 0.0)
        self.assertEqual(info["reward_terms"]["signal_progress"], 10.0)
        self.assertAlmostEqual(reward, 10.0)
        self.assert_reward_total_matches_terms(reward, info)

    def test_stage_annotates_ram_backed_vision_position_target(self):
        stage = FullSMBStage(env=RamPositionRetroEnv(), vision=StaticFullSMBVision())
        try:
            stage.reset(seed=3)
            np.testing.assert_allclose(
                stage.last_info["vision_position_target"],
                np.asarray([40.0 / 256.0, 176.0 / 240.0], dtype=np.float32),
            )
            self.assertEqual(
                stage.last_info["full_smb_signals"]["position"],
                (40.0, 176.0),
            )
            self.assertEqual(stage.last_info["full_smb_signals"]["progress"], 40.0)
            _observation, _reward, _terminated, _truncated, info = stage.step(
                SMBAction.RIGHT
            )
        finally:
            stage.close()

        # After the camera scrolls (scroll_x=200), the target is Mario's true
        # screen x (level_x - scroll_x = 304 - 200 = 104), not the raw
        # page-relative RAM byte (48).
        np.testing.assert_allclose(
            info["vision_position_target"],
            np.asarray([104.0 / 256.0, 160.0 / 240.0], dtype=np.float32),
        )
        self.assertEqual(
            info["vision_position_target_raw"]["player_page_x_address"],
            0x86,
        )
        self.assertEqual(
            info["vision_position_target_raw"]["player_screen_y_address"],
            0xCE,
        )
        self.assertEqual(
            info["vision_position_target_raw"]["player_level_page_address"],
            0x6D,
        )
        self.assertEqual(info["vision_position_target_raw"]["player_page_x"], 48.0)
        self.assertEqual(info["vision_position_target_raw"]["scroll_x"], 200.0)
        self.assertEqual(info["player_screen_x"], 104.0)
        self.assertEqual(info["player_screen_y"], 160.0)
        self.assertEqual(info["player_level_page"], 1.0)
        self.assertEqual(info["player_level_x"], 304.0)
        self.assertEqual(info["full_smb_signals"]["position"], (304.0, 160.0))
        self.assertEqual(info["full_smb_signals"]["progress"], 304.0)
        # camera_vec screen-x (element 1) must agree with player_camera_x
        # (element 3), both derived from the same scroll source.
        self.assertAlmostEqual(float(info["camera_vec"][1]), float(info["camera_vec"][3]))
        self.assertAlmostEqual(float(info["camera_vec"][1]), 104.0 / 256.0, places=6)

    def test_make_stable_retro_env_reports_missing_backend_setup(self):
        with patch.dict(sys.modules, {"retro": None}):
            with self.assertRaisesRegex(
                RuntimeError,
                "stable-retro is not installed",
            ) as cm:
                make_stable_retro_env()

        message = str(cm.exception)
        self.assertIn("SuperMarioBros-Nes", message)
        self.assertIn("python -m pip install -e '.[full-smb]'", message)
        self.assertIn("python -m retro.import local/full_smb/roms", message)
        self.assertIn("local/full_smb/checksums/SuperMarioBros-Nes.sha256", message)
        self.assertIn("legally obtained", message)

    def test_make_stable_retro_env_reports_missing_imported_rom_setup(self):
        def raise_missing_game(**_kwargs):
            raise FileNotFoundError("game data not found")

        retro = SimpleNamespace(make=raise_missing_game)
        with patch.dict(sys.modules, {"retro": retro}):
            with self.assertRaisesRegex(RuntimeError, "game data not found") as cm:
                make_stable_retro_env()

        message = str(cm.exception)
        self.assertIn("Required content", message)
        self.assertIn("SuperMarioBros-Nes", message)
        self.assertIn("Local ROM staging directory", message)
        self.assertIn("Checksum record", message)

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
        self.assertIsInstance(signals, GameSignals)
        self.assertEqual(signals.progress, 144.0)
        self.assertEqual(signals.score, 1234)
        self.assertEqual(signals.coins, 12)
        self.assertEqual(signals.collectibles["coins"], 12)
        self.assertEqual(signals.lives, 1)
        self.assertIsNone(signals.screen)
        self.assertIsNone(signals.level)
        self.assertIsNone(signals.power_state)
        self.assertFalse(signals.game_over)
        self.assertFalse(signals.completion)
        self.assertTrue(signals.death)
        self.assertTrue(signals.terminated)
        self.assertFalse(signals.truncated)
        self.assertFalse(signals.timeout)

    def test_signal_extractor_reads_nested_memory_variables(self):
        signals = extract_full_smb_signals(
            {
                "memory": {
                    "x_pos": 512,
                    "y_pos": 88,
                    "screen": (2, 1),
                    "world": 2,
                    "stage": 1,
                    "score": 3210,
                    "coins": 3,
                    "lives": 2,
                    "power_state": "fireball",
                    "flag_get": True,
                }
            },
            terminated=True,
            truncated=False,
        )

        self.assertEqual(signals.position, (512.0, 88.0))
        self.assertEqual(signals.screen, (2, 1))
        self.assertEqual(signals.level, "2-1")
        self.assertEqual(signals.world, 2)
        self.assertEqual(signals.stage, 1)
        self.assertEqual(signals.score, 3210)
        self.assertEqual(signals.coins, 3)
        self.assertEqual(signals.lives, 2)
        self.assertEqual(signals.power_state, "fire")
        self.assertTrue(signals.completion)
        self.assertFalse(signals.game_over)

    def test_signal_extractor_reads_timeout_power_and_game_over_state(self):
        signals = extract_full_smb_signals(
            {
                "variables": {
                    "game_over": 1,
                    "time_up": "true",
                    "status": 0,
                },
                "done_reason": "game over",
            },
            terminated=True,
            truncated=False,
        )

        self.assertTrue(signals.game_over)
        self.assertTrue(signals.death)
        self.assertTrue(signals.timeout)
        self.assertEqual(signals.power_state, "small")

    def test_signal_extractor_marks_negative_lives_as_death(self):
        signals = extract_full_smb_signals(
            {
                "xscrollHi": 0,
                "xscrollLo": 0,
                "score": 0,
                "coins": 0,
                "lives": -1,
            },
            terminated=True,
            truncated=False,
        )

        self.assertEqual(signals.lives, -1)
        self.assertTrue(signals.death)
        self.assertFalse(signals.completion)

    def test_full_smb_signal_extractor_satisfies_game_signal_contract(self):
        extractor = FullSMBSignalExtractor()
        signals = extractor.extract(
            {
                "xscrollHi": 1,
                "xscrollLo": 2,
                "screen_x": 3,
                "y_pos": 88,
                "score": 500,
                "coins": 4,
                "lives": 2,
                "goal_reached": True,
            },
            terminated=True,
            truncated=False,
        )

        self.assertEqual(extractor.game_name, "smb")
        self.assertIsInstance(signals, GameSignals)
        self.assertEqual(signals.position, (261.0, 88.0))
        self.assertEqual(signals.screen, (3, 0))
        self.assertEqual(signals.progress, 261.0)
        self.assertEqual(signals.score, 500)
        self.assertEqual(signals.collectibles, {"coins": 4})
        self.assertEqual(signals.lives, 2)
        self.assertTrue(signals.completion)
        self.assertFalse(signals.death)
        self.assertTrue(signals.terminated)
        self.assertFalse(signals.timeout)

        timeout = extractor.extract({}, terminated=False, truncated=True)
        self.assertTrue(timeout.timeout)
        self.assertTrue(timeout.truncated)

    def test_signal_extractor_reads_scroll_position_without_y_target(self):
        signals = extract_full_smb_signals(
            {
                "xscrollHi": 1,
                "xscrollLo": 44,
                "score": 100,
                "coins": 7,
                "lives": 2,
            },
            terminated=False,
            truncated=False,
        )

        self.assertEqual(signals.position, (300.0, 0.0))
        self.assertEqual(signals.progress, 300.0)
        self.assertEqual(signals.score, 100)
        self.assertEqual(signals.coins, 7)

    def test_signal_extractor_prefers_absolute_scroll_pair_over_coarse_scroll(self):
        signals = extract_full_smb_signals(
            {
                "scrolling": 16,
                "xscrollHi": 0,
                "xscrollLo": 68,
                "player_screen_x": 180,
                "player_screen_y": 176,
                "score": 0,
                "coins": 0,
                "lives": 2,
            },
            terminated=False,
            truncated=False,
        )

        self.assertEqual(signals.position, (248.0, 176.0))
        self.assertEqual(signals.progress, 248.0)
        self.assertEqual(signals.screen, (180, 176))

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
            observation, reward, terminated, truncated, info = stage.step(SMBAction.RIGHT)
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

    def test_preprocessing_contract_crops_hud_grayscales_and_encodes_camera(self):
        env = PreprocessingRetroEnv()
        vision = StaticFullSMBVision()
        stage = FullSMBStage(
            env=env,
            vision=vision,
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(4, 5),
                crop_margins=(1, 2, 1, 1),
                hud_policy="crop",
                hud_crop_top=1,
                color_mode="grayscale",
                normalization_mean=(0.5, 0.5, 0.5),
                normalization_std=(0.5, 0.5, 0.5),
                include_camera_state=True,
            ),
        )
        try:
            stage.reset(seed=3)
            observation, _reward, _terminated, _truncated, info = stage.step(SMBAction.RIGHT)
            batch = stage.encode_observation(observation, info)
        finally:
            stage.close()

        vision_frame = vision.observations[-1]
        self.assertEqual(tuple(vision_frame.shape), (4, 5, 3))
        torch.testing.assert_close(vision_frame[..., 0], vision_frame[..., 1])
        torch.testing.assert_close(vision_frame[..., 0], vision_frame[..., 2])
        self.assertGreaterEqual(float(vision_frame.min()), -1.0)
        self.assertLessEqual(float(vision_frame.max()), 1.0)

        observation_metadata = batch.metadata["observation"]
        self.assertEqual(observation_metadata["resize_shape"], (4, 5))
        self.assertEqual(observation_metadata["crop_margins"], (1, 2, 1, 1))
        self.assertEqual(observation_metadata["effective_crop_margins"], (2, 2, 1, 1))
        self.assertEqual(observation_metadata["hud_policy"], "crop")
        self.assertEqual(observation_metadata["color_mode"], "grayscale")
        self.assertEqual(
            observation_metadata["normalization"],
            {
                "input_range": (0.0, 1.0),
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
            },
        )
        self.assertTrue(observation_metadata["camera_state_enabled"])
        self.assertEqual(
            observation_metadata["frame_stack"].shape,
            (1, 2, 3, 4, 5),
        )
        self.assertTrue(observation_metadata["frame_mask"].all().item())

        expected_camera = torch.tensor(
            [[296.0 / 4096.0, 4.0 / 256.0, 5.0 / 240.0, 4.0 / 256.0]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            observation_metadata["camera_vec"],
            expected_camera,
        )
        np.testing.assert_allclose(info["camera_vec"], expected_camera.numpy()[0])
        self.assertEqual(batch.metadata["vision_fusion"]["c_state"], (8, 21))
        torch.testing.assert_close(batch.src_c[:, 17:21], expected_camera)

    def test_observation_config_rejects_invalid_preprocessing_values(self):
        with self.assertRaisesRegex(ValueError, "crop_margins"):
            FullSMBObservationConfig(crop_margins=(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "hud_policy"):
            FullSMBObservationConfig(hud_policy="mask")
        with self.assertRaisesRegex(ValueError, "color_mode"):
            FullSMBObservationConfig(color_mode="hsv")
        with self.assertRaisesRegex(ValueError, "normalization_std"):
            FullSMBObservationConfig(normalization_std=(1.0, 0.0, 1.0))

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
            observation_1, _reward_1, _terminated_1, _truncated_1, _info_1 = stage.step(
                SMBAction.RIGHT
            )
            saved = stage.save_emulator_state()

            observation_2, reward_2, terminated_2, truncated_2, info_2 = stage.step(SMBAction.LEFT)
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

            next_observation, reward, terminated, truncated, info = stage.step(SMBAction.RIGHT_JUMP)
            expected_action = full_smb_action(SMBAction.RIGHT_JUMP, env.buttons)
            expected_action[env.buttons.index("B")] = 1
            np.testing.assert_array_equal(
                env.actions[-1],
                expected_action,
            )
            self.assertEqual(next_observation.dtype, np.uint8)
            self.assertEqual(reward, 3.5)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["action"]["shared_name"], "RIGHT_JUMP")
            self.assertTrue(info["action"]["hold_run_button"])
            self.assertEqual(
                info["action"]["button_vector"][env.buttons.index("B")],
                1,
            )
            self.assertEqual(
                info["full_smb_signals"],
                {
                    "position": (306.0, 180.0),
                    "progress": 306.0,
                    "score": 100,
                    "health": None,
                    "inventory": {},
                    "collectibles": {"coins": 7},
                    "lives": 2,
                    "completion": True,
                    "death": False,
                    "timeout": False,
                    "terminated": True,
                    "truncated": False,
                    "objectives": {},
                    "termination_reason": "level_complete",
                    "coins": 7,
                    "screen": (6, 0),
                    "level": None,
                    "world": None,
                    "stage": None,
                    "power_state": None,
                    "game_over": False,
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

    def test_stage_marks_life_loss_as_terminal_death_boundary(self):
        env = LifeLossRetroEnv()
        stage = FullSMBStage(
            env=env,
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(16, 20),
            ),
            reward_config=FullSMBRewardConfig(death=-7.0),
        )
        try:
            observation = stage.reset(seed=123)
            next_observation, reward, terminated, truncated, info = stage.step(
                SMBAction.RIGHT
            )
            signals = info["full_smb_signals"]
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(signals["death"])
            self.assertTrue(signals["terminated"])
            self.assertEqual(signals["termination_reason"], "life_lost")
            self.assertEqual(reward, -7.0)
            self.assertEqual(info["reward_terms"]["death"], -7.0)
            batch = stage.encode_observation(next_observation, info)
            self.assertEqual(batch.metadata["episode"]["mask"].item(), 0.0)
            self.assertEqual(observation.shape, next_observation.shape)
        finally:
            stage.close()
        self.assertTrue(env.closed)

    def test_stage_can_disable_run_button_augmentation(self):
        env = GymnasiumRetroEnv()
        stage = FullSMBStage(
            env=env,
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(hold_run_button=False),
        )
        try:
            stage.reset(seed=123)
            _observation, _reward, _terminated, _truncated, info = stage.step(
                SMBAction.RIGHT_JUMP
            )
            np.testing.assert_array_equal(
                env.actions[-1],
                full_smb_action(SMBAction.RIGHT_JUMP, env.buttons),
            )
            self.assertFalse(info["action"]["hold_run_button"])
        finally:
            stage.close()
        self.assertTrue(env.closed)

    def test_legacy_done_api_preserves_timeout_as_truncation(self):
        env = LegacyRetroEnv()
        stage = FullSMBStage(env=env, vision=StaticFullSMBVision())
        try:
            stage.reset()
            _observation, reward, terminated, truncated, _info = stage.step(SMBAction.NOOP)
        finally:
            stage.close()

        self.assertEqual(reward, -1.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(_info["full_smb_signals"]["terminated"])
        self.assertTrue(_info["full_smb_signals"]["truncated"])

    def test_headless_random_agent_smoke_runs_without_rendering(self):
        stage = self.make_seeded_headless_stage(episode_length=3)
        env = stage.env
        try:
            result = run_headless_random_agent_smoke(
                stage,
                FullSMBSmokeConfig(
                    steps=5,
                    seed=11,
                    encode_observations=True,
                    reset_on_done=True,
                ),
            )
        finally:
            stage.close()

        self.assertEqual(result.requested_steps, 5)
        self.assertEqual(result.executed_steps, 5)
        self.assertEqual(len(result.action_ids), 5)
        self.assertEqual(result.resets, 2)
        self.assertEqual(result.completed_episodes, 1)
        self.assertEqual(result.terminated_count, 1)
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(result.encoded_observations, 7)
        self.assertEqual(env.render_calls, 0)
        self.assertTrue(env.closed)
        self.assertEqual(result.final_signals["terminated"], False)

    def test_deterministic_reset_smoke_compares_seeded_rollouts(self):
        result = run_deterministic_reset_smoke(
            lambda: self.make_seeded_headless_stage(episode_length=10),
            seed=23,
            steps=6,
            encode_observations=True,
        )

        self.assertTrue(result.deterministic, result.mismatch)
        self.assertIsNone(result.mismatch)
        self.assertEqual(result.first, result.second)
        self.assertEqual(len(result.first.action_ids), 6)
        self.assertEqual(result.first.encoded_observations, 7)


if __name__ == "__main__":
    unittest.main()
