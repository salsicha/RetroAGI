"""Tests for Full SMB environment capability checks."""

import unittest
from types import SimpleNamespace

import numpy as np

from retroagi.stages.full_smb import (
    FullSMBEnvironmentCheckConfig,
    FullSMBObservationConfig,
    FullSMBStage,
    run_full_smb_environment_check,
)


class CapabilityRetroEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self):
        self.seed_value = 0
        self.step_count = 0
        self.position = 0
        self.render_calls = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed_value = 0 if seed is None else int(seed)
        self.step_count = 0
        self.position = self.seed_value % 17
        return self._observation(), self._info()

    def step(self, action):
        action_value = int(np.asarray(action, dtype=np.int8).sum())
        self.step_count += 1
        self.position = (self.position * 5 + action_value + self.step_count) % 997
        return (
            self._observation(),
            float(self.position) / 100.0,
            False,
            False,
            self._info(),
        )

    def render(self):
        self.render_calls += 1
        return self._observation()

    def get_state(self):
        return {
            "seed_value": self.seed_value,
            "step_count": self.step_count,
            "position": self.position,
        }

    def set_state(self, state):
        self.seed_value = int(state["seed_value"])
        self.step_count = int(state["step_count"])
        self.position = int(state["position"])

    def close(self):
        self.closed = True

    def _observation(self):
        base = (self.seed_value + self.step_count + self.position) % 256
        return np.full((16, 16, 3), base, dtype=np.uint8)

    def _info(self):
        return {
            "x_pos": self.position,
            "y_pos": 96,
            "score": self.step_count * 10,
            "coins": self.step_count,
            "lives": 3,
        }


class NoopVision:
    def encode(self, _observation):
        raise AssertionError("environment capability checks should not encode vision")


class TestFullSMBEnvironmentCapabilities(unittest.TestCase):
    def make_stage_factory(self, config):
        def make_stage():
            return FullSMBStage(
                env=CapabilityRetroEnv(),
                vision=NoopVision(),
                observation_config=FullSMBObservationConfig(
                    frame_skip=config.frame_skip,
                    frame_stack=2,
                    resize_shape=None,
                ),
            )

        return make_stage

    @staticmethod
    def retro_module():
        return SimpleNamespace(
            __name__="retro",
            make=lambda **_kwargs: CapabilityRetroEnv(),
            data=SimpleNamespace(list_games=lambda: ["SuperMarioBros-Nes"]),
        )

    def test_environment_check_passes_with_seeded_backend(self):
        config = FullSMBEnvironmentCheckConfig(seed=13, steps=4, frame_skip=3)
        result = run_full_smb_environment_check(
            config,
            retro_module_factory=self.retro_module,
            stage_factory=self.make_stage_factory(config),
        )
        manifest = result.as_dict()

        self.assertTrue(result.passed, manifest["checks"])
        self.assertTrue(manifest["backend_probe"]["passed"])
        self.assertTrue(manifest["deterministic_reset"]["deterministic"])
        for check in (
            "backend_import",
            "game_registration",
            "rom_availability",
            "headless_reset",
            "render_reset",
            "save_load_state",
            "action_step",
            "frame_skip",
            "deterministic_seeding",
        ):
            self.assertTrue(manifest["checks"][check]["passed"], check)
        self.assertEqual(
            manifest["checks"]["frame_skip"]["details"]["frame_skip"],
            3,
        )
        self.assertEqual(manifest["content"]["game"], "SuperMarioBros-Nes")

    def test_environment_check_reports_missing_backend_without_stage_creation(self):
        calls = []

        def missing_retro():
            raise ModuleNotFoundError("No module named 'retro'")

        def make_stage():
            calls.append("called")
            return FullSMBStage(env=CapabilityRetroEnv(), vision=NoopVision())

        result = run_full_smb_environment_check(
            FullSMBEnvironmentCheckConfig(),
            retro_module_factory=missing_retro,
            stage_factory=make_stage,
        )
        manifest = result.as_dict()

        self.assertFalse(result.passed)
        self.assertEqual(calls, [])
        self.assertFalse(manifest["checks"]["backend_import"]["passed"])
        self.assertIn("stable-retro setup failed", manifest["checks"]["backend_import"]["reason"])
        self.assertFalse(manifest["checks"]["rom_availability"]["passed"])
        self.assertFalse(manifest["checks"]["deterministic_seeding"]["passed"])


if __name__ == "__main__":
    unittest.main()
