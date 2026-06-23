"""Tests for shared backend provider contracts."""

import unittest

import numpy as np

from retroagi.core import (
    BACKEND_PROVIDER_KINDS,
    BackendCapabilityProbeConfig,
    BackendCapabilitySpec,
    GameBackendSpec,
    GymnasiumBackendAdapter,
    probe_backend_capabilities,
)


class GymnasiumEnv:
    buttons = ("LEFT", "RIGHT", "A")

    def __init__(self):
        self.reset_seed = None
        self.actions = []
        self.closed = False

    def reset(self, seed=None):
        self.reset_seed = seed
        return "obs0", {"reset": True}

    def step(self, action):
        self.actions.append(action)
        return "obs1", 2.5, True, False, {"score": 10}

    def close(self):
        self.closed = True


class LegacySeedEnv:
    buttons = ("A",)

    def __init__(self):
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return "legacy_obs"

    def step(self, action):
        return "legacy_step", -1.0, True, {"TimeLimit.truncated": True}


class EnvStateOwner:
    buttons = ("A",)

    def __init__(self):
        self.state = {"tick": 0}

    def reset(self):
        return "obs"

    def step(self, action):
        return "obs", 0.0, False, False, {}

    def get_state(self):
        return dict(self.state)

    def set_state(self, state):
        self.state = dict(state)


class EmulatorStateOwner:
    def __init__(self):
        self.state = {"em_tick": 0}

    def get_state(self):
        return dict(self.state)

    def set_state(self, state):
        self.state = dict(state)


class EnvWithEmulatorState:
    buttons = ("A",)

    def __init__(self):
        self.em = EmulatorStateOwner()

    def reset(self):
        return "obs"

    def step(self, action):
        return "obs", 0.0, False, False, {}


class DeterministicCapabilityEnv:
    buttons = ("LEFT", "RIGHT")

    def __init__(self):
        self.seed_value = 0
        self.position = 0
        self.step_count = 0
        self.render_calls = 0

    def reset(self, seed=None):
        self.seed_value = 0 if seed is None else int(seed)
        self.position = self.seed_value % 11
        self.step_count = 0
        return self._observation(), {"seed": self.seed_value}

    def step(self, action):
        action_value = int(np.asarray(action).sum())
        self.step_count += 1
        self.position = (self.position * 3 + action_value + 1) % 257
        return (
            self._observation(),
            float(self.position) / 10.0,
            False,
            False,
            {"position": self.position, "step_count": self.step_count},
        )

    def render(self):
        self.render_calls += 1
        return self._observation().reshape(1, 3)

    def get_state(self):
        return {
            "seed_value": self.seed_value,
            "position": self.position,
            "step_count": self.step_count,
        }

    def set_state(self, state):
        self.seed_value = int(state["seed_value"])
        self.position = int(state["position"])
        self.step_count = int(state["step_count"])

    def _observation(self):
        return np.asarray(
            [self.seed_value, self.position, self.step_count],
            dtype=np.int64,
        )


class MissingCapabilityEnv:
    buttons = ("A",)

    def reset(self, seed=None):
        return "obs", {"seed": seed}

    def step(self, action):
        return "next", 0.0, False, False, {}


class TestBackendContracts(unittest.TestCase):
    def test_backend_spec_manifest_declares_provider_and_capabilities(self):
        spec = GameBackendSpec(
            name="stable-retro",
            provider_kind="stable_retro",
            entrypoint="retro.make",
            observation_api="RGB frame",
            action_api="MultiBinary button vector",
            capabilities=BackendCapabilitySpec(
                reset_seed=True,
                save_load_state=True,
                frame_step=True,
                action_repeat=True,
                render=True,
                headless=True,
                legacy_gym_step_api=True,
            ),
            metadata={"game": "SuperMarioBros-Nes"},
        )

        manifest = spec.to_manifest()

        self.assertIn("stable_retro", BACKEND_PROVIDER_KINDS)
        self.assertEqual(manifest["provider_kind"], "stable_retro")
        self.assertEqual(manifest["entrypoint"], "retro.make")
        self.assertTrue(manifest["capabilities"]["save_load_state"])
        self.assertTrue(manifest["capabilities"]["legacy_gym_step_api"])
        self.assertEqual(manifest["metadata"]["game"], "SuperMarioBros-Nes")

    def test_backend_spec_rejects_unknown_provider_kind(self):
        with self.assertRaisesRegex(ValueError, "provider_kind"):
            GameBackendSpec(
                name="bad",
                provider_kind="unknown",
                entrypoint="unit.make",
                observation_api="obs",
                action_api="action",
            )

    def test_gymnasium_adapter_normalizes_reset_step_buttons_and_close(self):
        env = GymnasiumEnv()
        adapter = GymnasiumBackendAdapter(env, context="unit backend")

        reset = adapter.reset(seed=123)
        step = adapter.step([0, 1, 1])
        adapter.close()

        self.assertEqual(adapter.buttons, ("LEFT", "RIGHT", "A"))
        self.assertEqual(reset.observation, "obs0")
        self.assertEqual(reset.info, {"reset": True})
        self.assertEqual(env.reset_seed, 123)
        self.assertEqual(step.observation, "obs1")
        self.assertEqual(step.reward, 2.5)
        self.assertTrue(step.terminated)
        self.assertFalse(step.truncated)
        self.assertEqual(step.info, {"score": 10})
        self.assertEqual(env.actions, [[0, 1, 1]])
        self.assertTrue(env.closed)

    def test_gymnasium_adapter_supports_legacy_seed_and_done_api(self):
        env = LegacySeedEnv()
        adapter = GymnasiumBackendAdapter(env)

        reset = adapter.reset(seed=77)
        step = adapter.step([1])

        self.assertEqual(env.seed_value, 77)
        self.assertEqual(reset.observation, "legacy_obs")
        self.assertEqual(reset.info, {})
        self.assertEqual(step.observation, "legacy_step")
        self.assertEqual(step.reward, -1.0)
        self.assertFalse(step.terminated)
        self.assertTrue(step.truncated)

    def test_gymnasium_adapter_uses_env_or_emulator_state_api(self):
        env_owner = EnvStateOwner()
        env_adapter = GymnasiumBackendAdapter(env_owner)
        env_adapter.set_state({"tick": 4})

        em_owner = EnvWithEmulatorState()
        em_adapter = GymnasiumBackendAdapter(em_owner)
        em_adapter.set_state({"em_tick": 9})

        self.assertEqual(env_adapter.get_state(), {"tick": 4})
        self.assertEqual(em_adapter.get_state(), {"em_tick": 9})

    def test_capability_probe_covers_deterministic_backend_matrix(self):
        env = DeterministicCapabilityEnv()
        adapter = GymnasiumBackendAdapter(env, context="deterministic backend")

        report = probe_backend_capabilities(
            adapter,
            BackendCapabilityProbeConfig(seed=42, action=2, action_repeat=3),
        )
        manifest = report.to_manifest()

        self.assertTrue(report.passed)
        self.assertEqual(manifest["failures"], {})
        self.assertEqual(
            {
                key: manifest[key]
                for key in (
                    "reset_seed",
                    "save_load_state",
                    "frame_step",
                    "action_repeat",
                    "render",
                    "headless",
                )
            },
            {
                "reset_seed": True,
                "save_load_state": True,
                "frame_step": True,
                "action_repeat": True,
                "render": True,
                "headless": True,
            },
        )
        self.assertEqual(env.render_calls, 1)

    def test_capability_probe_reports_missing_backend_features(self):
        report = probe_backend_capabilities(
            GymnasiumBackendAdapter(MissingCapabilityEnv(), context="limited backend"),
            BackendCapabilityProbeConfig(seed=1, action=0),
        )

        self.assertFalse(report.passed)
        self.assertTrue(report.reset_seed)
        self.assertTrue(report.frame_step)
        self.assertTrue(report.headless)
        self.assertFalse(report.save_load_state)
        self.assertFalse(report.action_repeat)
        self.assertFalse(report.render)
        self.assertIn("save_load_state", report.failures)
        self.assertIn("action_repeat", report.failures)
        self.assertIn("render", report.failures)


if __name__ == "__main__":
    unittest.main()
