"""Local Full SMB integration tests that require real stable-retro content."""

import unittest
from typing import Any

from retroagi.stages.full_smb import (
    FULL_SMB_PERCEPTION_REPLACE,
    FullSMBEnvConfig,
    FullSMBEnvironmentCheckConfig,
    FullSMBObservationConfig,
    FullSMBPlayConfig,
    FullSMBSegmentationVision,
    FullSMBSmokeConfig,
    FullSMBStage,
    FullSMBTrainingConfig,
    evaluate_full_smb_policy,
    play_full_smb_policy,
    run_full_smb_environment_check,
    run_headless_random_agent_smoke,
)
from retroagi.stages.full_smb.transfer import make_full_smb_policy_model


class _NoopVision:
    def encode(self, _observation: Any) -> Any:
        raise RuntimeError("local reset/step smoke does not encode observations")


class TestFullSMBLocalIntegration(unittest.TestCase):
    """Smoke real Full SMB content paths when local stable-retro setup exists."""

    env_config = FullSMBEnvConfig()
    observation_config = FullSMBObservationConfig(
        frame_skip=1,
        frame_stack=2,
        resize_shape=(16, 20),
    )

    @classmethod
    def setUpClass(cls):
        check_config = FullSMBEnvironmentCheckConfig(
            seed=31,
            steps=1,
            frame_skip=1,
            env_config=cls.env_config,
        )
        result = run_full_smb_environment_check(check_config)
        if not result.passed:
            raise unittest.SkipTest(_integration_skip_reason(result.as_dict()))

    def test_real_stable_retro_reset_and_step_smoke(self):
        stage = self._make_stage(_NoopVision())
        try:
            result = run_headless_random_agent_smoke(
                stage,
                FullSMBSmokeConfig(
                    steps=1,
                    seed=31,
                    encode_observations=False,
                    reset_on_done=False,
                    render=False,
                ),
            )
        finally:
            stage.close()

        self.assertEqual(result.executed_steps, 1)
        self.assertEqual(len(result.action_ids), 1)
        self.assertEqual(result.resets, 1)
        self.assertTrue(result.initial_observation_checksum)
        self.assertTrue(result.final_observation_checksum)
        self.assertIsInstance(result.final_signals, dict)

    def test_real_stable_retro_evaluate_and_play_smoke(self):
        model = make_full_smb_policy_model(
            architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
        )
        vision = FullSMBSegmentationVision(
            checkpoint=None,
            dim=16,
            depth=1,
            heads=4,
            drop=0.0,
            freeze=True,
        )
        runtime_config = FullSMBTrainingConfig(
            seed=37,
            architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
            evaluation_episodes=1,
            evaluation_max_steps=1,
            device="cpu",
            full_smb_vision_checkpoint=None,
            perception_mode=FULL_SMB_PERCEPTION_REPLACE,
            frame_skip=1,
            game_id=self.env_config.game,
            emulator_state=self.env_config.state,
            scenario=self.env_config.scenario,
        )

        evaluation = evaluate_full_smb_policy(
            model,
            config=runtime_config,
            make_stage=self._make_stage,
            vision=vision,
        )
        playback = play_full_smb_policy(
            model,
            config=runtime_config,
            play_config=FullSMBPlayConfig(
                max_steps=1,
                render=False,
                fps=0.0,
                interactive_controls=False,
                deterministic_policy=True,
            ),
            make_stage=self._make_stage,
            vision=vision,
        )

        self.assertEqual(evaluation.episodes, 1)
        self.assertEqual(evaluation.steps, 1)
        self.assertEqual(playback.steps, 1)
        self.assertEqual(playback.resets, 1)
        self.assertEqual(playback.control_mode, "policy")
        self.assertFalse(playback.render)

    @classmethod
    def _make_stage(cls, vision):
        return FullSMBStage(
            env_config=cls.env_config,
            observation_config=cls.observation_config,
            vision=vision,
        )


def _integration_skip_reason(report: dict[str, Any]) -> str:
    checks = report.get("checks", {})
    if isinstance(checks, dict):
        for name, check in checks.items():
            if isinstance(check, dict) and not check.get("passed", False):
                reason = check.get("reason") or "check did not pass"
                return f"Full SMB local integration content unavailable: {name}: {reason}"
    return "Full SMB local integration content unavailable"


if __name__ == "__main__":
    unittest.main()
