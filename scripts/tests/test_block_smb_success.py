"""Tests for Block SMB fixed-scenario success thresholds."""

import unittest

from retroagi.stages.block_smb import (
    FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS,
    evaluate_fixed_success_thresholds,
    evaluate_success_threshold,
)


class TestBlockSMBSuccessThresholds(unittest.TestCase):
    def test_thresholds_cover_every_fixed_scenario(self):
        self.assertEqual(
            set(FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS),
            {
                "level_1_flat.json",
                "level_2_gap.json",
                "level_3_stairs.json",
                "level_4_platforms.json",
            },
        )
        for threshold in FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS.values():
            self.assertEqual(threshold.min_success_rate, 1.0)
            self.assertEqual(threshold.min_episodes, 3)
            self.assertEqual(threshold.max_steps, 200)
            self.assertGreaterEqual(threshold.min_mean_return, 55.0)

    def test_threshold_diagnostics_explain_pass_and_fail_reasons(self):
        passing = evaluate_success_threshold(
            "level_1_flat.json",
            {"success_rate": 1.0, "return": 60.0},
            evaluation_episodes=3,
            evaluation_max_steps=200,
        )
        self.assertTrue(passing["threshold_met"])
        self.assertTrue(passing["meets_success_rate"])
        self.assertTrue(passing["meets_return"])

        too_few_episodes = evaluate_success_threshold(
            "level_1_flat.json",
            {"success_rate": 1.0, "return": 60.0},
            evaluation_episodes=1,
            evaluation_max_steps=200,
        )
        self.assertFalse(too_few_episodes["threshold_met"])
        self.assertFalse(too_few_episodes["enough_episodes"])

        low_return = evaluate_success_threshold(
            "level_1_flat.json",
            {"success_rate": 1.0, "return": 40.0},
            evaluation_episodes=3,
            evaluation_max_steps=200,
        )
        self.assertFalse(low_return["threshold_met"])
        self.assertFalse(low_return["meets_return"])

    def test_fixed_threshold_evaluation_returns_per_scenario_results(self):
        results = evaluate_fixed_success_thresholds(
            {
                "level_1_flat.json": {"success_rate": 1.0, "return": 60.0},
                "level_2_gap.json": {"success_rate": 0.0, "return": -1.0},
            },
            evaluation_episodes=3,
            evaluation_max_steps=200,
        )

        self.assertTrue(results["level_1_flat.json"]["threshold_met"])
        self.assertFalse(results["level_2_gap.json"]["threshold_met"])


if __name__ == "__main__":
    unittest.main()
