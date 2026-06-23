"""Tests for Full SMB fixed-task success thresholds."""

import unittest

from retroagi.stages.full_smb import (
    FIXED_FULL_SMB_SUCCESS_THRESHOLDS,
    evaluate_fixed_full_smb_success_thresholds,
    evaluate_full_smb_success_threshold,
    full_smb_task_catalog,
    summarize_fixed_full_smb_success_metrics,
)


class TestFullSMBSuccessThresholds(unittest.TestCase):
    def test_thresholds_cover_every_fixed_benchmark_task(self):
        catalog = full_smb_task_catalog()
        fixed_tasks = catalog.tasks_for_set("fixed_benchmark")

        self.assertEqual(
            set(FIXED_FULL_SMB_SUCCESS_THRESHOLDS),
            {task.name for task in fixed_tasks},
        )
        for task in fixed_tasks:
            threshold = FIXED_FULL_SMB_SUCCESS_THRESHOLDS[task.name]
            self.assertEqual(threshold.min_episodes, task.episodes)
            self.assertEqual(threshold.max_steps, task.max_steps)
            self.assertGreaterEqual(threshold.min_progress, 0.0)
            self.assertGreaterEqual(threshold.min_mean_return, 0.0)
            self.assertTrue(threshold.rationale)

    def test_threshold_diagnostics_explain_pass_and_fail_reasons(self):
        passing = evaluate_full_smb_success_threshold(
            "benchmark_1_1_start",
            {
                "max_progress": 3300.0,
                "completion_rate": 1.0,
                "survival_rate": 1.0,
                "mean_score": 900.0,
                "mean_coins": 0.0,
                "death_count": 0.0,
                "mean_return": 1.0,
            },
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )
        self.assertTrue(passing["threshold_met"])
        self.assertTrue(passing["meets_progress"])
        self.assertTrue(passing["meets_completion"])
        self.assertTrue(passing["within_death_budget"])

        low_progress = evaluate_full_smb_success_threshold(
            "benchmark_1_1_start",
            {
                "max_progress": 1200.0,
                "completion_rate": 1.0,
                "survival_rate": 1.0,
                "mean_score": 900.0,
                "mean_coins": 0.0,
                "death_count": 0.0,
                "mean_return": 1.0,
            },
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )
        self.assertFalse(low_progress["threshold_met"])
        self.assertFalse(low_progress["meets_progress"])

        too_many_deaths = evaluate_full_smb_success_threshold(
            "benchmark_1_1_start",
            {
                "max_progress": 3300.0,
                "completion_rate": 1.0,
                "survival_rate": 1.0,
                "mean_score": 900.0,
                "mean_coins": 0.0,
                "death_count": 1.0,
                "mean_return": 1.0,
            },
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )
        self.assertFalse(too_many_deaths["threshold_met"])
        self.assertFalse(too_many_deaths["within_death_budget"])

    def test_threshold_accepts_existing_evaluation_aliases(self):
        passing = evaluate_full_smb_success_threshold(
            "benchmark_1_2_start",
            {
                "progress": 2900.0,
                "success_rate": 2.0 / 3.0,
                "survival_rate": 2.0 / 3.0,
                "score": 700.0,
                "coins": 0.0,
                "deaths": 1.0,
                "return": 0.5,
            },
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )

        self.assertTrue(passing["threshold_met"])
        self.assertEqual(passing["observed"]["completion_rate"], 2.0 / 3.0)

    def test_fixed_threshold_evaluation_returns_per_task_results(self):
        results = evaluate_fixed_full_smb_success_thresholds(
            {
                "benchmark_1_1_start": {
                    "max_progress": 3300.0,
                    "completion_rate": 1.0,
                    "survival_rate": 1.0,
                    "mean_score": 900.0,
                    "mean_coins": 0.0,
                    "death_count": 0.0,
                    "mean_return": 1.0,
                },
                "benchmark_2_1_start": {
                    "max_progress": 800.0,
                    "completion_rate": 0.0,
                    "survival_rate": 0.0,
                    "mean_score": 0.0,
                    "mean_coins": 0.0,
                    "death_count": 3.0,
                    "mean_return": -1.0,
                },
            },
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )

        self.assertTrue(results["benchmark_1_1_start"]["threshold_met"])
        self.assertFalse(results["benchmark_2_1_start"]["threshold_met"])

    def test_tuning_summary_prioritizes_thresholds_before_raw_progress(self):
        passing_results = {
            "benchmark_1_1_start": {
                "max_progress": 3300.0,
                "completion_rate": 1.0,
                "survival_rate": 1.0,
                "mean_score": 900.0,
                "death_count": 0.0,
                "mean_return": 1.0,
            }
        }
        passing_thresholds = evaluate_fixed_full_smb_success_thresholds(
            passing_results,
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )
        unsolved_results = {
            "benchmark_1_1_start": {
                "max_progress": 9999.0,
                "completion_rate": 0.0,
                "survival_rate": 0.0,
                "mean_score": 9999.0,
                "death_count": 3.0,
                "mean_return": 9999.0,
            }
        }
        unsolved_thresholds = evaluate_fixed_full_smb_success_thresholds(
            unsolved_results,
            evaluation_episodes=3,
            evaluation_max_steps=2400,
        )

        passing = summarize_fixed_full_smb_success_metrics(
            passing_results,
            passing_thresholds,
        )
        unsolved = summarize_fixed_full_smb_success_metrics(
            unsolved_results,
            unsolved_thresholds,
        )

        self.assertEqual(passing["threshold_pass_rate"], 1.0)
        self.assertEqual(unsolved["threshold_pass_rate"], 0.0)
        self.assertGreater(passing["score"], unsolved["score"])


if __name__ == "__main__":
    unittest.main()
