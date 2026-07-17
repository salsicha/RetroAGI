"""Tests for action-distribution collapse statistics."""

import math
import unittest

from retroagi.core import action_distribution_stats
from retroagi.stages.block_smb.train import block_smb_action_count_metric_values


class TestActionDistributionStats(unittest.TestCase):
    def test_uniform_distribution_has_full_entropy_and_no_collapse(self):
        stats = action_distribution_stats(
            {index: 10 for index in range(6)},
            action_count=6,
        )

        self.assertEqual(stats["total_actions"], 60)
        self.assertAlmostEqual(stats["normalized_entropy"], 1.0)
        self.assertAlmostEqual(stats["dominant_share"], 1 / 6)
        self.assertFalse(stats["collapsed"])

    def test_historical_always_right_collapse_is_flagged(self):
        # The exact post-hoc observation from docs/issues.md: RIGHT=15934,
        # all other actions 0.
        stats = action_distribution_stats(
            {"1": 15934, "0": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            action_count=6,
        )

        self.assertTrue(stats["collapsed"])
        self.assertEqual(stats["dominant_action"], 1)
        self.assertEqual(stats["dominant_share"], 1.0)
        self.assertEqual(stats["entropy_nats"], 0.0)

    def test_near_collapse_respects_threshold(self):
        counts = {"1": 96, "2": 4}
        self.assertTrue(
            action_distribution_stats(counts, action_count=6)["collapsed"],
        )
        self.assertFalse(
            action_distribution_stats(
                counts,
                action_count=6,
                collapse_share_threshold=0.97,
            )["collapsed"],
        )

    def test_empty_counts_are_not_collapsed(self):
        stats = action_distribution_stats({}, action_count=6)

        self.assertEqual(stats["total_actions"], 0)
        self.assertIsNone(stats["dominant_action"])
        self.assertFalse(stats["collapsed"])

    def test_ignores_out_of_range_and_malformed_keys(self):
        stats = action_distribution_stats(
            {"1": 10, "9": 100, "bogus": 5, "-1": 3},
            action_count=6,
        )

        self.assertEqual(stats["total_actions"], 10)
        self.assertEqual(stats["dominant_action"], 1)

    def test_validates_arguments(self):
        with self.assertRaisesRegex(ValueError, "action_count"):
            action_distribution_stats({}, action_count=1)
        with self.assertRaisesRegex(ValueError, "collapse_share_threshold"):
            action_distribution_stats({}, action_count=6, collapse_share_threshold=0.0)

    def test_two_action_split_entropy(self):
        stats = action_distribution_stats({"0": 50, "1": 50}, action_count=6)

        self.assertAlmostEqual(stats["entropy_nats"], math.log(2))
        self.assertFalse(stats["collapsed"])


class TestBlockSMBActionMetrics(unittest.TestCase):
    def test_metric_values_include_entropy_and_collapse_flag(self):
        metrics = block_smb_action_count_metric_values(
            "eval",
            {"1": 100, "0": 0, "2": 0, "3": 0, "4": 0, "5": 0},
        )

        self.assertEqual(metrics["eval_action_collapse"], 1.0)
        self.assertEqual(metrics["eval_action_entropy"], 0.0)
        self.assertEqual(metrics["eval_action_dominant_share"], 1.0)

    def test_metric_values_for_mixed_actions(self):
        metrics = block_smb_action_count_metric_values(
            "train",
            {str(index): 10 for index in range(6)},
        )

        self.assertEqual(metrics["train_action_collapse"], 0.0)
        self.assertAlmostEqual(metrics["train_action_entropy"], 1.0)


if __name__ == "__main__":
    unittest.main()
