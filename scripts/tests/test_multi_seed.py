"""Tests for multi-seed evaluation aggregation."""

import unittest

import torch

from retroagi.core import (
    aggregate_seed_metrics,
    evaluate_over_seeds,
    evaluation_seeds,
)
from retroagi.stages.block_smb.train import (
    BlockSMBTrainingConfig,
    evaluate_block_smb_multi_seed,
    make_block_smb_model,
)
from retroagi.stages.block_smb.vision import BlockVisionTransformer


class TestSeedAggregation(unittest.TestCase):
    def test_evaluation_seeds_are_deterministic_and_distinct(self):
        seeds = evaluation_seeds(7, 3)

        self.assertEqual(seeds, (7, 1007, 2007))
        self.assertEqual(seeds, evaluation_seeds(7, 3))

    def test_aggregate_reports_mean_std_min_max(self):
        aggregated = aggregate_seed_metrics(
            [
                {"success_rate": 0.5, "mean_return": 10.0},
                {"success_rate": 0.7, "mean_return": 14.0},
                {"success_rate": 0.9, "mean_return": 18.0},
            ]
        )

        self.assertAlmostEqual(aggregated["success_rate"]["mean"], 0.7)
        self.assertAlmostEqual(aggregated["success_rate"]["min"], 0.5)
        self.assertAlmostEqual(aggregated["success_rate"]["max"], 0.9)
        self.assertGreater(aggregated["success_rate"]["std"], 0.0)
        self.assertEqual(aggregated["mean_return"]["count"], 3.0)

    def test_aggregate_skips_non_numeric_and_missing_keys(self):
        aggregated = aggregate_seed_metrics(
            [
                {"success_rate": 1.0, "note": "a", "only_here": 2.0},
                {"success_rate": 0.0, "note": "b"},
            ]
        )

        self.assertIn("success_rate", aggregated)
        self.assertNotIn("note", aggregated)
        self.assertNotIn("only_here", aggregated)

    def test_evaluate_over_seeds_collects_per_seed_payloads(self):
        result = evaluate_over_seeds(
            lambda seed: {"value": float(seed)},
            base_seed=3,
            seed_count=2,
        )

        self.assertEqual(result["seeds"], [3, 1003])
        self.assertEqual([entry["seed"] for entry in result["per_seed"]], [3, 1003])
        self.assertAlmostEqual(result["aggregate"]["value"]["mean"], 503.0)

    def test_seed_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "seed_count"):
            evaluation_seeds(0, 0)


class TestBlockSMBMultiSeedEvaluation(unittest.TestCase):
    def test_multi_seed_evaluation_aggregates_success_metrics(self):
        config = BlockSMBTrainingConfig(
            seed=5,
            epochs=1,
            episodes_per_epoch=1,
            rollout_steps=1,
            hidden_dim=8,
            fixed_scenarios=("level_1_flat.json",),
            evaluation_episodes=1,
            evaluation_max_steps=3,
            monte_carlo_validation_samples=0,
            monte_carlo_test_samples=0,
            device="cpu",
        )
        model = make_block_smb_model(config)

        result = evaluate_block_smb_multi_seed(
            model,
            config,
            device=torch.device("cpu"),
            vision_factory=BlockVisionTransformer,
            seed_count=2,
        )

        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["seeds"], [5, 1005])
        self.assertIn("success_rate", result["aggregate"])
        self.assertIn("mean_return", result["aggregate"])
        self.assertIn("action_collapse", result["aggregate"])
        for entry in result["per_seed"]:
            self.assertIn("success_rate", entry)


if __name__ == "__main__":
    unittest.main()
