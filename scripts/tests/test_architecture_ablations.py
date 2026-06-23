"""Tests for normalized architecture ablation variants."""

import argparse
import unittest

from retroagi.core import (
    ArchitectureAblationConfig,
    build_architecture_variant,
    parse_architecture_ablation_item,
)


class TestArchitectureAblations(unittest.TestCase):
    def test_parse_ablation_aliases_and_values(self):
        self.assertEqual(parse_architecture_ablation_item("vision=off"), ("vision_enabled", False))
        self.assertEqual(
            parse_architecture_ablation_item("target-network=auto"),
            ("target_network_mode", "auto"),
        )
        self.assertEqual(
            parse_architecture_ablation_item("controller-schedule=linear"),
            ("controller_schedule", "linear"),
        )

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_architecture_ablation_item("not_real=off")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_architecture_ablation_item("vision=maybe")

    def test_variant_maps_ablations_to_architecture_and_stage_args(self):
        variant = build_architecture_variant(
            {"hidden_dim": 8},
            [
                ("controller_schedule", "linear"),
                ("vision_enabled", False),
                ("world_model_enabled", False),
                ("critic_feedback_enabled", False),
                ("hierarchy_enabled", False),
                ("recurrent_state_enabled", False),
                ("checkpoint_transfer_enabled", True),
                ("target_network_mode", "off"),
                ("auxiliary_objectives_enabled", False),
            ],
        )

        self.assertIsInstance(variant.ablation, ArchitectureAblationConfig)
        self.assertEqual(
            variant.architecture_config,
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )
        self.assertEqual(
            variant.forward_kwargs,
            {"critic_feedback_enabled": False, "world_model_enabled": False},
        )
        self.assertEqual(
            variant.args_for_stage("synthetic-1d"),
            ["--critic-loss-weight", "0"],
        )
        block_args = variant.args_for_stage("block-smb")
        self.assertIn("--disable-vision", block_args)
        self.assertIn("--disable-world-model", block_args)
        self.assertIn("--disable-critic-feedback", block_args)
        self.assertIn("--disable-hierarchy", block_args)
        self.assertIn("--disable-recurrent-state", block_args)
        self.assertIn("--enable-checkpoint-transfer", block_args)
        self.assertIn("--target-network-mode", block_args)
        self.assertIn("--critic-loss-weight", block_args)
        self.assertEqual(variant.metadata()["ablation"]["controller_schedule"], "linear")


if __name__ == "__main__":
    unittest.main()
