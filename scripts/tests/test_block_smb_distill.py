"""Tests for Block SMB distillation data collection."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from retroagi.core import VisionOutput, VisionSpec
from retroagi.stages.block_smb.distill import (
    BlockSMBDistillationConfig,
    build_block_smb_distillation_scenarios,
    collect_scripted_distillation_examples,
    main as distill_main,
)


class StaticBlockVision:
    spec = VisionSpec(
        name="static_block_distill",
        semantic_classes=(
            "background",
            "mario",
            "platform",
            "coin",
            "goal",
            "enemy",
            "moving_platform",
        ),
        token_dim=4,
    )

    def encode(self, _observation):
        logits = torch.full((1, self.spec.num_classes, 2, 16), -8.0)
        logits[:, 1, :, 1] = 8.0
        logits[:, 2, :, :] = torch.maximum(logits[:, 2, :, :], torch.tensor(1.0))
        return VisionOutput(
            position=torch.tensor([[0.1, 0.8]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.zeros(1, 240, self.spec.token_dim),
            metadata={},
        )


def static_vision_factory():
    return StaticBlockVision()


class TestBlockSMBDistillation(unittest.TestCase):
    def test_monte_carlo_oracle_examples_are_collected(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=(),
            monte_carlo_samples=2,
            rollout_steps=24,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=24,
            device="cpu",
        )

        scenarios, scripts, summary = build_block_smb_distillation_scenarios(config)
        examples = collect_scripted_distillation_examples(
            config,
            vision_factory=static_vision_factory,
        )

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(summary["monte_carlo"]["sample_count"], 2)
        self.assertTrue(all(name in scripts for name, _scenario in scenarios))
        self.assertTrue(any(action == 2 for actions in scripts.values() for action in actions))
        self.assertGreater(len(examples), 0)
        self.assertTrue(
            all(example.scenario_name.startswith("block_smb_mc_v1.train.") for example in examples)
        )
        self.assertIn(2, {example.action for example in examples})

    def test_cli_passes_monte_carlo_distillation_config(self):
        with patch(
            "retroagi.stages.block_smb.distill.train_distilled_block_smb_policy",
            return_value={"ok": True},
        ) as train:
            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = distill_main(
                    [
                        "--checkpoint",
                        "data/block_smb/distilled.pth",
                        "--epochs",
                        "1",
                        "--monte-carlo-samples",
                        "3",
                        "--monte-carlo-seed",
                        "60002",
                        "--monte-carlo-family-weight",
                        "flat_run=1",
                        "--monte-carlo-validation-samples",
                        "4",
                        "--monte-carlo-test-samples",
                        "5",
                        "--monte-carlo-pass-rate-gate",
                        "0.8",
                        "--monte-carlo-family-pass-rate-gate",
                        "0.7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stream.getvalue()), {"ok": True})
        config = train.call_args.args[0]
        self.assertEqual(config.checkpoint_path, Path("data/block_smb/distilled.pth"))
        self.assertEqual(config.monte_carlo_samples, 3)
        self.assertEqual(config.monte_carlo_seed, 60002)
        self.assertEqual(config.monte_carlo_family_weights, {"flat_run": 1.0})
        self.assertEqual(config.monte_carlo_validation_samples, 4)
        self.assertEqual(config.monte_carlo_test_samples, 5)
        self.assertEqual(config.monte_carlo_pass_rate_gate, 0.8)
        self.assertEqual(config.monte_carlo_family_pass_rate_gate, 0.7)


if __name__ == "__main__":
    unittest.main()
