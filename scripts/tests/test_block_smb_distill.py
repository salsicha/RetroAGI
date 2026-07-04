"""Tests for Block SMB distillation data collection."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from retroagi.core import DEFAULT_PRIMITIVE_DURATION_BINS, SMBAction, VisionOutput, VisionSpec
from retroagi.stages.block_smb import distill as distill_module
from retroagi.stages.block_smb.distill import (
    BlockSMBDistillationConfig,
    DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
    build_block_smb_distillation_scenarios,
    collect_scripted_distillation_examples,
)
from retroagi.stages.block_smb.distill import (
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
            required_monte_carlo_families=(),
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

    def test_default_warm_start_covers_fixed_chained_and_full_smb_proxy(self):
        config = BlockSMBDistillationConfig(
            rollout_steps=24,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=24,
            device="cpu",
        )

        scenarios, scripts, summary = build_block_smb_distillation_scenarios(config)
        family_counts = summary["monte_carlo"]["coverage"]["family_counts"]

        self.assertEqual(summary["fixed_scenario_count"], len(config.fixed_scenarios))
        self.assertEqual(
            tuple(summary["monte_carlo"]["required_families"]),
            DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
        )
        for family in DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES:
            self.assertEqual(family_counts[family], 3)
        self.assertEqual(
            summary["scenario_count"],
            len(config.fixed_scenarios) + 3 * len(DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES),
        )
        self.assertTrue(any(".full_smb_opening_proxy." in name for name, _scenario in scenarios))
        self.assertTrue(all(name in scripts for name, _scenario in scenarios))
        self.assertTrue(any(action == 2 for actions in scripts.values() for action in actions))

    def test_scripted_examples_carry_primitive_duration_release_labels(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=("level_5_enemy_hop.json",),
            required_monte_carlo_families=(),
            rollout_steps=45,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=45,
            primitive_hazard_weight_multiplier=3.0,
            device="cpu",
        )

        examples = collect_scripted_distillation_examples(
            config,
            vision_factory=static_vision_factory,
        )

        jump_examples = [
            example
            for example in examples
            if example.action == int(SMBAction.RIGHT_JUMP)
        ]
        duration_examples = [
            example
            for example in examples
            if example.primitive_duration_mask > 0.0
        ]
        release_examples = [
            example
            for example in examples
            if example.primitive_release_mask > 0.0
        ]
        positive_release_examples = [
            example
            for example in examples
            if example.primitive_release > 0.0
        ]

        self.assertEqual(len(jump_examples), 18)
        self.assertEqual(len(duration_examples), 1)
        self.assertEqual(
            duration_examples[0].primitive_duration_bin,
            int(
                torch.abs(
                    torch.as_tensor(DEFAULT_PRIMITIVE_DURATION_BINS, dtype=torch.float32)
                    - 18.0
                )
                .argmin()
                .item()
            ),
        )
        self.assertEqual(len(release_examples), 18)
        self.assertEqual(len(positive_release_examples), 1)
        self.assertEqual(
            {example.primitive_post_release for example in jump_examples},
            {int(SMBAction.RIGHT)},
        )
        self.assertTrue(all(example.primitive_weight == 3.0 for example in jump_examples))

    def test_primitive_loss_uses_scripted_duration_release_targets(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=("level_5_enemy_hop.json",),
            required_monte_carlo_families=(),
            rollout_steps=45,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=45,
            device="cpu",
        )
        examples = collect_scripted_distillation_examples(
            config,
            vision_factory=static_vision_factory,
        )
        example = next(
            example for example in examples if example.primitive_duration_mask > 0.0
        )
        motor_primitives = SimpleNamespace(
            hold_duration_logits=torch.zeros(1, 1, len(DEFAULT_PRIMITIVE_DURATION_BINS)),
            release_logit=torch.zeros(1, 1),
            post_release_logits=torch.zeros(1, 1, len(SMBAction)),
        )

        loss = distill_module._block_smb_distillation_primitive_loss(
            motor_primitives,
            [example],
            device=torch.device("cpu"),
        )

        self.assertGreater(float(loss.item()), 0.0)

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
                        "--vision-checkpoint",
                        "data/pipeline/block_vit.pth",
                        "--epochs",
                        "1",
                        "--primitive-loss-weight",
                        "0.6",
                        "--primitive-hazard-weight-multiplier",
                        "3.5",
                        "--monte-carlo-samples",
                        "3",
                        "--monte-carlo-seed",
                        "60002",
                        "--monte-carlo-family-weight",
                        "flat_run=1",
                        "--monte-carlo-parameter-sweep",
                        "--monte-carlo-sweep-repeats-per-difficulty",
                        "2",
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
        self.assertEqual(config.vision_checkpoint, Path("data/pipeline/block_vit.pth"))
        self.assertEqual(config.monte_carlo_samples, 3)
        self.assertEqual(config.primitive_loss_weight, 0.6)
        self.assertEqual(config.primitive_hazard_weight_multiplier, 3.5)
        self.assertEqual(config.monte_carlo_seed, 60002)
        self.assertEqual(config.monte_carlo_family_weights, {"flat_run": 1.0})
        self.assertTrue(config.monte_carlo_parameter_sweep)
        self.assertEqual(config.monte_carlo_sweep_repeats_per_difficulty, 2)
        self.assertEqual(
            config.required_monte_carlo_families,
            DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
        )
        self.assertEqual(config.required_monte_carlo_repeats_per_difficulty, 1)
        self.assertEqual(config.monte_carlo_validation_samples, 4)
        self.assertEqual(config.monte_carlo_test_samples, 5)
        self.assertEqual(config.monte_carlo_pass_rate_gate, 0.8)
        self.assertEqual(config.monte_carlo_family_pass_rate_gate, 0.7)

    def test_training_config_preserves_distillation_vision_checkpoint(self):
        config = BlockSMBDistillationConfig(
            vision_checkpoint=Path("data/pipeline/block_vit.pth")
        )

        training_config = distill_module._training_config_from_distillation(config)

        self.assertEqual(
            training_config.vision_checkpoint_path,
            Path("data/pipeline/block_vit.pth"),
        )
        self.assertEqual(
            training_config.monte_carlo_train_samples_per_epoch,
            len(DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES)
            * len(distill_module.BLOCK_SMB_MC_DIFFICULTY_BINS),
        )


if __name__ == "__main__":
    unittest.main()
