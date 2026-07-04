"""Tests for architecture specs and registry plumbing."""

import unittest

import torch

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    BASELINE_ARCHITECTURE_SPEC,
    AgentWorldModelCritic,
    ArchitectureRegistry,
    ArchitectureSpec,
    StageSpec,
    architecture_names,
    build_architecture,
    get_architecture,
)


def tiny_stage(name="synthetic_1d"):
    return StageSpec(
        name=name,
        observation_kind="synthetic",
        action_kind="discrete",
        seq_len_a=2,
        ratio_ab=2,
        ratio_bc=2,
        vocab_size=6,
    )


class TestArchitectureRegistry(unittest.TestCase):
    def test_baseline_architecture_is_registered_with_metadata(self):
        self.assertIn(BASELINE_ARCHITECTURE_NAME, architecture_names())

        spec = get_architecture(BASELINE_ARCHITECTURE_NAME)
        metadata = spec.metadata()

        self.assertEqual(metadata["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertIn("synthetic_1d", metadata["supported_stage_names"])
        self.assertIn("block_smb", metadata["supported_stage_names"])
        self.assertIn("full_smb", metadata["supported_stage_names"])
        self.assertEqual(metadata["configurable_hyperparameters"]["hidden_dim"], 64)
        self.assertIn("checkpoint_compatibility_policy", metadata)

    def test_baseline_architecture_builds_agent_world_model_critic(self):
        model = build_architecture(
            BASELINE_ARCHITECTURE_NAME,
            tiny_stage(),
            {"hidden_dim": 16, "controller_schedule": "linear"},
        )

        self.assertIsInstance(model, AgentWorldModelCritic)
        src_a = torch.zeros((1, 2), dtype=torch.long)
        src_b = torch.zeros((1, 4), dtype=torch.long)
        src_c = torch.zeros((1, 8), dtype=torch.float32)

        outputs = model(src_a, src_b, src_c, tau=1.0)

        self.assertEqual(len(outputs), 7)
        self.assertEqual(outputs[0].shape, (1, 8))
        self.assertEqual(outputs[1].shape, (1, 8))
        self.assertEqual(outputs[3].shape, (1, 8))
        self.assertEqual(outputs[4].shape, (1, 2, 6))

    def test_removed_lstm_actor_is_not_registered(self):
        removed_name = "_".join(("single", "pass", "lstm", "conditioned", "actor"))

        self.assertNotIn(removed_name, architecture_names())
        with self.assertRaisesRegex(KeyError, "unknown architecture"):
            get_architecture(removed_name)

    def test_baseline_rejects_unsupported_stage_and_bad_hyperparameters(self):
        with self.assertRaisesRegex(ValueError, "does not support stage"):
            build_architecture(BASELINE_ARCHITECTURE_NAME, tiny_stage("other_game"))

        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            build_architecture(BASELINE_ARCHITECTURE_NAME, tiny_stage(), {"hidden_dim": 0})

        with self.assertRaisesRegex(ValueError, "controller_schedule"):
            build_architecture(
                BASELINE_ARCHITECTURE_NAME,
                tiny_stage(),
                {"controller_schedule": "unsupported"},
            )

        with self.assertRaisesRegex(ValueError, "unknown architecture config"):
            build_architecture(
                BASELINE_ARCHITECTURE_NAME,
                tiny_stage(),
                {"not_a_real_option": True},
            )

    def test_registry_rejects_duplicate_architecture_names(self):
        registry = ArchitectureRegistry()
        registry.register(BASELINE_ARCHITECTURE_SPEC)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(BASELINE_ARCHITECTURE_SPEC)

    def test_architecture_spec_requires_declared_contracts(self):
        with self.assertRaisesRegex(ValueError, "supported_stage_names"):
            ArchitectureSpec(
                name="bad",
                factory=lambda _stage, _config=None: torch.nn.Identity(),
                supported_stage_names=(),
                checkpoint_model_name="bad",
                checkpoint_compatibility_policy="strict",
                output_contract="bad.forward.v1",
            )

        with self.assertRaisesRegex(ValueError, "output_contract"):
            ArchitectureSpec(
                name="bad",
                factory=lambda _stage, _config=None: torch.nn.Identity(),
                supported_stage_names=("synthetic_1d",),
                checkpoint_model_name="bad",
                checkpoint_compatibility_policy="strict",
                output_contract="",
            )


if __name__ == "__main__":
    unittest.main()
