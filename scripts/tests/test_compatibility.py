"""Tests for startup compatibility validation."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from retroagi.core import (
    CompatibilityError,
    ModelConfig,
    StageSpec,
    build_checkpoint,
    validate_checkpoint_compatibility,
    validate_model_vision_compatibility,
    validate_stage_spec,
)
from retroagi.stages.block_smb import BLOCK_SMB_SPEC, BlockVisionTransformer
from scripts.vit.train_block_vit import TrainConfig, save_checkpoint


class TestCompatibilityValidation(unittest.TestCase):
    def test_accepts_block_vit_startup_contract(self):
        model = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        config = ModelConfig(name="block_smb_vit", hidden_dim=16, patch_size=16)

        validate_stage_spec(BLOCK_SMB_SPEC)
        validate_model_vision_compatibility(config, model.spec)

    def test_rejects_stage_action_vocab_mismatch(self):
        bad_stage = replace(BLOCK_SMB_SPEC, vocab_size=2)

        with self.assertRaisesRegex(CompatibilityError, "SMB actions"):
            validate_stage_spec(bad_stage)

    def test_rejects_model_vision_mismatch(self):
        model = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        bad_config = ModelConfig(name="wrong_model", hidden_dim=16)

        with self.assertRaisesRegex(CompatibilityError, "model name"):
            validate_model_vision_compatibility(bad_config, model.spec)

    def test_rejects_incompatible_checkpoint_before_state_load(self):
        model = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        checkpoint = build_checkpoint(
            stage="synthetic_1d",
            model_name="block_smb_vit",
            checkpoint_kind="vision_encoder",
            specs={"vision": model.spec},
            states={"model": model.state_dict(), "optimizer": {}},
        )

        with self.assertRaisesRegex(CompatibilityError, "stage"):
            validate_checkpoint_compatibility(
                checkpoint,
                stage=BLOCK_SMB_SPEC,
                model=ModelConfig(name="block_smb_vit", hidden_dim=16),
                vision=model.spec,
                checkpoint_kind="vision_encoder",
                required_states=("model", "optimizer"),
            )

    def test_block_vit_checkpoint_is_compatible_with_startup_contract(self):
        model = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = TrainConfig(model=ModelConfig(name="block_smb_vit", hidden_dim=16, patch_size=16))

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit.pth"
            save_checkpoint(path, model, optimizer, epoch=0, metrics={"mean_iou": 0.1}, config=config)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        normalized = validate_checkpoint_compatibility(
            checkpoint,
            stage=BLOCK_SMB_SPEC,
            model=config.model,
            vision=model.spec,
            checkpoint_kind="vision_encoder",
            required_states=("model", "optimizer"),
        )

        self.assertEqual(normalized["stage"], BLOCK_SMB_SPEC.name)
        self.assertEqual(normalized["model_name"], config.model.name)


if __name__ == "__main__":
    unittest.main()
