"""Tests for the shared versioned checkpoint schema."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from retroagi.core import (
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint as save_versioned_checkpoint,
    validate_checkpoint,
)
from retroagi.stages.block_smb import BlockVisionTransformer
from scripts.vit.train_block_vit import TrainConfig, save_checkpoint


class TestCheckpointSchema(unittest.TestCase):
    def make_checkpoint(self):
        return build_checkpoint(
            stage="block_smb",
            model_name="block_smb_vit",
            checkpoint_kind="vision_encoder",
            epoch=3,
            global_step=12,
            metrics={"mean_iou": 0.5},
            config={"environment": {"stage": "block_smb"}},
            specs={"vision": {"name": "block_smb_vit"}},
            states={"model": {"weight": torch.tensor([1.0])}},
            metadata={"unit": True},
        )

    def test_builds_valid_schema_v1_payload(self):
        checkpoint = self.make_checkpoint()

        self.assertEqual(checkpoint["checkpoint_schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(checkpoint["stage"], "block_smb")
        self.assertEqual(checkpoint["model_name"], "block_smb_vit")
        self.assertEqual(checkpoint["checkpoint_kind"], "vision_encoder")
        self.assertEqual(checkpoint["states"]["model"]["weight"].item(), 1.0)

    def test_round_trips_through_torch_file(self):
        checkpoint = self.make_checkpoint()
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pth"

            save_versioned_checkpoint(path, checkpoint)
            loaded = load_checkpoint(path)

        self.assertEqual(loaded["checkpoint_schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(loaded["states"]["model"]["weight"].item(), 1.0)

    def test_rejects_missing_and_unknown_schema_versions(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_schema_version"):
            validate_checkpoint({"states": {"model": {}}})
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint schema version"):
            validate_checkpoint({"checkpoint_schema_version": 999, "states": {"model": {}}})

    def test_block_vit_trainer_saves_shared_schema(self):
        model = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = TrainConfig()

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit.pth"
            save_checkpoint(
                path, model, optimizer, epoch=1, metrics={"mean_iou": 0.25}, config=config
            )
            loaded = load_checkpoint(path)

        self.assertEqual(loaded["stage"], "block_smb")
        self.assertEqual(loaded["model_name"], "block_smb_vit")
        self.assertEqual(loaded["checkpoint_kind"], "vision_encoder")
        self.assertIn("model", loaded["states"])
        self.assertIn("optimizer", loaded["states"])
        self.assertEqual(loaded["specs"]["vision"]["name"], model.spec.name)


if __name__ == "__main__":
    unittest.main()
