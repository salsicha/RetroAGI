"""Tests for typed experiment configuration contracts."""

import unittest
from pathlib import Path

from retroagi.core import (
    CheckpointConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from scripts.vit.train_block_vit import TrainConfig


class TestExperimentConfig(unittest.TestCase):
    def test_serializes_nested_sections_to_plain_data(self):
        config = ExperimentConfig(
            environment=EnvironmentConfig(stage="block_smb", seed=7, rollout_steps=32),
            model=ModelConfig(name="block_smb_vit", hidden_dim=64, patch_size=16),
            training=TrainingConfig(epochs=2, batch_size=4, samples_per_epoch=8),
            evaluation=EvaluationConfig(samples=4, metrics=("loss", "mean_iou")),
            checkpoints=CheckpointConfig(output_path=Path("data/block_vit/block_vit.pth")),
            name="smoke",
        )

        data = config.to_dict()

        self.assertEqual(data["environment"]["stage"], "block_smb")
        self.assertEqual(data["model"]["patch_size"], 16)
        self.assertEqual(data["evaluation"]["metrics"], ["loss", "mean_iou"])
        self.assertEqual(data["checkpoints"]["output_path"], "data/block_vit/block_vit.pth")

    def test_rejects_invalid_values_early(self):
        with self.assertRaisesRegex(ValueError, "rollout_steps"):
            EnvironmentConfig(stage="block_smb", rollout_steps=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            ModelConfig(name="model", dropout=1.0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            TrainingConfig(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "metrics"):
            EvaluationConfig(metrics=("",))
        with self.assertRaisesRegex(ValueError, "best_mode"):
            CheckpointConfig(best_mode="middle")

    def test_block_vit_train_config_uses_typed_sections(self):
        config = TrainConfig()

        self.assertEqual(config.environment.stage, "block_smb")
        self.assertEqual(config.model.name, "block_smb_vit")
        self.assertEqual(config.training.samples_per_epoch, 2048)
        self.assertEqual(config.evaluation.samples, 512)
        self.assertEqual(config.checkpoints.best_metric, "mean_iou")
        self.assertEqual(config.to_dict()["metadata"]["position_weight"], 2.0)


if __name__ == "__main__":
    unittest.main()
