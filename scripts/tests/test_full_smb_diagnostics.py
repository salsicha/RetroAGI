"""Tests for Full SMB perception diagnostics."""

import unittest

import numpy as np
import torch

from retroagi.core import VisionOutput, VisionSpec
from retroagi.stages.full_smb import (
    FullSMBPerceptionDiagnosticThresholds,
    collect_full_smb_perception_diagnostic_frames,
    evaluate_full_smb_perception,
)


class StableDiagnosticVision:
    spec = VisionSpec(
        name="stable_full_smb_diagnostic",
        semantic_classes=("sky", "ground", "mario", "coin"),
        token_dim=4,
    )

    def __init__(self):
        self.training = True
        self.eval_called = False
        self.train_called = False

    def eval(self):
        self.eval_called = True
        self.training = False

    def train(self):
        self.train_called = True
        self.training = True

    def encode(self, observation):
        frames = torch.as_tensor(observation)
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        batch = frames.shape[0]
        logits = torch.full((batch, self.spec.num_classes, 2, 2), -8.0)
        logits[:, 0, 0, 0] = 8.0
        logits[:, 1, 0, 1] = 8.0
        logits[:, 2, 1, 0] = 8.0
        logits[:, 3, 1, 1] = 8.0
        ids = logits.argmax(dim=1)
        return VisionOutput(
            position=torch.tensor([[0.25, 0.50]], dtype=torch.float32).repeat(batch, 1),
            semantic_logits=logits,
            semantic_ids=ids,
            tokens=torch.ones(batch, 4, self.spec.token_dim),
            metadata={"source": "stable"},
        )


class WeakDiagnosticVision(StableDiagnosticVision):
    def encode(self, observation):
        frames = torch.as_tensor(observation)
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        batch = frames.shape[0]
        logits = torch.zeros(batch, self.spec.num_classes, 2, 2)
        return VisionOutput(
            position=torch.ones(batch, 2, dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.zeros(batch, 4, self.spec.token_dim),
            metadata={"source": "weak"},
        )


class DiagnosticStage:
    def __init__(self):
        self.step_count = 0
        self.last_info = {}

    def reset(self, seed=None):
        self.step_count = 0
        self.last_info = self._info(x=0.0, y=0.5)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def step(self, action):
        self.step_count += 1
        self.last_info = self._info(x=0.25, y=0.5)
        observation = np.full((8, 8, 3), self.step_count, dtype=np.uint8)
        return observation, 0.0, False, False, self.last_info

    @staticmethod
    def _info(*, x, y):
        return {
            "state_vec": np.array([0.0, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "camera_vec": np.array([0.0, 0.0, 0.0, x]),
        }


class TestFullSMBPerceptionDiagnostics(unittest.TestCase):
    def test_full_smb_perception_diagnostic_accepts_stable_predictions(self):
        vision = StableDiagnosticVision()
        frames = torch.zeros(3, 8, 8, 3, dtype=torch.uint8)
        infos = [DiagnosticStage._info(x=0.25, y=0.50) for _ in range(3)]

        metrics = evaluate_full_smb_perception(
            vision,
            frames,
            infos,
            thresholds=FullSMBPerceptionDiagnosticThresholds(
                min_semantic_confidence=0.90,
                min_class_coverage=1.0,
                min_temporal_stability=1.0,
                max_position_rmse=0.0,
                min_position_within_tolerance=1.0,
                position_tolerance=0.0,
            ),
            batch_size=2,
        )

        self.assertFalse(metrics["bottleneck"])
        self.assertEqual(metrics["bottleneck_reasons"], [])
        self.assertGreater(metrics["semantic_confidence"], 0.99)
        self.assertEqual(metrics["class_coverage"], 1.0)
        self.assertEqual(metrics["temporal_stability"], 1.0)
        self.assertEqual(metrics["position_rmse"], 0.0)
        self.assertEqual(metrics["position_within_tolerance"], 1.0)
        self.assertTrue(vision.eval_called)
        self.assertTrue(vision.train_called)

    def test_full_smb_perception_diagnostic_flags_bottlenecks(self):
        vision = WeakDiagnosticVision()
        frames = torch.zeros(2, 8, 8, 3, dtype=torch.uint8)
        infos = [DiagnosticStage._info(x=0.0, y=0.0) for _ in range(2)]

        metrics = evaluate_full_smb_perception(
            vision,
            frames,
            infos,
            thresholds=FullSMBPerceptionDiagnosticThresholds(
                min_semantic_confidence=0.5,
                min_class_coverage=0.75,
                max_position_rmse=0.1,
                min_position_within_tolerance=1.0,
                position_tolerance=0.1,
            ),
        )

        self.assertTrue(metrics["bottleneck"])
        self.assertIn("semantic_confidence", metrics["bottleneck_reasons"])
        self.assertIn("class_coverage", metrics["bottleneck_reasons"])
        self.assertIn("position_rmse", metrics["bottleneck_reasons"])
        self.assertIn("position_consistency", metrics["bottleneck_reasons"])

    def test_collect_full_smb_perception_diagnostic_frames_uses_stage_rollout(self):
        stage = DiagnosticStage()
        trace = collect_full_smb_perception_diagnostic_frames(
            stage,
            samples=3,
            seed=11,
            rollout_steps=4,
        )

        self.assertEqual(trace.observations.shape, (3, 8, 8, 3))
        self.assertEqual(len(trace.infos), 3)
        self.assertEqual(len(trace.action_ids), 2)
        self.assertEqual(trace.reset_count, 1)
        self.assertEqual(trace.summary()["samples"], 3)


if __name__ == "__main__":
    unittest.main()
