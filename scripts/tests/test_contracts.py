"""Shared contract tests for implemented adapters and vision encoders."""

from collections.abc import Mapping
import unittest

import numpy as np
import torch

from retroagi.core import (
    LinearVisionEncoder,
    StageBatch,
    VisionOutput,
    VisionSpec,
    validate_stage_spec,
)
from retroagi.stages.block_smb import BLOCK_SMB_SPEC, BlockSMBStage, BlockVisionTransformer
from retroagi.stages.full_smb import FullSMBSegmentationVision


class StaticVisionEncoder:
    """Small deterministic encoder used to isolate adapter contract tests."""

    spec = VisionSpec(
        name="static_block_contract",
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

    def encode(self, observation):
        logits = torch.full((1, self.spec.num_classes, 2, 16), -10.0)
        for column in range(16):
            logits[0, column % self.spec.num_classes, :, column] = 10.0
        return VisionOutput(
            position=torch.tensor([[0.25, 0.75]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.linspace(-1, 1, 240 * self.spec.token_dim).view(1, 240, -1),
            metadata={"source": "static"},
        )


class TestStageAdapterContracts(unittest.TestCase):
    def adapter_cases(self):
        return (
            (
                "block_smb",
                lambda: BlockSMBStage(vision=StaticVisionEncoder()),
                BLOCK_SMB_SPEC,
                0,
            ),
        )

    def test_every_adapter_satisfies_stage_lifecycle_contract(self):
        for name, factory, expected_spec, action in self.adapter_cases():
            with self.subTest(adapter=name):
                stage = factory()
                try:
                    validate_stage_spec(stage.spec, context=name)
                    self.assertEqual(stage.spec, expected_spec)

                    observation = stage.reset(seed=123)
                    self.assertIsNotNone(observation)
                    self.assertIsInstance(stage.last_info, Mapping)
                    self.assertIn("state_vec", stage.last_info)

                    next_observation, reward, terminated, truncated, info = stage.step(action)
                    self.assertIsNotNone(next_observation)
                    self.assertIsInstance(float(reward), float)
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                    self.assertIsInstance(info, Mapping)

                    batch = stage.encode_observation(observation)
                    self.assertIsInstance(batch, StageBatch)
                    self.assertEqual(batch.src_a.shape, (1, stage.spec.seq_len_a))
                    self.assertEqual(batch.src_b.shape, (1, stage.spec.seq_len_b))
                    self.assertEqual(batch.src_c.shape, (1, stage.spec.seq_len_c))
                    self.assertIsNone(batch.target_a)
                    self.assertIsNone(batch.target_b)
                    self.assertIsNone(batch.target_c)
                    self.assertIsInstance(batch.metadata, Mapping)
                    self.assertIn("vision", batch.metadata)
                finally:
                    stage.env.close()


class TestVisionEncoderContracts(unittest.TestCase):
    def vision_cases(self):
        return (
            (
                "synthetic_1d",
                lambda: LinearVisionEncoder(vocab_size=20, token_dim=16),
                torch.arange(8),
            ),
            (
                "block_smb_vit",
                lambda: BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0).eval(),
                np.zeros((240, 256, 3), dtype=np.uint8),
            ),
            (
                "full_smb_deeplab",
                lambda: FullSMBSegmentationVision(checkpoint=None),
                torch.zeros(1, 3, 64, 64),
            ),
        )

    def assert_vision_output_contract(self, encoder, observation):
        with torch.no_grad():
            output = encoder.encode(observation)

        spec = encoder.spec
        self.assertIsInstance(spec, VisionSpec)
        self.assertIsInstance(output, VisionOutput)
        self.assertEqual(output.position.ndim, 2)
        self.assertEqual(output.position.shape[0], 1)
        self.assertEqual(output.position.shape[-1], spec.position_dim)
        self.assertTrue(torch.isfinite(output.position).all())
        self.assertTrue(torch.all((output.position >= 0) & (output.position <= 1)))

        self.assertEqual(output.semantic_logits.ndim, 4)
        self.assertEqual(output.semantic_logits.shape[0], 1)
        self.assertEqual(output.semantic_logits.shape[1], spec.num_classes)
        self.assertEqual(
            output.semantic_ids.shape,
            output.semantic_logits.shape[0:1] + output.semantic_logits.shape[2:],
        )
        self.assertEqual(output.semantic_ids.dtype, torch.long)
        self.assertGreaterEqual(int(output.semantic_ids.min()), 0)
        self.assertLess(int(output.semantic_ids.max()), spec.num_classes)

        self.assertEqual(output.tokens.ndim, 3)
        self.assertEqual(output.tokens.shape[0], 1)
        self.assertEqual(output.tokens.shape[-1], spec.token_dim)
        self.assertTrue(torch.isfinite(output.tokens).all())
        self.assertIsInstance(output.metadata, Mapping)

    def test_every_vision_encoder_satisfies_output_contract(self):
        seen_names = set()
        for name, factory, observation in self.vision_cases():
            with self.subTest(vision=name):
                try:
                    encoder = factory()
                except (ImportError, ModuleNotFoundError) as exc:
                    self.skipTest(f"{name} dependencies are unavailable: {exc}")
                self.assertEqual(encoder.spec.name, name)
                self.assertNotIn(encoder.spec.name, seen_names)
                seen_names.add(encoder.spec.name)
                self.assert_vision_output_contract(encoder, observation)


if __name__ == "__main__":
    unittest.main()
