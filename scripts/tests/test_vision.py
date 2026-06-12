"""Tests for the unified curriculum vision interface."""

import unittest
from pathlib import Path

import torch

from retroagi.core import LinearVisionEncoder, VisionOutput
from retroagi.stages.block_smb import BlockSMBStage, BlockVisionTransformer
from scripts.vit.train_vit import ViTSegmenter
from scripts.vit.train_block_vit import (
    build_ground_truth,
    class_weights,
    collect_procedural_frames,
    compute_loss,
    make_loader,
)



class TestVisionInterface(unittest.TestCase):
    def test_linear_encoder_uses_common_output(self):
        encoder = LinearVisionEncoder(vocab_size=20, token_dim=16)
        output = encoder.encode(torch.arange(8))

        self.assertIsInstance(output, VisionOutput)
        self.assertEqual(output.position.shape, (1, 1))
        self.assertEqual(output.semantic_logits.shape, (1, 20, 1, 8))
        self.assertEqual(output.semantic_ids.shape, (1, 1, 8))
        self.assertEqual(output.tokens.shape, (1, 8, 16))

    def test_block_vit_extracts_position_semantics_and_tokens(self):
        encoder = BlockVisionTransformer(dim=32, depth=1, heads=4, drop=0.0).eval()
        stage = BlockSMBStage(vision=encoder)
        try:
            observation = stage.reset(seed=3)
            output = encoder.encode(observation)
            loss, losses = encoder.training_loss(observation)
            targets = encoder.semantic_targets(observation)

            self.assertEqual(output.position.shape, (1, 2))
            self.assertEqual(output.semantic_logits.shape, (1, 7, 15, 16))
            self.assertEqual(output.semantic_ids.shape, (1, 15, 16))
            self.assertEqual(output.tokens.shape, (1, 240, 32))
            self.assertEqual(targets.shape, (1, 240, 256))
            self.assertEqual(encoder.patch_targets(observation).shape, (1, 15, 16))
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(losses), {"semantic", "position"})
            self.assertTrue(torch.all((output.position >= 0) & (output.position <= 1)))
        finally:
            stage.env.close()


    def test_procedural_trainer_executes_an_optimizer_step(self):
        frames = collect_procedural_frames(8, seed=12, rollout_steps=4)
        model = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        labels, positions = build_ground_truth(model, frames, batch_size=4)
        loader = make_loader(frames, labels, positions, batch_size=4, shuffle=False, seed=12)
        weights = class_weights(labels, model.spec.num_classes, torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        images, batch_labels, batch_positions = next(iter(loader))
        before = model.patch_embed.weight.detach().clone()
        loss, semantic_loss, position_loss = compute_loss(
            model,
            images,
            batch_labels,
            batch_positions,
            weights,
            position_weight=2.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertEqual(frames.shape, (8, 240, 256, 3))
        self.assertEqual(labels.shape, (8, 15, 16))
        self.assertEqual(positions.shape, (8, 2))
        self.assertTrue(torch.isfinite(semantic_loss))
        self.assertTrue(torch.isfinite(position_loss))
        self.assertFalse(torch.equal(before, model.patch_embed.weight))
    def test_block_stage_populates_hierarchical_streams_from_vision(self):
        encoder = BlockVisionTransformer(dim=32, depth=1, heads=4, drop=0.0).eval()
        stage = BlockSMBStage(vision=encoder)
        try:
            observation = stage.reset(seed=4)
            batch = stage.encode_observation(observation)

            self.assertEqual(batch.src_a.shape, (1, stage.spec.seq_len_a))
            self.assertEqual(batch.src_b.shape, (1, stage.spec.seq_len_b))
            self.assertEqual(batch.src_c.shape, (1, stage.spec.seq_len_c))
            self.assertIn("vision", batch.metadata)
            self.assertIsInstance(batch.metadata["vision"], VisionOutput)
        finally:
            stage.env.close()

    def test_existing_vit_checkpoint_loads_into_shared_architecture(self):
        checkpoint = Path("data/vit/vit_smb.pth")
        if not checkpoint.exists():
            self.skipTest("trained ViT checkpoint is not available")

        model = ViTSegmenter()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

        output = model.encode(torch.zeros(1, 3, 240, 256))
        self.assertEqual(output.semantic_logits.shape, (1, 13, 15, 16))
        self.assertEqual(output.position.shape, (1, 2))


class TestFullSMBVision(unittest.TestCase):
    def test_existing_deeplab_checkpoint_loads(self):
        from retroagi.stages.full_smb import FullSMBSegmentationVision

        checkpoint = Path("scripts/segmentation/MarioSegmentationModel.pth")
        if not checkpoint.exists():
            self.skipTest("trained DeepLab checkpoint is not available")

        model = FullSMBSegmentationVision(checkpoint=checkpoint)
        self.assertEqual(model.spec.num_classes, 6)
        self.assertEqual(model.spec.semantic_classes[-1], "mario")


if __name__ == "__main__":
    unittest.main()
