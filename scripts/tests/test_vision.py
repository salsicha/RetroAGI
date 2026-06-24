"""Tests for the unified curriculum vision interface."""

import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from retroagi.core import (
    LinearVisionEncoder,
    VisionOutput,
    build_checkpoint,
    save_checkpoint,
)
from retroagi.stages.block_smb import (
    BLOCK_SMB_SPEC,
    BlockSMBStage,
    BlockVisionTransformer,
    BlockVITPerceptionThresholds,
    evaluate_block_vit_perception,
    load_block_vit_checkpoint,
)
from retroagi.stages.full_smb import load_full_smb_vit_checkpoint
from scripts.vit.train_block_vit import (
    build_ground_truth,
    class_weights,
    collect_procedural_frames,
    compute_loss,
    make_loader,
)

GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def skip_unavailable_checkpoint(testcase: unittest.TestCase, checkpoint: Path, label: str) -> None:
    if not checkpoint.exists():
        testcase.skipTest(f"{label} checkpoint is not available")
    try:
        with checkpoint.open("rb") as handle:
            prefix = handle.read(len(GIT_LFS_POINTER_PREFIX))
    except OSError as exc:
        testcase.skipTest(f"{label} checkpoint cannot be read: {exc}")
    if prefix == GIT_LFS_POINTER_PREFIX:
        testcase.skipTest(f"{label} checkpoint is a Git LFS pointer without its blob")


class OracleBlockVisionTransformer(BlockVisionTransformer):
    def forward(self, observation):
        labels = self.patch_targets(observation)
        logits = torch.full(
            (labels.shape[0], self.spec.num_classes, *labels.shape[1:]),
            -12.0,
            device=labels.device,
        )
        logits.scatter_(1, labels.unsqueeze(1), 12.0)
        return VisionOutput(
            position=self.position_targets(observation),
            semantic_logits=logits,
            semantic_ids=labels,
            tokens=torch.zeros(
                labels.shape[0],
                labels.shape[1] * labels.shape[2],
                self.spec.token_dim,
                device=labels.device,
            ),
            metadata={},
        )


class BackgroundOnlyBlockVisionTransformer(BlockVisionTransformer):
    def forward(self, observation):
        image = torch.as_tensor(observation)
        if image.ndim == 3:
            batch_size = 1
        else:
            batch_size = image.shape[0]
        height, width = self.grid_size
        logits = torch.full(
            (batch_size, self.spec.num_classes, height, width),
            -12.0,
            device=self.pos_embed.device,
        )
        logits[:, 0] = 12.0
        return VisionOutput(
            position=torch.zeros(batch_size, 2, device=self.pos_embed.device),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.zeros(
                batch_size,
                height * width,
                self.spec.token_dim,
                device=self.pos_embed.device,
            ),
            metadata={},
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

    def test_block_vit_perception_diagnostic_accepts_oracle_predictions(self):
        encoder = OracleBlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        stage = BlockSMBStage(vision=encoder)
        try:
            frames = []
            observation = stage.reset(seed=5)
            frames.append(torch.as_tensor(observation))
            observation, _reward, _terminated, _truncated, _info = stage.step(1)
            frames.append(torch.as_tensor(observation))
            metrics = evaluate_block_vit_perception(
                encoder,
                torch.stack(frames),
                thresholds=BlockVITPerceptionThresholds(
                    min_accuracy=1.0,
                    min_foreground_accuracy=1.0,
                    min_mean_iou=1.0,
                    max_position_rmse=0.0,
                    min_position_within_tolerance=1.0,
                    position_tolerance=0.0,
                ),
                batch_size=1,
            )
        finally:
            stage.env.close()

        self.assertFalse(metrics["bottleneck"])
        self.assertEqual(metrics["bottleneck_reasons"], [])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["foreground_accuracy"], 1.0)
        self.assertEqual(metrics["mean_iou"], 1.0)
        self.assertEqual(metrics["position_rmse"], 0.0)

    def test_block_vit_perception_diagnostic_flags_bad_predictions(self):
        encoder = BackgroundOnlyBlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        stage = BlockSMBStage(vision=encoder)
        try:
            observation = stage.reset(seed=6)
            metrics = evaluate_block_vit_perception(
                encoder,
                torch.as_tensor(observation),
                thresholds=BlockVITPerceptionThresholds(
                    min_accuracy=0.99,
                    min_foreground_accuracy=0.99,
                    min_mean_iou=0.99,
                    max_position_rmse=0.001,
                    min_position_within_tolerance=0.99,
                    position_tolerance=0.001,
                ),
            )
        finally:
            stage.env.close()

        self.assertTrue(metrics["bottleneck"])
        self.assertIn("foreground_accuracy", metrics["bottleneck_reasons"])
        self.assertIn("mean_iou", metrics["bottleneck_reasons"])
        self.assertIn("position_rmse", metrics["bottleneck_reasons"])

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

    def make_block_vit_checkpoint(self, path: Path, model: BlockVisionTransformer) -> None:
        checkpoint = build_checkpoint(
            stage=BLOCK_SMB_SPEC.name,
            model_name=model.spec.name,
            checkpoint_kind="vision_encoder",
            states={"model": model.state_dict()},
            config={
                "model": {
                    "name": model.spec.name,
                    "hidden_dim": model.spec.token_dim,
                    "depth": len(model.encoder.layers),
                    "heads": int(model.encoder.layers[0].self_attn.num_heads),
                    "patch_size": model.patch_size,
                    "dropout": float(model.dropout.p),
                }
            },
            specs={"vision": asdict(model.spec)},
        )
        save_checkpoint(path, checkpoint)

    def test_block_vit_policy_loader_freezes_checkpoint_by_default(self):
        source = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit.pth"
            self.make_block_vit_checkpoint(path, source)

            result = load_block_vit_checkpoint(path, freeze=True)

        self.assertTrue(result.frozen)
        self.assertEqual(result.checkpoint["checkpoint_kind"], "vision_encoder")
        self.assertFalse(any(parameter.requires_grad for parameter in result.model.parameters()))
        self.assertFalse(result.model.training)
        for name, value in result.model.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name])

    def test_block_vit_policy_loader_normalizes_legacy_checkpoint(self):
        source = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_block_vit.pth"
            torch.save(
                {
                    "model_state": source.state_dict(),
                    "epoch": 3,
                    "metrics": {"mean_iou": 0.5},
                    "config": {
                        "dim": 16,
                        "depth": 1,
                        "heads": 4,
                        "patch_size": 16,
                        "dropout": 0.0,
                    },
                    "vision_spec": asdict(source.spec),
                },
                path,
            )

            result = load_block_vit_checkpoint(path, freeze=True)

        self.assertEqual(result.checkpoint["checkpoint_schema_version"], 1)
        self.assertTrue(result.checkpoint["metadata"]["legacy_checkpoint"])
        self.assertEqual(result.checkpoint["metadata"]["source_path"], str(path))
        self.assertEqual(result.checkpoint["metrics"]["mean_iou"], 0.5)
        for name, value in result.model.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name])

    def test_block_vit_policy_loader_can_enable_fine_tuning(self):
        source = BlockVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit.pth"
            self.make_block_vit_checkpoint(path, source)

            result = load_block_vit_checkpoint(path, freeze=False)

        self.assertFalse(result.frozen)
        self.assertTrue(all(parameter.requires_grad for parameter in result.model.parameters()))
        self.assertTrue(result.model.training)

    def test_existing_vit_checkpoint_loads_into_shared_architecture(self):
        checkpoint = Path("data/vit/vit_smb.pth")
        skip_unavailable_checkpoint(self, checkpoint, "trained ViT")

        result = load_full_smb_vit_checkpoint(checkpoint)

        self.assertTrue(result.checkpoint["metadata"]["legacy_checkpoint"])
        self.assertEqual(result.path, checkpoint)

        output = result.model.encode(torch.zeros(1, 3, 240, 256))
        self.assertEqual(output.semantic_logits.shape, (1, 13, 15, 16))
        self.assertEqual(output.position.shape, (1, 2))

    def test_full_smb_vit_loader_reports_git_lfs_pointer_without_blob(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "full_smb_vit.pth"
            checkpoint.write_text(
                "\n".join(
                    (
                        "version https://git-lfs.github.com/spec/v1",
                        "oid sha256:0123456789abcdef",
                        "size 12345",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Git LFS pointer"):
                load_full_smb_vit_checkpoint(checkpoint)


class TestFullSMBVision(unittest.TestCase):
    def test_full_smb_segmentation_defaults_to_vit_contract(self):
        from retroagi.stages.full_smb import FullSMBSegmentationVision

        model = FullSMBSegmentationVision(
            checkpoint=None,
            dim=16,
            depth=1,
            heads=4,
            drop=0.0,
        )
        output = model.encode(torch.zeros(1, 3, 64, 64))

        self.assertEqual(model.spec.name, "full_smb_vit")
        self.assertEqual(model.spec.num_classes, 13)
        self.assertEqual(model.spec.semantic_classes[8], "mario")
        self.assertEqual(output.semantic_logits.shape, (1, 13, 15, 16))
        self.assertEqual(output.semantic_ids.shape, (1, 15, 16))
        self.assertEqual(output.position.shape, (1, 2))
        self.assertEqual(output.tokens.shape, (1, 240, 16))

    def test_existing_full_smb_vit_checkpoint_loads(self):
        from retroagi.stages.full_smb import FullSMBSegmentationVision

        checkpoint = Path("data/vit/full_smb_vit.pth")
        skip_unavailable_checkpoint(self, checkpoint, "trained Full SMB ViT")

        model = FullSMBSegmentationVision(checkpoint=checkpoint)
        output = model.encode(torch.zeros(1, 3, 240, 256))

        self.assertTrue(model.frozen)
        self.assertEqual(model.checkpoint_path, checkpoint)
        self.assertEqual(model.checkpoint["checkpoint_schema_version"], 1)
        self.assertEqual(model.spec.name, "full_smb_vit")
        self.assertEqual(output.semantic_logits.shape, (1, 13, 15, 16))
        self.assertEqual(output.position.shape, (1, 2))

    def test_legacy_full_smb_vit_state_dict_loads(self):
        from retroagi.stages.full_smb import FullSMBSegmentationVision

        checkpoint = Path("data/vit/vit_smb.pth")
        skip_unavailable_checkpoint(self, checkpoint, "legacy Full SMB ViT")

        model = FullSMBSegmentationVision(checkpoint=checkpoint)
        output = model.encode(torch.zeros(1, 3, 240, 256))

        self.assertTrue(model.checkpoint["metadata"]["legacy_checkpoint"])
        self.assertEqual(model.spec.name, "full_smb_vit")
        self.assertEqual(output.semantic_ids.shape, (1, 15, 16))

    def test_existing_deeplab_checkpoint_loads(self):
        from retroagi.stages.full_smb import FullSMBDeepLabSegmentationVision

        checkpoint = Path("scripts/segmentation/MarioSegmentationModel.pth")
        skip_unavailable_checkpoint(self, checkpoint, "trained DeepLab")

        model = FullSMBDeepLabSegmentationVision(checkpoint=checkpoint)
        self.assertEqual(model.spec.num_classes, 6)
        self.assertEqual(model.spec.semantic_classes[-1], "mario")


if __name__ == "__main__":
    unittest.main()
