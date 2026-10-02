"""Tests for the unified curriculum vision interface."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from retroagi.core import LinearVisionEncoder, VisionOutput
from retroagi.core.scene_vision import (
    array_batches,
    evaluate_scene_vision,
    frame_targets,
    pixel_type_weights,
    save_scene_vision_checkpoint,
    stack_targets,
    train_scene_vision,
)
from retroagi.core.smb_pixel_types import PIXEL_TYPES, TYPE_ID, visible_window
from retroagi.core.smb_scene_labels import (
    ENEMY_KINDS,
    SUPPORTS,
    SceneObservation,
    scene_targets,
)
from retroagi.core.vision import SceneVisionTransformer
from retroagi.stages.block_smb import (
    BLOCK_SMB_SPEC,
    BlockSMBStage,
    BlockVisionTransformer,
    MarioScenarioEnv,
    load_block_vit_checkpoint,
)
from retroagi.stages.block_smb.vision_frames import VisionFrame
from retroagi.stages.full_smb import (
    FULL_SMB_SPEC,
    FullSMBStage,
    FullVisionTransformer,
    load_full_vit_checkpoint,
)


def block_frames(steps=40, actions=(1, 1, 2, 2, 2, 1)):
    """Frames, true labels and simulator flags from a short Block episode."""
    scenario = {
        "world_width": 400,
        "mario": [20, 208],
        "platforms": [[0, 220, 400, 20], [120, 160, 48, 10]],
        "coins": [[140, 140, 10, 10]],
        "enemies": [[200, 206, 180, 260, 0.5]],
        "goal": [380, 200, 16, 20],
    }
    env = MarioScenarioEnv()
    frames = []
    try:
        env.reset(scenario=scenario)
        for step in range(steps):
            env.step(actions[step % len(actions)])
            scene = env.scene_labels()
            frames.append(
                VisionFrame(
                    image=visible_window(env.render()),
                    labels=scene.types,
                    on_ground=scene.standing,
                    enemy_rects=tuple(tuple(r) for r in env.enemy_screen_rects()),
                    family="test",
                    route="scripted",
                    scene=scene,
                )
            )
    finally:
        env.close()
    return frames


def perfect_heads(targets: dict) -> dict:
    """Raw heads that decode to exactly the scene the targets describe."""
    one = torch.nn.functional.one_hot

    def batch(name):
        return torch.as_tensor(np.stack([t[name] for t in targets]))

    types = batch("types").long()
    kinds = batch("kind").clamp_min(0)
    return {
        "pixel_logits": one(types, len(PIXEL_TYPES)).permute(0, 3, 1, 2).float() * 20,
        "kind_logits": one(kinds, len(ENEMY_KINDS)).permute(0, 3, 1, 2).float() * 20,
        "facing_logits": one(batch("facing"), 2).float() * 20,
        "support_logits": one(batch("support"), len(SUPPORTS)).float() * 20,
    }


class OracleVision(BlockVisionTransformer):
    """Reports each frame's true scene, looked up by the picture."""

    def __init__(self, frames):
        super().__init__(dim=16, depth=1, heads=4)
        self.truth = {frame.image.tobytes(): scene_targets(frame.scene) for frame in frames}

    def heads(self, observation):
        images = np.asarray(observation).reshape(-1, 240, 256, 3)
        return perfect_heads([self.truth[image.tobytes()] for image in images])


class BackgroundOnlyVision(BlockVisionTransformer):
    """Sees nothing but sky: no objects, no terrain."""

    def __init__(self):
        super().__init__(dim=16, depth=1, heads=4)

    def heads(self, observation):
        images = np.asarray(observation).reshape(-1, 240, 256, 3)
        empty = {
            "types": np.zeros((240, 256), np.uint8),
            "kind": np.zeros((30, 32), np.int64),
            "facing": np.int64(1),
            "support": np.int64(0),
        }
        return perfect_heads([empty] * len(images))


class TestVisionInterface(unittest.TestCase):
    def test_linear_encoder_uses_common_output(self):
        encoder = LinearVisionEncoder(vocab_size=20, token_dim=16)
        output = encoder.encode(torch.arange(8))

        self.assertIsInstance(output, VisionOutput)
        self.assertEqual(output.position.shape, (1, 1))
        self.assertEqual(output.semantic_logits.shape, (1, 20, 1, 8))
        self.assertEqual(output.semantic_ids.shape, (1, 1, 8))
        self.assertEqual(output.tokens.shape, (1, 8, 16))

    def test_block_vit_is_the_shared_scene_model_and_reports_a_scene(self):
        encoder = BlockVisionTransformer(dim=32, depth=1, heads=4).eval()
        self.assertIsInstance(encoder, SceneVisionTransformer)
        self.assertEqual(encoder.spec.semantic_classes, PIXEL_TYPES)
        frames = block_frames(steps=2)
        with torch.no_grad():
            out = encoder.heads(np.stack([frame.image for frame in frames]))
            scenes = encoder.scene(np.stack([frame.image for frame in frames]))

        self.assertEqual(
            set(out), {"pixel_logits", "kind_logits", "facing_logits", "support_logits"}
        )
        self.assertEqual(out["pixel_logits"].shape, (2, len(PIXEL_TYPES), 240, 256))
        self.assertEqual(out["kind_logits"].shape, (2, len(ENEMY_KINDS), 30, 32))
        self.assertEqual(out["facing_logits"].shape, (2, 2))
        self.assertEqual(out["support_logits"].shape, (2, len(SUPPORTS)))
        self.assertEqual(len(scenes), 2)
        self.assertTrue(all(isinstance(scene, SceneObservation) for scene in scenes))

    def test_block_vit_refuses_pictures_that_are_not_256x240(self):
        encoder = BlockVisionTransformer(dim=16, depth=1, heads=4).eval()
        with self.assertRaisesRegex(ValueError, "never stretch"):
            encoder.heads(torch.zeros(1, 3, 120, 128))

    def test_shared_trainer_learns_from_screens_and_scene_targets(self):
        frames = block_frames(steps=4)
        images = np.stack([frame.image for frame in frames])
        targets = stack_targets([frame_targets(frame.scene) for frame in frames])
        model = BlockVisionTransformer(dim=16, depth=1, heads=4)
        before = model.kind_head.weight.detach().clone()
        seen = []

        best = train_scene_vision(
            model,
            array_batches(images, targets, batch_size=2, seed=0),
            epochs=2,
            steps_per_epoch=2,
            held_out_images=images[:2],
            held_out_targets={name: value[:2] for name, value in targets.items()},
            pixel_weights=pixel_type_weights(targets["types"]),
            device=torch.device("cpu"),
            warmup_steps=1,
            on_epoch=lambda epoch, metrics, improved: seen.append((epoch, improved)),
        )

        self.assertEqual([epoch for epoch, _ in seen], [1, 2])
        self.assertTrue(seen[0][1])
        self.assertEqual(
            set(best["held_out"]),
            {"pixels", "kind", "facing", "support", "total"},
        )
        self.assertFalse(torch.equal(before, model.kind_head.weight))

    def test_pixel_type_weights_lift_rare_types(self):
        labels = np.zeros((1, 240, 256), dtype=np.uint8)
        labels[0, :20] = TYPE_ID["ground"]
        labels[0, 100:110, 100:110] = TYPE_ID["mario"]
        weights = pixel_type_weights(labels)

        self.assertGreater(weights[TYPE_ID["mario"]], weights[TYPE_ID["ground"]])
        self.assertGreater(weights[TYPE_ID["ground"]], weights[TYPE_ID["background"]])
        shares = np.bincount(labels.ravel(), minlength=len(PIXEL_TYPES)) / labels.size
        self.assertAlmostEqual(float((weights.numpy() * shares).sum()), 1.0, places=5)

    def test_shared_measurements_score_the_true_scene_as_perfect(self):
        frames = block_frames()
        metrics = evaluate_scene_vision(OracleVision(frames), frames, batch_size=7)

        self.assertEqual(metrics["frames"], len(frames))
        self.assertEqual(metrics["pixels_correct"], 1.0)
        self.assertEqual(metrics["mario"]["found"], 1.0)
        self.assertEqual(metrics["mario"]["edge_error_px"]["max"], 0.0)
        self.assertEqual(metrics["mario"]["support"], 1.0)
        self.assertEqual(metrics["mario"]["facing"], 1.0)
        for name in ("enemies", "coins", "surfaces"):
            self.assertGreater(metrics[name]["true"], 0, name)
            self.assertEqual(metrics[name]["recall"], 1.0, name)
            self.assertEqual(metrics[name]["precision"], 1.0, name)
        self.assertEqual(metrics["enemy_kind_accuracy"], 1.0)

    def test_shared_measurements_flag_a_model_that_sees_only_sky(self):
        frames = block_frames()
        metrics = evaluate_scene_vision(BackgroundOnlyVision(), frames)

        self.assertLess(metrics["pixels_correct"], 1.0)
        self.assertEqual(metrics["mario"]["found"], 0.0)
        self.assertIsNone(metrics["mario"]["edge_error_px"])
        self.assertEqual(metrics["enemies"]["recall"], 0.0)
        self.assertEqual(metrics["surfaces"]["recall"], 0.0)

    def test_block_stage_populates_hierarchical_streams_from_vision(self):
        encoder = BlockVisionTransformer(dim=32, depth=1, heads=4).eval()
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

    def test_block_vit_policy_loader_freezes_checkpoint_by_default(self):
        source = BlockVisionTransformer(dim=16, depth=1, heads=4, refine_dim=8)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit_scene.pth"
            save_scene_vision_checkpoint(path, source, stage=BLOCK_SMB_SPEC.name, metrics={})

            result = load_block_vit_checkpoint(path, freeze=True)

        self.assertTrue(result.frozen)
        self.assertEqual(result.checkpoint["checkpoint_kind"], "vision_encoder")
        self.assertEqual(result.model.refine_dim, 8)
        self.assertFalse(any(parameter.requires_grad for parameter in result.model.parameters()))
        self.assertFalse(result.model.training)
        for name, value in result.model.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name])

    def test_block_vit_policy_loader_can_enable_fine_tuning(self):
        source = BlockVisionTransformer(dim=16, depth=1, heads=4)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit_scene.pth"
            save_scene_vision_checkpoint(path, source, stage=BLOCK_SMB_SPEC.name, metrics={})

            result = load_block_vit_checkpoint(path, freeze=False)

        self.assertFalse(result.frozen)
        self.assertTrue(all(parameter.requires_grad for parameter in result.model.parameters()))
        self.assertTrue(result.model.training)

    def test_block_vit_loader_rejects_another_games_weights(self):
        source = SceneVisionTransformer(name="full_smb_vit", dim=16, depth=1, heads=4)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full_smb_scene.pth"
            save_scene_vision_checkpoint(path, source, stage=BLOCK_SMB_SPEC.name, metrics={})

            with self.assertRaisesRegex(Exception, "full_smb_vit"):
                load_block_vit_checkpoint(path)


class FakeRetroEnv:
    """A stand-in emulator for building a Full SMB stage without the ROM."""

    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def reset(self, seed=None):
        return np.zeros((224, 240, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((224, 240, 3), dtype=np.uint8), 0.0, False, False, {}

    def close(self):
        pass


class TestFullSMBVision(unittest.TestCase):
    def test_full_vit_is_the_shared_scene_model_under_the_full_name(self):
        encoder = FullVisionTransformer(dim=16, depth=1, heads=4).eval()
        block = BlockVisionTransformer(dim=16, depth=1, heads=4).eval()
        self.assertIsInstance(encoder, SceneVisionTransformer)
        self.assertEqual(encoder.spec.name, "full_smb_vit")
        self.assertEqual(encoder.spec.semantic_classes, PIXEL_TYPES)
        with self.assertRaisesRegex(ValueError, "never stretch"):
            encoder.heads(torch.zeros(1, 3, 224, 240))
        picture = np.zeros((1, 240, 256, 3), dtype=np.uint8)
        with torch.no_grad():
            full, practice = encoder.heads(picture), block.heads(picture)
        # Same class, same outputs: only the learned numbers differ.
        self.assertEqual(set(full), set(practice))
        for name in full:
            self.assertEqual(full[name].shape, practice[name].shape, name)
        self.assertEqual(encoder.architecture()["metadata"], block.architecture()["metadata"])

    def test_full_vit_policy_loader_freezes_checkpoint_by_default(self):
        source = FullVisionTransformer(dim=16, depth=1, heads=4, refine_dim=8)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full_vit_scene.pth"
            save_scene_vision_checkpoint(path, source, stage=FULL_SMB_SPEC.name, metrics={})

            result = load_full_vit_checkpoint(path, freeze=True)

        self.assertTrue(result.frozen)
        self.assertEqual(result.path, path)
        self.assertEqual(result.checkpoint["checkpoint_kind"], "vision_encoder")
        self.assertIsInstance(result.model, FullVisionTransformer)
        self.assertEqual(result.model.refine_dim, 8)
        self.assertFalse(any(parameter.requires_grad for parameter in result.model.parameters()))
        self.assertFalse(result.model.training)
        for name, value in result.model.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name])

    def test_full_vit_policy_loader_can_enable_fine_tuning(self):
        source = FullVisionTransformer(dim=16, depth=1, heads=4)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "full_vit_scene.pth"
            save_scene_vision_checkpoint(path, source, stage=FULL_SMB_SPEC.name, metrics={})

            result = load_full_vit_checkpoint(path, freeze=False)

        self.assertFalse(result.frozen)
        self.assertTrue(all(parameter.requires_grad for parameter in result.model.parameters()))
        self.assertTrue(result.model.training)

    def test_full_vit_loader_rejects_another_games_weights(self):
        source = BlockVisionTransformer(dim=16, depth=1, heads=4)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "block_vit_scene.pth"
            save_scene_vision_checkpoint(path, source, stage=FULL_SMB_SPEC.name, metrics={})

            with self.assertRaisesRegex(Exception, "block_smb_vit"):
                load_full_vit_checkpoint(path)

    def test_full_vit_loader_names_the_trainer_when_the_checkpoint_is_missing(self):
        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "train_full_vit.py"):
                load_full_vit_checkpoint(Path(tmpdir) / "missing.pth")

    def test_full_stage_defaults_to_the_trained_full_vit_on_the_cpu(self):
        trained = FullVisionTransformer(dim=16, depth=1, heads=4)
        with patch("retroagi.stages.full_smb.adapter.load_full_vit_checkpoint") as load:
            load.return_value.model = trained
            stage = FullSMBStage(env=FakeRetroEnv())

        load.assert_called_once_with()
        self.assertIs(stage.vision, trained)
        self.assertFalse(stage.vision.training)

    def test_full_stage_reads_screens_through_the_shared_pixel_types(self):
        stage = FullSMBStage(
            env=FakeRetroEnv(), vision=FullVisionTransformer(dim=16, depth=1, heads=4)
        )
        observation = stage.reset(seed=0)
        with torch.no_grad():
            batch = stage.encode_observation(observation)

        self.assertEqual(observation.shape, (240, 256, 3))
        self.assertEqual(batch.metadata["vision"].metadata["semantic_classes"], PIXEL_TYPES)


if __name__ == "__main__":
    unittest.main()
