"""Tests for deterministic vision-to-hierarchy fusion."""

import unittest

import numpy as np
import torch

from retroagi.core import StageSpec, VisionHierarchyProjector, VisionOutput
from retroagi.stages.block_smb import BlockSMBObservationConfig, BlockSMBStage


class TestVisionHierarchyProjector(unittest.TestCase):
    def setUp(self):
        self.spec = StageSpec(
            name="test",
            observation_kind="vision",
            action_kind="test",
            seq_len_a=8,
            ratio_ab=2,
            ratio_bc=4,
            vocab_size=20,
        )
        self.projector = VisionHierarchyProjector(self.spec)

    def make_vision(self, position=None, token_offset=0.0):
        region_classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 1])
        column_classes = region_classes.repeat_interleave(2)
        logits = torch.full((1, 7, 2, 16), -10.0)
        for column, class_id in enumerate(column_classes):
            logits[0, class_id, :, column] = 10.0

        tokens = torch.linspace(-2, 2, 240 * 4).reshape(1, 240, 4) + token_offset
        return VisionOutput(
            position=torch.tensor(position or [[0.25, 0.75]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=tokens,
        )

    def test_spatial_semantics_feed_a_and_b_without_flat_sampling(self):
        batch = self.projector.project(self.make_vision())

        self.assertEqual(batch.src_a.tolist(), [[0, 1, 2, 3, 4, 5, 6, 1]])
        self.assertEqual(batch.src_b.tolist(), [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 1]])

    def test_foreground_presence_wins_over_more_confident_background_patch(self):
        logits = torch.full((1, 7, 2, 4), -20.0)
        logits[:, 0, 0, :] = 20.0
        logits[:, 0, 1, :] = 20.0

        for column, class_id in enumerate((1, 2, 5)):
            logits[0, 0, 1, column] = 9.0
            logits[0, class_id, 1, column] = 10.0

        probabilities = logits.softmax(dim=1)
        stream = self.projector._semantic_stream(probabilities, length=4)

        self.assertEqual(stream.tolist(), [[1, 2, 5, 0]])

    def test_c_stream_has_stable_position_semantic_state_and_token_slots(self):
        state = torch.linspace(-1, 1, 14)
        batch = self.projector.project(self.make_vision(), state=state)
        fusion = batch.metadata["vision_fusion"]

        self.assertEqual(fusion["c_position"], (0, 2))
        self.assertEqual(fusion["c_semantic_probabilities"], (2, 9))
        self.assertEqual(fusion["c_support_state"], (9, 9))
        self.assertEqual(fusion["c_state"], (9, 23))
        self.assertEqual(fusion["c_patch_tokens"], (23, 64))
        torch.testing.assert_close(batch.src_c[:, :2], torch.tensor([[0.25, 0.75]]))
        torch.testing.assert_close(batch.src_c[:, 9:23], state.unsqueeze(0))
        self.assertTrue(torch.all(batch.src_c[:, 23:] >= -1))
        self.assertTrue(torch.all(batch.src_c[:, 23:] <= 1))

    def test_position_and_tokens_change_only_their_declared_c_sections(self):
        state = torch.zeros(14)
        baseline = self.projector.project(self.make_vision(), state=state)
        changed_position = self.projector.project(
            self.make_vision(position=[[0.75, 0.25]]), state=state
        )
        changed_tokens = self.projector.project(self.make_vision(token_offset=1.0), state=state)

        torch.testing.assert_close(baseline.src_c[:, 2:], changed_position.src_c[:, 2:])
        self.assertFalse(torch.equal(baseline.src_c[:, :2], changed_position.src_c[:, :2]))
        torch.testing.assert_close(baseline.src_c[:, :23], changed_tokens.src_c[:, :23])
        self.assertFalse(torch.equal(baseline.src_c[:, 23:], changed_tokens.src_c[:, 23:]))
        torch.testing.assert_close(baseline.src_a, changed_tokens.src_a)
        torch.testing.assert_close(baseline.src_b, changed_tokens.src_b)

    def test_support_logits_feed_declared_c_section(self):
        vision = self.make_vision()
        vision.support_logits = torch.tensor([[-4.0, 4.0, -4.0]])
        vision.support_ids = torch.tensor([1])
        batch = self.projector.project(vision, state=torch.zeros(14))
        fusion = batch.metadata["vision_fusion"]

        self.assertEqual(fusion["c_support_state"], (9, 12))
        self.assertEqual(fusion["c_state"], (12, 26))
        self.assertEqual(fusion["c_patch_tokens"], (26, 64))
        support = batch.src_c[:, 9:12]
        self.assertEqual(int(support.argmax(dim=1).item()), 1)
        self.assertGreater(float(support[0, 1]), 0.99)

    def test_block_stage_normalizes_stacks_and_masks_observations(self):
        vision = self.make_vision()

        class CaptureVision:
            def __init__(self):
                self.inputs = []

            def encode(self, observation):
                self.inputs.append(observation)
                return vision

        capture = CaptureVision()
        stage = BlockSMBStage(
            vision=capture,
            observation_config=BlockSMBObservationConfig(frame_stack=3),
        )
        try:
            observation = np.full((240, 256, 3), 255, dtype=np.uint8)
            state_vec = np.array(
                [
                    2.0,
                    -2.0,
                    np.nan,
                    np.inf,
                    -np.inf,
                    0.5,
                    0.0,
                    1.0,
                    -1.0,
                    0.25,
                    -0.25,
                    0.75,
                    -0.75,
                    0.0,
                ],
                dtype=np.float32,
            )
            info = {"state_vec": state_vec}

            batch = stage.encode_observation(observation, info)
            observation_meta = batch.metadata["observation"]

            self.assertEqual(observation_meta["frame_stack"].shape, (1, 3, 3, 240, 256))
            self.assertEqual(observation_meta["frame_mask"].tolist(), [[False, False, True]])
            self.assertEqual(batch.metadata["episode"]["mask"].tolist(), [1.0])
            self.assertEqual(capture.inputs[-1].dtype, torch.float32)
            self.assertTrue(torch.all(capture.inputs[-1] == 1.0))
            self.assertTrue(torch.all(observation_meta["frame_stack"] >= 0.0))
            self.assertTrue(torch.all(observation_meta["frame_stack"] <= 1.0))

            c_state_start, c_state_end = batch.metadata["vision_fusion"]["c_state"]
            c_state = batch.src_c[:, c_state_start:c_state_end]
            self.assertTrue(torch.isfinite(c_state).all())
            self.assertTrue(torch.all(c_state >= -1.0))
            self.assertTrue(torch.all(c_state <= 1.0))

            next_observation = np.zeros((240, 256, 3), dtype=np.uint8)
            batch = stage.encode_observation(next_observation, info)
            self.assertEqual(
                batch.metadata["observation"]["frame_mask"].tolist(),
                [[False, True, True]],
            )
            self.assertTrue(torch.all(capture.inputs[-1] == 0.0))
        finally:
            stage.env.close()

    def test_block_stage_episode_mask_drops_on_truncation(self):
        vision = self.make_vision()

        class StaticVision:
            def encode(self, observation):
                return vision

        stage = BlockSMBStage(
            vision=StaticVision(),
            observation_config=BlockSMBObservationConfig(frame_stack=2),
        )
        try:
            stage.env.max_steps = 1
            stage.reset(seed=8)
            observation, _, terminated, truncated, info = stage.step(0)
            batch = stage.encode_observation(observation, info)

            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(batch.metadata["episode"]["mask"].tolist(), [0.0])
            self.assertEqual(batch.metadata["episode"]["truncated"], True)
            self.assertEqual(batch.metadata["observation"]["frame_mask"].tolist(), [[True, True]])
        finally:
            stage.env.close()

    def test_rejects_semantic_vocab_overflow(self):
        vision = self.make_vision()
        vision.semantic_logits = torch.zeros(1, 21, 2, 16)
        with self.assertRaisesRegex(ValueError, "exceed vocab_size"):
            self.projector.project(vision)

    def test_rejects_invalid_support_shape(self):
        vision = self.make_vision()
        vision.support_logits = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "support_logits"):
            self.projector.project(vision)

    def test_block_stage_uses_shared_projector_contract(self):
        vision = self.make_vision()

        class StaticVision:
            def encode(self, observation):
                return vision

        stage = BlockSMBStage(vision=StaticVision())
        try:
            observation = np.zeros((240, 256, 3), dtype=np.uint8)
            info = {"state_vec": np.zeros(14, dtype=np.float32)}
            batch = stage.encode_observation(observation, info)

            self.assertEqual(batch.src_a.shape, (1, 8))
            self.assertEqual(batch.src_b.shape, (1, 16))
            self.assertEqual(batch.src_c.shape, (1, 64))
            self.assertEqual(batch.metadata["vision_fusion"]["c_patch_tokens"], (23, 64))
        finally:
            stage.env.close()


if __name__ == "__main__":
    unittest.main()
