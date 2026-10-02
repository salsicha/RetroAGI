"""Exact per-pixel labels for real Super Mario Bros frames, read from game memory.

These tests need the Super Mario Bros cartridge file installed for stable-retro;
without it they are skipped.
"""

import unittest
from collections import Counter

import numpy as np

from retroagi.core.smb_pixel_types import TYPE_ID
from retroagi.stages.full_smb import pixel_labels as P
from retroagi.stages.full_smb.play import POLICY_TEST_LEVELS
from retroagi.stages.full_smb.vision_frames import LEVELS, played_frames


def _cartridge_available() -> bool:
    try:
        P.cartridge()
    except Exception:  # noqa: BLE001 - any failure to find or read the file means skip
        return False
    return True


@unittest.skipUnless(_cartridge_available(), "Super Mario Bros cartridge file not installed")
class TestCartridgeTables(unittest.TestCase):
    def test_block_drawing_table_covers_every_typed_code(self):
        _, drawings = P.cartridge()
        self.assertEqual(set(drawings), set(P.BLOCK_TYPES))
        self.assertEqual(len(drawings), sum(P._DRAWINGS_PER_GROUP))

    def test_known_block_drawings(self):
        _, drawings = P.cartridge()
        # (top-left, top-right, bottom-left, bottom-right) tiles.
        self.assertEqual(drawings[0x51], (0x45, 0x45, 0x47, 0x47))  # brick
        self.assertEqual(drawings[0x47], (0x47, 0x47, 0x47, 0x47))  # castle wall scenery
        self.assertEqual(drawings[0xC0], (0x53, 0x54, 0x55, 0x56))  # question block
        self.assertEqual(drawings[0x64], (0xC6, 0xC7, 0xC8, 0xC9))  # Bullet Bill cannon
        self.assertEqual(drawings[0x00], (0x24, 0x24, 0x24, 0x24))  # empty sky

    def test_square_type_uses_the_block_map_and_the_drawing(self):
        _, drawings = P.cartridge()
        table = 1
        brick, wall, coin = drawings[0x51], drawings[0x47], drawings[0xC2]
        self.assertEqual(P.square_type(0x51, brick, table), "brick")
        # The castle wall looks like a plain brick; the block map tells them apart.
        self.assertEqual(P.square_type(0x00, wall, table), "background")
        self.assertEqual(P.square_type(0x52, wall, table), "brick")
        # A square whose block was just collected still shows the coin.
        self.assertEqual(P.square_type(0x00, coin, table), "coin")
        # A drawing with no drawn pixels has no type.
        self.assertIsNone(P.square_type(0x5F, drawings[0x00], table))
        with self.assertRaises(P.Unexplained):
            P.square_type(0x00, (0x45, 0x47, 0x45, 0x47), table)  # not a block drawing


class TestSpriteOwners(unittest.TestCase):
    def test_only_claimants_that_draw_the_shape_qualify(self):
        # A hammer's piece also lies inside an active springboard's claimed range.
        claims = [(8, "enemy 32", 3), (5, "hammer", 2)]
        self.assertEqual(P.piece_owner(claims, 0x81), ("hammer", 2))
        self.assertEqual(P.piece_owner(claims, 0xF2), ("enemy 32", 3))

    def test_unknown_or_conflicting_owners_are_refused(self):
        with self.assertRaises(P.Unexplained):
            P.piece_owner([(8, "enemy 06", 0)], 0x81)  # a Goomba does not draw hammers
        with self.assertRaises(P.Unexplained):
            P.piece_owner([], 0x70)  # nobody claims the piece
        with self.assertRaises(P.Unexplained):
            # Mario's fireball and a fire bar draw the same shape at the same rank.
            P.piece_owner([(4, "fireball", 0), (4, "enemy 1D", 1)], 0x64)

    def test_every_listed_owner_has_a_type(self):
        for owner in P.OWNER_DRAWINGS:
            self.assertIn(P.owner_type(owner), TYPE_ID, owner)


@unittest.skipUnless(_cartridge_available(), "Super Mario Bros cartridge file not installed")
class TestRealFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [
            frame
            for _, frame in zip(range(150), played_frames("Level1-1", frames=600, seed=3, every=4))
        ]

    def test_vision_learns_every_level_and_policy_test_levels_are_among_them(self):
        self.assertEqual(len(set(LEVELS)), 10)
        self.assertLessEqual(set(POLICY_TEST_LEVELS), set(LEVELS))
        self.assertIn("Level1-1", POLICY_TEST_LEVELS)

    def test_saved_state_memory_is_the_game_memory(self):
        frame, before, after = self.frames[10]
        blocks = P.state_blocks(after)
        self.assertEqual(blocks["RAM"].shape, (2048,))
        self.assertEqual(blocks["NTAR"].shape, (2048,))
        self.assertEqual(blocks["PRAM"].shape, (32,))

    def test_every_frame_is_rebuilt_exactly_and_labelled(self):
        seen = Counter()
        for frame, before, after in self.frames:
            out = P.label_frame(frame, before, after, family="Level1-1")
            self.assertEqual(out.image.shape, (240, 256, 3))
            self.assertEqual(out.labels.shape, (240, 256))
            self.assertEqual(out.labels.dtype, np.uint8)
            # The score bar is background.
            self.assertTrue(
                (out.labels[8:32][~np.isin(out.labels[8:32], [TYPE_ID["mario"]])] == 0).all()
            )
            seen.update(np.unique(out.labels).tolist())
        for name in ("background", "mario", "ground", "question_block", "pipe", "enemy"):
            self.assertIn(TYPE_ID[name], seen, name)

    def test_padded_border_repeats_the_edge_labels(self):
        frame, before, after = self.frames[20]
        out = P.label_frame(frame, before, after)
        np.testing.assert_array_equal(out.labels[:8], np.repeat(out.labels[8:9], 8, axis=0))
        np.testing.assert_array_equal(out.labels[:, :8], np.repeat(out.labels[:, 8:9], 8, axis=1))

    def test_a_picture_memory_does_not_explain_is_refused(self):
        frame, before, after = self.frames[30]
        changed = frame.copy()
        changed[100, 100] = 255 - changed[100, 100]
        with self.assertRaises(P.Unexplained):
            P.label_frame(changed, before, after)

    def test_mario_standing_flag_matches_the_start(self):
        frame, before, after = self.frames[0]
        self.assertTrue(P.label_frame(frame, before, after).on_ground)


if __name__ == "__main__":
    unittest.main()
