"""Tests for the action vocabulary shared by Block SMB and Full SMB."""

import unittest

import numpy as np

from retroagi.core import (
    SMB_ACTIONS,
    SMBAction,
    block_smb_action,
    coerce_smb_action,
    full_smb_action,
)


class TestSMBActionVocabulary(unittest.TestCase):
    def test_ids_preserve_block_smb_action_order(self):
        self.assertEqual(
            [(action.name, action.value) for action in SMB_ACTIONS],
            [
                ("NOOP", 0),
                ("RIGHT", 1),
                ("RIGHT_JUMP", 2),
                ("LEFT", 3),
                ("LEFT_JUMP", 4),
                ("JUMP", 5),
            ],
        )
        self.assertEqual(block_smb_action(SMBAction.LEFT_JUMP), 4)
        self.assertIs(coerce_smb_action(2), SMBAction.RIGHT_JUMP)

    def test_full_smb_mapping_uses_button_names_not_positions(self):
        buttons = ("A", "RIGHT", "B", "LEFT", "START")
        mapped = full_smb_action(SMBAction.RIGHT_JUMP, buttons)

        np.testing.assert_array_equal(mapped, np.array([1, 1, 0, 0, 0], dtype=np.int8))

    def test_noop_releases_every_full_smb_button(self):
        mapped = full_smb_action(SMBAction.NOOP, ("LEFT", "RIGHT", "A", "B"))
        np.testing.assert_array_equal(mapped, np.zeros(4, dtype=np.int8))

    def test_invalid_action_and_button_layout_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "invalid SMB action"):
            coerce_smb_action(99)
        with self.assertRaisesRegex(ValueError, "missing.*A"):
            full_smb_action(SMBAction.JUMP, ("LEFT", "RIGHT", "B"))


if __name__ == "__main__":
    unittest.main()
