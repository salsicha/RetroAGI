"""Tests for the action vocabulary shared by Block SMB and Full SMB."""

import unittest

import numpy as np

from retroagi.core import (
    SMB_ACTION_SPECS,
    SMB_ACTIONS,
    SMBAction,
    ActionSpec,
    ContinuousControlSpec,
    action_backend_id,
    action_button_vector,
    block_smb_action,
    coerce_action_spec,
    coerce_smb_action,
    full_smb_action,
    smb_action_spec,
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
        self.assertEqual(action_backend_id(smb_action_spec(SMBAction.LEFT_JUMP)), 4)

    def test_full_smb_mapping_uses_button_names_not_positions(self):
        buttons = ("A", "RIGHT", "B", "LEFT", "START")
        mapped = full_smb_action(SMBAction.RIGHT_JUMP, buttons)

        np.testing.assert_array_equal(mapped, np.array([1, 1, 0, 0, 0], dtype=np.int8))

    def test_noop_releases_every_full_smb_button(self):
        mapped = full_smb_action(SMBAction.NOOP, ("LEFT", "RIGHT", "A", "B"))
        np.testing.assert_array_equal(mapped, np.zeros(4, dtype=np.int8))
        self.assertTrue(smb_action_spec(SMBAction.NOOP).is_noop)

    def test_generic_action_specs_support_buttons_release_and_continuous_axes(self):
        throttle = ActionSpec(
            name="throttle",
            stable_id=0,
            kind="continuous",
            continuous_controls=(ContinuousControlSpec("x", 0.75),),
        )
        jump = ActionSpec(name="jump", stable_id=1, buttons=("A",), backend_action_id=7)
        release = ActionSpec(name="release", stable_id=2, release_all=True)
        action_space = (throttle, jump, release)

        self.assertIs(coerce_action_spec(action_space, "jump"), jump)
        self.assertEqual(action_backend_id(jump), 7)
        np.testing.assert_array_equal(
            action_button_vector(jump, ("LEFT", "A")),
            np.array([0, 1], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            action_button_vector(release, ("LEFT", "A")),
            np.zeros(2, dtype=np.int8),
        )
        with self.assertRaisesRegex(ValueError, "continuous action"):
            action_backend_id(throttle)

    def test_invalid_action_and_button_layout_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "invalid SMB action"):
            coerce_smb_action(99)
        with self.assertRaisesRegex(ValueError, "missing.*A"):
            full_smb_action(SMBAction.JUMP, ("LEFT", "RIGHT", "B"))
        with self.assertRaisesRegex(ValueError, "not in this action space"):
            coerce_action_spec(SMB_ACTION_SPECS[:1], SMB_ACTION_SPECS[-1])
        with self.assertRaisesRegex(ValueError, "discrete action"):
            ActionSpec(
                name="bad",
                stable_id=0,
                continuous_controls=(ContinuousControlSpec("x", 0.5),),
            )


if __name__ == "__main__":
    unittest.main()
