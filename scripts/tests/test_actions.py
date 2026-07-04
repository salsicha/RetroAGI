"""Tests for the action vocabulary shared by Block SMB and Full SMB."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from retroagi.core import (
    SMB_ACTION_SPECS,
    SMB_ACTIONS,
    SMBAction,
    SMBJumpActionTerminator,
    SMBWalkActionLimiter,
    ActionSpec,
    ContinuousControlSpec,
    VisionOutput,
    action_backend_id,
    action_button_vector,
    block_smb_action,
    coerce_action_spec,
    coerce_smb_action,
    full_smb_action,
    is_smb_walk_action,
    smb_action_spec,
    smb_jump_release_action,
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

    def test_jump_release_actions_preserve_horizontal_intent(self):
        self.assertIs(smb_jump_release_action(SMBAction.RIGHT_JUMP), SMBAction.RIGHT)
        self.assertIs(smb_jump_release_action(SMBAction.LEFT_JUMP), SMBAction.LEFT)
        self.assertIs(smb_jump_release_action(SMBAction.JUMP), SMBAction.NOOP)

    def test_walk_limiter_releases_after_one_second_window(self):
        limiter = SMBWalkActionLimiter(max_walk_seconds=1.0, actions_per_second=2.0)

        self.assertTrue(is_smb_walk_action(SMBAction.RIGHT))
        self.assertTrue(is_smb_walk_action(SMBAction.LEFT))
        self.assertFalse(is_smb_walk_action(SMBAction.RIGHT_JUMP))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.NOOP))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))

    def test_walk_limiter_resets_on_non_walk_or_direction_change(self):
        limiter = SMBWalkActionLimiter(max_walk_seconds=1.0, actions_per_second=2.0)

        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.JUMP), int(SMBAction.JUMP))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.LEFT), int(SMBAction.LEFT))
        self.assertEqual(limiter.filter_action(SMBAction.LEFT), int(SMBAction.LEFT))
        self.assertEqual(limiter.filter_action(SMBAction.LEFT), int(SMBAction.NOOP))

    def test_jump_terminator_releases_after_vit_support_landing(self):
        terminator = SMBJumpActionTerminator()

        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT_JUMP),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(0)),
            ),
            int(SMBAction.RIGHT_JUMP),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(2)),
            ),
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT_JUMP),
        )

    def test_jump_terminator_releases_on_vit_enemy_contact(self):
        terminator = SMBJumpActionTerminator()
        self.assertEqual(
            terminator.filter_action(
                SMBAction.JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.JUMP),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.JUMP,
                batch=self._batch_with_vision(self._support_vision(0)),
            ),
            int(SMBAction.JUMP),
        )

        labels = torch.zeros(1, 5, 5, dtype=torch.long)
        labels[0, 2, 2] = 1
        labels[0, 3, 2] = 2
        vision = self._support_vision(
            0,
            semantic_ids=labels,
            semantic_classes=("background", "mario", "enemy"),
        )

        self.assertEqual(
            terminator.filter_action(SMBAction.JUMP, batch=self._batch_with_vision(vision)),
            int(SMBAction.NOOP),
        )

    @staticmethod
    def _batch_with_vision(vision: VisionOutput):
        return SimpleNamespace(metadata={"vision": vision})

    @staticmethod
    def _support_vision(
        support_id: int,
        *,
        semantic_ids: torch.Tensor | None = None,
        semantic_classes: tuple[str, ...] = ("background", "mario", "enemy"),
    ) -> VisionOutput:
        if semantic_ids is None:
            semantic_ids = torch.zeros(1, 5, 5, dtype=torch.long)
            semantic_ids[0, 2, 2] = 1
        semantic_logits = torch.zeros(
            semantic_ids.shape[0],
            len(semantic_classes),
            semantic_ids.shape[1],
            semantic_ids.shape[2],
        )
        semantic_logits.scatter_(1, semantic_ids.unsqueeze(1), 1.0)
        return VisionOutput(
            position=torch.zeros(1, 2),
            semantic_logits=semantic_logits,
            semantic_ids=semantic_ids,
            tokens=torch.zeros(1, 1, 4),
            metadata={
                "semantic_classes": semantic_classes,
                "support_classes": ("air", "ground", "platform"),
            },
            support_logits=torch.nn.functional.one_hot(
                torch.tensor([support_id]),
                num_classes=3,
            ).float(),
            support_ids=torch.tensor([support_id]),
        )


if __name__ == "__main__":
    unittest.main()
