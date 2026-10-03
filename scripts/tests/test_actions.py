"""Tests for the action vocabulary shared by Block SMB and Full SMB."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from retroagi.core.actions import SMBAction, full_smb_action
from retroagi.core.interfaces import VisionOutput


class TestSMBActionVocabulary(unittest.TestCase):
    def test_full_smb_mapping_uses_button_names_not_positions(self):
        buttons = ("A", "RIGHT", "B", "LEFT", "START")
        mapped = full_smb_action(SMBAction.RIGHT_JUMP, buttons)

        np.testing.assert_array_equal(mapped, np.array([1, 1, 0, 0, 0], dtype=np.int8))

    @staticmethod
    def _duration_motor(logits: list[float]) -> SimpleNamespace:
        return SimpleNamespace(
            hold_duration_logits=torch.tensor([[logits]]),
            duration_bin_values=torch.tensor([2.0, 4.0, 16.0]),
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
