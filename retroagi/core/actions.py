"""The six button actions shared by Block SMB and Full SMB."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable

import numpy as np


class SMBAction(IntEnum):
    """Stable action IDs shared by Block SMB and Full SMB."""

    NOOP = 0
    RIGHT = 1
    RIGHT_JUMP = 2
    LEFT = 3
    LEFT_JUMP = 4
    JUMP = 5


SMB_ACTIONS = tuple(SMBAction)
SMB_JUMP_ACTIONS = frozenset((SMBAction.RIGHT_JUMP, SMBAction.LEFT_JUMP, SMBAction.JUMP))

# The NES buttons each action holds down; every other button is released.
SMB_ACTION_BUTTONS = {
    SMBAction.NOOP: (),
    SMBAction.RIGHT: ("RIGHT",),
    SMBAction.RIGHT_JUMP: ("RIGHT", "A"),
    SMBAction.LEFT: ("LEFT",),
    SMBAction.LEFT_JUMP: ("LEFT", "A"),
    SMBAction.JUMP: ("A",),
}


def full_smb_action(action: SMBAction | int, buttons: Iterable[str]) -> np.ndarray:
    """The emulator's button vector (one 0 or 1 per button, in ``buttons`` order)."""
    names = tuple(str(button).upper() for button in buttons)
    pressed = SMB_ACTION_BUTTONS[SMBAction(action)]
    missing = sorted(set(pressed).difference(names))
    if missing:
        raise ValueError(
            f"the emulator has no {missing!r} button for action {SMBAction(action).name}"
        )
    return np.asarray([name in pressed for name in names], dtype=np.int8)
