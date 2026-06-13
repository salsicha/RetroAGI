"""Shared action vocabulary and backend mappings for SMB stages."""

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

_NES_BUTTONS = {
    SMBAction.NOOP: (),
    SMBAction.RIGHT: ("RIGHT",),
    SMBAction.RIGHT_JUMP: ("RIGHT", "A"),
    SMBAction.LEFT: ("LEFT",),
    SMBAction.LEFT_JUMP: ("LEFT", "A"),
    SMBAction.JUMP: ("A",),
}


def coerce_smb_action(action: SMBAction | int) -> SMBAction:
    """Return a named action or raise a clear error for an invalid ID."""
    try:
        return SMBAction(action)
    except (TypeError, ValueError) as exc:
        valid = ", ".join(f"{item.name}={item.value}" for item in SMB_ACTIONS)
        raise ValueError(f"invalid SMB action {action!r}; expected one of: {valid}") from exc


def block_smb_action(action: SMBAction | int) -> int:
    """Map a shared action to the Block SMB discrete action ID."""
    return int(coerce_smb_action(action))


def full_smb_action(action: SMBAction | int, buttons: Iterable[str]) -> np.ndarray:
    """Map a shared action to a stable-retro MultiBinary button vector."""
    action = coerce_smb_action(action)
    button_names = tuple(str(button).upper() for button in buttons)
    if len(set(button_names)) != len(button_names):
        raise ValueError(f"stable-retro button names must be unique: {button_names!r}")

    required = _NES_BUTTONS[action]
    missing = sorted(set(required).difference(button_names))
    if missing:
        raise ValueError(
            f"stable-retro button layout is missing {missing!r} for action {action.name}"
        )

    pressed = set(required)
    return np.asarray([name in pressed for name in button_names], dtype=np.int8)
