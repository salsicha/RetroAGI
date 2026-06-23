"""Game action contracts and SMB compatibility mappings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ContinuousControlSpec:
    """One normalized continuous control axis for a game action."""

    axis: str
    value: float

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("continuous control axis must be non-empty")
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(
                f"continuous control {self.axis!r} value must be in [-1, 1]"
            )


@dataclass(frozen=True)
class ActionSpec:
    """Stable per-game action identity and backend mapping metadata."""

    name: str
    stable_id: int
    kind: str = "discrete"
    buttons: tuple[str, ...] = ()
    continuous_controls: tuple[ContinuousControlSpec, ...] = ()
    release_buttons: tuple[str, ...] = ()
    release_all: bool = False
    backend_action_id: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action name must be non-empty")
        if self.stable_id < 0:
            raise ValueError("action stable_id must be non-negative")
        if self.backend_action_id is not None and self.backend_action_id < 0:
            raise ValueError("backend_action_id must be non-negative when declared")
        if self.kind not in {"discrete", "continuous", "hybrid"}:
            raise ValueError(
                f"action {self.name!r} kind must be discrete, continuous, or hybrid"
            )
        if len(set(self.buttons)) != len(self.buttons):
            raise ValueError(f"action {self.name!r} has duplicate buttons")
        if len(set(self.release_buttons)) != len(self.release_buttons):
            raise ValueError(f"action {self.name!r} has duplicate release buttons")
        if self.kind == "continuous" and self.buttons:
            raise ValueError(f"continuous action {self.name!r} cannot declare buttons")
        if self.kind == "discrete" and self.continuous_controls:
            raise ValueError(
                f"discrete action {self.name!r} cannot declare continuous controls"
            )
        if self.release_all and self.buttons:
            raise ValueError(f"release-all action {self.name!r} cannot press buttons")

    @property
    def is_noop(self) -> bool:
        return self.release_all and not self.buttons and not self.continuous_controls

    @property
    def backend_id(self) -> int:
        return self.stable_id if self.backend_action_id is None else self.backend_action_id


class SMBAction(IntEnum):
    """Stable action IDs shared by Block SMB and Full SMB."""

    NOOP = 0
    RIGHT = 1
    RIGHT_JUMP = 2
    LEFT = 3
    LEFT_JUMP = 4
    JUMP = 5


SMB_ACTIONS = tuple(SMBAction)

SMB_ACTION_SPECS = (
    ActionSpec(
        name="noop",
        stable_id=0,
        release_all=True,
        description="Release every NES button",
    ),
    ActionSpec(
        name="right",
        stable_id=1,
        buttons=("RIGHT",),
        description="Hold right",
    ),
    ActionSpec(
        name="right_jump",
        stable_id=2,
        buttons=("RIGHT", "A"),
        description="Hold right and jump",
    ),
    ActionSpec(
        name="left",
        stable_id=3,
        buttons=("LEFT",),
        description="Hold left",
    ),
    ActionSpec(
        name="left_jump",
        stable_id=4,
        buttons=("LEFT", "A"),
        description="Hold left and jump",
    ),
    ActionSpec(
        name="jump",
        stable_id=5,
        buttons=("A",),
        description="Hold jump",
    ),
)


def coerce_action_spec(
    action_space: Iterable[ActionSpec], action: ActionSpec | int | str
) -> ActionSpec:
    """Return an action spec from a per-game action space."""

    action_specs = tuple(action_space)
    if isinstance(action, ActionSpec):
        if action in action_specs:
            return action
        raise ValueError(f"action spec {action.name!r} is not in this action space")
    if isinstance(action, str):
        normalized = action.lower()
        for spec in action_specs:
            if spec.name == normalized:
                return spec
        valid = ", ".join(spec.name for spec in action_specs)
        raise ValueError(f"invalid action {action!r}; expected one of: {valid}")
    try:
        action_id = int(action)
    except (TypeError, ValueError) as exc:
        valid = ", ".join(f"{spec.name}={spec.stable_id}" for spec in action_specs)
        raise ValueError(f"invalid action {action!r}; expected one of: {valid}") from exc
    for spec in action_specs:
        if spec.stable_id == action_id:
            return spec
    valid = ", ".join(f"{spec.name}={spec.stable_id}" for spec in action_specs)
    raise ValueError(f"invalid action {action!r}; expected one of: {valid}")


def action_button_vector(action: ActionSpec, buttons: Iterable[str]) -> np.ndarray:
    """Map a per-game action spec to a backend MultiBinary button vector."""

    button_names = tuple(str(button).upper() for button in buttons)
    if len(set(button_names)) != len(button_names):
        raise ValueError(f"backend button names must be unique: {button_names!r}")

    required = tuple(button.upper() for button in action.buttons)
    missing = sorted(set(required).difference(button_names))
    if missing:
        raise ValueError(
            f"backend button layout is missing {missing!r} for action {action.name}"
        )

    release_buttons = tuple(button.upper() for button in action.release_buttons)
    release_missing = sorted(set(release_buttons).difference(button_names))
    if release_missing:
        raise ValueError(
            "backend button layout is missing "
            f"{release_missing!r} for action {action.name} release behavior"
        )

    pressed = set(required)
    if not action.release_all:
        pressed.difference_update(release_buttons)
    return np.asarray([name in pressed for name in button_names], dtype=np.int8)


def action_backend_id(action: ActionSpec) -> int:
    """Return the stage-native discrete ID for an action spec."""

    if action.kind == "continuous":
        raise ValueError(f"continuous action {action.name!r} has no discrete backend ID")
    return action.backend_id


def coerce_smb_action(action: SMBAction | int) -> SMBAction:
    """Return a named action or raise a clear error for an invalid ID."""
    try:
        return SMBAction(action)
    except (TypeError, ValueError) as exc:
        valid = ", ".join(f"{item.name}={item.value}" for item in SMB_ACTIONS)
        raise ValueError(f"invalid SMB action {action!r}; expected one of: {valid}") from exc


def smb_action_spec(action: SMBAction | int | str) -> ActionSpec:
    """Return the SMB game-profile action spec for a legacy SMB action input."""

    if isinstance(action, SMBAction):
        return coerce_action_spec(SMB_ACTION_SPECS, int(action))
    if isinstance(action, str):
        return coerce_action_spec(SMB_ACTION_SPECS, action)
    return coerce_action_spec(SMB_ACTION_SPECS, int(coerce_smb_action(action)))


def block_smb_action(action: SMBAction | int) -> int:
    """Map an SMB action spec to the Block SMB discrete action ID."""

    return action_backend_id(smb_action_spec(action))


def full_smb_action(action: SMBAction | int, buttons: Iterable[str]) -> np.ndarray:
    """Map an SMB action spec to a stable-retro MultiBinary button vector."""

    return action_button_vector(smb_action_spec(action), buttons)
