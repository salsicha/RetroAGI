"""Game action contracts and SMB compatibility mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Mapping

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


SMB_JUMP_ACTIONS = frozenset((SMBAction.RIGHT_JUMP, SMBAction.LEFT_JUMP, SMBAction.JUMP))
SMB_JUMP_RELEASE_ACTIONS = {
    SMBAction.RIGHT_JUMP: SMBAction.RIGHT,
    SMBAction.LEFT_JUMP: SMBAction.LEFT,
    SMBAction.JUMP: SMBAction.NOOP,
}
SMB_WALK_ACTIONS = frozenset((SMBAction.RIGHT, SMBAction.LEFT))
SMB_DEFAULT_ACTIONS_PER_SECOND = 60.0
SMB_MAX_WALK_ACTION_SECONDS = 1.0
SMB_SUPPORT_AIR = "air"
SMB_SUPPORT_GROUND = "ground"
SMB_SUPPORT_PLATFORM = "platform"
SMB_AGENT_CLASS_NAMES = frozenset(("mario", "player", "agent"))
SMB_ENEMY_CLASS_NAMES = frozenset(("enemy", "goomba", "koopa"))


def is_smb_jump_action(action: SMBAction | int) -> bool:
    return coerce_smb_action(action) in SMB_JUMP_ACTIONS


def is_smb_walk_action(action: SMBAction | int) -> bool:
    return coerce_smb_action(action) in SMB_WALK_ACTIONS


def smb_jump_release_action(action: SMBAction | int) -> SMBAction:
    """Return the non-jump action that preserves horizontal intent."""

    return SMB_JUMP_RELEASE_ACTIONS.get(coerce_smb_action(action), coerce_smb_action(action))


class SMBJumpActionTerminator:
    """Release SMB jump actions after ViT support says a jump has landed.

    The first jump frame is always allowed, even if the current observation is
    still grounded. Once the support-state output reports air, a subsequent
    ground/platform support state or enemy contact releases `A` while preserving
    horizontal direction.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._jump_active = False
        self._left_support = False
        self._suppress_until_non_jump = False

    def filter_action(
        self,
        action: SMBAction | int,
        *,
        batch: Any = None,
        vision: Any = None,
    ) -> int:
        action_value = coerce_smb_action(action)
        if vision is None and batch is not None:
            vision = _vision_from_batch(batch)
        support_name = _vision_support_name(vision)
        enemy_contact = _vision_enemy_contact(vision)

        if action_value not in SMB_JUMP_ACTIONS:
            self.reset()
            return int(action_value)

        if self._suppress_until_non_jump:
            return int(smb_jump_release_action(action_value))

        if not self._jump_active:
            self._jump_active = True
            self._left_support = support_name == SMB_SUPPORT_AIR
            return int(action_value)

        if enemy_contact:
            release = smb_jump_release_action(action_value)
            self._jump_active = False
            self._left_support = False
            self._suppress_until_non_jump = True
            return int(release)

        if support_name == SMB_SUPPORT_AIR:
            self._left_support = True
            return int(action_value)

        landed = (
            self._left_support
            and support_name in {SMB_SUPPORT_GROUND, SMB_SUPPORT_PLATFORM}
        )
        if landed:
            release = smb_jump_release_action(action_value)
            self._jump_active = False
            self._left_support = False
            self._suppress_until_non_jump = True
            return int(release)

        return int(action_value)


class SMBWalkActionLimiter:
    """Release pure SMB walk actions after a fixed continuous hold window."""

    def __init__(
        self,
        *,
        max_walk_seconds: float = SMB_MAX_WALK_ACTION_SECONDS,
        actions_per_second: float = SMB_DEFAULT_ACTIONS_PER_SECOND,
    ) -> None:
        if float(max_walk_seconds) <= 0.0:
            raise ValueError("max_walk_seconds must be positive")
        if float(actions_per_second) <= 0.0:
            raise ValueError("actions_per_second must be positive")
        self.max_walk_seconds = float(max_walk_seconds)
        self.actions_per_second = float(actions_per_second)
        self.max_walk_frames = max(
            1,
            int(math.ceil(self.max_walk_seconds * self.actions_per_second)),
        )
        self.reset()

    def reset(self) -> None:
        self._walk_action: SMBAction | None = None
        self._walk_frames = 0

    def filter_action(self, action: SMBAction | int) -> int:
        action_value = coerce_smb_action(action)
        if action_value not in SMB_WALK_ACTIONS:
            self.reset()
            return int(action_value)
        if self._walk_action != action_value:
            self._walk_action = action_value
            self._walk_frames = 0
        if self._walk_frames >= self.max_walk_frames:
            self.reset()
            return int(SMBAction.NOOP)
        self._walk_frames += 1
        return int(action_value)


def _vision_from_batch(batch: Any) -> Any:
    metadata = getattr(batch, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get("vision")
    return None


def _vision_support_name(vision: Any) -> str | None:
    if vision is None:
        return None
    support_id = _first_index(getattr(vision, "support_ids", None))
    if support_id is None:
        support_logits = _to_numpy(getattr(vision, "support_logits", None))
        if support_logits is not None and support_logits.size:
            support_id = int(
                np.asarray(support_logits)
                .reshape((-1, support_logits.shape[-1]))[0]
                .argmax()
            )
    if support_id is None:
        return None
    support_classes = _metadata_tuple(vision, "support_classes") or (
        SMB_SUPPORT_AIR,
        SMB_SUPPORT_GROUND,
        SMB_SUPPORT_PLATFORM,
    )
    if 0 <= support_id < len(support_classes):
        return str(support_classes[support_id]).lower()
    return None


def _vision_enemy_contact(vision: Any) -> bool:
    if vision is None:
        return False
    labels = _semantic_labels(vision)
    if labels is None or labels.size == 0:
        return False
    if labels.ndim == 3:
        labels = labels[0]
    if labels.ndim != 2:
        return False

    semantic_classes = _metadata_tuple(vision, "semantic_classes") or _metadata_tuple(
        vision, "checkpoint_classes"
    )
    agent_ids, enemy_ids = _semantic_contact_class_ids(semantic_classes, labels)
    if not agent_ids or not enemy_ids:
        return False

    agent_mask = np.isin(labels, tuple(agent_ids))
    if not bool(agent_mask.any()):
        return False
    rows, cols = np.nonzero(agent_mask)
    row_start = max(int(rows.min()) - 1, 0)
    row_end = min(int(rows.max()) + 2, labels.shape[0])
    col_start = max(int(cols.min()) - 1, 0)
    col_end = min(int(cols.max()) + 2, labels.shape[1])
    contact_window = labels[row_start:row_end, col_start:col_end]
    return bool(np.isin(contact_window, tuple(enemy_ids)).any())


def _semantic_labels(vision: Any) -> np.ndarray | None:
    semantic_ids = _to_numpy(getattr(vision, "semantic_ids", None))
    if semantic_ids is not None:
        return np.asarray(semantic_ids)
    semantic_logits = _to_numpy(getattr(vision, "semantic_logits", None))
    if semantic_logits is None or semantic_logits.ndim < 3:
        return None
    return np.asarray(semantic_logits).argmax(axis=1)


def _semantic_contact_class_ids(
    semantic_classes: tuple[str, ...],
    labels: np.ndarray,
) -> tuple[set[int], set[int]]:
    if semantic_classes:
        lowered = tuple(str(name).lower() for name in semantic_classes)
        agent_ids = {
            index for index, name in enumerate(lowered) if name in SMB_AGENT_CLASS_NAMES
        }
        enemy_ids = {
            index for index, name in enumerate(lowered) if name in SMB_ENEMY_CLASS_NAMES
        }
        return agent_ids, enemy_ids

    class_count = int(labels.max()) + 1 if labels.size else 0
    if class_count == 7:
        return {1}, {5}
    if class_count == 13:
        return {8}, {6, 7}
    if class_count == 6:
        return {5}, {3}
    return set(), set()


def _metadata_tuple(vision: Any, key: str) -> tuple[str, ...]:
    metadata = getattr(vision, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ()
    value = metadata.get(key)
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _first_index(value: Any) -> int | None:
    array = _to_numpy(value)
    if array is None or array.size == 0:
        return None
    return int(np.asarray(array).reshape(-1)[0])


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
