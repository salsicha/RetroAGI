"""Actions: a verb aimed at one object the vision reports.

The skill chooses an action; the action predictor gives the action's
successful end state; the target tracker says where its object will be; the
predictive controller computes the buttons that reach it
(docs/action-predictor-controller.md).

An action's object is a (list, slot) pair of smb_observer.packed_lists: slot i
of a list is item i there, the same slot the scene encoder makes a token of.
"""

from dataclasses import dataclass
from typing import Optional

from .smb_observer import SCENE_SLOTS

VERBS = ("hold", "run_to", "approach", "land_on", "jump_over", "stomp", "back_away")

# What each verb may aim at (hold aims at nothing).
SURFACE_LISTS = ("surfaces", "moving_platforms", "pipes", "blocks")
VERB_TARGETS = {
    "hold": (),
    # Walk or run to a point on a surface (on the floor Mario stands on, or
    # one he walks off onto).
    "run_to": SURFACE_LISTS,
    # Run up to the object of the next jump and stop at the best takeoff.
    "approach": (*SURFACE_LISTS, "enemies", "gaps"),
    # Jump onto a surface.
    "land_on": SURFACE_LISTS,
    # Jump past an enemy or across a gap.
    "jump_over": ("enemies", "gaps"),
    # Jump onto an enemy.
    "stomp": ("enemies",),
    # Retreat from an enemy.
    "back_away": ("enemies",),
}

# The executor's movement mode for each verb.
VERB_MODES = {
    "hold": "hold",
    "run_to": "run",
    "approach": "run",
    "back_away": "run",
    "land_on": "jump",
    "jump_over": "jump",
    "stomp": "jump",
}

# The scene encoder's token order: a summary, Mario, then every slot of every
# list in SCENE_SLOTS order (SceneEncoder._objects), then the picture bands.
POINTER_LISTS = tuple(SCENE_SLOTS)
_OFFSETS = {}
_offset = 2
for _name in POINTER_LISTS:
    _OFFSETS[_name] = _offset
    _offset += SCENE_SLOTS[_name]
POINTER_TOKENS = _offset - 2  # object slots a pointer can name


def pointer_index(target) -> int:
    """The scene token index of a (list, slot) target."""
    name, slot = target
    if name not in _OFFSETS or not 0 <= slot < SCENE_SLOTS[name]:
        raise ValueError(f"no such scene slot: {target!r}")
    return _OFFSETS[name] + slot


def pointer_target(index: int):
    """The (list, slot) a scene token index names (pointer_index's inverse)."""
    for name in POINTER_LISTS:
        offset = _OFFSETS[name]
        if offset <= index < offset + SCENE_SLOTS[name]:
            return name, index - offset
    raise ValueError(f"token {index} is not an object slot")


@dataclass(frozen=True)
class ActionToken:
    """A verb and the scene slot it is aimed at (None for hold)."""

    verb: str
    target: Optional[tuple] = None

    def __post_init__(self):
        if self.verb not in VERBS:
            raise ValueError(f"unknown verb {self.verb!r}")
        if self.verb == "hold":
            if self.target is not None:
                raise ValueError("hold aims at nothing")
            return
        if self.target is None:
            raise ValueError(f"{self.verb} needs an object")
        name, _ = self.target
        if name not in VERB_TARGETS[self.verb]:
            raise ValueError(f"{self.verb} cannot aim at {name}")
        pointer_index(self.target)  # a real slot

    @property
    def mode(self) -> str:
        return VERB_MODES[self.verb]


def allowed_lists(verb: str) -> tuple:
    return VERB_TARGETS[verb]
