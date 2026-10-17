"""Actions: a verb (run, jump or wait) and, for a run or a jump, the object
the vision reports that it ends on.

The skill chooses an action; in training the teacher carries it out, later
the world model predicts where it ends and how long it takes
(core/world_model) and the executor reaches that.

An action's object is a (list, slot) pair of smb_observer.packed_lists: slot
i of a list is item i there, the same slot the scene encoder makes a token
of.
"""

from dataclasses import dataclass
from typing import Optional

from .smb_observer import SCENE_SLOTS

# Three verbs (user, 2026-10-17): run (on the ground, along or off a floor;
# in the air, a steer), jump and wait (stay where Mario is). A run's or a
# jump's pointer names where Mario ends up: the surface he ends on, or the
# enemy he lands on (a stomp). A wait aims at nothing. How long a wait
# lasts, like whether a run keeps its speed into the jump after it, is part
# of carrying the action out.
VERBS = ("run", "jump", "wait")

# What each verb may aim at: where Mario can end up (a wait: nothing).
SURFACE_LISTS = ("surfaces", "moving_platforms", "pipes", "blocks")
VERB_TARGETS = {
    "run": (*SURFACE_LISTS, "enemies"),
    "jump": (*SURFACE_LISTS, "enemies"),
    "wait": (),
}

# The executor's movement mode for each verb.
VERB_MODES = {"run": "run", "jump": "jump", "wait": "hold"}

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
    """A verb and the scene slot of where it ends (None for a wait)."""

    verb: str
    target: Optional[tuple] = None

    def __post_init__(self):
        if self.verb not in VERBS:
            raise ValueError(f"unknown verb {self.verb!r}")
        if self.verb == "wait":
            if self.target is not None:
                raise ValueError("a wait aims at nothing")
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


# The verb an executor command was made with, for commands that carry no
# action (the teacher's own, given in training): its mode's plain verb.
MODE_VERBS = {"hold": "wait", "run": "run", "jump": "jump"}
# An action as numbers in the skill's history (encode_action).
ACTION_WIDTH = len(VERBS) + 2


def encode_action(action: Optional[ActionToken], destination):
    """[ACTION_WIDTH]: the verb (one-hot; for a command with no action, its
    mode's verb, MODE_VERBS) and where the executor was sent (``destination``,
    a tokens.SkillToken from Mario's feet: x / 256 and y / 256; none for a
    hold)."""
    import torch

    vector = torch.zeros(ACTION_WIDTH)
    verb = action.verb if action is not None else MODE_VERBS[destination.mode]
    vector[VERBS.index(verb)] = 1.0
    if destination is not None and destination.mode != "hold":
        vector[-2:] = torch.tensor((destination.x / 256, destination.y / 256))
    return vector
