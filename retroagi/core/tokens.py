"""The strategy switch and the tokens the policy layers pass down: tactic, skill.

- The strategy is a switch, set by whoever runs the agent, not chosen by a
  layer: a StrategyToken holds what the run is for and which side the goal is
  on (the agent cannot see the goal). speed_run: finish as fast as possible;
  max_coins: collect as many coins as possible; careful: finish without
  risk, however long it takes. A Block SMB layout sets it (its strategy
  course, else speed_run, and the side its goal is on); in Full SMB it is set
  for the run (the goal is always to the right).
- The tactic layer holds a TacticToken over many actions: advance,
  alternate_route, hold_area or retreat. Advance and alternate route go
  right, the level's way; retreat goes back to the left, whether to back away
  from something or to get back to a goal or an enemy behind Mario; hold area
  faces the goal (tactic_direction).
- The skill layer emits a SkillToken: a skill (advance, jump a gap, climb,
  descend, stomp, retreat or wait), a direction, and which object in the
  scene is the target, if any. Avoiding enemies is part of every skill, not
  a skill of its own.
- The action layer turns the skill token into an action and a frame count
  for the executor (smb_executor).

In Block SMB each layer learns from explicit tokens given by a teacher. At
play time a layer's token comes only from the layer above, and the tactic
layer reads the switch.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .smb_observer import SCENE_SLOTS

STRATEGIES = ("speed_run", "max_coins", "careful")
TACTICS = ("advance", "alternate_route", "hold_area", "retreat")
SKILLS = ("advance", "jump_gap", "climb", "descend", "stomp", "retreat", "wait")
# What a skill's target can be: one slot of these scene lists, or nothing.
TARGET_LISTS = ("surfaces", "enemies", "moving_platforms")
TARGETS = tuple((name, slot) for name in TARGET_LISTS for slot in range(SCENE_SLOTS[name]))
NO_TARGET = len(TARGETS)  # the pointer value meaning "no target"


def _direction(value: int) -> int:
    if value not in (-1, 1):
        raise ValueError(f"direction must be -1 (left) or 1 (right), got {value!r}")
    return value


@dataclass(frozen=True)
class StrategyToken:
    kind: str
    direction: int = 1

    def __post_init__(self):
        if self.kind not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.kind!r}")
        _direction(self.direction)


@dataclass(frozen=True)
class TacticToken:
    stance: str
    direction: int = 1

    def __post_init__(self):
        if self.stance not in TACTICS:
            raise ValueError(f"unknown tactic {self.stance!r}")
        _direction(self.direction)


@dataclass(frozen=True)
class SkillToken:
    kind: str
    direction: int = 1
    target: Optional[tuple[str, int]] = None  # (scene list, slot) or None

    def __post_init__(self):
        if self.kind not in SKILLS:
            raise ValueError(f"unknown skill {self.kind!r}")
        _direction(self.direction)
        if self.target is not None and self.target not in TARGETS:
            raise ValueError(f"unknown skill target {self.target!r}")

    @property
    def pointer(self) -> int:
        """The target as one number: its index in TARGETS, or NO_TARGET."""
        return NO_TARGET if self.target is None else TARGETS.index(self.target)


DEFAULT_STRATEGY = StrategyToken("speed_run", 1)


def tactic_direction(stance: str, goal_side: int) -> int:
    """The way a tactic goes: advance and alternate route right (the level's
    way), retreat back to the left; hold area faces the goal."""
    _direction(goal_side)
    if stance == "hold_area":
        return goal_side
    return -1 if stance == "retreat" else 1


def tactic_token(stance: str, switch: StrategyToken) -> TacticToken:
    """A tactic with its direction (tactic_direction; hold area faces the
    strategy switch's goal side)."""
    return TacticToken(stance, tactic_direction(stance, switch.direction))


STRATEGY_WIDTH = len(STRATEGIES) + 1
TACTIC_WIDTH = len(TACTICS) + 1
SKILL_WIDTH = len(SKILLS) + 1 + len(TARGETS) + 1


def encode_strategy(token: StrategyToken) -> torch.Tensor:
    """[STRATEGY_WIDTH]: one-hot strategy, then direction (-1 or 1)."""
    vector = torch.zeros(STRATEGY_WIDTH)
    vector[STRATEGIES.index(token.kind)] = 1.0
    vector[-1] = float(token.direction)
    return vector


def encode_tactic(token: TacticToken) -> torch.Tensor:
    """[TACTIC_WIDTH]: one-hot stance, then direction (-1 or 1)."""
    vector = torch.zeros(TACTIC_WIDTH)
    vector[TACTICS.index(token.stance)] = 1.0
    vector[-1] = float(token.direction)
    return vector


def encode_skill(token: SkillToken) -> torch.Tensor:
    """[SKILL_WIDTH]: one-hot skill, direction, one-hot target pointer."""
    vector = torch.zeros(SKILL_WIDTH)
    vector[SKILLS.index(token.kind)] = 1.0
    vector[len(SKILLS)] = float(token.direction)
    vector[len(SKILLS) + 1 + token.pointer] = 1.0
    return vector


def token_layout() -> dict:
    """The token vocabularies, stored with checkpoints and compared on load."""
    return {
        "strategies": list(STRATEGIES),
        "tactics": list(TACTICS),
        "skills": list(SKILLS),
        "targets": [list(target) for target in TARGETS],
    }
