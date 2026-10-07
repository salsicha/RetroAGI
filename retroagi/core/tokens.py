"""Strategy context, categorical tactics, and spatial commands for predictive control.

- The strategy is a switch, set by whoever runs the agent, not chosen by a
  layer: a StrategyToken holds what the run is for and which side the goal is
  on (the agent cannot see the goal). speed_run: finish in the least time;
  max_points: finish with the most points, a point for every coin collected
  and every enemy killed. A Block SMB layout sets it (its family's strategy,
  else speed_run, and the side its goal is on); in Full SMB it is set for the
  run (the goal is always to the right).
- The tactic layer holds a TacticToken over many actions: the way Mario is to
  move. Forward is right, the level's way; backward is left:
  - advance: go forward on the level (getting past enemies, jumping gaps,
    and stomping are all part of it);
  - retreat: go backward on the level;
  - climb_forward / climb_backward: go up onto something higher, ahead or
    behind;
  - descend_forward / descend_backward: go down to something lower, ahead or
    behind;
  - hold_ground: stay at the current spot on the supporting platform, moving
    with that platform until the tactic ends.
- The skill layer reads the tactic and strategy context and chooses a
  run/jump/hold destination.
- The predictive executor reads that command and visual feedback to choose buttons.

In Block SMB each learner receives a teacher command from above. At play
time the skill receives the tactic and the executor receives the skill command.
"""

from dataclasses import dataclass

import torch

STRATEGIES = ("speed_run", "max_points")
TACTICS = (
    "advance",
    "retreat",
    "climb_forward",
    "climb_backward",
    "descend_forward",
    "descend_backward",
    "hold_ground",
)
BACKWARD_TACTICS = frozenset(("retreat", "climb_backward", "descend_backward"))
SKILL_MODES = ("run", "jump", "hold")
# Pixel destinations relative to Mario's feet at the decision frame. Positive
# x is right; positive y is down. These are coordinates, not object slot IDs.
SKILL_X = tuple(range(-256, 257))
SKILL_Y = tuple(range(-240, 241))
SKILL_WIDTH = len(SKILL_MODES) + 2


@dataclass(frozen=True)
class SkillToken:
    """A movement mode and a destination relative to Mario's foot center.

    Hold anchors the current spot on the supporting platform; x/y are ignored.
    Run/jump coordinates describe the destination, not a button duration.
    """

    mode: str
    x: int
    y: int

    def __post_init__(self):
        if self.mode not in SKILL_MODES:
            raise ValueError(f"unknown skill mode {self.mode!r}")
        if self.x not in SKILL_X or self.y not in SKILL_Y:
            raise ValueError("skill destination must be integer pixels within the local screen")


def encode_skill(token: SkillToken) -> torch.Tensor:
    vector = torch.zeros(SKILL_WIDTH)
    vector[SKILL_MODES.index(token.mode)] = 1.0
    if token.mode != "hold":
        vector[-2:] = torch.tensor((token.x / 256, token.y / 256))
    return vector


def skill_picks(token: SkillToken) -> dict:
    return {
        "mode": SKILL_MODES.index(token.mode),
        "x": SKILL_X.index(token.x),
        "y": SKILL_Y.index(token.y),
    }


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

    def __post_init__(self):
        if self.stance not in TACTICS:
            raise ValueError(f"unknown tactic {self.stance!r}")


DEFAULT_STRATEGY = StrategyToken("speed_run", 1)


def tactic_token(stance: str) -> TacticToken:
    """A categorical tactic with no separate direction channel."""
    return TacticToken(stance)


STRATEGY_WIDTH = len(STRATEGIES) + 1
TACTIC_WIDTH = len(TACTICS)


def encode_strategy(token: StrategyToken) -> torch.Tensor:
    """[STRATEGY_WIDTH]: one-hot strategy, then the goal's side (-1 or 1)."""
    vector = torch.zeros(STRATEGY_WIDTH)
    vector[STRATEGIES.index(token.kind)] = 1.0
    vector[-1] = float(token.direction)
    return vector


def encode_tactic(token: TacticToken) -> torch.Tensor:
    """[TACTIC_WIDTH]: one-hot tactic only."""
    vector = torch.zeros(TACTIC_WIDTH)
    vector[TACTICS.index(token.stance)] = 1.0
    return vector


def token_layout() -> dict:
    """The token vocabularies, stored with checkpoints and compared on load."""
    from .actions import SMB_ACTIONS

    return {
        "strategies": list(STRATEGIES),
        "tactics": list(TACTICS),
        "executor_actions": [action.name for action in SMB_ACTIONS] + ["HOLD_GROUND"],
        "skill": {
            "strategy_context": "strategy_switch_v1",
            "modes": list(SKILL_MODES),
            "x": list(SKILL_X),
            "y": list(SKILL_Y),
            "reference": "mario_feet_relative_pixels",
            "history": 16,
        },
        "executor": "predictive_spatial_v1",
    }
