"""The strategy switch and the tactic token the action layer is given.

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
    stomping and waiting for the way to clear are all part of it);
  - retreat: go backward on the level;
  - climb_forward / climb_backward: go up onto something higher, ahead or
    behind;
  - descend_forward / descend_backward: go down to something lower, ahead or
    behind.
- The action layer turns the tactic token into a button action and a frame
  count for the executor (smb_executor).

In Block SMB the action layer learns from explicit tactic tokens given by a
teacher; at play time the token comes only from the tactic layer, which reads
the switch.
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
)
BACKWARD_TACTICS = frozenset(("retreat", "climb_backward", "descend_backward"))


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
        if self.direction != tactic_direction(self.stance):
            raise ValueError(f"{self.stance} goes {tactic_direction(self.stance)}")


DEFAULT_STRATEGY = StrategyToken("speed_run", 1)


def tactic_direction(stance: str) -> int:
    """The way a tactic goes: forward (right, the level's way) or backward (left)."""
    return -1 if stance in BACKWARD_TACTICS else 1


def tactic_token(stance: str) -> TacticToken:
    """A tactic with its direction."""
    return TacticToken(stance, tactic_direction(stance))


STRATEGY_WIDTH = len(STRATEGIES) + 1
TACTIC_WIDTH = len(TACTICS) + 1


def encode_strategy(token: StrategyToken) -> torch.Tensor:
    """[STRATEGY_WIDTH]: one-hot strategy, then the goal's side (-1 or 1)."""
    vector = torch.zeros(STRATEGY_WIDTH)
    vector[STRATEGIES.index(token.kind)] = 1.0
    vector[-1] = float(token.direction)
    return vector


def encode_tactic(token: TacticToken) -> torch.Tensor:
    """[TACTIC_WIDTH]: one-hot tactic, then its direction (-1 or 1)."""
    vector = torch.zeros(TACTIC_WIDTH)
    vector[TACTICS.index(token.stance)] = 1.0
    vector[-1] = float(token.direction)
    return vector


def token_layout() -> dict:
    """The token vocabularies, stored with checkpoints and compared on load."""
    return {"strategies": list(STRATEGIES), "tactics": list(TACTICS)}
