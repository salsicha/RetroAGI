"""Block Pong as the practice world (framework/interfaces.Practice), and
the state read from it.

A practice world is one point of Block Pong from a situation drawn at
random from the game's own ranges: a serve to come (either way, the ball
going up or down) or a ball already in flight (anywhere between the
paddles, either way, at any of the game's vertical speeds), each paddle
anywhere in its range (a ball coming toward us at least 38 frames off),
the frame count anywhere in the opponent's rhythm,
and our knob where our paddle rests or, as after the real game is reset,
fresh at its middle. A skill scene ("return") is a ball already coming
toward us, passed once our paddle has turned it.

The point ends the play: lost is a death, won is the task finished.
Progress is RETURN for every return and WON for the point won, so the
chooser's second rule (most progress a frame) keeps the rally going and
wins it when it can; nothing is scored as points.
"""

import random

from retroagi.framework.interfaces import Step

from .block_pong import PongState, step
from .knob import FULL, START, TOP, Knob, reading

RETURN, WON = 16.0, 64.0
BUDGET = 4000  # frames a point may take
SPEEDS = (-3, -2, -1, 0, 1, 2, 3)
PRACTICES = ("level", "return")
TOWARD_US = 150  # the furthest right a ball coming toward us starts


def resting_charge(paddle: int, rng: random.Random) -> int:
    """A knob charge whose reading holds our paddle at ``paddle`` (its top
    rests one above the reading or on it), on the 60 steps a knob that has
    been turned keeps from the start."""
    charges = range(START % FULL, TOP, FULL)
    return min(charges, key=lambda c: (abs(reading(c) - 1 - paddle), rng.random()))


def random_situation(seed: int, kind: str = "level") -> PongState:
    """The start of a practice world of ``kind`` made from ``seed``."""
    rng = random.Random(seed * 7919 + PRACTICES.index(kind))
    ours = rng.randint(38, 219)
    fresh = rng.random() < 0.2 and kind == "level"
    knob = Knob() if fresh else Knob(resting_charge(ours, rng))
    state = PongState(ours=ours, drawn=ours, theirs=2 * rng.randint(0, 103), frame=rng.randrange(256), knob=knob)
    if kind == "level" and rng.random() < 0.4:
        state.across, state.speed = rng.choice((-1, 1)), rng.choice((-1, 1))
        state.ball_x = 52 if state.across < 0 else 205
        state.wait = rng.randint(1, 63)
        return state
    state.across = 1 if kind == "return" else rng.choice((-1, 1))
    # (a ball coming toward us starts far enough off to be met: TOWARD_US)
    state.ball_x = rng.randint(70, TOWARD_US) if state.across == 1 else rng.randint(70, 186)
    state.ball_y = rng.randint(46, 205)
    state.speed = rng.choice(SPEEDS[1:-1]) if rng.random() < 0.9 else rng.choice((-3, 3))
    return state


class BlockPongWorld:
    """One point of Block Pong."""

    settled = True  # (nothing in Pong is ever unsettled)

    def __init__(self, start: PongState, kind: str = "level"):
        self.start, self.kind = start, kind
        self.budget = BUDGET
        self.state = start.copy()
        self.best = 0.0

    def reset(self) -> None:
        self.state = self.start.copy()
        self.best = 0.0

    def step(self, action: int) -> Step:
        point = step(self.state, action).point
        self.best = max(self.best, self.progress)
        if point:
            return Step(over=True, died=point < 0, finished=point > 0)
        return Step()

    @property
    def won(self) -> int:
        return self.state.points[0] - self.start.points[0]

    @property
    def progress(self) -> float:
        return RETURN * (self.state.returns - self.start.returns) + WON * self.won

    @property
    def furthest(self) -> float:
        return self.best

    score = 0.0

    @property
    def passed(self) -> bool:
        return self.kind == "return" and self.state.returns > self.start.returns

    def snapshot(self):
        return self.state.copy(), self.best

    def restore(self, saved) -> None:
        state, self.best = saved
        self.state = state.copy()

    def close(self) -> None:
        pass


class TruthSource:
    """The state from Block Pong's own: a copy of it (a lookup)."""

    def reset(self) -> None:
        pass

    def read(self, world) -> PongState:
        return world.state.copy()

    def acted(self, action: int) -> None:
        pass
