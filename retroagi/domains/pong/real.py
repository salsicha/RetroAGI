"""The real Pong as the agent plays and is scored in it: a whole game to 21
in the emulator (game.py), and the state from the game's memory.

Nothing here learns, and nothing learned is shaped by it: the real game
only scores (docs/requirements.md). The state from memory is Block Pong's
own state as the game keeps it (the same parts, the same units: scripts/
dev/pong_compare.py shows they agree), with the hidden knob followed from
the agent's own presses: it starts afresh at every reset (knob.py), and
its rule is exact.
"""

from retroagi.framework.interfaces import Step

from .block_pong import PongState
from .game import RealPong
from .knob import Knob
from .world import RETURN, WON

SCORING = 63  # the serve's wait as the point is scored (block_pong.SERVE_WAIT)


def state_from_memory(m, knob: Knob, returns: int = 0) -> PongState:
    """Block Pong's state for the game's memory ``m`` and the knob."""
    scored = int(m.serve_wait == SCORING and m.ball_y != 0)
    return PongState(m.ball_x, m.ball_y, m.across, m.speed, m.our_paddle, m.our_drawn, m.their_paddle, m.frame,
                     m.serve_wait, scored, (m.our_score, m.their_score), knob, returns)


class RealPongWorld:
    """A whole game of the real Pong, from its start."""

    settled = True

    def __init__(self, level: str = "Start", start_state: bytes = None):
        self.game = RealPong(level)
        self.start_state = start_state
        self.points, self.returns, self.last = (0, 0), 0, None
        self.screen = None  # the last screen shown (what seeing reads)

    def reset(self) -> None:
        self.screen = self.game.reset()
        if self.start_state is not None:
            self.game.load(self.start_state)
            self.screen = None  # nothing of the loaded state is drawn until its first frame
        self.returns, self.last = 0, self.game.memory()
        self.points = (self.last.our_score, self.last.their_score)

    def step(self, action: int) -> Step:
        self.screen, _, over = self.game.step(action)
        m = self.game.memory()
        if self.last is not None and self.last.across == 1 and m.across == -1 and m.ball_x > 128:
            self.returns += 1
        self.last = m
        self.points = (m.our_score, m.their_score)
        if over:
            return Step(over=True, died=m.their_score > m.our_score, finished=m.our_score > m.their_score)
        return Step()

    @property
    def progress(self) -> float:
        return RETURN * self.returns + WON * self.points[0]

    furthest = progress

    @property
    def score(self) -> float:
        return float(self.points[0] - self.points[1])

    def close(self) -> None:
        self.game.close()


class MemorySource:
    """The state from the real game's memory, the knob followed from the
    agent's presses."""

    def __init__(self):
        self.knob = Knob(step=0)

    def reset(self) -> None:
        self.knob = Knob(step=0)

    def read(self, world) -> PongState:
        state = state_from_memory(world.game.memory(), self.knob, world.returns)
        return state.copy()

    def acted(self, action: int) -> None:
        self.knob.press(action)


DEFAULT_VISION = "data/pong_vision/vision.pt"


class VisionSource:
    """The state from the screen alone: the vision network's reading of
    each frame (vision.py), followed by the tracker (seeing.py), with the
    knob followed from the agent's presses. None (the agent waits) until
    the tracker follows the ball."""

    def __init__(self, model):
        from .seeing import Seeing

        self.model, self.seeing = model, Seeing()
        self.reset()

    def reset(self) -> None:
        self.knob = Knob(step=0)
        self.seeing.reset()

    def read(self, world):
        from .vision import read

        if world.screen is None:
            return None
        return self.seeing.observe(read(self.model, world.screen), self.knob)

    def acted(self, action: int) -> None:
        self.knob.press(action)
        self.seeing.acted(action)
