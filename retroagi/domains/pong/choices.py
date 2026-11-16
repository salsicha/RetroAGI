"""The choices open in a state of Pong and the state as the outcome network
reads it (framework/interfaces.ChoiceSpace).

When the ball is coming toward us the choices are the five returns, one
for each part of the paddle (controller.PARTS); otherwise, the three rests
(controller.REST_TOPS). The state is two token groups:

- the paddles (the agent's own token): our paddle's top and where it is
  drawn, the knob's reading and charge, whether its step has grown and a
  button was held, their paddle's top, and the frame count's place in the
  opponent's eight-frame rhythm and in our paddle's two-frame one;
- the ball (one slot): its left and top, its direction across, its
  vertical speed, whether it is in play, and the frames before a serve.

The returns are choices about the ball, the rests about no thing.
"""

import numpy as np

from retroagi.framework.interfaces import Candidate
from retroagi.framework.outcome_network import ChoiceSpec, Kind, TokenGroup

from .controller import PARTS, REST_TOPS, Choice, coming
from .world import RETURN, WON

KINDS = ("return", "rest")
GROUPS = (
    TokenGroup("paddles", 0, 9, len(REST_TOPS), (100.0, 100.0, 100.0, 1000.0, 1.0, 1.0, 100.0, 8.0, 2.0)),
    TokenGroup("ball", 1, 6, len(PARTS), (100.0, 100.0, 1.0, 2.0, 1.0, 64.0)),
)
SPEC = ChoiceSpec(GROUPS, (Kind("return", 1, 0, len(PARTS)), Kind("rest", 0, 0, len(REST_TOPS))))


class PongSpace:
    spec = SPEC
    new_ground = RETURN  # every return is new ground: what was tried before is forgotten

    def candidates(self, state):
        if coming(state):
            return [Candidate(Choice("return", part), 0, "ball", k) for k, part in enumerate(PARTS)]
        return [Candidate(Choice("rest", k), 1, None, k) for k in range(len(REST_TOPS))]

    def tokens(self, state) -> tuple:
        knob = state.knob
        paddles = np.array([state.ours, state.drawn, knob.reading(), knob.charge, knob.step == 60, knob.held,
                            state.theirs, state.frame % 8, state.frame % 2], np.float32)
        ball = np.array([[state.ball_x, state.ball_y, state.across, state.speed, state.ball_y != 0, state.wait]],
                        np.float32)
        return (paddles, ball), {"ball": 0}

    def place(self, state) -> tuple:
        return RETURN * state.returns + WON * state.points[0], None

    def signature(self, candidate, state) -> tuple:
        return candidate.choice.kind, candidate.choice.part

    def waits(self, candidate) -> bool:
        return candidate.choice.kind == "rest"

    def still(self, candidate) -> bool:
        return False  # (the ball always moves)

    def changing(self, state) -> bool:
        return True


SPACE = PongSpace()
