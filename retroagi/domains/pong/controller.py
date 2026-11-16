"""The controller: carries out one choice, a button a frame, in plain
arithmetic (no network: docs/requirements.md).

The choices:

- "return" with a part of the paddle (PARTS: its top edge, upper part,
  middle, lower part, bottom edge, which send the ball back steeply up, up,
  level, down or steeply down): meet the ball coming toward us with that
  part, then rest at the middle until the ball comes back. Its events:
  "returned" (the opponent sent it back), "scored" (the ball went out of
  play: the point is ours), "missed" (it got past our paddle: the point is
  theirs), "infeasible" (no way to meet it with that part in time, or no
  ball coming).
- "rest" at a height (REST_TOPS): move the paddle there and wait until the
  ball comes toward us (event "turned").

Our paddle follows a knob the buttons turn (knob.py), so the controller
moves it by choosing the knob's charge to turn to: pressing toward it
until one more press would pass it (a press moves it 60 a frame once
grown). To meet the ball, it picks a plan by trying a few, each in Block
Pong's own rules (the ball's flight, the knob, the paddle; the opponent
plays no part in the ball's way to us) from the present state, and keeps
the first that meets the ball with the chosen part: one charge to turn
to, or, since a paddle at rest sits only every 5 or 6 pixels (the knob's
reading steps by 2, a press by about 5) and a part is 4 tall, one charge
and then its neighbour from a frame timed so the paddle meets the ball on
its way between them. A millisecond or a few, once a choice.
"""

from dataclasses import dataclass
from typing import Optional

from .block_pong import OUR_TURN_X, PongState, band, step
from .game import DOWN, STAY, UP
from .knob import FULL, GROWING, TOP, Knob, reading

PARTS = (-2, -1, 0, 1, 2)  # the ball's vertical speed after it (band)
# For each part, the ball's height minus the paddle's top aimed at: inside
# the part's four pixels, and within the paddle's reach whether the ball's
# height is odd or even (block_pong._reaches).
AIM = {-2: -1, -1: 2, 0: 6, 1: 10, 2: 14}
REST_TOPS = (70, 121, 172)
MIDDLE = 121
# Charges tried for a return: the one that should rest the paddle where
# the aim needs it, then the nearest others, up to TRIES.
TRIES = 7
# The longest a planned meeting is looked ahead for (frames).
LOOK_AHEAD = 400
# A switch to a neighbouring charge is tried at each of the last SWITCH
# frames before the ball reaches the paddle (the paddle closes half the way
# to the knob's reading every second frame).
SWITCH = 14


@dataclass(frozen=True)
class Choice:
    kind: str  # "return" or "rest"
    part: int = 0  # return: one of PARTS; rest: an index into REST_TOPS


def next_step(knob: Knob) -> int:
    """How far the knob turns if a button is pressed this frame."""
    if not knob.held:
        return knob.step
    return knob.step + 1 if knob.step < GROWING else FULL


def toward(knob: Knob, target: int) -> int:
    """The button that turns the knob toward the charge ``target``: none
    once one more press would take it further from it than it is."""
    gap, turn = target - knob.charge, next_step(knob)
    if abs(gap) <= turn / 2 and turn:
        return STAY
    if gap > 0:
        return DOWN if knob.charge + turn < TOP else STAY
    return UP if knob.charge > turn else STAY


def charge_for(top: int, knob: Knob) -> int:
    """The charge, of those the knob reaches from where it is in whole
    presses, that rests our paddle's top nearest ``top``."""
    first = knob.charge % FULL or FULL
    return min(range(first, TOP, FULL), key=lambda c: abs(reading(c) - 1 - top))


def coming(state: PongState) -> bool:
    return state.ball_y != 0 and state.across == 1 and state.ball_x <= OUR_TURN_X[1]


def meeting(state: PongState, plan: tuple, frames: int = LOOK_AHEAD) -> Optional[tuple]:
    """(frames until, the ball's vertical speed after) of our paddle's
    meeting with the ball, the knob turned by ``plan`` from ``state``, in
    Block Pong's rules; None if the ball gets past or the point ends first.
    ``plan``: ((until frame, charge), ...), the last until the end."""
    s = state.copy()
    returns = s.returns
    for frame in range(frames):
        point = step(s, toward(s.knob, target_of(plan, frame))).point
        if s.returns > returns:
            return frame + 1, s.speed
        if point or s.ball_y == 0 or s.ball_x > OUR_TURN_X[1]:
            return None
    return None


def target_of(plan: tuple, frame: int) -> int:
    for until, target in plan:
        if frame < until:
            return target
    return plan[-1][1]


def arrival(state: PongState) -> Optional[tuple]:
    """(frames until, the ball's height) when the ball reaches our
    paddle's first column (None if it is not coming)."""
    s = state.copy()
    for frame in range(LOOK_AHEAD):
        if s.ball_x >= OUR_TURN_X[0] or s.ball_y == 0:
            return (frame, s.ball_y) if s.ball_x >= OUR_TURN_X[0] and s.ball_y else None
        s.drawn = s.ours = -100  # (no paddle in the way)
        step(s, STAY)
    return None


def plans(state: PongState, part: int):
    """The plans tried for meeting the ball with ``part``, likeliest
    first: the charges nearest the one that rests the paddle where the aim
    needs it, then switches between neighbouring charges."""
    reach = arrival(state)
    if reach is None:
        return
    frames, y = reach
    aimed = charge_for(y - AIM[part], state.knob)
    tries = sorted((c for c in range(aimed - FULL * (TRIES // 2), aimed + FULL * (TRIES // 2) + 1, FULL)
                    if 0 < c < TOP), key=lambda c: abs(c - aimed))
    for target in tries:
        yield ((LOOK_AHEAD, target),)
    for first in tries[:3]:
        for then in (first - FULL, first + FULL):
            if 0 < then < TOP:
                for switch in range(max(0, frames - SWITCH), frames + 2):
                    yield ((switch, first), (LOOK_AHEAD, then))


class Controller:
    def __init__(self):
        self.choice, self.event = None, None
        self.target, self.phase, self.returns = None, None, 0
        self.plan, self.frame = None, 0

    def observe(self, state) -> None:
        pass

    def another(self) -> "Controller":
        return Controller()

    def start(self, choice: Choice, state: PongState) -> None:
        self.choice, self.event, self.returns = choice, None, state.returns
        if choice.kind == "rest":
            self.phase, self.target = "rest", charge_for(REST_TOPS[choice.part], state.knob)
            if coming(state):
                self.event = "turned"
            return
        self.phase, self.frame = "meet", 0
        if coming(state):
            want = band(AIM[choice.part])
            for plan in plans(state, choice.part):
                met = meeting(state, plan)
                if met is not None and met[1] == want:
                    self.plan = plan
                    return
        self.event = "infeasible"

    def press(self, state: PongState) -> int:
        if self.phase == "meet":
            self.target = target_of(self.plan, self.frame)
            self.frame += 1
            if state.returns > self.returns:
                self.phase, self.target = "away", charge_for(MIDDLE, state.knob)
            elif state.ball_y == 0 or state.ball_x > OUR_TURN_X[1] or state.across != 1:
                self.event = "missed"
                return STAY
        if self.phase == "away":
            if state.ball_y == 0:
                self.event = "scored"
                return STAY
            if state.across == 1:
                self.event = "returned"
        if self.phase == "rest" and coming(state):
            self.event = "turned"
        return toward(state.knob, self.target)
