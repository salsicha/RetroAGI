"""Block Pong: the fast practice world for Pong, its rules in plain
arithmetic, measured in the real game's memory (docs/pong.md,
scripts/dev/pong_measure.py, scripts/dev/pong_compare.py). It loads nothing
of the game: no picture, no program. Its state is read by lookup.

Units are the game's: pixels, heights growing downward, one step a frame.
The agent plays the right-hand paddle with STAY, UP and DOWN.

Each frame, in the game's order:

1. The ball, while it is in play:
   - a paddle the ball overlaps turns it: ours while the ball is at x 188
     to 192 and its height, rounded down to even, is from 2 above our
     paddle's drawn top to 15 below it (4 above for a ball falling 2 a
     frame from an odd height); theirs while the ball is at x 64 to
     68 and its height is from 4 above their top to 14 below. The ball then
     goes back across, and up or down by where it met the paddle (BANDS);
   - at the top wall (height 45 or less, going up) or the bottom one (206
     or more, going down) it turns, with a pixel more that frame;
   - it moves one pixel across and its vertical speed up or down;
   - at x 205 (past us) or 52 (past them) the point is scored: the ball
     stops there a frame, then is out of play for SERVE_WAIT frames, then
     is served from the middle toward the side that lost the point, one
     pixel a frame up from height 127 or down from 129, keeping the
     direction up or down it had.
2. Their paddle moves 2 pixels toward lining its top up with the ball's
   height (toward 0 while the ball is out of play), except on every eighth
   frame; at a serve it is first put at 128.
3. Our paddle: what is drawn is where it was; on every second frame it
   moves halfway to the knob's reading (knob.py).
"""

from dataclasses import dataclass, field, replace
from typing import ClassVar

from .game import DOWN, STAY, UP
from .knob import Knob, follow

OUR_TURN_X, THEIR_TURN_X = (188, 192), (64, 68)
OUR_REACH, THEIR_REACH = (-2, 15), (-4, 14)  # heights relative to the paddle's top that it turns
TOP_WALL, BOTTOM_WALL = 45, 206
PAST_US, PAST_THEM = 205, 52
SERVE_WAIT = 63
SERVE_X = {-1: 127, 1: 129}  # by the direction across it is served in
SERVE_Y = {-1: 127, 1: 129}  # by the direction up or down
THEIR_SERVE_TOP = 128
OPPONENT_STEP, OPPONENT_REST = 2, 8
WIN = 21
TOP, BOTTOM = 38, 219  # our paddle's top's range


def band(meet: int) -> int:
    """The ball's vertical speed after a paddle, by its height minus the
    paddle's top."""
    if meet <= -4:
        return -3
    if meet <= 0:
        return -2
    if meet <= 4:
        return -1
    if meet <= 8:
        return 0
    if meet <= 12:
        return 1
    return 2


def _reaches(y: int, speed: int, drawn: int) -> bool:
    """Whether our paddle, drawn with its top at ``drawn``, turns a ball at
    height ``y``: the game draws in two-scanline rows, so the height counts
    rounded down to even, and a ball falling 2 a frame from an odd height
    is caught from a row higher."""
    meet = (y & ~1) - drawn
    top = OUR_REACH[0] - (2 if speed == 2 and y % 2 else 0)
    return top <= meet <= OUR_REACH[1]


@dataclass
class PongState:
    """Everything Block Pong keeps (the real game's names in docs/pong.md).

    ball_x, ball_y: the ball's left and top (ball_y 0: out of play);
    across: +1 toward us (right), -1 toward them; speed: pixels a frame up
    (negative) or down; ours: our paddle's top, drawn: where it is drawn
    this frame; theirs: their paddle's top; frame: the game's frame count
    (0 to 255); wait: frames until the serve (0: in play); scored: the ball
    has just passed a side and stops a frame; points: (ours, theirs);
    returns: the times our paddle has turned the ball.

    The agent reads this state as it is (a lookup), and may decide at any
    frame."""

    decidable: ClassVar[bool] = True

    ball_x: int = 0
    ball_y: int = 0
    across: int = -1
    speed: int = -1
    ours: int = 128
    drawn: int = 128
    theirs: int = THEIR_SERVE_TOP
    frame: int = 0
    wait: int = 0
    scored: int = 0
    points: tuple = (0, 0)
    knob: Knob = field(default_factory=Knob)
    returns: int = 0

    def copy(self) -> "PongState":
        return replace(self, knob=replace(self.knob))


@dataclass
class Frame:
    """What one frame led to: +1 we scored, -1 they did, 0 neither; game over."""

    point: int = 0
    over: bool = False


def step(state: PongState, action: int) -> Frame:
    """One frame of Block Pong, in place."""
    s, point = state, 0
    if s.scored or s.ball_y == 0:
        s.wait -= 1
        if s.scored:
            s.ball_y, s.scored = 0, 0
        elif s.wait <= 0:
            s.wait = 0
            s.across = 1 if s.across > 0 else -1  # (none yet at the game's start: toward them)
            s.speed = 1 if s.speed >= 0 else -1
            s.ball_x, s.ball_y = SERVE_X[s.across], SERVE_Y[s.speed]
            s.theirs = THEIR_SERVE_TOP
    elif s.ball_x in (PAST_US, PAST_THEM):
        point = 1 if s.ball_x == PAST_THEM else -1
        s.points = (s.points[0] + (point > 0), s.points[1] + (point < 0))
        s.scored, s.wait = 1, SERVE_WAIT
    else:
        x, y = s.ball_x, s.ball_y
        if OUR_TURN_X[0] <= x <= OUR_TURN_X[1] and _reaches(y, s.speed, s.drawn):
            s.returns += s.across == 1
            s.across, s.speed = -1, band(y - s.drawn)
        elif THEIR_TURN_X[0] <= x <= THEIR_TURN_X[1] and THEIR_REACH[0] <= y - s.theirs <= THEIR_REACH[1]:
            s.across, s.speed = 1, band(y - s.theirs)
        if y <= TOP_WALL and s.speed < 0:
            s.speed, y = -s.speed, y + 1
        elif y >= BOTTOM_WALL and s.speed > 0:
            s.speed, y = -s.speed, y - 1
        s.ball_x, s.ball_y = x + s.across, y + s.speed
    if s.frame % OPPONENT_REST:
        gap = s.ball_y - s.theirs
        s.theirs += OPPONENT_STEP if gap > 0 else -OPPONENT_STEP if gap < 0 else 0
    s.knob.press(action)
    s.drawn = s.ours
    if s.frame % 2:
        s.ours = follow(s.ours, s.knob.reading())
    s.frame = (s.frame + 1) % 256
    return Frame(point, max(s.points) >= WIN)
