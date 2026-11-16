"""Following the real Pong from what the vision network reads (vision.py):
the exact state the agent plays on, as Block Pong keeps it.

The screen does not show everything exactly: heights come in two-line rows,
the ball's speed and direction are not drawn, nor is the game's frame count
(their paddle rests every eighth frame, ours moves every second) or the
serve's wait. So the tracker keeps every exact state the screen allows
(hypotheses), and each frame:
1. steps each by Block Pong's own rules with the button the agent pressed
   (block_pong.step; the knob is the agent's own, followed exactly);
2. weighs each by how likely the network finds what it predicts there: the
   ball's left and top, our paddle's drawn top, theirs, the scores; the
   weights add up frame after frame, older frames fading (FORGET: the
   network's slight but steady preference between two heights the screen
   draws alike is no evidence, and summed for ever it dropped the right one);
3. drops those far less likely than the best (MARGIN), and when even the
   best is unlikely (FLOOR: the network or the rules went wrong somewhere)
   starts again from what the screen shows.
It starts (and starts again) only while the ball is in play, and gives
nothing (the agent waits) until the likely states agree on the ball's
direction and speed (SETTLED), a frame or two. What it gives is the most
likely state, with the returns counted from what it gave (the ball turned
from coming toward us to going away on our side, as the real game's own
count is kept), so that starting again forgets none.

Nothing here learns; it is seeing's own part: the network and the rules
of the practice world, which were measured in the game.
"""

import itertools
from typing import Optional

import numpy as np

from .block_pong import PongState, step
from .knob import Knob, follow

MARGIN = 12.0  # log-likelihood below the best at which a state is dropped
FORGET = 0.8  # each frame's weight carried to the next
SETTLED = 3.0  # states within this of the best must agree on the ball's direction and speed
FLOOR = -9.0  # the best state's log-likelihood this frame below which it starts again
MOST = 400  # states kept at most
PLAUSIBLE = np.log(0.01)  # values the network gives at least this are tried when starting


def likely(logp: np.ndarray, at_least: float = PLAUSIBLE, most: int = 3) -> list:
    order = np.argsort(-logp)[:most]
    return [int(v) for v in order if logp[v] >= at_least] or [int(order[0])]


class Seeing:
    def __init__(self):
        self.starts = 0  # times followed from the screen afresh (for inspection)
        self.reset()

    def reset(self) -> None:
        self.states: list = []  # (log weight, PongState)
        self.returns, self.points = 0, (0, 0)
        self.best: Optional[PongState] = None
        self.given: Optional[PongState] = None  # the last state given

    def _fit(self, s: PongState, seen: dict) -> float:
        lp = seen["ball_y"][s.ball_y] + seen["our_drawn"][s.drawn] + seen["theirs"][s.theirs]
        lp += seen["our_score"][min(s.points[0], 21)] + seen["their_score"][min(s.points[1], 21)]
        if s.ball_y:
            lp += seen["ball_x"][s.ball_x]
        return float(lp)

    def _start(self, seen: dict, knob: Knob) -> list:
        points = (int(np.argmax(seen["our_score"])), int(np.argmax(seen["their_score"])))
        out = []
        for x, y, drawn, theirs in itertools.product(likely(seen["ball_x"], most=2), likely(seen["ball_y"]),
                                                     likely(seen["our_drawn"], most=2), likely(seen["theirs"], most=2)):
            if y == 0:
                continue
            for across, speed, frame in itertools.product((-1, 1), range(-3, 4), range(8)):
                for ours in {drawn, follow(drawn, knob.reading())}:
                    out.append(PongState(x, y, across, speed, ours, drawn, theirs, frame, 0, 0, points,
                                         Knob(knob.charge, knob.step, knob.held), self.returns))
        return out

    def observe(self, seen: dict, knob: Knob) -> Optional[PongState]:
        """The state for this frame, from the network's reading ``seen``
        (vision.read) and the agent's knob; None while nothing is followed."""
        scored = [(FORGET * w + f, f, s) for w, s in self.states for f in (self._fit(s, seen),)]
        if not scored or max(f for _, f, _ in scored) < FLOOR:
            if np.argmax(seen["ball_y"]) == 0:
                self.states, self.best = [], None
                return None
            self.starts += 1
            fresh = self._start(seen, knob)
            scored = [(f, f, s) for s in fresh for f in (self._fit(s, seen),)]
        top = max(w for w, _, _ in scored)
        kept = sorted(((w, s) for w, _, s in scored if w >= top - MARGIN), key=lambda ws: -ws[0])
        unique, seen_keys = [], set()
        for w, s in kept:
            key = (s.ball_x, s.ball_y, s.across, s.speed, s.ours, s.drawn, s.theirs, s.frame % 8, s.wait, s.scored)
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append((w - top, s))
        self.states = unique[:MOST]
        self.best = best = self.states[0][1]
        if any((s.across, s.speed) != (best.across, best.speed) for w, s in self.states if w >= -SETTLED):
            return None
        last = self.given
        if last is not None and last.ball_y and last.across == 1 and best.across == -1 and best.ball_x > 128:
            self.returns += 1
        self.points = best.points
        self.given = given = best.copy()
        given.returns = self.returns
        return given

    def acted(self, action: int) -> None:
        """Every followed state one frame on with the agent's button."""
        for _, s in self.states:
            step(s, action)
