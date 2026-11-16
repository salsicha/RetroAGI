"""The tracker that follows Pong from what vision reads (retroagi/domains/
pong/seeing.py), on Block Pong with a stand-in for the network that sees as
the screen does: heights only in two-line rows, nothing of the speed, the
direction, the frame count or the serve's wait."""

import numpy as np

from retroagi.domains.pong.block_pong import PongState, step
from retroagi.domains.pong.knob import Knob
from retroagi.domains.pong.seeing import Seeing

NONE = np.log(1e-6)


def _values(*likely):
    logp = np.full(256, NONE)
    for v in likely:
        logp[v] = np.log(1.0 / len(likely))
    return logp - np.logaddexp.reduce(logp)


def seen(s: PongState) -> dict:
    """What the screen shows of ``s``, as the network would give it."""
    pair = (s.drawn + 1) // 2
    out = {"ball_y": _values(0) if s.ball_y == 0 else _values(s.ball_y & ~1, s.ball_y | 1),
           "ball_x": _values(s.ball_x), "our_drawn": _values(2 * pair - 1, 2 * pair), "theirs": _values(s.theirs)}
    for name, v in (("our_score", s.points[0]), ("their_score", s.points[1])):
        logp = np.full(22, NONE)
        logp[v] = 0.0
        out[name] = logp
    return out


def test_the_tracker_settles_on_the_exact_state():
    from retroagi.domains.pong.controller import Controller
    from retroagi.domains.pong.scripted_chooser import ScriptedChooser
    from retroagi.framework.agent import Agent
    from retroagi.domains.pong.world import TruthSource

    rng = np.random.default_rng(0)
    truth = PongState(ball_x=150, ball_y=100, across=1, speed=1, ours=140, drawn=140, theirs=100, frame=37,
                      knob=Knob(step=0))
    agent = Agent(TruthSource(), ScriptedChooser, Controller, 0)
    agent.reset(0, "points")
    seeing, knob = Seeing(), Knob(step=0)
    exact = counted = 0
    for t in range(6000):
        got = seeing.observe(seen(truth), knob)
        state = truth.copy()
        action = agent.step(state) if rng.random() > 0.1 else int(rng.integers(3))
        if t > 20 and truth.ball_y:
            counted += 1
            exact += got is not None and (got.ball_x, got.ball_y, got.across, got.speed, got.ours, got.drawn,
                                          got.theirs, got.frame % 8, got.points) == (
                truth.ball_x, truth.ball_y, truth.across, truth.speed, truth.ours, truth.drawn, truth.theirs,
                truth.frame % 8, truth.points)
        knob.press(action)
        seeing.acted(action)
        if step(truth, action).over:
            break
    assert counted > 3000
    assert exact / counted > 0.97, (exact, counted, seeing.starts)
