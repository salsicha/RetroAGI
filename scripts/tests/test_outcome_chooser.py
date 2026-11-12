"""The chooser's three rules (framework/chooser.py) on given
predictions."""

import numpy as np

from retroagi.domains.smb.choices import CARES
from retroagi.domains.smb.controller import USUAL, State
from retroagi.domains.smb.domain import DOMAIN
from retroagi.domains.smb.physics import NESPlayerMotion
from retroagi.domains.smb.planner import Enemy, Solid
from retroagi.framework.chooser import DEATH, FINISHED, FRAMES, INFEASIBLE, POINTS, PROGRESS, STUCK
from retroagi.framework.outcome_network import PREDICTIONS

# (kind, modifier) of the choices used, each carried out the usual way (controller.CARE).
SHORT, LONG, BACK = ((3, run * CARES + USUAL) for run in range(3))
STOMP = (1, USUAL)
WAIT = (4, 0)
FLOOR = Solid(0, 208, 400, 240, "ground", ("ground", 0))
GOOMBA = Enemy((150, 198, 160, 204), -0.5, 0.0, ("slots", 0))
STATE = State(body=(100, 196, 110, 208), motion=NESPlayerMotion(), grounded=True, solids=[FLOOR], enemies=[GOOMBA])


class Given:
    """A predictor answering ``rows`` {(kind, modifier): predictions};
    every other choice infeasible."""

    def __init__(self, rows):
        self.rows = rows

    def predict(self, state, candidates):
        p = np.zeros((len(candidates), len(PREDICTIONS)), np.float32)
        p[:, INFEASIBLE] = 1.0
        for k, c in enumerate(candidates):
            if (c.kind, c.modifier) in self.rows:
                p[k] = self.rows[(c.kind, c.modifier)]
        return p


def chooser_for(mode, rows):
    return DOMAIN.chooser(Given(rows), mode)


def choose(mode, rows):
    """The chooser's pick (kind, modifier) when the predictor gives ``rows``."""
    chooser = chooser_for(mode, rows)
    chooser.decide(STATE)
    options, _, best = chooser.last
    return options[best].kind, options[best].modifier


def row(death=0.0, progress=0.0, frames=16.0, points=0.0, finished=0.0, stuck=0.0):
    p = np.zeros(len(PREDICTIONS), np.float32)
    p[DEATH], p[PROGRESS], p[FRAMES], p[POINTS], p[FINISHED], p[STUCK] = death, progress, frames, points, finished, stuck
    return p


def test_dont_die_comes_before_progress():
    rows = {LONG: row(death=0.6, progress=96), WAIT: row(death=0.0, progress=0, frames=8)}
    assert choose("speed_run", rows) == WAIT  # the wait


def test_among_safe_choices_the_most_progress_a_frame():
    rows = {SHORT: row(progress=32, frames=16), LONG: row(progress=96, frames=40), WAIT: row(frames=8)}
    assert choose("speed_run", rows) == LONG


def test_points_count_in_max_points_only():
    rows = {STOMP: row(progress=40, frames=40, points=5), LONG: row(progress=60, frames=40)}
    assert choose("speed_run", rows) == LONG
    assert choose("max_points", rows) == STOMP


def test_when_everything_risks_death_the_safest_is_taken():
    rows = {LONG: row(death=0.5, progress=96), BACK: row(death=0.2, progress=-48)}
    assert choose("speed_run", rows) == BACK  # the step back


def test_a_choice_taken_from_a_spot_is_not_repeated_there_until_new_ground():
    rows = {LONG: row(progress=96, frames=40), SHORT: row(progress=32, frames=20), WAIT: row(frames=8)}
    chooser = chooser_for("speed_run", rows)
    still = State(body=STATE.body, motion=STATE.motion, grounded=True, solids=[FLOOR], enemies=[])
    picks = [chooser.decide(still) for _ in range(3)]
    assert [(c.verb, c.x) for c in picks] == [("run", 201.0), ("run", 137.0), ("wait", 0.0)]
    # Nothing new left: the best again.
    assert chooser.decide(still).x == 201.0
    # Further right, all is forgotten.
    ahead = State(body=(140, 196, 150, 208), motion=STATE.motion, grounded=True, solids=[FLOOR], enemies=[])
    assert chooser.decide(ahead).x == 241.0


def test_a_wait_with_enemies_on_screen_forgets_what_was_tried():
    # The long run is tried from the spot; with a Goomba on screen the short
    # run is next, then (both tried) the wait; after it, the long run again.
    rows = {LONG: row(progress=96, frames=40), SHORT: row(progress=32, frames=20), WAIT: row(frames=8)}
    chooser = chooser_for("speed_run", rows)
    picks = [chooser.decide(STATE) for _ in range(4)]
    assert [(c.verb, c.x) for c in picks] == [("run", 201.0), ("run", 137.0), ("wait", 0.0), ("run", 201.0)]


def test_what_could_not_be_carried_out_stays_refused_after_a_wait():
    rows = {LONG: row(progress=96, frames=40), SHORT: row(progress=32, frames=20), WAIT: row(frames=8)}
    chooser = chooser_for("speed_run", rows)
    first = chooser.decide(STATE)
    assert first.x == 201.0
    # The long run could not be carried out: after the short run and a wait,
    # the short run again, not the long one.
    picks = [chooser.decide(STATE, "infeasible", first)]
    picks += [chooser.decide(STATE, "arrived", picks[-1]) for _ in range(2)]
    assert [(c.verb, c.x) for c in picks] == [("run", 137.0), ("wait", 0.0), ("run", 137.0)]


def test_reaching_the_flag_counts_only_when_likely():
    rows = {LONG: row(progress=96, frames=40), BACK: row(progress=8, frames=40, finished=0.02)}
    assert choose("speed_run", rows) == LONG
    rows[BACK] = row(progress=8, frames=40, finished=0.9)
    assert choose("speed_run", rows) == BACK


def test_waiting_is_tried_after_max_waits_at_a_spot():
    # Every way on a little risky, waiting safe: after MAX_WAITS waits the
    # least risky way on is taken.
    from retroagi.framework.chooser import MAX_WAITS

    rows = {LONG: row(death=0.08, progress=96, frames=40), WAIT: row(frames=8, progress=4)}
    chooser = chooser_for("speed_run", rows)
    picks = [chooser.decide(STATE).verb for _ in range(MAX_WAITS + 1)]
    assert picks == ["wait"] * MAX_WAITS + ["run"]
