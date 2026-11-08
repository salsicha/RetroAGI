"""The chooser's three rules (core/outcome_chooser.py) on given
predictions."""

import numpy as np

from retroagi.core import outcome_chooser
from retroagi.core.outcome_chooser import DEATH, FINISHED, FRAMES, INFEASIBLE, POINTS, PROGRESS, STUCK, OutcomeChooser
from retroagi.core.outcome_network import PREDICTIONS
from retroagi.core.smb_controller import State
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import Enemy, Solid

FLOOR = Solid(0, 208, 400, 240, "ground", ("ground", 0))
GOOMBA = Enemy((150, 198, 160, 204), -0.5, 0.0, ("slots", 0))
STATE = State(body=(100, 196, 110, 208), motion=NESPlayerMotion(), grounded=True, solids=[FLOOR], enemies=[GOOMBA])


def choose(monkeypatch, mode, rows):
    """The chooser's pick when the network predicts ``rows`` {(kind,
    modifier): predictions}; every other choice infeasible."""
    def fake(model, tokens, choices, device):
        p = np.zeros((len(choices), len(PREDICTIONS)), np.float32)
        p[:, INFEASIBLE] = 1.0
        for k, (kind, _, modifier) in enumerate(choices):
            if (kind, modifier) in rows:
                p[k] = rows[(kind, modifier)]
        return p

    monkeypatch.setattr(outcome_chooser, "predict", fake)
    chooser = OutcomeChooser(None, mode)
    chooser.decide(STATE)
    options, _, best = chooser.last
    return options[best][1], options[best][3]


def row(death=0.0, progress=0.0, frames=16.0, points=0.0, finished=0.0, stuck=0.0):
    p = np.zeros(len(PREDICTIONS), np.float32)
    p[DEATH], p[PROGRESS], p[FRAMES], p[POINTS], p[FINISHED], p[STUCK] = death, progress, frames, points, finished, stuck
    return p


def test_dont_die_comes_before_progress(monkeypatch):
    rows = {(3, 1): row(death=0.6, progress=96), (4, 0): row(death=0.0, progress=0, frames=8)}
    assert choose(monkeypatch, "speed_run", rows) == (4, 0)  # the wait


def test_among_safe_choices_the_most_progress_a_frame(monkeypatch):
    rows = {(3, 0): row(progress=32, frames=16), (3, 1): row(progress=96, frames=40), (4, 0): row(frames=8)}
    assert choose(monkeypatch, "speed_run", rows) == (3, 1)


def test_points_count_in_max_points_only(monkeypatch):
    rows = {(1, 0): row(progress=40, frames=40, points=5), (3, 1): row(progress=60, frames=40)}
    assert choose(monkeypatch, "speed_run", rows) == (3, 1)
    assert choose(monkeypatch, "max_points", rows) == (1, 0)


def test_when_everything_risks_death_the_safest_is_taken(monkeypatch):
    rows = {(3, 1): row(death=0.5, progress=96), (3, 2): row(death=0.2, progress=-48)}
    assert choose(monkeypatch, "speed_run", rows) == (3, 2)  # the step back


def test_a_choice_just_ended_at_once_is_not_taken_again(monkeypatch):
    rows = {(3, 1): row(progress=96, frames=40), (3, 0): row(progress=32, frames=20)}
    first = choose(monkeypatch, "speed_run", rows)
    assert first == (3, 1)
    chooser = OutcomeChooser(None, "speed_run")
    last = chooser.decide(STATE)
    again = chooser.decide(STATE, "blocked", last)
    assert again != last


def test_a_choice_leaving_mario_stuck_gives_way_to_one_that_does_not(monkeypatch):
    # Down into a valley is quick progress but a dead end; back up the
    # staircase is slower but leaves a way on.
    rows = {(3, 1): row(progress=40, frames=20, stuck=0.9), (3, 2): row(progress=-30, frames=30)}
    assert choose(monkeypatch, "speed_run", rows) == (3, 2)


def test_with_nothing_moving_a_wait_is_not_followed_by_another(monkeypatch):
    rows = {(4, 0): row(progress=1, frames=8), (3, 2): row(progress=-30, frames=30)}
    still = State(body=STATE.body, motion=STATE.motion, grounded=True, solids=[FLOOR], enemies=[])
    monkeypatch.setattr(outcome_chooser, "predict", lambda model, tokens, choices, device: np.array(
        [rows.get((k, m), row()) if (k, m) in rows else np.array([0, 1, 0, 0, 0, 16, 0], np.float32)
         for k, _, m in choices], np.float32))
    chooser = OutcomeChooser(None, "speed_run")
    first = chooser.decide(still)
    assert first.verb == "wait"
    assert chooser.decide(still, "waited", first).verb == "run"
