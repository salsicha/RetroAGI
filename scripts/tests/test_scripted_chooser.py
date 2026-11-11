"""The scripted chooser (domains/smb/scripted_chooser.py): its choices
from a hand-made state."""

from retroagi.domains.smb.controller import Choice, State
from retroagi.domains.smb.physics import NESPlayerMotion
from retroagi.domains.smb.planner import Goal, Solid
from retroagi.domains.smb.scripted_chooser import CLOSER, EDGE, ScriptedChooser

FLOOR = Solid(0, 208, 200, 240, "ground", ("ground", 0))
BEYOND = Solid(248, 208, 400, 240, "ground", ("ground", 1))


def state_at(x):
    return State(body=(x - 5, 192, x + 5, 208), motion=NESPlayerMotion(), grounded=True,
                 solids=[FLOOR, BEYOND])


def test_a_run_closer_after_an_infeasible_jump_stops_at_the_floors_end():
    # 40 pixels before the far side's landing is inside the pit.
    last = Choice("jump", Goal("land", BEYOND.ref, BEYOND.x0, BEYOND.x1, top=BEYOND.top))
    choice = ScriptedChooser("speed_run").decide(state_at(120), "infeasible", last)
    assert choice.verb == "run"
    assert choice.x == FLOOR.x1 - EDGE < BEYOND.x0 - CLOSER
    # At the floor's end already: a step back for a run-up.
    choice = ScriptedChooser("speed_run").decide(state_at(FLOOR.x1 - EDGE), "infeasible", last)
    assert choice.verb == "run" and choice.x < FLOOR.x1 - EDGE


def test_far_from_a_pit_a_run_ends_before_the_floors_end():
    choice = ScriptedChooser("speed_run").decide(state_at(100), None, None)
    assert choice.verb == "run"
    assert choice.x <= FLOOR.x1 - EDGE
    # Near the floor's end, the jump across.
    choice = ScriptedChooser("speed_run").decide(state_at(170), None, None)
    assert choice.verb == "jump" and choice.goal.ref == BEYOND.ref


def test_a_staircase_down_is_run_off_not_jumped():
    top = Solid(0, 160, 200, 240, "ground", ("surfaces", 0))
    lower = Solid(200, 176, 216, 240, "ground", ("surfaces", 1))
    state = State(body=(185, 144, 195, 160), motion=NESPlayerMotion(), grounded=True, solids=[top, lower])
    choice = ScriptedChooser("speed_run").decide(state, None, None)
    assert choice.verb == "run" and choice.x > top.x1 - EDGE


def test_an_enemy_beyond_a_pipe_waits_for_the_pipe():
    from retroagi.domains.smb.planner import Enemy

    pipe = Solid(150, 176, 182, 240, "pipe", ("pipes", 0))
    goomba = Enemy((230, 194, 246, 208), -0.5, 0.0, ("slots", 0))
    state = State(body=(130, 192, 140, 208), motion=NESPlayerMotion(), grounded=True,
                  solids=[FLOOR, pipe], enemies=[goomba])
    choice = ScriptedChooser("speed_run").decide(state, None, None)
    assert choice.verb == "jump" and choice.goal.kind == "land" and choice.goal.ref == pipe.ref


def test_a_pipes_end_is_jumped_down_from_not_run_off():
    # The floor's labels end at the pipe's sides.
    pipe = Solid(100, 176, 132, 240, "pipe", ("pipes", 0))
    left, right = Solid(0, 208, 102, 240, "ground", ("ground", 0)), Solid(130, 208, 400, 240, "ground", ("ground", 1))
    state = State(body=(115, 160, 125, 176), motion=NESPlayerMotion(), grounded=True, solids=[left, pipe, right])
    choice = ScriptedChooser("speed_run").decide(state, None, None)
    assert choice.verb == "jump" and choice.goal.ref == right.ref


def test_an_infeasible_pass_tries_a_jump_back_then_runs_away():
    from retroagi.domains.smb.planner import Enemy

    goomba = Enemy((140, 194, 156, 208), -0.5, 0.0, ("slots", 0))
    state = State(body=(115, 192, 125, 208), motion=NESPlayerMotion(), grounded=True,
                  solids=[FLOOR], enemies=[goomba])
    chooser = ScriptedChooser("speed_run")
    back = chooser.decide(state, "infeasible", Choice("jump", Goal("pass", goomba.ref, side=1)))
    assert back.verb == "jump" and back.goal.kind == "pass" and back.goal.side == -1
    away = chooser.decide(state, "infeasible", back)
    assert away.verb == "run" and away.x < 120
    # One still far off is waited for.
    far = Enemy((300, 194, 316, 208), -0.5, 0.0, ("slots", 1))
    distant = State(body=(115, 192, 125, 208), motion=NESPlayerMotion(), grounded=True,
                    solids=[FLOOR], enemies=[far])
    assert chooser.decide(distant, "infeasible", Choice("jump", Goal("pass", far.ref, side=1))).verb == "wait"
