"""The scripted chooser (stages/full_smb/scripted_chooser.py): its choices
from a hand-made state."""

from retroagi.core.smb_controller import Choice, State
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_planner import Goal, Solid
from retroagi.stages.full_smb.scripted_chooser import CLOSER, EDGE, ScriptedChooser

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
    from retroagi.core.smb_planner import Enemy

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
