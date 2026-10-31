"""Without the teacher the executor drives to what the skill aimed at (user,
2026-10-31: "make executor drive to object the skill aimed at"): a run ends
inside the aimed thing, found again in every picture; a jump keeps its
takeoff plan."""

import math
import os
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from retroagi.core.smb_aim import AIM_INSET, aimed_thing, refound
from retroagi.core.smb_ground_control import GroundMove
from retroagi.core.smb_scene_labels import BlockView, EnemyView, MarioView, SceneObservation, Surface
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken


def scene(mario_x=100, surfaces=(), pipes=(), blocks=(), enemies=(), uncut=()):
    return SceneObservation(
        MarioView((mario_x, 196, mario_x + 10, 208), True, "ground", True),
        surfaces=tuple(surfaces),
        pipes=tuple(pipes),
        blocks=tuple(blocks),
        enemies=tuple(enemies),
        uncut_surfaces=tuple(uncut),
    )


def test_an_aimed_thing_has_its_real_ends():
    # Seams joined; an end at the picture's edge is open; pipes and blocks by
    # their boxes; nothing for an enemy, a moving surface or an empty slot.
    seen = scene(
        surfaces=(Surface(8, 59, 207, False), Surface(63, 180, 208, False), Surface(200, 248, 208, True)),
        pipes=((180, 144, 212, 208),),
        blocks=(BlockView((120, 128, 136, 144), "question_block"),),
    )
    floor = aimed_thing(seen, ("surfaces", 1))
    assert floor.left == -math.inf and floor.right == 180 and floor.top == 208
    assert (floor.seen_left, floor.seen_right) == (8, 180)
    pipe = aimed_thing(seen, ("pipes", 0))
    assert (pipe.left, pipe.right, pipe.top) == (180, 212, 144)
    assert aimed_thing(seen, ("blocks", 0)).top == 128
    assert aimed_thing(seen, ("surfaces", 2)) is None
    assert aimed_thing(seen, ("enemies", 0)) is None
    assert aimed_thing(seen, ("surfaces", 7)) is None
    # A floor cut where an enemy stands: the enemy is no end.
    cut = scene(
        surfaces=(Surface(8, 120, 208, False), Surface(140, 200, 208, False)),
        uncut=(Surface(8, 200, 208, False),),
        enemies=(EnemyView((124, 194, 136, 208), "walker"),),
    )
    assert aimed_thing(cut, ("surfaces", 0)).right == 200


def test_an_end_coming_into_view_closes_without_reaching_past_a_pit():
    floor = aimed_thing(scene(surfaces=(Surface(8, 248, 208, False),)), ("surfaces", 0))
    assert floor.right == math.inf
    # Scrolled 100 pixels: the floor ends at 180, a pit, more floor at 220.
    later = scene(surfaces=(Surface(8, 180, 208, False), Surface(220, 248, 208, False)))
    found = refound(floor.shifted(-100), later)
    assert found.right == 180 and found.place(500) == 180 - AIM_INSET


def test_a_run_ends_inside_the_aimed_floor_as_its_end_comes_into_view():
    first = scene(mario_x=100, surfaces=(Surface(8, 248, 208, False),))
    feedback = SimpleNamespace(displacement=0.0)
    floor = aimed_thing(first, ("surfaces", 0))
    move = GroundMove(feedback, 200, 0.0, 208, 0, aim=floor.shifted(-105))
    assert move._goal(first) == 200  # no end in view: the destination
    # Mario has run 60 pixels and the picture scrolled with him: the floor's
    # end, 150 pixels ahead of where he started, is in view.
    feedback.displacement = 60.0
    later = scene(mario_x=100, surfaces=(Surface(8, 195, 208, False),))
    assert move._goal(later) == (195 - 105 + 60) - AIM_INSET


def test_a_jump_keeps_its_takeoff_plan_where_it_is_aimed():
    # Landing on the aimed thing's top, inside its ends, steering toward it in
    # the air did worse in 1-1 (2026-10-31): a jump goes to its destination.
    seen = scene(mario_x=100, surfaces=(Surface(8, 130, 208, False), Surface(170, 248, 208, False)))
    feedback = SpatialFeedback()
    feedback.observe(seen)
    plan = feedback.begin(SkillToken("jump", 40, -2), seen, aim=("surfaces", 1))
    assert plan.action in (2, 4, 5)
    assert feedback.flight.goal == (105 + 40, 208 - 2)


def test_with_the_teacher_or_no_aim_the_executor_goes_to_the_destination():
    seen = scene(mario_x=100, surfaces=(Surface(8, 130, 208, False), Surface(170, 248, 208, False)))
    feedback = SpatialFeedback()
    feedback.observe(seen)
    feedback.begin(SkillToken("jump", 40, -2), seen)
    assert feedback.flight.goal == (105 + 40, 208 - 2)
    feedback.begin(SkillToken("run", 30, 0), seen)
    assert feedback.travel.aim is None and feedback.travel._goal(seen) == 30


def test_the_agent_aims_the_executor_only_without_the_teacher():
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    class Pictures:
        """Vision standing in: the same picture every frame."""

        def __init__(self, picture):
            self.picture = picture

        def eval(self):
            return self

        def scene(self, screens):
            return [self.picture for _ in screens]

    seen = scene(mario_x=100, surfaces=(Surface(8, 248, 208, False),))
    aims = []
    original = SpatialFeedback.begin

    def recorded(self, destination, picture, proposed=None, aim=None):
        aims.append(aim)
        return original(self, destination, picture, proposed, aim=aim)

    SpatialFeedback.begin = recorded
    try:
        import torch

        from retroagi.core.action_tokens import VERBS

        for teacher in (True, False):
            policy = LayeredSMBPolicy().eval()
            with torch.no_grad():  # the untrained skill runs, aimed at something
                policy.skill.heads["verb"].weight.zero_()
                policy.skill.heads["verb"].bias.fill_(-20)
                policy.skill.heads["verb"].bias[VERBS.index("run")] = 20
            agents = SMBAgents(
                VisionObserver(Pictures(seen)),
                policy,
                "cpu",
                world_model=not teacher,
                carry_out=(lambda number, action: SkillToken("run", 20, 0)) if teacher else None,
            )
            agents.reset(0)
            aims.clear()
            import numpy as np

            for _ in range(3):
                agents.act([np.zeros((240, 256, 3), np.uint8)], [0])
            chosen = [a for a in aims if a is not None]
            assert bool(chosen) == (not teacher)
    finally:
        SpatialFeedback.begin = original
