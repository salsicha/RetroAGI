"""Regression cases found in the all-family skill audit."""

import os
from collections import deque

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.core.smb_executor import HOLD_GROUND, ActionPlan, SMBExecutor
from retroagi.core.smb_scene_labels import (
    EnemyView,
    MarioView,
    SceneObservation,
    Surface,
    scene_from_labels,
)
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv


def test_clipped_floor_does_not_turn_camera_scroll_into_a_hold_correction():
    executor = SMBExecutor()
    executor.start(ActionPlan(HOLD_GROUND, 32))

    def picture(x):
        return SceneObservation(
            MarioView((x, 208, x + 10, 220), True, "ground", True),
            surfaces=(Surface(8, 248, 220, False),),
        )

    assert executor.press(picture(104)) == 0
    assert executor.press(picture(85)) == 0


def test_reset_camera_is_already_at_its_stationary_first_frame_position():
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={"mario": [104, 208], "world_width": 447, "platforms": [[0, 220, 447, 20]]}
        )
        camera = env.camera_x
        env.step(0)
        assert env.camera_x == camera == 19
    finally:
        env.close()


@pytest.mark.parametrize("distance", [57, 65, 72, 80, 85])
def test_running_takeoff_reaches_the_narrow_platform_instead_of_overshooting(distance):
    # These commands are the verified teacher endpoints from platform_hop.
    target = SkillToken("jump", distance, -22)
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "mario": [60, 208],
                "world_width": 380,
                "mario_velocity": [2.5, 0],
                "platforms": [
                    [0, 220, 90, 20],
                    [60 + distance - 5, 198, 24, 10],
                    [200, 220, 180, 20],
                ],
                "goal": [60 + distance - 5, 178, 24, 20],
                "goal_requires_support": True,
                "single_jump_attempt": True,
            }
        )
        feedback = SpatialFeedback(speed=2.5, velocities=deque([2.5, 2.5]))
        scene = scene_from_labels(env.scene_labels())
        feedback.observe(scene)
        plan = feedback.begin(target, scene, ActionPlan(2, 26))
        assert feedback.flight is not None
        executor = SMBExecutor()
        executor.start(plan, flight=feedback.flight)
        for f in range(100):
            if f:
                scene = scene_from_labels(env.scene_labels())
                feedback.observe(scene)
            _, _, done, _, _ = env.step(executor.press(scene))
            if done:
                break
        assert env._goal_credited
    finally:
        env.close()


def test_execution_measures_running_takeoff_before_launching_a_checked_jump():
    env = MarioScenarioEnv()
    feedback = SpatialFeedback()
    executor = SMBExecutor()
    try:
        env.reset(
            scenario={
                "mario": [60, 208],
                "world_width": 380,
                "mario_velocity": [2.5, 0],
                "platforms": [[0, 220, 90, 20], [112, 198, 24, 10], [158, 220, 222, 20]],
                "goal": [116, 178, 16, 20],
                "goal_requires_support": True,
                "single_jump_attempt": True,
            }
        )
        launched = False
        for frame in range(60):
            scene = scene_from_labels(env.scene_labels())
            feedback.observe(scene)
            if executor.idle or executor.finished:
                executor.end("done")
                # Preserve the physical landing point while observing motion.
                destination = SkillToken("jump", round(122 - env.mario["x"] - 5), -22)
                plan = feedback.begin(destination, scene, ActionPlan(2, 26))
                executor.start(plan, flight=feedback.flight)
                if feedback.flight:
                    launched = True
                    assert frame >= 2 and feedback.flight.prediction.safe
            _, _, done, _, _ = env.step(executor.press(scene))
            if done:
                break
        assert launched
        assert env._goal_credited
    finally:
        env.close()


def test_a_new_airborne_command_releases_the_previous_jump_hold():
    feedback = SpatialFeedback()
    air = SceneObservation(MarioView((80, 150, 90, 162), True, "air", False))
    plan = feedback.begin(SkillToken("jump", 80, 30), air, ActionPlan(2, 14))
    assert plan == ActionPlan(1, 14)


def test_moving_enemy_contact_waits_for_motion_instead_of_assuming_a_static_arc():
    feedback = SpatialFeedback(speed=-2.5)
    scene = SceneObservation(
        MarioView((160, 208, 170, 220), False, "ground", True),
        enemies=(EnemyView((100, 210, 110, 220), "walker"),),
    )
    proposal = ActionPlan(4, 32)
    assert feedback.begin(SkillToken("jump", -60, -10), scene, proposal) == ActionPlan(
        HOLD_GROUND, 1
    )
    assert feedback.status == "observing_motion"


def test_bridge_carry_is_not_mistaken_for_takeoff_momentum():
    feedback = SpatialFeedback()

    def picture(left, mario):
        return SceneObservation(
            MarioView((mario, 208, mario + 10, 220), True, "moving_platform", True),
            moving_platforms=((left, 220, left + 64, 228),),
            surfaces=(Surface(8, 40, 220, False),),
        )

    feedback.observe(picture(80, 100))
    feedback.observe(picture(82, 102))
    assert feedback.displacement == 2
    assert feedback.speed == 0


def test_run_destination_completion_compensates_camera_motion():
    feedback = SpatialFeedback()

    def picture(x, edge):
        return SceneObservation(
            MarioView((x, 208, x + 10, 220), True, "ground", True),
            surfaces=(Surface(8, edge, 220, False),),
        )

    first = picture(85, 180)
    feedback.observe(first)
    feedback.begin(SkillToken("run", 5, 0), first, ActionPlan(1, 32))
    feedback.observe(picture(85, 177))
    assert not feedback.arrived()
    feedback.observe(picture(85, 175))
    assert feedback.arrived()


def test_scrolling_past_a_required_retreat_ends_the_unrecoverable_episode():
    from retroagi.stages.block_smb.tactic_schedule import segment

    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "mario": [20, 208],
                "world_width": 500,
                "platforms": [[0, 220, 500, 20]],
                "goal": [450, 200, 20, 20],
                "tactics": [
                    segment("advance", reach_x=200),
                    segment("retreat", -1, reach_x=60),
                    segment("advance"),
                ],
            }
        )
        for _ in range(200):
            _, _, done, _, info = env.step(1)
            if done:
                break
        assert done and info["off_route"] and not info["death"]
        assert env.camera_x > 60
        assert not env._goal_credited
    finally:
        env.close()
