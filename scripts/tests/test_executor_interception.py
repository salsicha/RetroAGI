"""Regressions from alternate-route and stomp validation failures."""

import pytest

from retroagi.core.smb_collision import stomp_contact, walker_body
from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_trajectory import Track
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv


def test_stomp_requires_physical_overlap_and_post_floor_downward_velocity():
    enemy = walker_body((89, 210, 99, 220))
    assert enemy == (89, 210, 99, 216)
    assert walker_body((89.1, 210, 99.10000000001, 220))[3] == 216
    assert not stomp_contact((80, 208, 90, 220), enemy, 0)
    assert not stomp_contact((81, 206, 91, 218), enemy, 4)
    assert stomp_contact((81, 204, 91, 216), enemy, 4)
    assert not stomp_contact((79, 204, 89, 216), enemy, 4)
    assert walker_body((89, 204, 105, 220)) == (89, 204, 105, 220)


def test_observed_patrol_forecast_reflects_at_both_turns():
    track = Track((113, 210, 123, 220), "walker", [(1, 0)] * 4, [90, 118])
    assert track.forecast(10)[0] == 113
    assert track.forecast(40)[0] == 97
    assert track.forecast(66)[0] == 113


def execute(scenario, target, frames=170):
    env = MarioScenarioEnv()
    env.reset(scenario=scenario)
    feedback, executor = SpatialFeedback(), SMBExecutor()
    buttons, statuses = [], []
    launched = False
    try:
        for _ in range(frames):
            scene = scene_from_labels(env.scene_labels())
            feedback.observe(scene)
            if executor.idle or executor.finished or executor.reconsider:
                executor.end("recheck")
                plan = feedback.begin(target, scene)
                executor.start(plan, flight=feedback.flight, travel=feedback.travel)
            button = executor.press(scene)
            feedback.executed(button, scene)
            buttons.append(button)
            statuses.append(feedback.flight.status if feedback.flight else feedback.status)
            _, _, done, _, info = env.step(button)
            launched |= button in (2, 4, 5)
            if info.get("death"):
                pytest.fail(f"died after {len(buttons)} frames: {statuses[-8:]}")
            if env.stomped or (launched and env.mario["on_ground"]):
                return dict(env.mario), env.stomped, buttons, statuses
            if done:
                break
        pytest.fail(f"never landed/stomped: {statuses[-8:]}, mario={env.mario}")
    finally:
        env.close()


@pytest.mark.parametrize("spawn,ledge,dx", [(67, 72, 10), (74, 71, -6)])
def test_back_away_then_reverse_onto_overhead_platform(spawn, ledge, dx):
    mario, stomped, buttons, _ = execute(
        {
            "world_width": 351,
            "mario": [spawn, 208],
            "platforms": [[0, 220, 351, 20], [ledge, 176, 40, 10]],
        },
        SkillToken("jump", dx, -44),
    )
    assert mario["y"] == 164
    assert 3 in buttons[: buttons.index(2)]
    assert not stomped


@pytest.mark.parametrize(
    "enemy,dx",
    [
        ([99, 206, 99, 99, 0, 1], 50),
        ([102, 206, 97.8, 117.8, 0.6, -1], 58),
        ([107, 206, 100.4, 120.4, 0.6, -1], 63),
        ([112, 206, 89.4, 117.4, 0.9, 1], 80),
        ([115, 206, 107.8, 135.8, 0.9, -1], 67),
    ],
)
def test_stomp_destination_executes_without_floor_contact_death_or_stationary_deadlock(enemy, dx):
    _, stomped, buttons, _ = execute(
        {
            "world_width": 340,
            "mario": [40, 204],
            "platforms": [[0, 220, 340, 20]],
            "enemies": [enemy],
            "goal_on_stomp": True,
            "goal": [enemy[0] - 2, 186, 16, 20],
        },
        SkillToken("jump", dx, -6),
    )
    assert stomped
    jumping = [i for i, b in enumerate(buttons) if b in (2, 4, 5)]
    assert jumping == list(range(jumping[0], jumping[-1] + 1))


def test_landing_reconciles_small_speed_error_that_changes_jump_physics():
    from collections import deque

    from retroagi.core.smb_physics import NESPlayerMotion
    from retroagi.core.smb_scene_labels import MarioView, SceneObservation

    feedback = SpatialFeedback(
        previous=SceneObservation(MarioView((40, 206, 50, 218), True, "air", False)),
        motion=NESPlayerMotion(x_speed=30),
        motion_ready=True,
        velocities=deque([1, 2, 1], maxlen=4),
    )
    feedback.observe(SceneObservation(MarioView((42, 208, 52, 220), True, "ground", True)))
    assert feedback.motion.x_speed == 24  # Slow-jump profile, despite <0.75px error.


def test_slow_approaching_walker_needs_overlap_margin_before_the_floor_landing():
    _, stomped, _, _ = execute(
        {
            "world_width": 360,
            "mario": [40, 204],
            "platforms": [[0, 220, 360, 20]],
            "enemies": [[147, 206, 0, 360, 0.3, -1]],
            "goal_on_stomp": True,
            "goal": [145, 186, 16, 20],
        },
        SkillToken("jump", 89, -9),
    )
    assert stomped
