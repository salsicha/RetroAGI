"""Closed-loop holding uses observations and emits only emulator button actions."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.core.actions import SMBAction
from retroagi.core.smb_executor import HOLD_GROUND, ActionPlan, SMBExecutor
from retroagi.core.smb_scene_labels import MarioView, SceneObservation, Surface, scene_from_labels
from retroagi.stages.block_smb.env import MarioScenarioEnv


def picture(left=80, mario=96, top=200):
    return SceneObservation(
        MarioView((mario, top - 16, mario + 12, top), True, "moving_platform", True),
        moving_platforms=((left, top, left + 64, top + 8),),
        surfaces=(Surface(left, left + 64, top, True),),
    )


def test_hold_corrects_both_directions_and_tracks_camera_and_platform_motion():
    executor = SMBExecutor()
    executor.start(ActionPlan(HOLD_GROUND, 10))
    assert executor.press(picture()) == SMBAction.NOOP
    # Whole picture translates, with the same relative spot on the platform.
    assert executor.press(picture(left=100, mario=116, top=198)) == SMBAction.NOOP
    assert executor.press(picture(left=100, mario=122, top=198)) == SMBAction.LEFT
    assert executor.press(picture(left=100, mario=110, top=198)) == SMBAction.RIGHT


def test_consecutive_holds_keep_anchor_and_another_action_releases_it():
    executor = SMBExecutor()
    executor.start(ActionPlan(HOLD_GROUND, 2))
    executor.press(picture())
    assert executor.press(picture(mario=103)) == SMBAction.LEFT
    executor.end("done")
    executor.start(ActionPlan(HOLD_GROUND, 1))
    assert executor.press(picture(mario=103)) == SMBAction.LEFT
    executor.end("done")
    executor.start(ActionPlan(SMBAction.NOOP, 1))
    executor.press()
    executor.end("done")
    executor.start(ActionPlan(HOLD_GROUND, 1))
    assert executor.press(picture(mario=103)) == SMBAction.NOOP


def test_missing_support_releases_buttons_without_reanchoring_to_another_platform():
    executor = SMBExecutor()
    executor.start(ActionPlan(HOLD_GROUND, 5))
    executor.press(picture())
    assert executor.press(None) == SMBAction.NOOP
    assert executor.press(picture(left=180, mario=196)) == SMBAction.NOOP
    assert executor.press(picture(mario=103)) == SMBAction.LEFT


@pytest.mark.parametrize("moving,direction", [(False, 1), (True, -1), (True, 1)])
@pytest.mark.parametrize("velocity", [-2.0, 2.0])
def test_hold_recovers_from_drift_on_static_and_reversing_platforms(moving, direction, velocity):
    platform = {"x": 90, "y": 220, "w": 96, "h": 12}
    if moving:
        platform.update(moving=[60, 150, 0.7], direction=direction)
    env = MarioScenarioEnv()
    executor = SMBExecutor()
    try:
        env.reset(
            scenario={
                "world_width": 320,
                "mario": [130, 204],
                "platforms": [platform],
                "mario_velocity": [velocity, 0],
                "frame_budget": 400,
            }
        )
        anchor = env.mario["x"] - env.platforms[0]["rect"].x
        errors, buttons = [], []
        for frame in range(240):
            if executor.finished:
                executor.end("done")
            if executor.idle:
                executor.start(ActionPlan(HOLD_GROUND, 8))
            observation = scene_from_labels(env.scene_labels())
            button = executor.press(observation)
            buttons.append(button)
            _, _, done, _, info = env.step(button)
            assert not done, info
            assert env.mario["on_ground"]
            errors.append(abs(env.mario["x"] - env.platforms[0]["rect"].x - anchor))
        assert set(buttons) <= {SMBAction.NOOP, SMBAction.LEFT, SMBAction.RIGHT}
        assert SMBAction.LEFT in buttons and SMBAction.RIGHT in buttons
        # Existing momentum needs stopping distance before returning to the spot.
        assert max(errors) <= 20
        assert max(errors[-40:]) <= 2
    finally:
        env.close()
