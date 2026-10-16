"""The executor's first-order kinematics: each tracked sprite's position and
velocity, and Mario's, as the world model and the collision trackers read
them (user, 2026-10-16: vision reports boxes only)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import replace

import pytest

from retroagi.core.smb_kinematics import MARIO_STATE, SPRITE_SLOTS, SPRITE_STATE, kinematics
from retroagi.core.smb_observer import packed_lists
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_trajectory import LOST_FRAMES
from retroagi.stages.block_smb.env import MarioScenarioEnv

S = {name: i for i, name in enumerate(SPRITE_STATE)}
M = {name: i for i, name in enumerate(MARIO_STATE)}


def play(enemies, frames, button=0, hide=()):
    """Mario on a long floor with ``enemies``; the executor observes every
    frame (the frames in ``hide`` with no enemy seen). Returns the last
    scene, the executor's state and the tracks' identities per frame."""
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 400,
                "mario": [40, 208],
                "platforms": [[0, 220, 400, 20]],
                "enemies": enemies,
                "goal": [380, 200, 16, 20],
            }
        )
        spatial, identities = SpatialFeedback(), []
        for frame in range(frames):
            scene = scene_from_labels(env.scene_labels())
            if frame in hide:
                scene = replace(scene, enemies=())
            spatial.observe(scene)
            spatial.executed(button, scene)
            identities.append([t.identity for t in spatial.tracks.tracks])
            env.step(button)
        return scene, spatial, identities
    finally:
        env.close()


def test_a_slow_walkers_speed_is_estimated_from_its_sightings():
    # 0.3 pixels a frame: its box moves 0 or 1 pixel between two frames.
    scene, spatial, _ = play([[150, 206, 60, 300, 0.3, -1]], 24)
    sprites, _ = kinematics(scene, spatial)
    walker = sprites[0]
    assert walker[S["present"]] == 1.0 and walker[S["seen"]] == 1.0
    assert walker[S["vx"]] == pytest.approx(-0.3, abs=0.05)
    assert walker[S["vy"]] == pytest.approx(0.0, abs=0.05)
    box, mario = scene.enemies[0].box, scene.mario.box
    assert walker[S["x"]] == pytest.approx((box[0] + box[2]) / 2 - (mario[0] + mario[2]) / 2)
    assert not sprites[1:].any()


def test_a_walker_missed_for_a_few_frames_keeps_its_identity_and_speed():
    scene, spatial, identities = play([[150, 206, 60, 300, 0.3, -1]], 30, hide=(18, 19, 20))
    assert identities[17] == identities[29] and identities[18] == []
    sprites, _ = kinematics(scene, spatial)
    assert sprites[0, S["vx"]] == pytest.approx(-0.3, abs=0.05)


def test_a_walker_unseen_for_longer_is_a_new_sprite():
    hidden = tuple(range(18, 18 + LOST_FRAMES + 2))
    _, _, identities = play([[150, 206, 60, 300, 0.3, -1]], 34, hide=hidden)
    assert identities[17] != identities[33]


def test_estimates_follow_the_scenes_slot_order():
    # The nearer walker is in slot 0, as its scene token is.
    scene, spatial, _ = play(
        [[220, 206, 200, 300, 0.5, 1], [120, 206, 60, 300, 0.4, -1]], 20
    )
    sprites, _ = kinematics(scene, spatial)
    lists = packed_lists(scene)
    mario = scene.mario.box
    for slot, enemy in enumerate(lists["enemies"]):
        middle = (enemy.box[0] + enemy.box[2]) / 2 - (mario[0] + mario[2]) / 2
        assert sprites[slot, S["x"]] == pytest.approx(middle)
    assert sprites[0, S["vx"]] == pytest.approx(-0.4, abs=0.08)
    assert sprites[1, S["vx"]] == pytest.approx(0.5, abs=0.08)
    assert sprites.shape == (SPRITE_SLOTS, len(SPRITE_STATE))


def test_marios_own_speed_and_contact_are_estimated():
    scene, spatial, _ = play([], 30, button=1)
    _, mario = kinematics(scene, spatial)
    assert mario[M["vx"]] > 1.0 and mario[M["grounded"]] == 1.0
    assert mario[M["vx"]] == pytest.approx(spatial.speed)
