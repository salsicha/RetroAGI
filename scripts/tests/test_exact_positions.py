"""In training, jumps at a walker aim where the simulator will have moved it:
the ground truth the target tracker's predicted collision point will stand
for once it is trained (user, 2026-10-14/15)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_trajectory import VisualTracks, aimed_goal
from retroagi.stages.block_smb.controller_teacher import exact_positions
from retroagi.stages.block_smb.env import MarioScenarioEnv


def scene_with_walker(speed):
    env = MarioScenarioEnv()
    env.reset(
        scenario={
            "world_width": 256,
            "mario": [60, 204],
            "platforms": [[0, 220, 256, 20]],
            "enemies": [[150, 206, 120, 200, speed, -1]],
            "goal": [230, 200, 16, 20],
        }
    )
    tracks = VisualTracks()
    scene = scene_from_labels(env.scene_labels())
    tracks.observe(scene, None)
    return env, scene, tracks


def test_a_walker_is_aimed_at_where_it_is_unless_positions_are_exact():
    env, scene, tracks = scene_with_walker(0.5)
    try:
        (walker,) = tracks.tracks
        box = scene.mario.box
        args = (walker, (0, 0), tracks, scene, NESPlayerMotion(), box, 1)
        middle = (walker.box[0] + walker.box[2]) / 2
        # Without the truth (and before the target tracker is trained): where
        # it is now. In training: where it will be when the jump lands.
        assert aimed_goal(*args)[0] == middle
        tracks.exact = exact_positions(env)
        assert aimed_goal(*args)[0] < middle - 2
    finally:
        env.close()


def test_a_walking_walker_is_aimed_at_where_it_will_be_when_the_jump_lands():
    env, scene, tracks = scene_with_walker(0.5)
    try:
        (walker,) = tracks.tracks
        tracks.exact = exact_positions(env)
        goal = aimed_goal(walker, (0, 0), tracks, scene, NESPlayerMotion(), scene.mario.box, 1)
        middle = (walker.box[0] + walker.box[2]) / 2
        # Walking left at half a pixel a frame through the jump's flight.
        assert middle - 40 < goal[0] < middle - 2
    finally:
        env.close()
