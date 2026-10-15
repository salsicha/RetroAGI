"""The ground truth of when and where each enemy and Mario collide (the target
trackers' stand-in until they are trained)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from retroagi.stages.block_smb.collision_truth import collisions
from retroagi.stages.block_smb.env import MarioScenarioEnv


def env_with_walker(x, speed, direction):
    env = MarioScenarioEnv()
    env.reset(
        scenario={
            "world_width": 320,
            "mario": [60, 204],
            "platforms": [[0, 220, 320, 20]],
            "enemies": [[x, 206, 0, 300, speed, direction]],
            "goal": [300, 200, 16, 20],
        }
    )
    for _ in range(4):  # Mario settles on the floor
        env.step(0)
    return env


def test_an_approaching_walker_hits_a_standing_mario_when_it_reaches_him():
    env = env_with_walker(100, 0.5, -1)
    try:
        m, e = env.mario, env.enemies[0]
        (found,) = collisions(env)
        gap = e["x"] - (m["x"] + m["w"])
        assert found.kind == "hit" and found.closing == 0.5
        assert abs(found.frames - gap / 0.5) <= 2
        assert found.x == m["x"] + m["w"] / 2 and found.y == m["y"] + m["h"]
    finally:
        env.close()


def test_a_faster_walker_comes_sooner_and_one_walking_away_never():
    slow, fast, away = env_with_walker(100, 0.4, -1), env_with_walker(100, 0.8, -1), None
    try:
        assert collisions(fast)[0].frames < collisions(slow)[0].frames
        away = env_with_walker(100, 0.5, 1)
        assert collisions(away) == []
    finally:
        for env in (slow, fast, away):
            if env is not None:
                env.close()


def test_coming_down_onto_a_walker_is_a_stomp():
    env = env_with_walker(64, 0.0, -1)
    try:
        m, e = env.mario, env.enemies[0]
        # Mario's feet falling 3 pixels a frame from above the walker's top.
        x0 = e["x"]
        path = [(x0, e["y"] - 30 + 3 * t, x0 + m["w"], e["y"] - 30 + 3 * t + m["h"]) for t in range(1, 30)]
        (found,) = collisions(env, path)
        assert found.kind == "stomp"
        assert abs(found.y - e["y"]) <= 3
    finally:
        env.close()
