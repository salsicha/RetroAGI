"""Block SMB's enemies under the real game's rules (docs/choose-and-control-
plan.md phase 2; user, 2026-11-07): no patrol; an enemy wakes when the
screen reaches it, walks half a pixel a frame, turns back at a wall and
falls off a ledge."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from retroagi.domains.smb.sim.env import MarioScenarioEnv

FLOOR = 220


def play(scenario, frames, action=0):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        xs = []
        for _ in range(frames):
            env.step(action)
            xs.append([(e["x"], e["y"]) for e in env.enemies])
        return xs
    finally:
        env.close()


def walker(x, y=FLOOR - 14, **more):
    return {"x": x, "y": y, "game_rules": True, **more}


def test_a_walker_waits_off_screen_until_the_screen_reaches_it():
    scenario = {"world_width": 1200, "mario": [40, FLOOR - 20], "platforms": [[0, FLOOR, 1200, 20]],
                "enemies": [walker(900)]}
    xs = play(scenario, 60)
    assert xs[-1][0][0] == 900  # Mario stood still: it never came on screen


def test_a_walker_walks_at_half_a_pixel_and_turns_at_a_wall():
    scenario = {"world_width": 400, "mario": [20, FLOOR - 20],
                "platforms": [[0, FLOOR, 400, 20], [100, FLOOR - 32, 32, 32]],
                "enemies": [walker(160)]}
    xs = play(scenario, 120)
    assert abs((xs[0][0][0] - xs[39][0][0]) - 0.5 * 39) <= 1  # walking left at 0.5
    # It met the pipe's side (at 132) and turned back.
    assert min(x for (x, _), in xs) >= 132 - 1 and xs[-1][0][0] > min(x for (x, _), in xs)


def test_a_walker_falls_off_a_ledge():
    scenario = {"world_width": 400, "mario": [20, FLOOR - 20],
                "platforms": [[0, FLOOR, 400, 20], [150, FLOOR - 40, 100, 8]],
                "enemies": [walker(160, y=FLOOR - 40 - 14)]}
    xs = play(scenario, 120)
    assert xs[-1][0][1] > xs[0][0][1] + 20  # walked off the raised floor and fell
