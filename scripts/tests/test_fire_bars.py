"""Fire bars in Block SMB (measured in Worlds 1-4's and 5-4's memory: the
bar's turn, and its fireballs as drawn): that a bar turns once round in 205
frames with its fireballs at whole steps of its turn, 8 apart, where the
game draws them; that a fireball hurts and cannot be stomped or knocked
out by a shell; that the state shows it going round its hub and the
planner predicts where it will be; and that the generator's pieces have
the game's heights."""

import math

from retroagi.domains.smb.planner import enemy_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FIREBAR_HEIGHTS, FLOOR, TILE, random_level, skill_scene
from retroagi.domains.smb.sim.world import SimSource, SimWorld

HUB_X, HUB_Y = 304, 144  # (the block the bar turns on)


def scene(angle=0.0, clockwise=False, balls=6, mario=(40.0, FLOOR - 12)):
    return {"world_width": 1000, "scroll_x": 115, "mario": list(mario), "frame_budget": 3000,
            "platforms": [[0, FLOOR, 1000, 32], [HUB_X, HUB_Y, TILE, TILE]], "platform_kinds": ["ground", "ground"],
            "enemies": [{"kind": "firebar", "x": HUB_X, "y": HUB_Y, "angle": angle, "clockwise": clockwise,
                         "balls": balls}],
            "task": {"family": "game_level"}}


def balls_of(world):
    return [e for e in world.env.enemies if e.get("kind") == "firebar"]


def test_the_fireballs_are_where_the_game_draws_them():
    # 5-4's long bar at step 19 of its turn: each fireball from the first, as drawn by the game
    game = [(0, 0), (-4, 6), (-9, 13), (-13, 19), (-18, 26), (-22, 33), (-27, 39), (-31, 46), (-36, 53), (-40, 59),
            (-45, 66), (-49, 73)]
    world = SimWorld(scene(angle=19.4, balls=12))
    world.reset()
    balls = balls_of(world)
    first = balls[0]
    for ball, (gx, gy) in zip(balls, game):
        assert abs(ball["x"] - first["x"] - gx) <= 1 and abs(ball["y"] - first["y"] - gy) <= 1
    assert (first["x"], first["y"]) == (HUB_X + 4, HUB_Y + 4)


def test_a_bar_turns_once_round_in_205_frames_by_whole_steps():
    world = SimWorld(scene(angle=0.0, clockwise=True))
    world.reset()
    tip = balls_of(world)[-1]
    places = []
    for _ in range(205):
        world.step(0)
        places.append((round(tip["x"], 3), round(tip["y"], 3)))
    assert len(set(places)) == block.FIREBAR_STEPS  # (32 places, each held 6 or 7 frames)
    assert places[0][0] == places[5][0] and places[40][0] > HUB_X + 8 + 30  # (clockwise: to the right a quarter round on)
    assert abs(tip["angle"] - 205 * block.FIREBAR_TURN % block.FIREBAR_STEPS) < 1e-6


def test_a_fireball_hurts_and_cannot_be_stomped_or_knocked_out():
    # (Mario standing under the bar's block, the bar turning down onto him)
    world = SimWorld({**scene(angle=18.5, mario=(HUB_X + 2, FLOOR - 12)), "platforms": [[0, FLOOR, 1000, 32]],
                      "platform_kinds": ["ground"],
                      "enemies": [{"kind": "firebar", "x": HUB_X, "y": FLOOR - 48, "angle": 18.5}]})
    world.reset()
    over = None
    for frame in range(60):
        over = world.step(0)
        if over.over:
            break
    assert over.died and frame < 30
    # dropped on from above, it still hurts
    world = SimWorld(scene(angle=0.0, mario=(HUB_X + 2, 20.0)))
    world.reset()
    for _ in range(200):
        over = world.step(0)
        if over.over:
            break
    assert over.died and not world.env.stomped
    assert all(not b["dead"] for b in balls_of(world))


def test_the_state_shows_it_going_round_and_the_planner_follows_it():
    world = SimWorld(scene(angle=3.0, clockwise=False, mario=(200.0, FLOOR - 12)))
    world.reset()
    state = SimSource().read(world)
    tips = [e for e in state.enemies if e.kind == "turner"]
    assert len(tips) == 6 and all(not e.stompable for e in tips)
    enemy_paths(tips, state.solids, 60)
    tip = max(tips, key=lambda e: math.hypot(e.box[0] - HUB_X, e.box[1] - HUB_Y))
    sim_tip = balls_of(world)[-1]
    worst = 0.0
    for frame in range(1, 61):
        world.step(0)
        predicted = tip.path[frame]
        worst = max(worst, math.hypot(predicted[0] - sim_tip["x"], predicted[1] - sim_tip["y"]))
    assert worst <= 5.0  # (half a step of its turn, 40 out)


def test_fire_bar_pieces_have_the_games_heights():
    scenes = [skill_scene(seed, "firebar") for seed in range(30)]
    bars = [e for s in scenes for e in s["enemies"] if e["kind"] == "firebar"]
    heights = {round((FLOOR - b["y"]) / TILE) for b in bars}
    assert heights <= set(FIREBAR_HEIGHTS) and len(heights) >= 4
    assert any(not b["clockwise"] for b in bars) and any(b["clockwise"] for b in bars)
    assert any(e.get("kind") == "firebar" for seed in range(60) for e in random_level(seed)["enemies"])
