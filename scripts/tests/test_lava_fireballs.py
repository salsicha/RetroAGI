"""Lava fireballs in Block SMB (measured in World 2-4's memory): one jumps
out of its pit from under the screen, rises to the game's heights and falls
back, and jumps again after one of the game's waits; it hurts and cannot be
stomped; the planner follows its leap; the generator puts them in pits."""

from retroagi.domains.smb.planner import enemy_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, TILE, random_level, skill_scene
from retroagi.domains.smb.sim.world import SimSource, SimWorld

PIT = (320, 352)


def scene(wait=1, turn=1, mario=(40.0, FLOOR - 12)):
    return {"world_width": 1000, "scroll_x": 115, "mario": list(mario), "frame_budget": 3000,
            "platforms": [[0, FLOOR, PIT[0], 32], [PIT[1], FLOOR, 1000 - PIT[1], 32]],
            "platform_kinds": ["ground", "ground"],
            "enemies": [{"kind": "lava", "x": PIT[0] + 8, "wait": wait, "turn": turn}], "task": {"family": "game_level"}}


def lava_of(world):
    return next(e for e in world.env.enemies if e.get("kind") == "lava")


def test_it_leaps_as_high_as_the_games_and_again_after_a_wait():
    tops, starts = [], []
    for turn in range(1, 40):
        world = SimWorld(scene(turn=turn))
        world.reset()
        lava, ys, jumps = lava_of(world), [], []
        for frame in range(700):
            before = lava["vy"]
            world.step(0)
            ys.append(lava["y"] - block.LAVA_BOX[1])
            if lava["vy"] < before - 1:  # (a jump: its speed turned up at once)
                jumps.append(frame)
        starts.extend(b - a for a, b in zip(jumps, jumps[1:]))
        tops.append(min(ys[:200]))
        # (the game's first frames out: 251, 245, 239)
        assert abs(ys[0] - 251.5) <= 1.0 and abs(ys[2] - 239) <= 1.5
    # the game's apexes: 73 and 90 in 2-4
    assert 68 <= min(tops) <= 74 and 89 <= max(tops) <= 94
    waits = set(starts)
    assert waits <= {block.LAVA_WAIT * w for w in block.LAVA_WAITS} and len(waits) >= 3


def test_it_hurts_from_below_and_cannot_be_stomped():
    # (Mario over the pit's edge as it comes up beside him: it reaches him)
    world = SimWorld(scene(wait=30, mario=(PIT[0] + 4, 120.0)))
    world.reset()
    world.env.mario["on_ground"] = False
    for _ in range(200):
        step = world.step(0)
        if step.over:
            break
    assert step.over and step.died and not world.env.stomped


def test_the_state_shows_it_and_the_planner_follows_its_leap():
    world = SimWorld(scene(wait=1, mario=(250.0, FLOOR - 12)))
    world.reset()
    for _ in range(12):
        world.step(0)
    state = SimSource().read(world)
    lava = next(e for e in state.enemies if e.kind == "fountain")
    assert not lava.stompable and lava.vy < 0
    enemy_paths([lava], state.solids, 60)
    sim = lava_of(world)
    for frame in range(1, 61):
        world.step(0)
        assert abs(lava.path[frame][1] - sim["y"]) < 1e-6


def test_the_generator_puts_lava_fireballs_in_pits():
    scenes = [skill_scene(seed, "lava") for seed in range(20)]
    for s in scenes:
        floor = [p for p in s["platforms"] if isinstance(p, list) and p[1] == FLOOR]
        for e in s["enemies"]:
            if e["kind"] == "lava":
                middle = e["x"] + 8
                assert not any(p[0] <= middle < p[0] + p[2] for p in floor)  # (over no floor: in a pit)
    assert any(len([e for e in s["enemies"] if e["kind"] == "lava"]) > 1 for s in scenes)
    assert any(e.get("kind") == "lava" for seed in range(60) for e in random_level(seed)["enemies"])
