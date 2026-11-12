"""Hammer Bros in Block SMB (sim/env.py, measured in World 3-1's memory):
how one walks, jumps from floor to floor and throws hammers, how the
state shows them, how the planner predicts a hammer, and the generator's
piece of them."""

from retroagi.domains.smb.perception.state import hammer_bro_box, thrown_box
from retroagi.domains.smb.planner import Enemy, Solid, enemy_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import (
    FLOOR,
    HIGH_ROW,
    LOW_ROW,
    SCENE_KINDS,
    TILE,
    random_level,
    skill_scene,
)
from retroagi.domains.smb.sim.world import SimSource, SimWorld

ROW = [x for x in range(160, 160 + 10 * TILE, TILE)]


def scene(bros, mario_x=40.0, rows=(LOW_ROW, HIGH_ROW)):
    """A floor, two brick rows over x 160..320 and the given Hammer Bros
    (each (x, the top it stands on, its first throw, its first jump))."""
    platforms = [[0, FLOOR, 2000, 32]] + [[x, top, TILE, TILE] for top in rows for x in ROW]
    return {
        "world_width": 2000, "scroll_x": 115, "mario": [mario_x, FLOOR - 12],
        "platforms": platforms, "platform_kinds": ["ground"] + ["brick"] * (len(platforms) - 1),
        "enemies": [{"x": float(x), "y": float(top - 14), "kind": "hammer_bro", "game_rules": True,
                     "throw_wait": throw, "jump_wait": jump} for x, top, throw, jump in bros],
        "task": {"family": "game_level"},
    }


def bro_of(world):
    return next(e for e in world.env.enemies if e.get("kind") == "hammer_bro")


def test_a_hammer_bro_walks_back_and_forth_and_throws_hammers_at_mario():
    world = SimWorld(scene([(200, FLOOR, 5, 10_000)]))
    world.reset()
    bro = bro_of(world)
    assert bro["y"] + bro["h"] - 4 == FLOOR and (bro["w"], bro["h"]) == (8, 24)
    xs, first = [], None
    for frame in range(400):
        world.step(0)
        xs.append(bro["x"])
        hammers = [e for e in world.env.enemies if e.get("kind") == "hammer"]
        if hammers and first is None:
            first = frame
    assert min(xs) == 192 and max(xs) == 208  # 8 pixels either way, a quarter pixel a frame
    assert abs(xs[1] - xs[0]) == 0.25
    hammers = [e for e in world.env.enemies if e.get("kind") == "hammer"]
    assert first == 4 and 7 <= len(hammers) <= 9  # every 49 or 50 frames
    assert all(not h["stompable"] for h in hammers)


def test_a_hammer_is_held_then_thrown_toward_mario_through_everything():
    world = SimWorld(scene([(200, FLOOR, 1, 10_000)]))
    world.reset()
    world.step(0)
    hammer = next(e for e in world.env.enemies if e.get("kind") == "hammer")
    held, ys = [], []
    for _ in range(20):
        world.step(0)
        held.append(hammer["held"])
        ys.append(hammer["y"])
    thrown = held.index(0)
    assert thrown == block.HAMMER_HOLD - 2  # (held from the frame it appeared)
    assert ys[thrown + 1] - ys[thrown] == block.HAMMER_THROW + block.HAMMER_GRAVITY  # up at about 2 a frame
    ys = []
    for _ in range(60):
        x = hammer["x"]
        world.step(0)
        assert hammer["x"] - x == -1.0  # toward Mario, at his left
        ys.append(hammer["y"])
    steps = [b - a for a, b in zip(ys, ys[1:])]
    assert abs(steps[1] - steps[0] - block.HAMMER_GRAVITY) < 1e-9
    assert max(ys) > LOW_ROW + TILE  # it fell through the brick rows


def test_a_hammer_bro_jumps_to_the_floor_above_and_back_down_through_it():
    world = SimWorld(scene([(200, FLOOR, 10_000, 3)]))
    world.reset()
    bro = bro_of(world)
    stood = []
    for _ in range(600):
        world.step(0)
        if bro["on_ground"]:
            feet = bro["y"] + bro["h"] - 4
            if not stood or stood[-1] != feet:
                stood.append(feet)
    assert stood[:3] == [FLOOR, LOW_ROW, FLOOR] or stood[:3] == [FLOOR, LOW_ROW, HIGH_ROW]


def test_mario_stomps_a_hammer_bro_and_a_hammer_kills_him():
    world = SimWorld(scene([(200, FLOOR, 10_000, 10_000)], rows=()))
    world.reset()
    bro = bro_of(world)
    env = world.env
    env.mario.update(x=bro["x"] - 1, y=bro["y"] - env.mario["h"] - 2, vy=3.0, on_ground=False)
    env.motion.y_speed = 3
    for _ in range(3):
        world.step(0)
    assert bro["dead"]
    world = SimWorld(scene([(120, FLOOR, 1, 10_000)], mario_x=60.0, rows=()))
    world.reset()
    ends = [world.step(0) for _ in range(200)]
    assert any(s.died for s in ends)


def test_the_state_shows_a_hammer_bro_as_a_walker_and_its_hammers_as_thrown():
    world = SimWorld(scene([(200, FLOOR, 1, 10_000)]))
    world.reset()
    for _ in range(30):
        world.step(0)
    kinds = sorted((e.kind, e.stompable) for e in SimSource().read(world).enemies)
    assert kinds == [("thrown", False), ("walker", True)]


def test_the_planner_has_a_thrown_thing_fall_through_everything():
    row = Solid(96, 144, 160, 160, "block", ("blocks", 0))
    hammer = Enemy((120.0, 100.0, 128.0, 108.0), 1.0, -2.0, ("misc", 0), stompable=False, kind="thrown")
    enemy_paths([hammer], [row], 120)
    assert hammer.at(10)[0] == 130.0 and abs(hammer.at(1)[1] - (100.0 - 2.0 + 0.0625)) < 1e-9
    assert hammer.at(120)[1] > 160  # down through the row
    from retroagi.domains.smb.planner import THROWN_FALL, THROWN_GRAVITY

    assert (THROWN_GRAVITY, THROWN_FALL) == (block.HAMMER_GRAVITY, block.HAMMER_FALL)


def test_vision_s_boxes_for_a_hammer_bro_and_a_hammer_are_the_game_s():
    assert hammer_bro_box((100, 40, 116, 64)) == (104, 43, 112, 67)
    assert thrown_box((100, 40, 108, 56)) == (100.0, 44.0, 108.0, 52.0)


def test_the_generator_has_a_piece_of_hammer_bros():
    assert "hammer_bros" in SCENE_KINDS
    for seed in range(20):
        level = skill_scene(seed, "hammer_bros")
        bros = [e for e in level["enemies"] if e.get("kind") == "hammer_bro"]
        rows = {p[1] for p, k in zip(level["platforms"], level["platform_kinds"]) if k in ("brick", "question_block")}
        assert 1 <= len(bros) <= 2 and rows == {LOW_ROW, HIGH_ROW}
        assert {b["y"] + 14 for b in bros} <= {FLOOR, LOW_ROW}
    assert any(e.get("kind") == "hammer_bro" for seed in range(40) for e in random_level(seed)["enemies"])
