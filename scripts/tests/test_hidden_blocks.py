"""Blocks invisible until hit in Block SMB (where the game's levels not
scored on hold them, and how its program meets them): unseen until Mario's
head hits one from below, nothing else meets it; hit, it shows, solid all
round, and gives its coin or lets out a 1-up; the generator puts them as
the game does, and Mario can get through."""

from collections import Counter

from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, TILE, _goomba, random_level, skill_scene
from retroagi.domains.smb.sim.passable import passable
from retroagi.domains.smb.sim.world import SimSource, SimWorld

RIGHT, JUMP = 1, 5


def scene(top, content="coin", mario=(100, FLOOR - 12), enemies=()):
    return {"world_width": 1000, "scroll_x": 115, "mario": list(mario), "frame_budget": 3000,
            "platforms": [[0, FLOOR, 1000, 32], [200, top, TILE, TILE]], "platform_kinds": ["ground", "hidden"],
            "block_contents": {1: content}, "enemies": list(enemies), "task": {"family": "game_level"}}


def hidden_of(world):
    return world.env.platforms[1]


def test_unhit_it_is_walked_and_fallen_through_and_not_seen():
    world = SimWorld(scene(FLOOR - TILE))  # (one tile up: in his way)
    world.reset()
    assert not any(s.ref == ("platforms", 1) for s in SimSource().read(world).solids)
    while world.env.mario["x"] < 260:
        world.step(RIGHT)
    assert not hidden_of(world).get("used") and world.env.mario["on_ground"]
    world = SimWorld(scene(FLOOR - 4 * TILE, mario=(202, FLOOR - 120)))
    world.reset()
    for _ in range(60):
        world.step(0)
    assert world.env.mario["y"] + world.env.mario["h"] == FLOOR  # (fell through it to the floor)


def test_his_head_hits_it_from_below_and_it_shows_solid_with_its_coin():
    world = SimWorld(scene(FLOOR - 4 * TILE, mario=(202, FLOOR - 12)))
    world.reset()
    score, tops = world.env.score, []
    for _ in range(40):
        world.step(JUMP)
        tops.append(world.env.mario["y"])
    plat = hidden_of(world)
    assert plat["used"] and world.env.score == score + block.COIN_POINTS
    assert min(tops) == FLOOR - 3 * TILE  # (his head stopped at its bottom)
    assert any(s.ref == ("platforms", 1) and s.kind == "block" for s in SimSource().read(world).solids)
    world.env.mario.update(x=202.0, y=FLOOR - 120.0, vy=0.0, on_ground=False)
    for _ in range(60):
        world.step(0)
    assert world.env.mario["on_ground"] and world.env.mario["y"] + world.env.mario["h"] == FLOOR - 4 * TILE


def test_a_1up_comes_out_and_is_worth_nothing_here():
    world = SimWorld(scene(FLOOR - 4 * TILE, content="1up", mario=(202, FLOOR - 12)))
    world.reset()
    score = world.env.score
    for _ in range(12):
        world.step(JUMP)
    assert [i["kind"] for i in world.env.items] == ["1up"]
    for _ in range(200):
        world.step(0)
        if not world.env.items:
            break
    assert world.env.score == score and not world.env.mario["big"]


def test_enemies_walk_through_it():
    goomba = _goomba(240, FLOOR)
    world = SimWorld(scene(FLOOR - TILE, mario=(40, FLOOR - 12), enemies=[goomba]))
    world.reset()
    walker = world.env.enemies[0]
    for _ in range(160):
        world.step(0)
    assert walker["x"] < 200 and not walker["dead"]  # (through it, not turned back)


def test_the_generator_puts_them_as_the_game_does_and_mario_gets_through():
    ways = Counter()
    for seed in range(60):
        s = skill_scene(seed, "hidden")
        hidden = [k for k, kind in enumerate(s["platform_kinds"]) if kind == "hidden"]
        assert hidden and all(s["block_contents"][k] in ("coin", "1up") for k in hidden)
        tops = sorted(s["platforms"][k][1] for k in hidden)
        ways[len(hidden), "1up" in {s["block_contents"][k] for k in hidden}] += 1
        if seed < 8:
            assert passable(s)["passable"], seed
    assert {(1, False), (1, True), (4, False), (6, False)} <= set(ways)
    assert any("hidden" in random_level(seed)["platform_kinds"] for seed in range(60))
