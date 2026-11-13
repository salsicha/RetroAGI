"""Flying Koopa Paratroopas in Block SMB (sim/env.py; measured in Worlds 1-3
and 7-3's memory): the red one's swing up and down, the green one's across
with its bob, that a stomp leaves a walking Koopa of its colour, how the
state shows them, and where the generator puts them."""

from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, random_level
from retroagi.domains.smb.sim.world import SimSource, SimWorld


def scene(kind, phase=0):
    return {
        "world_width": 2000, "scroll_x": 115, "mario": [40.0, FLOOR - 12],
        "platforms": [[0, FLOOR, 2000, 32]], "platform_kinds": ["ground"],
        "enemies": [{"x": 200.0, "y": float(FLOOR - 9 * 16 - 14), "kind": kind, "game_rules": True, "phase": phase}],
        "task": {"family": "game_level"},
    }


def swing(kind, frames=800):
    world = SimWorld(scene(kind))
    world.reset()
    koopa = world.env.enemies[0]
    home = koopa["home"]
    places = []
    for _ in range(frames):
        world.step(0)
        places.append((koopa["x"] - home[0], koopa["y"] - home[1]))
    return world, koopa, places


def test_a_red_paratroopa_swings_down_and_back_every_400_frames():
    _, _, places = swing("red_paratroopa")
    ys = [y for _, y in places]
    assert {x for x, _ in places} == {0.0}
    assert min(ys) >= 0 and abs(max(ys) - block.RED_FLIGHT[0]) < 0.1
    lowest = max(range(400), key=ys.__getitem__)  # (the first swing's)
    assert abs(lowest + 1 - block.RED_FLIGHT[1] / 2) <= 1


def test_a_green_flyer_swings_left_and_back_bobbing():
    _, _, places = swing("green_flyer")
    xs, ys = [x for x, _ in places], [y for _, y in places]
    assert abs(min(xs) + block.GREEN_FLIGHT[0]) < 0.1 and max(xs) <= 0
    assert abs(min(ys) + block.GREEN_BOB[0]) < 0.1 and abs(max(ys) - block.GREEN_BOB[0]) < 0.1


def test_the_state_shows_a_flyer_moving_and_a_stomp_leaves_a_walking_koopa_of_its_colour():
    for kind, walker in (("red_paratroopa", "pacer"), ("green_flyer", "walker")):
        world = SimWorld(scene(kind))
        world.reset()
        for _ in range(50):
            world.step(0)
        enemy = SimSource().read(world).enemies[0]
        koopa = world.env.enemies[0]
        assert enemy.kind == "flier" and enemy.stompable
        assert (enemy.vx, enemy.vy) == (koopa["speed"] * koopa["direction"], koopa["vy"])
        koopa.update(wings=False, vy=0.0)  # (what a stomp does: env._touch_koopa)
        for _ in range(200):
            world.step(0)
        assert koopa["on_ground"] and koopa["y"] + koopa["h"] + koopa["foot_offset"] == FLOOR
        assert SimSource().read(world).enemies[0].kind == walker


def test_random_levels_have_flying_paratroopas():
    kinds = [e.get("kind") for seed in range(60) for e in random_level(seed)["enemies"]]
    assert kinds.count("red_paratroopa") > 0 and kinds.count("green_flyer") > 0
