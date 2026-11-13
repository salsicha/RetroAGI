"""Red Koopa Troopas in Block SMB (sim/env.py; measured in Worlds 1-3 and
3-3's memory: half a pixel a frame, turned back where the floor under its
middle would end): how one paces a ledge, that its shell does not, how the
state and the planner show it, and the generator's share of them."""

from retroagi.domains.smb.planner import Enemy, Solid, enemy_paths
from retroagi.domains.smb.sim.levels import FLOOR, TILE, random_level
from retroagi.domains.smb.sim.world import SimSource, SimWorld

LEDGE = (160, FLOOR - 4 * TILE, 160 + 5 * TILE)  # a brick row's left end, top, right end


def scene(kind="red_koopa", x=200.0):
    """A floor and a brick row 5 tiles long, 4 up, a Koopa on the row."""
    left, top, right = LEDGE
    platforms = [[0, FLOOR, 2000, 32]] + [[bx, top, TILE, TILE] for bx in range(left, right, TILE)]
    return {
        "world_width": 2000, "scroll_x": 115, "mario": [40.0, FLOOR - 12],
        "platforms": platforms, "platform_kinds": ["ground"] + ["brick"] * (len(platforms) - 1),
        "enemies": [{"x": x, "y": float(top - 14), "kind": kind, "game_rules": True}],
        "task": {"family": "game_level"},
    }


def koopa_of(world):
    return next(e for e in world.env.enemies if e.get("kind") == "koopa")


def test_a_red_koopa_paces_its_ledge_turning_where_its_middle_would_leave_it():
    world = SimWorld(scene())
    world.reset()
    koopa = koopa_of(world)
    middles = []
    for _ in range(600):
        world.step(0)
        middles.append(koopa["x"] + koopa["w"] / 2)
    left, top, right = LEDGE
    assert koopa["y"] + koopa["h"] + koopa["foot_offset"] == top  # still on the row
    assert left <= min(middles) < left + 1 and right - 1 <= max(middles) < right
    assert abs(middles[1] - middles[0]) == 0.5


def test_a_green_koopa_walks_off_the_ledge_and_a_red_koopas_shell_falls_off_it():
    world = SimWorld(scene("koopa"))
    world.reset()
    for _ in range(200):
        world.step(0)
    assert koopa_of(world)["y"] + koopa_of(world)["h"] + koopa_of(world)["foot_offset"] == FLOOR
    world = SimWorld(scene())
    world.reset()
    koopa = koopa_of(world)
    koopa.update(shell="moving", speed=3.0, direction=-1, kick_grace=0)
    for _ in range(60):
        world.step(0)
    assert koopa["y"] + koopa["h"] + koopa["foot_offset"] == FLOOR


def test_the_state_calls_a_walking_red_koopa_a_pacer_and_its_shell_a_walker():
    world = SimWorld(scene())
    world.reset()
    world.step(0)
    assert [e.kind for e in SimSource().read(world).enemies] == ["pacer"]
    koopa_of(world).update(shell="still", speed=0.0)
    world.step(0)
    assert [e.kind for e in SimSource().read(world).enemies] == ["walker"]


def test_the_planner_turns_a_pacer_where_block_smb_does():
    world = SimWorld(scene(x=170.0))
    world.reset()
    world.step(0)
    state = SimSource().read(world)
    enemy = state.enemies[0]
    enemy_paths(state.enemies, state.solids, 120)
    koopa = koopa_of(world)
    for frame in range(1, 121):
        world.step(0)
        box = enemy.path[frame]
        assert abs(box[0] - koopa["x"]) <= 0.5, frame
    walker = Enemy(enemy.box, enemy.vx, 0.0, ("e", 0), kind="walker")
    enemy_paths([walker], state.solids, 120)
    assert walker.path[120][3] > enemy.path[120][3]  # a walker falls off the row


def test_random_levels_have_red_koopas():
    kinds = [e.get("kind") for seed in range(40) for e in random_level(seed)["enemies"]]
    assert 0 < kinds.count("red_koopa") < kinds.count("koopa") + kinds.count("red_koopa")
