"""Lakitu and his Spinies in Block SMB (sim/env.py; measured in World 8-2's
memory): how Lakitu follows Mario and throws eggs, how an egg falls and
hatches, that a Spiny cannot be stomped and Lakitu can, how the state and
the planner show them, and the generator's piece of him."""

from retroagi.domains.smb.planner import enemy_paths
from retroagi.domains.smb.sim import env as block
from retroagi.domains.smb.sim.levels import FLOOR, SCENE_KINDS, TILE, random_level, skill_scene
from retroagi.domains.smb.sim.world import SimSource, SimWorld


def scene(lakitu_x=150.0, until=1e9, mario_x=100.0):
    return {
        "world_width": 2000, "scroll_x": 115, "mario": [mario_x, FLOOR - 12],
        "platforms": [[0, FLOOR, 2000, 32]], "platform_kinds": ["ground"],
        "enemies": [{"x": lakitu_x, "y": 0.0, "kind": "lakitu", "game_rules": True, "until": until}],
        "task": {"family": "game_level"},
    }


def lakitu_of(world):
    return next(e for e in world.env.enemies if e.get("kind") == "lakitu")


def apart(world):
    lakitu, mario = lakitu_of(world), world.env.mario
    return (lakitu["x"] + lakitu["w"] / 2) - (mario["x"] + mario["w"] / 2)


def test_lakitu_swings_to_and_fro_over_a_still_mario_fastest_over_him():
    world = SimWorld(scene())
    world.reset()
    lakitu = lakitu_of(world)
    seen = []
    for _ in range(1200):
        world.step(0)
        seen.append((apart(world), lakitu["lakitu_speed"]))
    assert lakitu["y"] == block.LAKITU_Y + block.LAKITU_BOX[1]
    later = seen[300:]
    assert -block.LAKITU_REACH - 3 <= min(d for d, _ in later) and max(d for d, _ in later) <= block.LAKITU_REACH + 3
    assert max(s for d, s in later if abs(d) < 4) == 20 and min(s for d, s in later if abs(d) > 56) <= 6


def test_lakitu_is_faster_while_mario_scrolls_the_screen():
    world = SimWorld(scene())
    world.reset()
    env, lakitu = world.env, lakitu_of(world)
    lakitu["x"] = env.mario["x"] + env.mario["w"] / 2 - lakitu["w"] / 2  # right over him
    for scrolled, speed, base in ((0.0, 0, 20), (1.0, 16, 47), (2.0, 32, 63)):
        env.scrolled, env.motion.x_speed = scrolled, speed
        before = lakitu["x"]
        env._update_lakitu(lakitu)
        assert lakitu["lakitu_speed"] == base
        lakitu["x"] = before


def test_eggs_come_every_128_frames_rise_fall_and_hatch_into_spinies_walking_at_mario():
    world = SimWorld(scene(lakitu_x=200.0))
    world.reset()
    eggs, hatched = [], []
    for frame in range(300):
        world.step(0)
        for e in world.env.enemies:
            if e.get("kind") == "spiny" and id(e) not in [id(x) for _, x in eggs]:
                eggs.append((frame, e))
    assert [f for f, _ in eggs][:2] == [0, 128]
    first = eggs[0][1]
    assert not first.get("egg") and first["on_ground"] and first["speed"] == block.SPINY_SPEED
    assert first["y"] + first["h"] + first["foot_offset"] == FLOOR
    mario_middle = world.env.mario["x"] + world.env.mario["w"] / 2
    assert first["direction"] == (1 if mario_middle > first["x"] + first["w"] / 2 else -1)


def test_a_spiny_cannot_be_stomped_and_lakitu_can():
    world = SimWorld(scene(lakitu_x=400.0))
    world.reset()
    env = world.env
    spiny = {"kind": "spiny", "x": env.mario["x"], "y": env.mario["y"] + env.mario["h"] + 2.0, "vx": 0.0, "vy": 0.0,
             "w": 10, "h": 6, "foot_offset": 4, "speed": 0.0, "direction": 1, "patrol_min": -1e9, "patrol_max": 1e9,
             "edge_aware": False, "on_ground": True, "dead": False, "stompable": False, "game_rules": True, "awake": True}
    env.enemies.append(spiny)
    env.mario.update(y=spiny["y"] - env.mario["h"] - 4, on_ground=False)
    env.motion.y_speed = 3
    assert world.step(0).died
    world = SimWorld(scene(lakitu_x=400.0))
    world.reset()
    lakitu, mario = lakitu_of(world), world.env.mario
    lakitu["throw_timer"] = 1000  # (no egg thrown into Mario on the way)
    mario.update(x=lakitu["x"], y=lakitu["y"] - mario["h"] - 2, on_ground=False)
    world.env.motion.y_speed = 3
    step = world.step(0)
    assert not step.died and lakitu["dead"]


def test_lakitu_leaves_once_mario_is_past_his_stretch():
    world = SimWorld(scene(until=120.0, mario_x=130.0))
    world.reset()
    lakitu = lakitu_of(world)
    xs = []
    for _ in range(40):
        world.step(0)
        xs.append(lakitu["x"])
    assert all(b - a == -block.LAKITU_LEAVE for a, b in zip(xs, xs[1:]))
    assert not [e for e in world.env.enemies if e.get("kind") == "spiny"]


def test_the_state_shows_lakitu_flying_and_an_egg_which_the_planner_drops_where_block_smb_does():
    world = SimWorld(scene(lakitu_x=200.0))
    world.reset()
    world.step(0)
    state = SimSource().read(world)
    kinds = {e.kind: e for e in state.enemies}
    assert kinds["flier"].stompable and not kinds["egg"].stompable
    egg = kinds["egg"]
    enemy_paths([egg], state.solids, 90)
    real = next(e for e in world.env.enemies if e.get("kind") == "spiny")
    for frame in range(1, 91):
        world.step(0)
        assert abs(egg.path[frame][0] - real["x"]) <= 0.5 and abs(egg.path[frame][1] - real["y"]) <= 0.5, frame


def test_random_levels_and_skill_scenes_have_lakitu():
    assert "lakitu" in SCENE_KINDS
    assert any(e.get("kind") == "lakitu" for e in skill_scene(1, "lakitu")["enemies"])
    assert sum(e.get("kind") == "lakitu" for seed in range(40) for e in random_level(seed)["enemies"]) > 0
