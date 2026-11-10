"""Question blocks, mushrooms and big Mario in Block SMB (env.py), as the game
has them."""

from retroagi.stages.block_smb.env import BIG_H, COIN_POINTS, HURT_FRAMES, MUSHROOM_POINTS, MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state

FLOOR = 208


def scene(contents, enemies=()):
    env = MarioScenarioEnv(world_width=1024)
    env.draws = False
    env.reset(scenario={
        "world_width": 1024,
        "mario": [100, FLOOR - 12],
        # a question block four tiles up, right above Mario; a brick beside it
        "platforms": [[0, FLOOR, 1024, 32], [96, FLOOR - 64, 16, 16], [112, FLOOR - 64, 16, 16]],
        "platform_kinds": ["ground", "question_block", "brick"],
        "block_contents": {1: contents},
        "enemies": [{"x": x, "y": FLOOR - 14, "game_rules": True} for x in enemies],
        "task": {"family": "game_level"},
    })
    return env


def jump(env, frames=40, direction=0):
    for k in range(frames):
        env.step({0: 5, 1: 2}[direction] if k < 20 else (1 if direction else 0))


def test_a_question_block_gives_its_coin_once():
    env = scene("coin")
    jump(env)
    assert env.score == COIN_POINTS and env.platforms[1]["used"]
    jump(env)
    assert env.score == COIN_POINTS


def test_a_mushroom_comes_out_moves_and_makes_mario_big():
    env = scene("mushroom")
    jump(env)
    assert len(env.items) == 1 and env.items[0]["vx"] > 0
    feet = env.mario["y"] + env.mario["h"]
    for _ in range(200):  # chase it right
        env.step(1)
        if env.mario["big"]:
            break
    assert env.mario["big"] and env.mario["h"] == BIG_H and env.score == MUSHROOM_POINTS
    assert abs(env.mario["y"] + env.mario["h"] - feet) <= 1 and not env.items


def test_big_mario_survives_a_hit_shrinking_and_breaks_bricks():
    env = scene("mushroom", enemies=[200])  # on screen, so awake
    env.mario.update(big=True, w=12, h=BIG_H, y=FLOOR - BIG_H, x=114.0)
    snapshot = snapshot_env_state(env)
    jump(env)  # under the brick
    assert env.platforms[2]["rect"].width == 0  # broken
    restore_env_state(env, snapshot)
    assert env.platforms[2]["rect"].width == 16 and env.mario["big"]
    for _ in range(400):
        _, _, terminated, _, info = env.step(0)
        if not env.mario["big"] or terminated:
            break
    assert not info["death"] and not env.mario["big"] and env.mario["safe"] == HURT_FRAMES - 1


def koopa_scene(enemies, mario=(100, FLOOR - 12)):
    env = MarioScenarioEnv(world_width=1024)
    env.draws = False
    env.reset(scenario={
        "world_width": 1024,
        "mario": list(mario),
        "platforms": [[0, FLOOR, 1024, 32]],
        "enemies": enemies,
        "task": {"family": "game_level"},
    })
    return env


def test_a_koopa_stomped_leaves_a_shell_that_a_touch_kicks_and_that_kills():
    # Mario dropping onto a Koopa.
    env = koopa_scene([{"x": 150, "y": FLOOR - 14, "game_rules": True, "kind": "koopa"}], mario=(146, 140))
    koopa = env.enemies[0]
    assert (koopa["w"], koopa["h"]) == (12, 12) and koopa["y"] + koopa["h"] + koopa["foot_offset"] == FLOOR
    # Stomped: a still shell, and points.
    for k in range(60):
        env.step(0)
        if koopa.get("shell"):
            break
    assert koopa["shell"] == "still" and env.score == 5 and not koopa["dead"]
    # Once he has landed, walk into it: kicked away, him unhurt.
    for _ in range(40):
        env.step(0)
    toward = 3 if koopa["x"] < env.mario["x"] else 1
    for _ in range(60):
        _, _, terminated, _, info = env.step(toward)
        if koopa["shell"] == "moving" or terminated:
            break
    assert koopa["shell"] == "moving" and not info["death"]
    # A moving shell kills the Goomba it meets.
    env = koopa_scene([{"x": 200, "y": FLOOR - 14, "game_rules": True, "kind": "koopa"},
                       {"x": 140, "y": FLOOR - 14, "game_rules": True}], mario=(20, FLOOR - 12))
    env.enemies[0].update(shell="moving", speed=3.0, direction=-1, awake=True)
    for _ in range(40):
        env.step(0)
    assert env.enemies[1]["dead"]


def test_a_moving_shell_hurts_mario():
    env = koopa_scene([{"x": 200, "y": FLOOR - 14, "game_rules": True, "kind": "koopa"}])
    env.enemies[0].update(shell="moving", speed=3.0, direction=-1, awake=True)
    for _ in range(60):
        _, _, terminated, _, info = env.step(0)
        if terminated:
            break
    assert info["death"]


def test_a_paratroopa_hops_as_the_games_and_a_stomp_takes_its_wings():
    env = koopa_scene([{"x": 200, "y": FLOOR - 14, "game_rules": True, "kind": "paratroopa"}], mario=(20, FLOOR - 12))
    para = env.enemies[0]
    tops, landings = [], []
    for k in range(120):
        env.step(0)
        tops.append(para["y"])
        if para["on_ground"]:
            landings.append(k)
    # 5-1's: about 44 pixels high, a hop every 55 frames or so.
    assert 40 <= max(tops) - min(tops) <= 48
    hops = [b - a for a, b in zip(landings, landings[1:]) if b - a > 1]
    assert hops and all(50 <= h <= 60 for h in hops)
    # Mario dropping on it: its wings are gone, it walks.
    env = koopa_scene([{"x": 150, "y": FLOOR - 14, "game_rules": True, "kind": "paratroopa"}], mario=(146, 100))
    para = env.enemies[0]
    for _ in range(80):
        env.step(0)
        if not para.get("wings"):
            break
    assert not para["wings"] and env.score == 5 and para.get("shell") is None


def cannon_scene(mario_x):
    env = MarioScenarioEnv(world_width=1024)
    env.draws = False
    env.reset(scenario={
        "world_width": 1024,
        "mario": [mario_x, FLOOR - 12],
        "platforms": [[0, FLOOR, 1024, 32], [160, FLOOR - 32, 16, 32]],
        "platform_kinds": ["ground", "cannon"],
        "task": {"family": "game_level"},
    })
    return env


def test_a_cannon_fires_bullet_bills_toward_mario_through_everything():
    from retroagi.stages.block_smb.env import BULLET_SPEED, CANNON_FRAMES

    env = cannon_scene(40)
    for _ in range(CANNON_FRAMES[0]):
        env.step(0)
    (bullet,) = env.enemies
    assert bullet["kind"] == "bullet" and bullet["direction"] == -1 and bullet["y"] == FLOOR - 32 + 6
    x = bullet["x"]
    env.step(0)
    assert bullet["x"] == x - BULLET_SPEED and not bullet["dead"]
    # Saved and restored with the cannon's turn: the same next shot.
    saved = snapshot_env_state(env)
    for _ in range(CANNON_FRAMES[1]):
        env.step(0)
    shots = len(env.enemies)
    restore_env_state(env, saved)
    for _ in range(CANNON_FRAMES[1]):
        env.step(0)
    assert len(env.enemies) == shots
    # Toward Mario on the other side.
    env = cannon_scene(200)
    for _ in range(CANNON_FRAMES[0]):
        env.step(0)
    assert env.enemies[0]["direction"] == 1


def test_a_piranha_plant_rises_as_the_games_and_stays_down_while_mario_is_near():
    from retroagi.stages.block_smb.piranha import EXPOSED_FRAMES, HIDDEN_FRAMES, RISE_FRAMES

    def plant_scene(mario_x):
        env = MarioScenarioEnv(world_width=1024)
        env.draws = False
        env.reset(scenario={
            "world_width": 1024,
            "mario": [mario_x, FLOOR - 12],
            "platforms": [[0, FLOOR, 1024, 32], [160, FLOOR - 48, 32, 48]],
            "platform_kinds": ["ground", "pipe"],
            "enemies": [{"kind": "piranha_plant", "x": 168, "pipe_top": FLOOR - 48}],
            "task": {"family": "game_level"},
        })
        return env

    env = plant_scene(20)
    plant = env.enemies[0]
    for _ in range(RISE_FRAMES):
        env.step(0)
    # Fully out: its 10 by 6 box 4 to 10 pixels above the pipe's top.
    assert (plant["w"], plant["h"]) == (10, 6) and plant["y"] + plant["h"] == FLOOR - 48 - 4
    for _ in range(EXPOSED_FRAMES + RISE_FRAMES + HIDDEN_FRAMES - 1):
        env.step(0)
    assert plant["exposed"] == 0
    # Mario beside its pipe: it stays down past its time.
    env = plant_scene(150)
    plant = env.enemies[0]
    plant["plant_tick"] = 2 * RISE_FRAMES + EXPOSED_FRAMES  # just gone down
    for _ in range(HIDDEN_FRAMES + 60):
        env.step(0)
    assert plant["exposed"] == 0
