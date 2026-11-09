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
