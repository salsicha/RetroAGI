"""Canonical policy inputs and NES physics that both games share."""


def test_nes_snapshot_restores_fractional_physics():
    from retroagi.stages.block_smb.env import MarioScenarioEnv
    from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state

    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={"mario": [43, 196], "platforms": [[0, 208, 512, 32]], "world_width": 512}
        )
        for _ in range(11):
            env.step(1)
        saved = snapshot_env_state(env)
        for _ in range(8):
            env.step(2)
        expected = (dict(env.motion.__dict__), env.mario["x"], env.mario["y"])
        restore_env_state(env, saved)
        for _ in range(8):
            env.step(2)
        assert expected == (dict(env.motion.__dict__), env.mario["x"], env.mario["y"])
    finally:
        env.close()


def test_landing_clears_fractional_vertical_force():
    from retroagi.core.smb_physics import NESPlayerMotion

    motion = NESPlayerMotion(y_speed=3, y_force=224, y_fraction=64)
    motion.vertical_contact()
    assert motion.y_speed == motion.y_force == 0
    assert motion.y_fraction == 64
    motion.bounce()
    assert motion.y_speed == -4


def test_nes_enemy_damage_body_is_distinct_from_floor_probe():
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "mario": [43, 196],
                "platforms": [[0, 208, 256, 32]],
                "enemies": [[120, 194, 90, 200, 0.5, -1]],
            }
        )
        for _ in range(4):
            env.step(0)
        enemy = env.enemies[0]
        assert (enemy["w"], enemy["h"]) == (10, 6)
        assert enemy["y"] + enemy["h"] == 204
        assert enemy["on_ground"]
    finally:
        env.close()
