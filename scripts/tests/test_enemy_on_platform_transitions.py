"""Enemy clearance must survive landing release and a subsequent walk-off."""

import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def platform_scenario(*, hard=False):
    return dict(
        world_width=352,
        mario=[36, 204] if hard else [33, 204],
        platforms=[[0, 220, 352, 20], [113, 173, 80, 47] if hard else [119, 191, 104, 29]],
        enemies=[[147, 159, 135, 159, 0.6, -1] if hard else [169, 177, 157, 181, 0.2, -1]],
        goal=[324, 204, 16, 16],
        goal_requires_support=True,
        task={"family": "enemy_on_platform"},
    )


def test_mount_certificate_hands_the_first_grounded_frame_to_the_next_enemy_jump():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=platform_scenario(hard=True))
        # NES walking accelerates slowly; the 47 px mount needs a run-up.
        for _ in range(35):
            env.step(1)
        saved = snapshot_env_state(env)
        valid = safe_jump_holds(env, local_objective(env), 1)
        assert 14 in valid and 16 in valid
        assert snapshot_env_state(env) == saved
        # The 16-frame hold is released mid-air and lands beside the enemy on
        # the 14th frame after release.
        for action in [2] * 16 + [1] * 14:
            _, _, _, _, info = env.step(action)
        assert env.mario["on_ground"] and not info["death"]
        landed = snapshot_env_state(env)
        # The button is already up, so the controller hands back the first
        # grounded frame and a jump from there survives ...
        assert safe_jump_holds(env, local_objective(env), 1)
        # ... whereas the former two forced release frames walk into the enemy.
        restore_env_state(env, landed)
        env.step(1)
        _, _, _, _, info = env.step(1)
        assert info["death"]
    finally:
        env.close()
