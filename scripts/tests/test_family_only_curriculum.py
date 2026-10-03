"""Family-only evaluation must preserve coverage and the real task contract."""

import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import (
    local_objective,
    safe_jump_holds,
    terrain_oracle,
)
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_leftward_mount_has_geometry_and_a_successful_left_jump_route():
    scene = {
        "mario": [205, 200],
        "platforms": [[150, 220, 106, 20], [70, 170, 90, 10]],
        "goal": [85, 150, 16, 20],
    }
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scene)
        target = local_objective(env)
        assert target.kind == "mount" and target.direction == -1
        assert safe_jump_holds(env, target, -1)
        actions = terrain_oracle(scene)
        assert 4 in actions
        for action in actions:
            _, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        assert env._goal_credited
    finally:
        env.close()


# Eighteen moving-bridge layouts, each route-checked, take over a minute on a busy machine.
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "family,expected",
    [("retreat_recovery", {"flat", "gap", "mount"}), ("moving_bridge", {"wide", "narrow"})],
)
def test_expanded_families_keep_all_variants_physically_solvable(family, expected):
    observed = set()
    for i in range(18):
        sample = sample_block_smb_monte_carlo_scenario(
            split="train",
            seed=20260908,
            sample_index=i,
            family=family,
            difficulty=("easy", "medium", "hard")[i % 3],
        )
        assert sample.reachability["reachable"]
        observed.add(sample.parameters["variant"])
    assert observed == expected


def complete_evidence():
    return {
        "success_rate": 1.0,
        "coverage": {"missing_families": []},
        "families": {f: {"success_rate": 1.0} for f in BLOCK_SMB_MC_FAMILIES},
        "difficulty_bins": {
            f"{f}:{d}": {"success_rate": 1.0}
            for f in BLOCK_SMB_MC_FAMILIES
            for d in BLOCK_SMB_MC_DIFFICULTY_BINS
        },
    }


def test_unlabelled_overhead_obstacles_leave_a_safe_floor_route():
    scene = {
        "mario": [20, 200],
        "platforms": [[0, 220, 384, 20], [90, 176, 178, 10]],
        "enemies": [[106, 162, 106, 106, 0]],
        "goal": [350, 200, 16, 20],
        "world_width": 384,
    }
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scene)
        assert local_objective(env).kind == "finish"
        scene["goal"] = [220, 156, 16, 20]
        env.reset(scenario=scene)
        assert local_objective(env).kind == "mount"
    finally:
        env.close()


def test_finish_overshoot_does_not_request_obstacles_beyond_the_goal():
    scene = {
        "mario": [20, 100],
        "platforms": [[0, 120, 40, 10], [70, 160, 40, 10], [140, 120, 40, 10], [200, 220, 56, 20]],
        "goal": [220, 200, 16, 20],
    }
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scene)
        env.mario.update(x=242.0, y=204.0, on_ground=True, _platform=env.platforms[-1])
        target = local_objective(env)
        assert target.kind == "retreat" and target.direction == -1
    finally:
        env.close()
