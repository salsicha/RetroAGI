"""Physical credit, counterfactual labels, and completion for the remaining curriculum."""

import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

# The composed families (tactic_families) are checked through the layered
# agent's executor instead (test_tactic_families).
FAMILIES = "wait_timing pit_leap pipe_mount enemy_hop stair_climb single_gap retreat_recovery platform_chain moving_bridge enemy_patrol enemy_gap".split()


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(family, difficulty="hard", seed=2):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=seed, sample_index=0, family=family, difficulty=difficulty
    )


@pytest.mark.parametrize("family", ("pit_leap", "pipe_mount"))
def test_counterfactual_labels_preserve_state_and_require_real_landing(family):
    item = sample(family)
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=item.scenario)
        before = snapshot_env_state(env)
        valid = safe_jump_holds(env, local_objective(env), 1)
        assert snapshot_env_state(env) == before
        assert valid and len(valid) < 16
        assert not env._goal_credited and not env._attempt_failed
        for hold in (min(valid), max(valid)):
            env.reset(scenario=item.scenario)
            for frame in range(64):
                _, _, done, _, info = env.step(2 if frame < hold else 1)
                if done:
                    break
            assert env._goal_credited
            assert env.mario["on_ground"]
            assert env.mario["_platform"]["rect"].top == env.goal.bottom
        env.reset(scenario=item.scenario)
        for frame in range(64):
            _, _, done, _, info = env.step(2 if frame == 0 else 1)
            if done:
                break
        assert done and not env._goal_credited
    finally:
        env.close()


def test_airborne_contact_with_pipe_goal_never_counts_as_a_mount():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample("pipe_mount").scenario)
        env.mario.update(x=float(env.goal.x), y=float(env.goal.top), vy=-2.0, on_ground=False)
        _, _, done, _, info = env.step(0)
        assert not done and not env._goal_credited and not env.mario["on_ground"]
    finally:
        env.close()


def test_retreat_rewards_approaching_left_goal():
    env = MarioScenarioEnv()
    try:
        item = sample("retreat_recovery")
        rewards = {}
        for action in (1, 3):
            env.reset(scenario=item.scenario)
            rewards[action] = sum(env.step(action)[1] for _ in range(24))
        assert rewards[3] > 0 > rewards[1]
    finally:
        env.close()


def test_stair_metadata_matches_actual_geometry_and_tiers_vary():
    heights = []
    for difficulty in ("easy", "medium", "hard"):
        item = sample("stair_climb", difficulty)
        platforms = item.scenario["platforms"]
        step_height = item.parameters["step_height"]
        assert [220 - p[1] for p in platforms] == [0, step_height, 2 * step_height, 3 * step_height]
        heights.append(step_height)
    assert heights == sorted(set(heights))


def test_platform_chain_varies_terrain_and_keeps_wide_far_shore():
    terrains = set()
    for difficulty in ("easy", "medium", "hard"):
        item = sample("platform_chain", difficulty)
        terrain = tuple(map(tuple, item.scenario["platforms"]))
        terrains.add(terrain)
        assert terrain[-1][2] >= 56
        assert item.scenario["goal"][0] >= terrain[-1][0]
    assert len(terrains) == 3


def test_mixed_sections_sample_distinct_compositions():
    compositions = {
        tuple(sample("mixed_section", seed=seed).parameters["sections"]) for seed in range(6)
    }
    assert len(compositions) > 1


@pytest.mark.parametrize("family", ("wait_timing", "moving_bridge"))
def test_bridge_families_leave_wait_choice_to_policy(family):
    item = sample(family)
    assert item.scenario["require_bridge_before_goal"]
    assert "a_level_action" not in item.parameters
