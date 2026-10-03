"""Generated failure situations must be reachable and enter normal learning."""

import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import local_objective
from retroagi.stages.block_smb.monte_carlo import (
    block_smb_monte_carlo_family_specs,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.transfer_failure_families import TRANSFER_FAILURE_FAMILIES


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(family, difficulty="hard", *, split="validation", seed=915301):
    return sample_block_smb_monte_carlo_scenario(
        family=family, difficulty=difficulty, split=split, seed=seed, sample_index=0
    )


@pytest.mark.parametrize("family", TRANSFER_FAILURE_FAMILIES)
def test_replay_is_deterministic_and_splits_have_distinct_layouts(family):
    a = sample(family, split="train")
    assert a.to_dict() == sample(family, split="train").to_dict()
    b = sample(family, split="test")
    assert a.sample_seed != b.sample_seed
    assert a.parameters != b.parameters or a.scenario["mario"] != b.scenario["mario"]
    assert a.scenario_id != b.scenario_id


def test_stair_gap_is_a_real_gap_after_the_highest_step():
    for difficulty in ("easy", "medium", "hard"):
        item = sample("stair_gap", difficulty)
        platforms = item.scenario["platforms"]
        takeoff, landing = platforms[3:5]
        assert takeoff[1] == landing[1] == 148
        assert landing[0] - takeoff[0] - takeoff[2] == item.parameters["gap_width"]
        assert not any(p[0] <= takeoff[0] + takeoff[2] < p[0] + p[2] for p in platforms)
        assert (
            min(p[2] for p in platforms)
            >= block_smb_monte_carlo_family_specs()["stair_gap"].constraints[
                "minimum_landing_width"
            ]
        )


def test_landing_enemy_starts_airborne_and_needs_a_new_grounded_decision():
    item = sample("landing_enemy")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=item.scenario)
        assert not env.mario["on_ground"]
        assert env.enemies[0]["direction"] == -1
        for frame, action in enumerate(item.oracle["actions"]):
            was_grounded = env.mario["on_ground"]
            if action == 2:
                assert frame > 0 and was_grounded
                assert local_objective(env).kind == "enemy"
                break
            env.step(action)
        else:
            pytest.fail("Landing recovery route never jumped")
    finally:
        env.close()


def test_platform_enemy_patrol_stays_on_the_raised_surface():
    item = sample("enemy_on_platform")
    left, top, width, _ = item.scenario["platforms"][1]
    x, y, low, high, speed, direction = item.scenario["enemies"][0]
    assert y + 14 == top
    assert left < low <= x <= high < left + width - 14
    assert speed > 0 and direction == -1
