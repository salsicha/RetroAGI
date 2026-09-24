"""Generated failure situations must be reachable and enter normal learning."""

import json
from pathlib import Path

import pytest
import torch

from retroagi.stages.block_smb.demonstrations import (
    collect_demonstrations,
    with_robust_demonstrations,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import LOCAL_TRAVERSAL_FAMILIES, local_objective
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    block_smb_monte_carlo_family_specs,
    sample_block_smb_monte_carlo_scenario,
    validate_block_smb_monte_carlo_oracle,
)
from retroagi.stages.block_smb.policy_recovery import RECOVERY_FAMILIES, repair_policy_actions
from retroagi.stages.block_smb.transfer_failure_families import TRANSFER_FAILURE_FAMILIES
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_tall_pipe_traversal import PhaseIntentPolicy, rollout


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


def test_new_families_append_without_renumbering_cached_family_labels():
    original = (
        "flat_run single_gap stair_climb platform_chain moving_bridge enemy_hop enemy_patrol "
        "enemy_gap enemy_stomp retreat_recovery wait_timing chained_obstacles chained_enemy_gauntlet "
        "full_smb_opening_proxy mixed_section tall_pipe_jump pipe_mount pit_leap stomp_mount "
        "stomp_recovery platform_hop bridge_wait bridge_mount bridge_dismount"
    ).split()
    assert BLOCK_SMB_MC_FAMILIES[:24] == tuple(original)
    assert BLOCK_SMB_MC_FAMILIES[24:] == TRANSFER_FAILURE_FAMILIES
    assert set(TRANSFER_FAILURE_FAMILIES) <= LOCAL_TRAVERSAL_FAMILIES & RECOVERY_FAMILIES
    config = json.loads(Path("scripts/configs/block_smb_full_volume_revision2.json").read_text())
    assert all(config["monte_carlo_family_weights"][f] > 0 for f in TRANSFER_FAILURE_FAMILIES)


@pytest.mark.parametrize("family", TRANSFER_FAILURE_FAMILIES)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_new_family_routes_complete_in_training_collector(family, difficulty):
    item = sample(family, difficulty)
    assert item.parameters["family_revision"] == (3 if family == "piranha_avoidance" else 1)
    assert item.reachability["reachable"]
    trajectory = rollout(item, PhaseIntentPolicy(), steps=320, use_oracle_actions=True)
    assert trajectory.success
    assert any(t.info.get("primitive_valid_hold_frames") for t in trajectory.transitions)
    assert len(trajectory.transitions) == item.reachability["completion_steps"]


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


@pytest.mark.parametrize("family", TRANSFER_FAILURE_FAMILIES)
def test_actual_policy_failures_supply_complete_recovery_suffixes(family):
    item = sample(family)
    actions = [1] * 180
    top_arrival = None
    if family == "stair_gap":
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=item.scenario)
            for frame, action in enumerate(item.oracle["actions"]):
                if env.mario["on_ground"] and env.mario.get("_platform") is env.platforms[3]:
                    top_arrival = frame
                    actions = list(item.oracle["actions"][:frame]) + [1] * 100
                    break
                env.step(action)
            assert top_arrival is not None
        finally:
            env.close()
    assert not validate_block_smb_monte_carlo_oracle(item.scenario, actions)["reachable"]
    repairs = repair_policy_actions(item.scenario, actions)
    assert 0 < len(repairs) <= 3
    for repair in repairs:
        start = repair["supervision_start_frame"]
        assert repair["actions"][:start] == actions[:start]
        assert validate_block_smb_monte_carlo_oracle(item.scenario, repair["actions"])["reachable"]
    if top_arrival is not None:
        assert any(r["supervision_start_frame"] >= top_arrival for r in repairs)


@pytest.mark.parametrize("family", TRANSFER_FAILURE_FAMILIES)
def test_new_training_demonstrations_keep_jump_credit_and_mask_forced_release(family):
    item = sample(family, split="train")
    family_id = BLOCK_SMB_MC_FAMILIES.index(family)
    cases = with_robust_demonstrations([(family_id, item)], 915302)
    data = collect_demonstrations(
        cases, tiny_config(walk_duration_primitives=False), StaticBlockVision
    )
    assert (data.actor_mask & (data.action == 2)).any()
    walked = (data.motor_action == 1) | (data.motor_action == 3)
    jumped = torch.isin(data.motor_action, torch.tensor([2, 4, 5]))
    release = walked & jumped.roll(1)
    release[0] = False
    suppressed = release.roll(1) & walked
    suppressed[0] = False
    assert release.any()
    assert not data.actor_mask[release | suppressed].any()
