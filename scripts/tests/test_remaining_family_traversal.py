"""Physical credit, counterfactual labels, and completion for the remaining curriculum."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
)
from retroagi.stages.block_smb.pipe_traversal import training_rollout_steps
from retroagi.stages.block_smb.train import (
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
    train_block_smb_epoch,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_tall_pipe_traversal import PhaseIntentPolicy, rollout

FAMILIES = "wait_timing pit_leap pipe_mount enemy_hop stair_climb single_gap retreat_recovery platform_chain moving_bridge mixed_section full_smb_opening_proxy enemy_patrol enemy_gap chained_obstacles chained_enemy_gauntlet".split()


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


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_revised_oracles_complete_through_real_controller(family, difficulty):
    item = sample(family, difficulty)
    trajectory = rollout(
        item,
        PhaseIntentPolicy(),
        steps=training_rollout_steps(60, item.scenario),
        use_oracle_actions=True,
    )
    assert item.parameters["family_revision"] == 2
    assert trajectory.success
    assert len(trajectory.transitions) == item.reachability["completion_steps"]
    assert not any(t.info.get("jump_overreach") for t in trajectory.transitions)
    if family != "retreat_recovery":
        assert any(t.info.get("primitive_valid_hold_frames") for t in trajectory.transitions)


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
            rewards[action] = sum(env.step(action)[1] for _ in range(8))
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
        sample("mixed_section", seed=seed).parameters["composition"] for seed in range(6)
    }
    assert compositions == {"enemy_gap_pipe", "enemy_two_pipes"}


@pytest.mark.parametrize("family", ("wait_timing", "moving_bridge"))
def test_bridge_families_leave_wait_choice_to_policy(family):
    item = sample(family)
    assert item.scenario["require_bridge_before_goal"]
    assert "a_level_action" not in item.parameters


def test_partial_traversal_metrics_expose_evaluation_timeout():
    item = sample("chained_obstacles")
    samples = replace(
        sample_block_smb_monte_carlo_split(split="validation", seed=2, sample_count=0),
        samples=(item,),
    )
    with patch(
        "retroagi.stages.block_smb.train.sample_block_smb_monte_carlo_parameter_sweep",
        return_value=samples,
    ):
        result = evaluate_block_smb_monte_carlo(
            PhaseIntentPolicy(),
            tiny_config(evaluation_episodes=1, evaluation_max_steps=1),
            split="validation",
            sample_count=1,
            stratified_repeats_per_difficulty=3,
            device=torch.device("cpu"),
            vision_factory=StaticBlockVision,
        )
    metrics = result["families"]["chained_obstacles"]["traversal_metrics"]
    assert metrics["episodes"] == metrics["timeouts"] == 1
    assert metrics["finishes_after_local_clear"] == 0


def test_real_optimizer_consumes_local_safe_sets_and_long_composites():
    config = tiny_config(rollout_steps=60, use_oracle_actions=True, generated_scenarios=0)
    model = make_block_smb_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    items = [sample("pipe_mount"), sample("enemy_gap"), sample("chained_obstacles")]
    metrics, _ = train_block_smb_epoch(
        model,
        optimizer,
        [(s.scenario_id, s.scenario) for s in items],
        config,
        epoch=0,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
    )
    assert metrics["training_rollout_steps_max"] >= 240
    assert metrics["loss_primitive_outcome"] > 0
    assert metrics["optimizer_updates"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
