"""Family-only evaluation must preserve coverage and the real task contract."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.bridge_traversal import bridge_phase
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import (
    local_objective,
    safe_jump_holds,
    terrain_oracle,
)
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    evaluate_block_smb_monte_carlo_gates,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.pipe_traversal import training_rollout_steps
from retroagi.stages.block_smb.train import (
    collect_trajectory,
    evaluate_block_smb,
    make_block_smb_model,
)
from retroagi.stages.full_smb.transfer import block_smb_checkpoint_transfer_source_gate
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_unlabelled_gap_receives_local_goal_without_monte_carlo_provenance():
    scene = {
        "mario": [20, 200],
        "platforms": [[0, 220, 100, 20], [150, 220, 106, 20]],
        "goal": [220, 200, 16, 20],
    }
    config = tiny_config()
    stage = BlockSMBStage(env=MarioScenarioEnv(), scenario=scene, vision=StaticBlockVision())
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                make_block_smb_model(config).eval(),
                stage,
                "external-gap",
                rollout_steps=2,
                seed=1,
                deterministic=True,
                device=torch.device("cpu"),
                skill_goal_conditioning=True,
            )
        assert all(t.info["skill_phase"] == "gap" for t in trajectory.transitions)
        assert "metadata" not in scene
    finally:
        stage.env.close()


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


def test_approach_does_not_commit_to_unsafe_bridge_boarding():
    scene = {
        "task": {"family": "moving_bridge"},
        "mario": [20, 200],
        "platforms": [
            [0, 220, 90, 20],
            {"x": 106, "y": 220, "w": 50, "h": 20, "moving": [96, 146, 0.5]},
            [172, 220, 84, 20],
        ],
        "goal": [230, 200, 16, 20],
    }
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scene)
        assert training_rollout_steps(60, scene) >= 240
        assert env._require_bridge_before_goal
        assert bridge_phase(env, False) == "approach"
        for _ in range(20):
            env.step(1)
        assert bridge_phase(env, False) == "wait"
        assert not env._bridge_boarded
    finally:
        env.close()


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


def test_gate_rejects_missing_coverage_or_one_failed_difficulty_despite_high_average():
    evidence = complete_evidence()

    def gate(e):
        return evaluate_block_smb_monte_carlo_gates(
            e, pass_rate_gate=0.95, family_pass_rate_gate=0.9
        )["gate_met"]

    assert gate(evidence)
    failed = deepcopy(evidence)
    failed["difficulty_bins"]["retreat_recovery:hard"]["success_rate"] = 0.8
    assert not gate(failed)
    missing = deepcopy(evidence)
    del missing["families"]["retreat_recovery"]
    assert not gate(missing)
    empty = deepcopy(evidence)
    empty["difficulty_bins"] = {}
    assert not gate(empty)


def test_family_only_summary_uses_validation_and_never_passes_without_evidence():
    config = tiny_config(fixed_scenarios=(), monte_carlo_validation_samples=630)
    result = {
        "success_rate": 0.99,
        "mean_return": 50,
        "gates": {"gate_met": True},
        "action_counts": {"0": 0, "1": 100, "2": 40},
        "action_collapse": {"all_noop": False},
    }
    with patch(
        "retroagi.stages.block_smb.train.evaluate_block_smb_monte_carlo", return_value=result
    ):
        evaluated = evaluate_block_smb(
            make_block_smb_model(config),
            config,
            device=torch.device("cpu"),
            vision_factory=StaticBlockVision,
        )
    assert evaluated["evaluation_suite"] == "families"
    assert evaluated["success_rate"] == 0.99 and evaluated["success_thresholds_met"]
    assert evaluated["fixed_scenarios"] == {}
    empty = evaluate_block_smb(
        make_block_smb_model(config),
        replace(config, monte_carlo_validation_samples=0),
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
    )
    assert not empty["success_thresholds_met"]


def test_family_only_transfer_requires_actual_validation_actions_and_gate():
    checkpoint = {
        "config": {"fixed_scenarios": [], "monte_carlo_validation_samples": 630},
        "metrics": {
            "eval_monte_carlo_validation_gate_met": 1,
            "eval_monte_carlo_validation_action_count_0": 0,
            "eval_monte_carlo_validation_action_count_1": 100,
            "eval_monte_carlo_validation_action_count_2": 40,
        },
    }
    gate = block_smb_checkpoint_transfer_source_gate(checkpoint)
    assert not gate["fixed_required"]
    assert gate["transfer_source_gate_met"]
    checkpoint["metrics"]["eval_monte_carlo_validation_gate_met"] = 0
    assert not block_smb_checkpoint_transfer_source_gate(checkpoint)["transfer_source_gate_met"]
    checkpoint["metrics"] = {"eval_monte_carlo_validation_gate_met": 1}
    assert not block_smb_checkpoint_transfer_source_gate(checkpoint)["transfer_source_gate_met"]


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
        _, _, _, _, info = env.step(0)
        assert info["terrain_direction"] == 1
        assert info["support_forward_dx"] == info["support_right_dx"]
    finally:
        env.close()
