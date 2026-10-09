"""Spatial demonstrations must complete the actual lesson through the controller."""

from copy import deepcopy

import pytest

from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.stages.block_smb.controller_teacher import (
    destination,
    route_available,
    teacher_target,
    trial,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import snapshot_env_state
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.teacher_replay import reference_vision, replay
from retroagi.stages.block_smb.teacher_tokens import episode_teacher


def sample(family, index=103, difficulty="medium"):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family=family, difficulty=difficulty
    ).scenario


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "family,index,difficulty",
    [
        ("choice_alternate_route", 103, "medium"),
        ("enemy_stomp", 106, "hard"),
        ("platform_chain", 106, "hard"),
        ("pit_leap", 106, "hard"),
        ("speed_run_retreat", 103, "medium"),
        ("max_points_descend_forward", 100, "easy"),
        pytest.param("max_points_descend_forward", 106, "hard", marks=pytest.mark.timeout(600)),
        ("max_points_retreat", 106, "hard"),
        ("tall_pipe_jump", 106, "hard"),
    ],
)
def test_teacher_completes_regression_layout(family, index, difficulty):
    scenario = sample(family, index, difficulty)
    observer = reference_vision()[0] if "strategy_reference" in scenario else None
    result = replay(scenario, family=family, observer=observer)
    assert result["won"], (family, result["frames"], result["commands"][-5:])
    assert result["points"] >= scenario.get("strategy_objective", {}).get("points", 0)
    if "strategy_reference" in scenario:
        assert scenario["strategy_reference"]["execution"] == "spatial"
        assert result["frames"] <= scenario["strategy_objective"]["deadline"]


def test_trial_restores_simulator_and_does_not_mutate_observation_feedback():
    scenario = sample("enemy_stomp", 106, "hard")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        state = episode_teacher(scenario)
        state.scene = scene_from_labels(env.scene_labels())
        state.execution = SpatialFeedback()
        state.execution.observe(state.scene)
        before = snapshot_env_state(env)
        motion = deepcopy(state.execution.motion)
        goal = destination(env, state, None)
        assert goal is not None
        trial(env, state, goal, teacher_target(env))
        assert snapshot_env_state(env) == before
        assert state.execution.motion == motion
        assert state.execution.destination is None
    finally:
        env.close()


def test_teacher_keeps_unfinished_route_accessible_to_forward_only_camera():
    scenario = sample("choice_alternate_route")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        assert route_available(env)
        target = teacher_target(env)
        assert target.platform_index is not None
        env.camera_x = env.platforms[target.platform_index]["rect"].right + 1
        assert not route_available(env)
    finally:
        env.close()


def test_strategy_reward_remains_a_target_before_terminal_goal():
    scenario = sample("max_points_advance")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        env._tactic_index = len(env._tactics) - 1
        target = teacher_target(env)
        assert target.kind in ("coin", "mount", "stomp")
        assert target.kind != "finish"
    finally:
        env.close()


def test_local_prefix_is_not_reused_as_a_certified_complete_route(monkeypatch):
    from retroagi.stages.block_smb import policy_recovery
    from retroagi.stages.block_smb.teacher_tokens import _remember_route, teacher_plan

    scenario = sample("flat_run")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        state = episode_teacher(scenario)
        state.notes.pop("initial_route", None)
        _remember_route(env, state, [1], complete=False)
        calls = []

        def no_complete_route(*args, **kwargs):
            calls.append(kwargs["prefix"])
            return None

        monkeypatch.setattr(policy_recovery, "coached_suffix", no_complete_route)
        assert teacher_plan(env, state, certify_holds=True) == (None, ())
        assert calls == [False]
    finally:
        env.close()


def test_monster_teacher_retreats_out_of_low_tunnel_before_jumping():
    scenario = sample("monster_retreat", 103, "medium")
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        state = episode_teacher(scenario)
        goal = destination(env, state, None)
        assert goal is not None and goal.mode == "run" and goal.x < 0
        result = trial(env, state, goal, teacher_target(env), save=True)
        assert result.safe
        mario = result.snapshot["mario"]
        assert mario["x"] + mario["w"] <= scenario["tactics"][0]["keep_behind"]
    finally:
        env.close()
