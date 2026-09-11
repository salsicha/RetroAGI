"""Bridge labels must be executable by the policy, with matching goals."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.stages.block_smb.demonstrations import collect_demonstrations, varied_demonstration
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.primitive_execution import BlockSMBPrimitiveExecutor
from retroagi.stages.block_smb.skills import requested_block_smb_skill_goal
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(params=["bridge_mount", "bridge_dismount"])
def bridge_case(request):
    return sample_block_smb_monte_carlo_scenario(
        family=request.param,
        split="train",
        seed=101,
        sample_index=0,
        difficulty="easy",
        validate_reachability=False,
    )


def motor(index):
    logits = torch.full((1, 1, 16), -100.0)
    logits[..., index] = 100
    return SimpleNamespace(hold_duration_logits=logits, duration_bin_values=torch.arange(1, 17))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_duration_labels_reproduce_landings_through_policy_executor(bridge_case, difficulty):
    case = sample_block_smb_monte_carlo_scenario(
        family=bridge_case.family,
        split="validation",
        seed=101,
        sample_index=0,
        difficulty=difficulty,
        validate_reachability=False,
    )
    actions = case.oracle["actions"]
    departure = actions.index(2)
    hold = actions.count(2)
    index = NES_JUMP_FRAMES.index(hold)
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        executor = BlockSMBPrimitiveExecutor(env, adaptive_duration=False)
        raw = motor(index)
        for frame in range(320):
            intent = executor.committed_action
            if intent is None:
                intent = 0 if frame < departure else 2
            mapped = executor.motor_parameters(intent, raw)
            execution = executor.execute(
                intent,
                motor_primitives=mapped,
                support_override="ground" if env.mario["on_ground"] else "air",
            )
            _, _, done, truncated, _ = env.step(execution.action)
            if done or truncated:
                break
        assert env._goal_credited
        assert raw.duration_bin_values.tolist() == list(range(1, 17))
    finally:
        env.close()


def test_demonstrations_preserve_bridge_goal_and_have_successful_alternate(bridge_case):
    data = collect_demonstrations([(0, bridge_case)], tiny_config(), StaticBlockVision)
    goal = requested_block_smb_skill_goal(bridge_case.scenario)
    assert torch.all(data.goal == goal)
    assert data.actor_mask[data.action == 0].all()
    alternative = varied_demonstration(bridge_case, 101, robust=True)
    assert alternative is not None
    assert alternative.oracle["actions"] != bridge_case.oracle["actions"]


def test_batched_and_sequential_bridge_goals_and_waits_match(bridge_case):
    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import (
        block_smb_policy_scenario,
        collect_trajectory,
        make_block_smb_model,
    )
    from scripts.block_smb_batched_evaluation import evaluate_batched

    config = tiny_config(evaluation_max_steps=12, ranked_candidate_search=False)
    config = replace(config, ablation=replace(config.ablation, recurrent_state_enabled=False))
    model = make_block_smb_model(config).eval()
    goals = []
    original = model.forward

    def forward(*args, **kwargs):
        goals.append(kwargs["skill_goal"].clone())
        result = original(*args, **kwargs)
        model.last_policy_logits_a[..., :6] = -100
        model.last_policy_logits_a[..., 0] = 100
        model.last_selected_action_id = 0
        return result

    model.forward = forward
    result = evaluate_batched(model, [bridge_case], config, StaticBlockVision, return_actions=True)
    batched_goals = goals.copy()
    goals.clear()
    stage = BlockSMBStage(
        scenario=block_smb_policy_scenario(bridge_case.scenario, True), vision=StaticBlockVision()
    )
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                model,
                stage,
                bridge_case.scenario_id,
                rollout_steps=12,
                seed=101,
                deterministic=True,
                device=torch.device("cpu"),
                ablation=config.ablation,
            )
        assert result["actions"][0] == [t.action for t in trajectory.transitions]
        expected = requested_block_smb_skill_goal(bridge_case.scenario)
        assert all(torch.equal(g, expected) for g in batched_goals + goals)
    finally:
        stage.env.close()


def test_training_keeps_physical_duration_labels_through_release(bridge_case):
    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import collect_trajectory, make_block_smb_model

    model = make_block_smb_model(tiny_config()).eval()
    stage = BlockSMBStage(scenario=bridge_case.scenario, vision=StaticBlockVision())
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                model,
                stage,
                bridge_case.scenario_id,
                rollout_steps=320,
                seed=101,
                deterministic=True,
                device=torch.device("cpu"),
                demonstration_actions=bridge_case.oracle["actions"],
            )
        assert trajectory.success
        jump_rows = [t for t in trajectory.transitions if t.info.get("primitive_valid_hold_frames")]
        assert jump_rows and any(t.action == 1 for t in jump_rows)
        for step in jump_rows:
            assert step.duration_bin_values.tolist() == list(NES_JUMP_FRAMES)
            assert step.info["primitive_target_hold"] in step.info["primitive_valid_hold_frames"]
        waits = [t for t in trajectory.transitions if t.action == 0]
        assert all("primitive_outcome_target" not in t.info for t in waits)
    finally:
        stage.env.close()
