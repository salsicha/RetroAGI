"""Stomp contact geometry, duration decisions, and single-attempt diagnostics."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pygame
import pytest
import torch

from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
)
from retroagi.stages.block_smb.stomp import stomp_coaching_target, stomp_collision_geometry
from retroagi.stages.block_smb.train import (
    _action_from_model,
    block_smb_duration_coaching_loss,
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_tall_pipe_traversal import PhaseIntentPolicy, rollout


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def geometry(x=120, y=192, vy=4, enemy_x=110):
    return stomp_collision_geometry(
        pygame.Rect(x, y, 14, 16), pygame.Rect(enemy_x, 206, 12, 14), vy
    )


def legacy_sample():
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=2, sample_index=0, family="stomp_mount", difficulty="easy"
    )
    scenario = dict(sample.scenario)
    scenario["enemies"] = [[94, 206, 94, 94, 0.0]]
    return replace(sample, scenario=scenario)


def held_policy(hold):
    policy = PhaseIntentPolicy(finish=False)
    policy.last_motor_primitives.hold_duration_logits.fill_(-30.0)
    policy.last_motor_primitives.hold_duration_logits[..., hold - 1] = 30.0
    return policy


def test_collision_footprints_anchor_edge_stomp_outside_goal_proxy():
    contact = geometry()
    assert abs(127 - 116) > 8  # Outside the obsolete proxy-center tolerance.
    assert contact["stomp"]
    assert stomp_coaching_target([contact], 16) == (16.0, "success")
    assert not geometry(x=122)["stomp"]  # Exactly touching side edges.
    assert not geometry(vy=-4)["stomp"]  # Rising contact.
    assert not geometry(y=207)["stomp"]  # Side hit below the stomp window.


def test_coach_compares_bodies_at_contact_time_and_not_old_target_positions():
    earlier = geometry(x=90, y=150, vy=-2, enemy_x=95)
    assert stomp_coaching_target([earlier, geometry()], 16) == (16.0, "success")
    short = geometry(x=90, enemy_x=110)
    later = geometry(x=145, y=204, vy=0, enemy_x=114)
    assert stomp_coaching_target([short, later], 6) == (7.0, "undershoot")
    assert stomp_coaching_target([geometry(x=140)], 16) == (15.0, "overshoot")
    assert stomp_coaching_target([earlier], 8) == (None, "no_contact")


def test_categorical_coaching_changes_the_executed_mode_not_just_its_mean():
    logits = torch.full((1, 1, 16), -3.0)
    logits[..., 5] = 3.0
    logits[..., 11] = 2.9  # Mean near nine, but execution picks six.
    logits.requires_grad_()
    step = SimpleNamespace(
        hold_duration_logits=logits,
        duration_bin_values=torch.arange(1, 17),
        info={"primitive_target_hold": 9},
    )
    optimizer = torch.optim.SGD([logits], lr=1.0)
    assert int(logits.argmax()) + 1 == 6
    for _ in range(15):
        optimizer.zero_grad()
        loss = block_smb_duration_coaching_loss(step, device=torch.device("cpu"))
        loss.backward()
        optimizer.step()
    assert int(logits.argmax()) + 1 == 9


def test_actual_success_and_safe_miss_have_consistent_labels_rewards_and_spans():
    sample = legacy_sample()
    success = rollout(sample, held_policy(8))
    assert success.success
    assert all(
        t.info["primitive_target_hold"] == 8
        for t in success.transitions
        if "primitive_target_hold" in t.info
    )
    miss = rollout(sample, held_policy(16))
    last = miss.transitions[-1]
    assert last.done and not miss.success and not last.info["death"]
    assert last.info["stomp_outcome"] == "overshoot"
    assert last.info["reward_terms"]["stomp_miss"] == -10.0
    assert miss.total_return < 0
    assert len(miss.transitions) < 60
    assert last.episode_mask == 0
    assert last.next_batch.metadata["episode"]["terminated"]
    assert miss.spans[0].termination_reason == "failure"
    assert miss.spans[0].failure_category == "overshoot"
    short = rollout(sample, held_policy(1))
    assert short.transitions[-1].info["stomp_outcome"] == "collision"
    assert short.transitions[-1].info["death"]
    assert short.total_return < 0
    timeout = rollout(sample, held_policy(16), steps=5)
    assert timeout.transitions[-1].info["stomp_outcome"] == "budget_timeout"
    assert timeout.spans[0].termination_reason == "evaluator_truncation"
    assert timeout.spans[0].failure_category == "budget_timeout"


def test_supplied_jump_conditions_b_world_model_and_motor_before_prediction():
    model = make_block_smb_model(tiny_config(ranked_candidate_search=True))
    model.eval()
    with torch.no_grad():
        model.agent.fc_out_A.weight.zero_()
        model.agent.fc_out_A.bias.fill_(-10)
        model.agent.fc_out_A.bias[1] = 10  # The free actor strongly prefers RIGHT.
    seen = []
    handle = model.world_model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["primitive_context"].detach().clone()),
        with_kwargs=True,
    )
    stage = BlockSMBStage(scenario=legacy_sample().scenario, vision=StaticBlockVision())
    try:
        batch = stage.encode_observation(stage.reset(seed=0))
        with torch.no_grad():
            outputs = _action_from_model(model, batch, deterministic=True, tau=1.0, forced_action=2)
        assert outputs[0] == 2
        assert (
            int(outputs[4][-1][0, -1].argmax()) == 1
        )  # Genuine logits remain available for oracle CE.
        assert len(seen) == 1  # Ranked search cannot substitute a different supplied intent.
        assert float(seen[0][0, -1, 0]) == pytest.approx(2 / 5)
        assert int(model.last_motor_primitives.button_combo_logits[0, -1].argmax()) == 2
        assert float(outputs[1]) == 0.0  # No policy credit for a supplied A action.
    finally:
        handle.remove()
        stage.env.close()


def test_new_motion_phases_reverse_before_the_release_window_closes_and_stay_reachable():
    directions = set()
    turns = set()
    for difficulty in ("medium", "hard"):
        for seed in range(12):
            sample = sample_block_smb_monte_carlo_scenario(
                split="validation",
                seed=seed,
                sample_index=0,
                family="stomp_mount",
                difficulty=difficulty,
            )
            assert sample.reachability["reachable"]
            assert sample.parameters["family_revision"] == 2
            env = MarioScenarioEnv()
            try:
                env.reset(scenario=sample.scenario)
                initial = env.enemies[0]["direction"]
                directions.add(initial)
                for frame in range(1, 17):
                    env.step(2)
                    if env.enemies[0]["direction"] != initial:
                        turns.add(frame)
                        break
                else:
                    pytest.fail("enemy never reversed within the controllable hold window")
            finally:
                env.close()
    assert directions == {-1, 1}
    assert len(turns) >= 3


def test_evaluation_aggregates_real_misses_separately_from_timeouts():
    samples = replace(
        sample_block_smb_monte_carlo_split(split="validation", seed=2, sample_count=0),
        samples=(legacy_sample(),),
    )
    for steps, outcome in ((120, "overshoot"), (5, "budget_timeout")):
        with patch(
            "retroagi.stages.block_smb.train.sample_block_smb_monte_carlo_parameter_sweep",
            return_value=samples,
        ):
            result = evaluate_block_smb_monte_carlo(
                held_policy(16),
                tiny_config(evaluation_episodes=2, evaluation_max_steps=steps),
                split="validation",
                sample_count=1,
                stratified_repeats_per_difficulty=3,
                device=torch.device("cpu"),
                vision_factory=StaticBlockVision,
            )
        family = result["families"]["stomp_mount"]
        assert family["stomp_outcome_counts"] == {outcome: 2}
        assert family["success_rate"] == 0.0
        assert family["failures"][0]["stomp_outcome_counts"] == {outcome: 2}


def test_oracle_stomp_has_one_complete_arc_and_no_off_policy_reinforce():
    sample = legacy_sample()
    with patch(
        "retroagi.stages.block_smb.train.block_smb_oracle_actions_for_rollout",
        return_value=(2,) * 8 + (1,) * 52,
    ):
        trajectory = rollout(sample, held_policy(16), use_oracle_actions=True)
    assert trajectory.success
    supervised = [t for t in trajectory.transitions if "primitive_target_hold" in t.info]
    assert len(supervised) == len(trajectory.transitions)
    assert {t.info["primitive_target_hold"] for t in supervised} == {8.0}
    assert all(float(t.log_prob) == 0.0 for t in trajectory.transitions)
    jumps = [s for s in trajectory.spans if s.command.get("primitive") == "jump"]
    assert len(jumps) == 1
    assert jumps[0].command["held_frames"] == 8


def test_wait_categorical_target_accounts_for_four_frame_bins():
    logits = torch.full((1, 1, 16), -30.0)
    logits[..., 4] = 30.0
    step = SimpleNamespace(
        hold_duration_logits=logits,
        duration_bin_values=torch.arange(1, 17),
        info={"primitive_target_hold": 20, "primitive_duration_scale": 4},
    )
    assert float(block_smb_duration_coaching_loss(step, device=torch.device("cpu"))) < 1e-6


@pytest.mark.parametrize("oracle", [False, True])
def test_stomp_duration_coaching_backpropagates_through_a_complete_training_attempt(oracle):
    from retroagi.stages.block_smb.train import train_block_smb_epoch

    torch.manual_seed(7)
    config = tiny_config(rollout_steps=60, use_oracle_actions=oracle, generated_scenarios=0)
    model = make_block_smb_model(config)
    before = {
        name: p.detach().clone() for name, p in model.named_parameters() if "duration" in name
    }
    assert before
    sample = sample_block_smb_monte_carlo_scenario(
        split="train", seed=4, sample_index=0, family="stomp_mount", difficulty="hard"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metrics, _ = train_block_smb_epoch(
        model,
        optimizer,
        [(sample.scenario_id, sample.scenario)],
        config,
        epoch=0,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
    )
    assert metrics["train_total_actions"] > 20
    assert metrics["loss_primitive_outcome"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(
        not torch.equal(before[name], p.detach())
        for name, p in model.named_parameters()
        if name in before
    )


def test_ranked_candidate_conditioning_preserves_actor_policy_gradients():
    model = make_block_smb_model(tiny_config(ranked_candidate_search=True))
    model.eval()
    stage = BlockSMBStage(scenario=legacy_sample().scenario, vision=StaticBlockVision())
    try:
        batch = stage.encode_observation(stage.reset(seed=0))
        output = _action_from_model(model, batch, deterministic=True, tau=1.0)
        assert model.last_selected_action_id is not None
        assert output[0] == model.last_selected_action_id
        assert int(model.last_motor_primitives.button_combo_logits[0, -1].argmax()) == output[0]
        (-output[1]).backward()
        grad = model.agent.fc_out_A.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    finally:
        stage.env.close()
