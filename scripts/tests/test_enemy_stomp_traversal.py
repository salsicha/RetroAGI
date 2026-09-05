"""Composite stomp credit, recovery, coaching, and scenario diversity."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    _goal_reached as oracle_goal_reached,
)
from retroagi.stages.block_smb.monte_carlo import (
    block_smb_monte_carlo_oracle_actions,
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
    validate_block_smb_monte_carlo_oracle,
)
from retroagi.stages.block_smb.pipe_traversal import training_rollout_steps
from retroagi.stages.block_smb.skills import achieved_block_smb_skill_goals
from retroagi.stages.block_smb.stomp import stomp_coaching_target
from retroagi.stages.block_smb.train import (
    _goal_reached,
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
    train_block_smb_epoch,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_stomp_coaching import geometry
from scripts.tests.test_tall_pipe_traversal import PhaseIntentPolicy, rollout


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def legacy_sample():
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=2, sample_index=0, family="enemy_stomp", difficulty="easy"
    )
    scenario = dict(sample.scenario)
    scenario.pop("require_stomp_before_goal")
    scenario.update(mario=[20, 200], enemies=[[109, 206, 109, 109, 0]], goal=[230, 200, 16, 20])
    return replace(sample, scenario=scenario)


def test_saved_family_requires_stomp_before_finish_and_oracle_cannot_bypass_it():
    sample = legacy_sample()
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        bypass = [1] * 15 + [2] * 16 + [1] * 39
        for action in bypass:
            _, _, done, _, info = env.step(action)
        assert not done and not info["death"] and not info["stomp_completed"]
        assert env.mario["x"] + env.mario["w"] > env.goal.left
        assert not _goal_reached(env) and not oracle_goal_reached(env)
        assert not validate_block_smb_monte_carlo_oracle(sample.scenario, bypass)["reachable"]
        env.reset(scenario=sample.scenario)
        contact = None
        for frame, action in enumerate([2] * 16 + [1] * 104):
            _, _, done, _, info = env.step(action)
            if (info.get("stomp_geometry") or {}).get("stomp"):
                contact = frame
                assert info["stomp_completed"] and not done
                assert not _goal_reached(env)
            if done:
                break
        assert contact is not None and done and _goal_reached(env)
        assert contact < frame
    finally:
        env.close()


def test_successful_stomp_is_coached_to_contact_then_bounce_and_finish_are_separate():
    policy = PhaseIntentPolicy()
    trajectory = rollout(legacy_sample(), policy)
    assert trajectory.success
    contacts = [
        i
        for i, t in enumerate(trajectory.transitions)
        if (t.info.get("stomp_geometry") or {}).get("stomp")
    ]
    assert len(contacts) == 1
    contact = contacts[0]
    coached = [t for t in trajectory.transitions if t.info.get("primitive_target_phase") == "stomp"]
    assert len(coached) == contact + 1
    assert {t.info["primitive_target_hold"] for t in coached} == {16.0}
    assert {t.info["primitive_target_x"] for t in coached} == {115.0}
    assert not any(t.info.get("jump_overreach") for t in coached)
    jump = next(s for s in trajectory.spans if s.command.get("primitive") == "jump")
    assert jump.termination_reason == "success" and jump.end_frame == contact
    assert jump.events[-1]["outcome"] == "stomp"
    bounce = next(s for s in trajectory.spans if s.command.get("primitive") == "bounce_recovery")
    assert bounce.start_frame == contact + 1
    assert trajectory.transitions[bounce.end_frame].info["mario"]["on_ground"]
    assert not any(e["event"] == "liftoff" for e in bounce.events)
    assert policy.goals[0].any() and all(not g.any() for g in policy.goals[contact + 1 :])
    assert any(
        g["goal_type"] == "enemy_clear" for g in achieved_block_smb_skill_goals(trajectory.spans)
    )


def test_persistent_jump_request_cannot_start_a_primitive_during_stomp_bounce():
    trajectory = rollout(legacy_sample(), PhaseIntentPolicy(finish=False))
    recovery = [t for t in trajectory.transitions if t.info.get("skill_phase") == "bounce_recovery"]
    assert recovery
    assert any(
        t.action == 2 and t.info.get("skill_phase") == "finish" for t in trajectory.transitions
    )
    assert all(t.action == 1 and t.expected_hold is None for t in recovery)
    for span in trajectory.spans:
        if span.command.get("primitive") == "jump":
            assert trajectory.transitions[span.start_frame].info["skill_phase"] != "bounce_recovery"


def test_leftward_recovery_coaching_uses_the_direction_of_the_interception():
    assert stomp_coaching_target([geometry(x=140)], 8, direction=-1) == (9.0, "undershoot")
    assert stomp_coaching_target([geometry(x=90)], 8, direction=-1) == (7.0, "overshoot")


def test_training_budget_includes_old_replay_scenarios_but_evaluation_honors_its_limit():
    sample = legacy_sample()
    assert training_rollout_steps(60, sample.scenario) == 160
    assert training_rollout_steps(200, sample.scenario) == 200
    trajectory = rollout(sample, PhaseIntentPolicy(), steps=60)
    assert len(trajectory.transitions) == 60 and not trajectory.success
    assert trajectory.transitions[-1].info["stomp_completed"]


def test_revision_two_oracles_stomp_and_finish_and_fixed_spawn_jump_does_not_solve_all_tiers():
    starts, walks, holds, directions = set(), set(), set(), set()
    fixed_results = []
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(4):
            sample = sample_block_smb_monte_carlo_scenario(
                split="validation",
                seed=seed,
                sample_index=0,
                family="enemy_stomp",
                difficulty=difficulty,
            )
            assert sample.parameters["family_revision"] == 2
            assert sample.reachability["reachable"] and sample.reachability["stomp_completed"]
            assert sample.reachability["completion_steps"] <= 160
            assert "stomp_window" not in sample.parameters
            actions = block_smb_monte_carlo_oracle_actions(sample.scenario)
            assert validate_block_smb_monte_carlo_oracle(sample.scenario, actions)[
                "stomp_completed"
            ]
            starts.add(sample.parameters["spawn_x"])
            walks.add(sample.parameters["oracle_approach_frames"])
            holds.add(sample.parameters["oracle_hold_frames"])
            directions.add(sample.parameters["enemy_initial_direction"])
            fixed_results.append(
                validate_block_smb_monte_carlo_oracle(sample.scenario, [2] * 16 + [1] * 144)[
                    "reachable"
                ]
            )
    assert len(starts) > 1 and len(walks) > 1 and len(holds) > 1
    assert directions == {-1, 1}
    assert not all(fixed_results)


def test_evaluation_counts_stomps_separately_from_finishing_after_stomp():
    samples = replace(
        sample_block_smb_monte_carlo_split(split="validation", seed=2, sample_count=0),
        samples=(legacy_sample(),),
    )
    for steps, finish in ((60, 0), (120, 2)):
        with patch(
            "retroagi.stages.block_smb.train.sample_block_smb_monte_carlo_parameter_sweep",
            return_value=samples,
        ):
            result = evaluate_block_smb_monte_carlo(
                PhaseIntentPolicy(),
                tiny_config(evaluation_episodes=2, evaluation_max_steps=steps),
                split="validation",
                sample_count=1,
                stratified_repeats_per_difficulty=3,
                device=torch.device("cpu"),
                vision_factory=StaticBlockVision,
            )
        metrics = result["families"]["enemy_stomp"]["enemy_stomp_metrics"]
        assert metrics == {
            "episodes": 2,
            "stomp_successes": 2,
            "finish_after_stomp_successes": finish,
            "stomp_success_rate": 1.0,
            "finish_after_stomp_success_rate": finish / 2,
        }


def test_real_optimizer_can_train_a_complete_composite_oracle():
    sample = sample_block_smb_monte_carlo_scenario(
        split="train", seed=2, sample_index=0, family="enemy_stomp", difficulty="hard"
    )
    config = tiny_config(rollout_steps=60, use_oracle_actions=True, generated_scenarios=0)
    model = make_block_smb_model(config)
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
    assert metrics["train_total_actions"] > 60
    assert metrics["training_rollout_steps_max"] == 160
    assert metrics["loss_primitive_outcome"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_unreachable_opening_jump_is_coached_against_enemy_distance():
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=0, family="enemy_stomp", difficulty="hard"
    )
    trajectory = rollout(sample, PhaseIntentPolicy(finish=False), steps=45)
    first = trajectory.transitions[0]
    assert first.info["jump_overreach"]
    assert first.info["primitive_target_phase"] == "stomp"
    assert first.info["primitive_target_x"] < sample.scenario["goal"][0]
    assert not any(t.info["stomp_completed"] for t in trajectory.transitions)


def test_recovery_policy_credit_combines_jump_and_release_intents():
    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import _action_from_model

    stage = BlockSMBStage(scenario=legacy_sample().scenario, vision=StaticBlockVision())
    try:
        batch = stage.encode_observation(stage.reset(seed=0))
        # The policy assigns almost all mass to RIGHT_JUMP. Recovery releases
        # the button, so the executed RIGHT has almost all projected mass too.
        result = _action_from_model(
            PhaseIntentPolicy(finish=False),
            batch,
            deterministic=True,
            tau=1.0,
            recovering_from_stomp=True,
        )
        assert result[0] == 1
        assert float(result[1]) == pytest.approx(0.0, abs=1e-6)
        assert not result[6].started and not result[6].active
    finally:
        stage.env.close()
