"""Bridge-dependent geometry, safe departure windows, and phase transitions."""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from retroagi.stages.block_smb.bridge_traversal import bridge_safe_wait_frames, bridge_walk_state
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
    validate_block_smb_monte_carlo_oracle,
)
from retroagi.stages.block_smb.pipe_traversal import training_rollout_steps
from retroagi.stages.block_smb.skills import achieved_block_smb_skill_goals
from retroagi.stages.block_smb.train import (
    block_smb_duration_coaching_loss,
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
    train_block_smb_epoch,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_tall_pipe_traversal import rollout


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(difficulty="hard", seed=2):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=seed, sample_index=0, family="bridge_wait", difficulty=difficulty
    )


class PhasePolicy(torch.nn.Module):
    """Only requests wait or locomotion; real primitives and physics execute."""

    def __init__(self, moving_action=1):
        super().__init__()
        self.moving_action = moving_action
        holds = torch.full((1, 1, 16), -30.0)
        holds[..., -1] = 30.0
        self.last_motor_primitives = SimpleNamespace(
            hold_duration_logits=holds, duration_bin_values=torch.arange(1, 17)
        )

    def forward(self, a, b, c, **kwargs):
        goal = kwargs.get("skill_goal")
        action = 0 if goal is not None and goal.any() else self.moving_action
        logits = torch.full((1, a.shape[1], 6), -30.0)
        logits[..., action] = 30.0
        return a.float(), c.clone(), torch.zeros_like(c), a.float(), logits, b, b, None


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_controller_waits_boards_rides_and_finishes(difficulty):
    trajectory = rollout(sample(difficulty), PhasePolicy(), steps=240)
    assert trajectory.success
    last = trajectory.transitions[-1]
    assert last.info["bridge_boarded"] and last.info["bridge_crossed"]
    assert last.info["skill_phase"] == "finish"
    assert any(t.info.get("skill_phase") == "ride" for t in trajectory.transitions)
    assert any(t.info.get("bridge_departure_safe") for t in trajectory.transitions)
    waits = [s for s in trajectory.spans if s.command.get("primitive") == "wait"]
    assert waits and all(s.termination_reason == "success" for s in waits)
    for span in waits:
        assert span.command["held_frames"] == span.duration
        assert trajectory.transitions[span.end_frame].info["bridge_wait_release"] == "event"
        assert span.events[-1]["safe_departure"]
    assert any(
        g["goal_type"] == "wait_pass" for g in achieved_block_smb_skill_goals(trajectory.spans)
    )


def test_safe_windows_certify_actual_walking_support_and_differ_across_phases():
    windows = []
    for difficulty in ("easy", "medium", "hard"):
        item = sample(difficulty)
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=item.scenario)
            safe = bridge_safe_wait_frames(env)
            assert safe
            windows.append(set(safe))
            for wait in {min(safe), max(safe), safe[len(safe) // 2]}:
                env.reset(scenario=item.scenario)
                for _ in range(wait):
                    env.step(0)
                boarded = False
                for _ in range(48):
                    _, _, done, _, _ = env.step(1)
                    state = bridge_walk_state(env)
                    assert state is not None  # Every step has actual engine support.
                    if state.stably_boarded():
                        boarded = True
                        break
                    assert not done
                assert boarded
        finally:
            env.close()
    assert not set.intersection(*windows)  # A constant wait cannot cover all three.


def test_wide_gap_is_not_jumpable_even_without_the_goal_credit_gate():
    scenario = copy.deepcopy(sample().scenario)
    scenario.pop("metadata")
    scenario["require_bridge_before_goal"] = False
    scenario["platforms"] = [p for p in scenario["platforms"] if not isinstance(p, dict)]
    for approach in (0, 4, 8, 12):
        result = validate_block_smb_monte_carlo_oracle(
            scenario, [0] * 4 + [1] * approach + [2] * 64 + [1] * 140
        )
        assert not result["reachable"]
        assert result["rejection_reason"] == "fall_death"


def test_goal_credit_requires_actual_boarding_and_far_shore_support():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample().scenario)
        env.mario.update(x=350.0, y=204.0, on_ground=True)
        _, _, done, _, info = env.step(0)
        assert not done and not env._goal_credited and not info["bridge_boarded"]
    finally:
        env.close()


def test_wait_reward_only_pays_noop_before_a_safe_departure():
    item = sample()
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=item.scenario)
        _, _, _, _, info = env.step(0)
        assert info["reward_terms"]["wait_survival"] > 0
        env.reset(scenario=item.scenario)
        _, _, _, _, info = env.step(1)
        assert info["reward_terms"]["wait_survival"] == 0
        env.reset(scenario=item.scenario)
        for _ in range(64):
            if 1 in bridge_safe_wait_frames(env):
                _, _, _, _, info = env.step(0)
                assert info["reward_terms"]["wait_survival"] == 0
                break
            env.step(0)
        else:
            pytest.fail("never reached a safe departure")
    finally:
        env.close()


def test_duration_loss_accepts_any_bin_in_the_safe_window():
    for chosen, expected_small in ((3, True), (4, True), (15, False)):
        logits = torch.full((1, 1, 16), -30.0)
        logits[..., chosen] = 30.0
        step = SimpleNamespace(
            hold_duration_logits=logits,
            duration_bin_values=torch.arange(1, 17),
            info={
                "primitive_target_hold": 16,
                "primitive_duration_scale": 4,
                "primitive_valid_hold_frames": [16, 20],
            },
        )
        loss = float(block_smb_duration_coaching_loss(step, device=torch.device("cpu")))
        assert (loss < 1e-6) == expected_small


def test_boarding_jump_targets_bridge_support_and_keeps_success_credit():
    # Actual support must override even a disagreeing distance heuristic.
    with patch("retroagi.stages.block_smb.train.jump_overreach", return_value=True):
        trajectory = rollout(sample("medium"), PhasePolicy(moving_action=2), steps=160)
    jumps = [
        s
        for s in trajectory.spans
        if s.command.get("primitive") == "jump"
        and s.termination_reason == "success"
        and trajectory.transitions[s.start_frame].info.get("primitive_target_phase") == "board"
    ]
    assert jumps
    first = trajectory.transitions[jumps[0].start_frame]
    assert not first.info.get("jump_overreach")
    assert first.info["primitive_target_hold"] == jumps[0].command["held_frames"]
    assert first.info["primitive_target_x"] < 285


def test_budget_extension_and_component_metrics_keep_timeouts_visible():
    item = sample()
    assert training_rollout_steps(60, item.scenario) == 240
    samples = replace(
        sample_block_smb_monte_carlo_split(split="validation", seed=2, sample_count=0),
        samples=(item,),
    )
    with patch(
        "retroagi.stages.block_smb.train.sample_block_smb_monte_carlo_parameter_sweep",
        return_value=samples,
    ):
        result = evaluate_block_smb_monte_carlo(
            PhasePolicy(),
            tiny_config(evaluation_episodes=2, evaluation_max_steps=60),
            split="validation",
            sample_count=1,
            stratified_repeats_per_difficulty=3,
            device=torch.device("cpu"),
            vision_factory=StaticBlockVision,
        )
    metrics = result["families"]["bridge_wait"]["bridge_metrics"]
    assert metrics["episodes"] == 2 and metrics["safe_departures"] == 2
    assert metrics["event_releases"] >= 2 and metrics["timer_releases"] == 0
    assert metrics["boardings"] == 2 and metrics["finishes_after_boarding"] == 0
    assert result["families"]["bridge_wait"]["success_rate"] == 0


def test_oracle_waits_have_complete_spans_and_real_training_updates_are_finite():
    item = sample()
    trajectory = rollout(item, PhasePolicy(), steps=240, use_oracle_actions=True)
    assert trajectory.success
    waits = [s for s in trajectory.spans if s.command.get("primitive") == "wait"]
    assert len(waits) == 2
    assert all(s.termination_reason == "success" for s in waits)
    assert trajectory.transitions[0].info["primitive_valid_hold_frames"]
    config = tiny_config(rollout_steps=60, use_oracle_actions=True, generated_scenarios=0)
    model = make_block_smb_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metrics, _ = train_block_smb_epoch(
        model,
        optimizer,
        [(item.scenario_id, item.scenario)],
        config,
        epoch=0,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
    )
    assert metrics["training_rollout_steps_max"] == 240
    assert metrics["train_total_actions"] > 60
    assert metrics["loss_primitive_outcome"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_unsafe_eight_frame_timer_does_not_earn_wait_skill_credit():
    policy = PhasePolicy()
    policy.last_motor_primitives.hold_duration_logits.fill_(-30.0)
    policy.last_motor_primitives.hold_duration_logits[..., 1] = 30.0
    trajectory = rollout(sample(), policy, steps=40)
    first = next(s for s in trajectory.spans if s.command.get("primitive") == "wait")
    assert first.duration == 8
    assert trajectory.transitions[first.end_frame].info["bridge_wait_release"] == "timer"
    assert first.events[-1]["safe_departure"] is False
    assert not achieved_block_smb_skill_goals([first])
