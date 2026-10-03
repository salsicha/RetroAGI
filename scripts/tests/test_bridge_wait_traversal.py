"""Bridge-dependent geometry, safe departure windows, and phase transitions."""

import copy
from types import SimpleNamespace

import pytest
import torch

from retroagi.stages.block_smb.bridge_traversal import bridge_safe_wait_frames, bridge_walk_state
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    validate_block_smb_monte_carlo_oracle,
)


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
