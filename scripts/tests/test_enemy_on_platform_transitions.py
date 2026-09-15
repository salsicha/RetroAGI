"""Enemy clearance must survive landing release and a subsequent walk-off."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from retroagi.stages.block_smb.demonstrations import collect_demonstrations
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds
from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
from retroagi.stages.block_smb.policy_recovery import repair_policy_actions
from retroagi.stages.block_smb.primitive_execution import teacher_route_reachable
from scripts.block_smb_batched_evaluation import evaluate_batched
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_full_smb_failure_families import sample
from scripts.tests.test_tall_pipe_traversal import PhaseIntentPolicy, rollout


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def platform_scenario(*, hard=False):
    return dict(
        world_width=352,
        mario=[36, 204] if hard else [33, 204],
        platforms=[[0, 220, 352, 20], [113, 173, 80, 47] if hard else [119, 191, 104, 29]],
        enemies=[[147, 159, 135, 159, 0.6, -1] if hard else [169, 177, 157, 181, 0.2, -1]],
        goal=[324, 204, 16, 16],
        goal_requires_support=True,
        task={"family": "enemy_on_platform"},
    )


def test_mount_certificate_includes_release_before_next_enemy_jump():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=platform_scenario(hard=True))
        for _ in range(8):
            env.step(1)
        saved = snapshot_env_state(env)
        valid = safe_jump_holds(env, local_objective(env), 1)
        assert 14 in valid and 16 not in valid
        assert snapshot_env_state(env) == saved
        # The excluded hold lands alive, but the required release kills Mario.
        for action in [2] * 16 + [1] * 10:
            _, _, _, _, info = env.step(action)
        assert env.mario["on_ground"] and not info["death"]
        env.step(1)
        _, _, _, _, info = env.step(1)
        assert info["death"]
    finally:
        env.close()


class ClearThenFinishPolicy(PhaseIntentPolicy):
    """Expose any resurrected enemy goal by turning back toward it."""

    def forward(self, a, b, c, **kwargs):
        goal = kwargs["skill_goal"]
        requested = goal.any(-1)
        action = torch.where(requested, 2, 1)
        action = torch.where(requested & (c[:, 12] > 190 / 352), 4, action)
        logits = torch.full((len(c), a.shape[1], 6), -30.0)
        logits.scatter_(2, action[:, None, None].expand(-1, a.shape[1], 1), 30)
        self.last_policy_logits_a = logits
        holds = torch.full((len(c), 1, 16), -30.0)
        holds[..., 7] = 30
        self.last_motor_primitives = SimpleNamespace(
            hold_duration_logits=holds,
            duration_bin_values=torch.arange(1, 17),
            hold_duration=torch.full((len(c), 1), 8.0),
        )
        return a.float(), c.clone(), torch.zeros_like(c), a.float(), logits, b, b, None


@pytest.mark.parametrize("batched", [False, True])
def test_completed_enemy_goal_does_not_return_when_walking_off_platform(batched):
    scenario = platform_scenario()
    scenario["mario"] = [120, 175]
    case = replace(sample("enemy_on_platform", "easy"), scenario=scenario)
    policy = ClearThenFinishPolicy()
    if batched:
        out = evaluate_batched(
            policy,
            [case],
            tiny_config(
                walk_duration_primitives=False,
                ablation={"recurrent_state_enabled": False},
                engine_support_override=True,
                skill_goal_conditioning=True,
                ranked_candidate_search=False,
                evaluation_max_steps=160,
            ),
            StaticBlockVision,
            return_actions=True,
        )
        assert not out["failures"]
        assert 4 not in out["actions"][0]
    else:
        trajectory = rollout(case, policy, steps=160, walk_duration_primitives=False)
        assert trajectory.success
        assert all(t.action != 4 for t in trajectory.transitions)
        assert any(t.info.get("skill_phase") == "finish" for t in trajectory.transitions)


def test_recovery_preserves_release_masks_at_splice():
    case = replace(sample("enemy_on_platform", "easy", split="train"), scenario=platform_scenario())
    # Landing on the platform with the next enemy still ahead.
    actions = [1] * 10 + [2] * 7 + [1] * 120
    # Preserve dataset provenance independently of the explicit task identity.
    case.scenario["metadata"] = {
        "block_smb_monte_carlo": {"family": "enemy_on_platform", "split": "train"}
    }
    repairs = repair_policy_actions(case.scenario, actions)
    repair = next(r for r in repairs if r["recovery_reason"] == "landing_recovery")
    start = repair["supervision_start_frame"]
    assert repair["actions"][start : start + 2] == [1, 1]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        assert teacher_route_reachable(env, repair["actions"])
        invalid = repair["actions"][:start] + repair["actions"][start + 2 :]
        assert not teacher_route_reachable(env, invalid)
    finally:
        env.close()
    data = collect_demonstrations(
        [(BLOCK_SMB_MC_FAMILIES.index("enemy_on_platform"), replace(case, oracle=repair))],
        tiny_config(walk_duration_primitives=False),
        StaticBlockVision,
    )
    assert data.forced_release[:2].all()
    assert not data.actor_mask[:2].any()
    assert (data.actor_mask & (data.action == 2)).any()


@pytest.mark.parametrize("index", [35, 53, 56, 101, 161])
def test_previous_hard_fallback_routes_obey_executor_timing(index):
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

    case = sample_block_smb_monte_carlo_scenario(
        family="enemy_on_platform",
        split="train",
        seed=20260908,
        sample_index=index,
        difficulty="hard",
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        assert teacher_route_reachable(env, case.oracle["actions"])
    finally:
        env.close()


def test_dangerous_mount_duration_gets_an_executor_valid_repair():
    case = platform_scenario(hard=True)
    case["metadata"] = {"block_smb_monte_carlo": {"family": "enemy_on_platform"}}
    repairs = repair_policy_actions(case, [1] * 8 + [2] * 16 + [1] * 12)
    repair = next(
        r
        for r in repairs
        if r["supervision_start_frame"] == 8 and r["recovery_reason"] == "duration"
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case)
        assert teacher_route_reachable(env, repair["actions"])
    finally:
        env.close()
