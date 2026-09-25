"""Shared tactical labels retain physical behavior and joint learning."""

from dataclasses import replace
from functools import lru_cache

import pytest
import torch

from retroagi.core.models import TACTIC_STANCES
from retroagi.stages.block_smb.demonstrations import (
    collect_demonstrations,
    demonstrated_tactic_sets,
    fit_demonstrations,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import snapshot_env_state
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.policy_recovery import combine_demonstrations, repair_policy_actions
from retroagi.stages.block_smb.primitive_execution import teacher_route_reachable
from retroagi.stages.block_smb.tactics import TACTICAL_FAMILIES, compatible_actions, tactic_label
from retroagi.stages.block_smb.train import make_block_smb_model
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@lru_cache(None)
def sample(family, difficulty="medium"):
    return sample_block_smb_monte_carlo_scenario(
        family=family,
        difficulty=difficulty,
        split="train",
        seed=101,
        sample_index=0,
    )


def config():
    return tiny_config(
        motion_observations=True,
        hazard_observations=True,
        adaptive_duration_control=False,
        walk_duration_primitives=False,
    )


@lru_cache(None)
def demonstrations(family):
    case = sample(family)
    return collect_demonstrations(
        [(BLOCK_SMB_MC_FAMILIES.index(family), case)], config(), StaticBlockVision
    )


@pytest.mark.parametrize("family", sorted(TACTICAL_FAMILIES - {"piranha_avoidance"}))
def test_each_added_family_supplies_tactic_and_compatible_motor_targets(family):
    data = demonstrations(family)
    mask = data.actor_mask & (data.tactic >= 0)
    assert mask.any()
    assert data.tactic_actions[mask, data.action[mask]].all()
    # Every trained jump still has a physically certified duration target.
    jumps = mask & ((data.action == 2) | (data.action == 4))
    assert data.valid_durations[jumps].any(dim=-1).all()
    assert (data.tactic[data.forced_release] == -1).all()


@pytest.mark.parametrize(
    "family,consumed",
    [
        ("wait_timing", True),
        ("bridge_wait", True),
        ("bridge_mount", False),
        ("bridge_dismount", False),
    ],
)
def test_tactical_wait_labels_do_not_disable_consumed_wait_durations(family, consumed):
    data = demonstrations(family)
    waits = data.actor_mask & (data.action == 0)
    assert waits.any()
    assert (data.tactic[waits] == TACTIC_STANCES.index("hold_area")).all()
    assert (data.duration_consumed[waits] == consumed).all()


def test_leftward_goal_is_advance_and_ordinary_families_are_not_labeled():
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario=dict(
                world_width=256,
                mario=[190, 204],
                platforms=[[0, 220, 256, 20]],
                goal=[35, 204, 16, 16],
            )
        )
        advance = TACTIC_STANCES.index("advance")
        retreat = TACTIC_STANCES.index("retreat")
        assert tactic_label(env, action=3, family="retreat_recovery") == advance
        assert tactic_label(env, action=1, family="retreat_recovery") == retreat
        assert tactic_label(env, family="retreat_recovery") == advance
        assert compatible_actions(env, advance) == [False, False, False, True, True, False]
        assert tactic_label(env, action=1, family="flat_run") == -1
        assert tactic_label(env, family="single_gap") == -1
    finally:
        env.close()


def test_blocked_pipe_requests_retreat_only_when_fallback_is_supported():
    env = MarioScenarioEnv()
    try:
        for floor_left, expected in [(0, "retreat"), (79, None)]:
            env.reset(
                scenario=dict(
                    world_width=256,
                    mario=[66, 204],
                    platforms=[[floor_left, 220, 256 - floor_left, 20], [80, 130, 32, 90]],
                    goal=[230, 204, 16, 16],
                )
            )
            before = snapshot_env_state(env)
            label = tactic_label(env, family="tall_pipe_jump")
            assert snapshot_env_state(env) == before
            assert label == (-1 if expected is None else TACTIC_STANCES.index(expected))
    finally:
        env.close()


@pytest.mark.parametrize(
    "family", ["wait_timing", "moving_bridge", "bridge_mount", "bridge_dismount"]
)
def test_bridge_online_labels_cover_hold_and_advance_without_mutating_physics(family):
    case = sample(family)
    env = MarioScenarioEnv()
    labels = set()
    try:
        env.reset(scenario=case.scenario)
        for action in case.oracle["actions"]:
            if env.mario["on_ground"]:
                before = snapshot_env_state(env)
                labels.add(tactic_label(env, family=family))
                assert snapshot_env_state(env) == before
            _, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        assert env._goal_credited
        assert {TACTIC_STANCES.index("advance"), TACTIC_STANCES.index("hold_area")} <= labels
    finally:
        env.close()


def test_successful_route_variants_accept_multiple_stances_at_identical_observations():
    data = demonstrations("wait_timing")
    # Two independently successful routes can choose an early departure or
    # keep waiting. Union their labels just as we union their motor choices.
    first = (data.actor_mask & (data.tactic >= 0)).nonzero().flatten()[0]
    rows = {f: getattr(data, f)[first : first + 1].clone() for f in data.__dataclass_fields__}
    advance = replace(
        type(data)(**rows),
        tactic=torch.tensor([0]),
        tactic_actions=torch.tensor([[False, True, True, False, False, False]]),
    )
    hold = replace(
        type(data)(**rows),
        tactic=torch.tensor([2]),
        tactic_actions=torch.tensor([[True, False, False, False, False, False]]),
    )
    combined = combine_demonstrations([advance, hold])
    labels, actions = demonstrated_tactic_sets(combined)
    assert labels.tolist() == [[True, False, True, False]] * 2
    assert actions.tolist() == [[True, True, True, False, False, False]] * 2


def test_bridge_enemy_and_recovery_batch_updates_all_three_levels():
    torch.manual_seed(7)
    data = combine_demonstrations(
        [demonstrations(f) for f in ("wait_timing", "enemy_stomp", "stair_climb", "mixed_section")]
    )
    model = make_block_smb_model(config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_demonstrations(
        model,
        optimizer,
        data,
        steps=4,
        batch_size=128,
        decision_durations_only=True,
        walk_durations=False,
    )
    for module in (
        model.tactics_network.encoder,
        model.tactics_network.stance_head,
        model.agent.transformer_A,
        model.agent.transformer_B,
    ):
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
    assert model.last_demonstration_metrics["tactic_action_loss"] > 0
    assert model.last_demonstration_metrics["tactic_loss"] > 0


@pytest.mark.parametrize("family", ["wait_timing", "moving_bridge"])
def test_bad_bridge_departures_produce_successful_tactical_recoveries(family):
    case = sample(family)
    failed = [1] * 80
    repairs = repair_policy_actions(case.scenario, failed, max_repairs=2)
    assert repairs and repairs[0]["recovery_reason"] == "tactic"
    env = MarioScenarioEnv()
    try:
        for repair in repairs:
            start = repair["supervision_start_frame"]
            assert repair["actions"][:start] == failed[:start]
            env.reset(scenario=case.scenario)
            assert teacher_route_reachable(env, repair["actions"])
    finally:
        env.close()


@pytest.mark.parametrize("family", ["wait_timing", "enemy_stomp"])
def test_online_losses_train_tactics_and_skill_for_added_families(family):
    from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage
    from retroagi.stages.block_smb.train import collect_trajectory, compute_block_smb_losses

    torch.manual_seed(5)
    model = make_block_smb_model(config())
    case = sample(family)
    stage = BlockSMBStage(
        scenario=case.scenario,
        vision=StaticBlockVision(),
        observation_config=BlockSMBObservationConfig(
            motion_observations=True, hazard_observations=True
        ),
    )
    try:
        trajectory = collect_trajectory(
            model,
            stage,
            case.scenario_id,
            rollout_steps=320,
            seed=0,
            deterministic=True,
            device=torch.device("cpu"),
            use_oracle_actions=True,
            adaptive_duration_control=False,
            walk_duration_primitives=False,
        )
        assert trajectory.success
        labels = [t for t in trajectory.transitions if t.tactic_target >= 0]
        assert labels
        losses = compute_block_smb_losses(
            model, trajectory.transitions, config(), torch.device("cpu")
        )
        assert losses["tactic_supervised_steps"].item() == len(labels)
        (gradient,) = torch.autograd.grad(
            losses["loss_tactic"],
            model.tactics_network.stance_head.weight,
            retain_graph=True,
        )
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        gradients = torch.autograd.grad(
            losses["loss_tactic_action"],
            tuple(model.agent.transformer_A.parameters()),
            allow_unused=True,
        )
        assert any(
            g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients
        )
    finally:
        stage.env.close()


def test_evaluation_separates_family_tactics_from_piranha(monkeypatch):
    from retroagi.stages.block_smb import train
    from retroagi.stages.block_smb.monte_carlo import BlockSMBMonteCarloSampleSet

    cases = [
        sample_block_smb_monte_carlo_scenario(
            family=family, difficulty="medium", split="validation", seed=101, sample_index=0
        )
        for family in ("wait_timing", "enemy_stomp")
    ]
    sample_set = BlockSMBMonteCarloSampleSet(
        schema_version=cases[0].schema_version,
        distribution_id=cases[0].distribution_id,
        split="validation",
        seed=0,
        samples=tuple(cases),
    )
    monkeypatch.setattr(train, "sample_block_smb_monte_carlo_split", lambda **kwargs: sample_set)
    cfg = replace(config(), evaluation_max_steps=12)
    evaluation = train.evaluate_block_smb_monte_carlo(
        make_block_smb_model(cfg),
        cfg,
        split="validation",
        sample_count=2,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
    )
    assert set(evaluation["tactics_by_family"]) == {"wait_timing", "enemy_stomp"}
    for metrics in evaluation["tactics_by_family"].values():
        assert metrics["decisions"] > 0
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["action_agreement"] <= 1
    assert evaluation["piranha_tactics"]["decisions"] == 0
    assert evaluation["piranha_tactics"]["accuracy"] is None
