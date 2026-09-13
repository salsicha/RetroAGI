"""Real bridge/pipe failure suffixes, observation consistency, and retention."""

from dataclasses import replace
from functools import lru_cache

import pytest
import torch

from retroagi.core.smb_geometry import geometry_features
from retroagi.stages.block_smb.demonstrations import (
    collect_demonstrations,
    demonstration_sample_weights,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    validate_block_smb_monte_carlo_oracle,
)
from retroagi.stages.block_smb.policy_recovery import (
    collect_policy_recovery,
    combine_demonstrations,
    repair_policy_actions,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@lru_cache(None)
def sample(family, difficulty, index):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=50000, sample_index=index, family=family, difficulty=difficulty
    )


def assert_completed(scenario, repairs):
    assert repairs
    for repair in repairs:
        assert validate_block_smb_monte_carlo_oracle(scenario, repair["actions"], max_steps=640)[
            "reachable"
        ]


def test_early_bridge_departure_gets_successful_opening_and_closing_continuations():
    case = sample("bridge_mount", "easy", 666)
    actions = [0] * 37 + [2] * 32 + [1] * 100
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    assert {r["closing_window"] for r in repairs} == {False, True}
    assert all(r["supervision_start_frame"] == 37 for r in repairs)
    for repair in repairs:
        start = repair["supervision_start_frame"]
        assert repair["actions"][:start] == actions[:start]
        assert repair["actions"][start] == 0  # Jumping here is physically impossible.


def test_post_stomp_pipe_stall_is_repaired_and_only_suffix_is_supervised():
    case = sample("chained_obstacles", "easy", 333)
    actions = [1] * 10 + [2] * 7 + [1] * 150
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    assert any(r["recovery_reason"] == "stall" for r in repairs)
    repair = next(r for r in repairs if r["recovery_reason"] == "stall")
    start = repair["supervision_start_frame"]
    assert start > 32  # The first enemy was stomped before this pipe failure.
    assert repair["actions"][:start] == actions[:start]
    config = tiny_config(walk_duration_primitives=False)
    repaired = replace(case, oracle=repair)
    data = collect_demonstrations([(11, repaired)], config, StaticBlockVision)
    assert len(data.action) == len(repair["actions"]) - start
    assert data.recovery.all()
    assert int(data.action[0]) == 2 and data.actor_mask[0]
    assert float(data.c[0, 23]) == pytest.approx(min(start / 200, 1))
    assert data.valid_durations[0].any()


def test_short_second_pipe_jump_gets_certified_longer_hold_and_complete_recovery():
    case = sample("chained_obstacles", "hard", 351)
    actions = [1] * 10 + [2] * 9 + [1] * 21 + [2] * 9 + [1] * 32 + [2] * 11 + [1] * 160
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    assert any(
        r["recovery_reason"] == "duration" and r["supervision_start_frame"] == 81 for r in repairs
    )


@pytest.mark.parametrize("left", [False, True])
def test_next_platform_retains_blocking_pipe_at_wall_contact(left):
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario=dict(
                world_width=400,
                mario=[86 if not left else 130, 204],
                platforms=[[0, 220, 400, 20], [100, 170, 30, 50], [260, 160, 30, 60]],
                goal=[20 if left else 360, 200, 16, 20],
                task_direction=-1 if left else 1,
            )
        )
        env._terrain_left = left
        features = geometry_features(env)["state_vec"]
        assert features[19] == 0
        assert features[20] == pytest.approx(-50 / 240)
    finally:
        env.close()


def test_repair_retention_balances_small_correction_set_against_large_original_set():
    case = sample("chained_obstacles", "easy", 333)
    config = tiny_config(walk_duration_primitives=False)
    original = collect_demonstrations([(11, case)], config, StaticBlockVision)
    # Same groups but very different row counts: recovery cannot disappear
    # merely because there are many more canonical demonstrations.
    repaired = replace(original, recovery=torch.ones_like(original.recovery))
    data = combine_demonstrations([original] * 8 + [repaired])
    weights = demonstration_sample_weights(data)
    decision = data.actor_mask
    for phase in data.phase.unique():
        for action in data.action[decision & (data.phase == phase)].unique():
            group = decision & (data.phase == phase) & (data.action == action)
            assert weights[group & data.recovery].sum() == pytest.approx(
                weights[group & ~data.recovery].sum().item()
            )
    assert weights.sum() == pytest.approx(1)


def test_held_out_policy_failures_cannot_enter_training_recovery_pool():
    case = sample("chained_obstacles", "easy", 333)
    with pytest.raises(ValueError, match="train split"):
        collect_policy_recovery(
            [dict(scenario=case.scenario, scenario_id=case.scenario_id, seed=0, actions=[1])],
            tiny_config(),
            StaticBlockVision,
        )


def test_best_checkpoint_contains_the_current_evaluated_epoch(monkeypatch, tmp_path):
    from retroagi.stages.block_smb import train

    def update(model, *args, **kwargs):
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return {"episodes": 1}, None

    evaluation = dict(
        mean_return=0,
        success_rate=1,
        tuning_metrics=dict(threshold_pass_rate=1, score=1),
        monte_carlo_validation=dict(
            primitive_metrics=dict(
                jump_spans=1,
                landing_rate=1,
                duration_gap_mean=0,
            )
        ),
    )
    monkeypatch.setattr(train, "train_block_smb_epoch", update)
    monkeypatch.setattr(train, "evaluate_block_smb", lambda *args, **kwargs: evaluation)
    config = tiny_config(
        generated_scenarios=0,
        checkpoint_path=tmp_path / "policy.pth",
        save_checkpoints=True,
        evaluation_interval_epochs=1,
    )
    train.train_and_evaluate_block_smb(config, vision_factory=StaticBlockVision)
    current = torch.load(config.checkpoint_path, weights_only=False)
    best = torch.load(tmp_path / "policy.best_primitives.pth", weights_only=False)
    assert best["epoch"] == current["epoch"] == 1
    assert best["global_step"] == current["global_step"] == 1
    for key, value in current["states"]["model"].items():
        assert torch.equal(value, best["states"]["model"][key])


def test_numbered_epochs_rehearse_actual_policy_repairs_with_bounded_retention(monkeypatch):
    from retroagi.stages.block_smb import demonstrations, policy_recovery, train

    base = tiny_config(
        epochs=4,
        generated_scenarios=0,
        numeric_policy_learning_rate=0.003,
        policy_recovery_samples_per_bin=1,
    )
    config = replace(
        base,
        demonstration_rehearsal_updates=1,
        ablation=replace(base.ablation, recurrent_state_enabled=False),
    )
    case = sample("chained_obstacles", "easy", 333)
    data = collect_demonstrations([(11, case)], config, StaticBlockVision)
    repaired = replace(data, recovery=torch.ones_like(data.recovery))
    monkeypatch.setattr(demonstrations, "build_balanced_demonstrations", lambda *args: data)

    def update(*args, recovery_records, **kwargs):
        recovery_records.append({"from_policy": True})
        return {"episodes": 1}, None

    def collect(records, *args):
        assert records == [{"from_policy": True}]
        return repaired

    seen = []

    def fit(model, optimizer, batch, **kwargs):
        seen.append((len(batch.action), int(batch.recovery.sum())))
        return 0.0

    monkeypatch.setattr(train, "train_block_smb_epoch", update)
    monkeypatch.setattr(policy_recovery, "collect_policy_recovery", collect)
    monkeypatch.setattr(demonstrations, "fit_demonstrations", fit)
    train.train_and_evaluate_block_smb(config, vision_factory=StaticBlockVision)
    n = len(data.action)
    assert seen == [(2 * n, n), (3 * n, 2 * n), (4 * n, 3 * n), (4 * n, 3 * n)]


def test_impossible_local_pipe_takeoff_has_no_invented_duration_target():
    from scripts.tests.test_stomp_coaching import held_policy
    from scripts.tests.test_tall_pipe_traversal import rollout

    case = sample("chained_obstacles", "easy", 333)
    case = replace(case, scenario=dict(case.scenario, enemies=[], mario=[20, 200]))
    trajectory = rollout(case, held_policy(7), steps=20)
    assert trajectory.transitions[0].info["primitive_unreachable"]
    assert trajectory.transitions[0].info["jump_overreach"]
    assert not any("primitive_target_hold" in step.info for step in trajectory.transitions)


def test_suffix_walk_targets_do_not_cross_into_the_next_recovery_episode():
    class SupportVision(StaticBlockVision):
        def encode(self, observation):
            return replace(super().encode(observation), support_logits=torch.zeros(1, 3))

    case = sample("chained_obstacles", "easy", 333)
    repairs = repair_policy_actions(case.scenario, [1] * 10 + [2] * 7 + [1] * 150)
    case = replace(case, oracle=next(r for r in repairs if r["recovery_reason"] == "stall"))
    config = tiny_config(walk_duration_primitives=False)
    single = collect_demonstrations([(11, case)], config, SupportVision)
    paired = collect_demonstrations([(11, case), (11, case)], config, SupportVision)
    n = len(single.action)
    assert single.action[-1] == 1 and single.actor_mask[-1]
    assert torch.equal(paired.next_c[n - 1], single.next_c[-1])
    assert not torch.equal(paired.next_c[n - 1], paired.c[n])
