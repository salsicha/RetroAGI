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


# Epoch-25 easy stair failure: two successful jumps, followed by a final-riser
# arrival at frame 45. The policy then walked against the wall until timeout.
STAIR_ARRIVAL = [2] * 10 + [1] * 17 + [2] * 5 + [1] * 13


@pytest.mark.parametrize(
    "tail,reason",
    [
        ([1], "landing_recovery"),
        ([1] * 120, "pause_recovery"),
        ([0] * 120, "pause_recovery"),
        ([1] * 8 + [2] + [1] * 120, "retry_recovery"),
    ],
)
def test_stair_final_arrival_pause_and_failed_retry_get_successful_suffixes(tail, reason):
    from retroagi.core.smb_coaching import training_target

    case = sample("stair_climb", "easy", 60)
    actions = STAIR_ARRIVAL + tail
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    assert len(repairs) <= 3
    repair = next(
        r for r in repairs if r["recovery_reason"] == reason and r["supervision_start_frame"] >= 45
    )
    start = repair["supervision_start_frame"]
    assert start >= 45
    assert repair["actions"][:start] == actions[:start]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        from retroagi.stages.block_smb.primitive_execution import JumpReleaseState

        release = JumpReleaseState()
        for action in actions[:start]:
            _, _, _, _, info = env.step(action)
            release.observe(env, action, info)
        assert env.mario["on_ground"]
        assert training_target(env).platform_index == 3
        assert (
            repair["actions"][start : start + release.remaining]
            == [release.action] * release.remaining
        )
        assert repair["actions"][start + release.remaining] == 2
    finally:
        env.close()
    if reason == "pause_recovery":
        # The short settling pause and the extended wall stall both survive
        # the cap; repairs at earlier steps cannot crowd out the last riser.
        assert {r["recovery_reason"] for r in repairs} == {
            "landing_recovery",
            "stall",
            "pause_recovery",
        }
        assert all(r["supervision_start_frame"] >= 45 for r in repairs)
        assert start >= 62
    if reason == "retry_recovery":
        pause = next(r for r in repairs if r["recovery_reason"] == "pause_recovery")
        assert pause["supervision_start_frame"] >= start + 16


def test_stair_repairs_respect_a_smaller_budget_and_zero_budget():
    case = sample("stair_climb", "easy", 60)
    actions = STAIR_ARRIVAL + [1] * 120
    repairs = repair_policy_actions(case.scenario, actions, max_repairs=1)
    assert_completed(case.scenario, repairs)
    assert len(repairs) == 1
    assert repairs[0]["supervision_start_frame"] == 45
    assert repair_policy_actions(case.scenario, actions, max_repairs=0) == []


def test_numbered_epoch_collects_bounded_train_stair_trajectories():
    from retroagi.stages.block_smb.train import (
        make_block_smb_model,
        make_block_smb_optimizer,
        train_block_smb_epoch,
    )

    case = sample_block_smb_monte_carlo_scenario(
        split="train", seed=914601, sample_index=0, family="stair_climb", difficulty="easy"
    )
    config = tiny_config(
        episodes_per_epoch=2,
        rollout_steps=2,
        policy_recovery_samples_per_bin=1,
        use_oracle_actions=False,
    )
    model = make_block_smb_model(config)
    optimizer = make_block_smb_optimizer(model, config)
    records = []
    train_block_smb_epoch(
        model,
        optimizer,
        [(case.scenario_id, case.scenario)],
        config,
        0,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
        recovery_records=records,
    )
    assert len(records) == 1
    assert records[0]["scenario_id"] == case.scenario_id
    assert records[0]["actions"]
    assert records[0]["seed"] == config.seed


def test_stair_landing_release_frames_are_not_learned_as_walk_choices():
    from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
    from retroagi.stages.block_smb.primitive_execution import BlockSMBPrimitiveExecutor

    case = sample("stair_climb", "easy", 60)
    actions = [2] * 10 + [1] * 17 + [2] * 7 + [1] * 16 + [2] * 9 + [1] * 27
    case = replace(case, oracle={**case.oracle, "actions": actions})
    data = collect_demonstrations(
        [(BLOCK_SMB_MC_FAMILIES.index("stair_climb"), case)],
        tiny_config(walk_duration_primitives=False),
        StaticBlockVision,
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        executor = BlockSMBPrimitiveExecutor(env, default_hold_frames=10, walk_primitives=False)
        for frame in range(40):
            execution = executor.execute(
                2,
                support_override="ground" if env.mario["on_ground"] else "air",
                enemy_contact_override=False,
            )
            env.step(execution.action)
            if execution.landed:
                break
        else:
            pytest.fail("The first stair jump did not land")
        assert frame == 25
        assert execution.action == 1
        suppressed = executor.execute(2, support_override="ground", enemy_contact_override=False)
        assert suppressed.action == 1 and not suppressed.started
        # Walking on these two frames is imposed by the executor even when
        # the policy requests jump. It must not compete with the next jump's
        # positive actor label during demonstration rehearsal.
        assert data.action[frame : frame + 2].tolist() == [1, 1]
        assert not data.actor_mask[frame : frame + 2].any()
        assert data.action[frame + 2] == 2 and data.actor_mask[frame + 2]
    finally:
        env.close()


def test_stair_release_mask_migrates_caches_without_crossing_episode_boundaries():
    from dataclasses import fields

    from retroagi.stages.block_smb.demonstrations import without_walk_commitments
    from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES

    case = sample("stair_climb", "easy", 60)
    actions = [2] * 10 + [1] * 17 + [2] * 7 + [1] * 16 + [2] * 9 + [1] * 27
    data = collect_demonstrations(
        [(BLOCK_SMB_MC_FAMILIES.index("stair_climb"), replace(case, oracle={"actions": actions}))],
        tiny_config(walk_duration_primitives=False),
        StaticBlockVision,
    )
    legacy = replace(
        data,
        actor_mask=data.actor_mask.clone(),
        forced_release=torch.zeros_like(data.forced_release),
    )
    legacy.actor_mask[legacy.motor_action == 1] = True
    migrated = without_walk_commitments(legacy, [0])
    assert torch.equal(migrated.actor_mask, data.actor_mask)
    assert torch.equal(without_walk_commitments(migrated, [0]).actor_mask, data.actor_mask)
    # A new recovery episode may start after frame zero; its walking choices
    # must not inherit a jump commitment from the preceding episode.
    separated = replace(legacy, **{f.name: getattr(legacy, f.name)[24:27] for f in fields(legacy)})
    assert separated.motor_action.tolist() == [2, 1, 1]
    assert without_walk_commitments(separated, [0, 1]).actor_mask[1:].all()


def test_successful_rollouts_do_not_use_failure_recovery_budget(monkeypatch):
    from retroagi.stages.block_smb import train

    case = sample_block_smb_monte_carlo_scenario(
        split="train", seed=914601, sample_index=0, family="piranha_avoidance", difficulty="easy"
    )
    config = tiny_config(episodes_per_epoch=3, rollout_steps=2, policy_recovery_samples_per_bin=1)
    model = train.make_block_smb_model(config)
    optimizer = train.make_block_smb_optimizer(model, config)
    collect = train.collect_trajectory
    calls = []

    def controlled_result(*args, **kwargs):
        trajectory = collect(*args, **kwargs)
        # Exercise success followed by failure in the same family/difficulty bin.
        last = trajectory.transitions[-1]
        last.done = True
        last.info = {**last.info, "goal_reached": not calls}
        calls.append(kwargs["seed"])
        return trajectory

    monkeypatch.setattr(train, "collect_trajectory", controlled_result)
    records = []
    train.train_block_smb_epoch(
        model,
        optimizer,
        [(case.scenario_id, case.scenario)],
        config,
        0,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
        recovery_records=records,
    )
    assert len(records) == 1 and records[0]["seed"] == config.seed + 1


def test_plant_recovery_retains_later_attempt_with_small_budget():
    case = sample("piranha_avoidance", "easy", 810)
    actions = [0] * 8 + [2] + [1] * 65
    repairs = repair_policy_actions(case.scenario, actions, max_repairs=1)
    assert len(repairs) == 1
    assert repairs[0]["supervision_start_frame"] > 20
    assert_completed(case.scenario, repairs)
    start = repairs[0]["supervision_start_frame"]
    assert repairs[0]["actions"][:start] == actions[:start]
