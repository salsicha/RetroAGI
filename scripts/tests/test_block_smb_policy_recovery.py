"""Real bridge/pipe failure suffixes, observation consistency, and retention."""

from dataclasses import replace
from functools import lru_cache

import pytest
import torch

from retroagi.core.smb_geometry import FEATURE_NAMES, geometry_features
from retroagi.stages.block_smb.demonstrations import (
    collect_demonstrations,
    demonstration_rows,
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


@pytest.mark.parametrize("departure", [37, 38])
def test_early_bridge_departure_repairs_wait_for_the_robust_window(departure):
    # Frame 37 is physically impossible; frame 38 is the thin opening edge,
    # where only the longest hold lands. Both repairs wait, then depart.
    case = sample("bridge_mount", "easy", 666)
    actions = [0] * departure + [2] * 32 + [1] * 100
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    assert all(r["supervision_start_frame"] == departure for r in repairs)
    assert all(r["recovery_reason"] == "takeoff" for r in repairs)
    for repair in repairs:
        start = repair["supervision_start_frame"]
        assert repair["actions"][:start] == actions[:start]
        assert repair["actions"][start] == 0
        assert repair["actions"].index(2) >= departure + 10
        assert repair["actions"].count(2) == 32  # The longest certified hold.


def test_waiting_past_the_robust_window_is_repaired():
    case = sample("bridge_mount", "easy", 666)
    actions = [0] * 240
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    first = repairs[0]
    assert first["recovery_reason"] == "departure_window"
    assert first["actions"][first["supervision_start_frame"]] == 2


# Stomps the first enemy, lands, then walks into the first pipe until timeout.
POST_STOMP_STALL = [1] * 25 + [2] * 20 + [1] * 196


def test_post_stomp_pipe_stall_is_repaired_and_only_suffix_is_supervised():
    case = sample("chained_obstacles", "easy", 333)
    actions = POST_STOMP_STALL
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
    # The failed prefix is kept only as unsupervised context for memory.
    assert len(data.action) == len(repair["actions"])
    assert int(data.context.sum()) == start and not data.actor_mask[:start].any()
    data = demonstration_rows(data, ~data.context)
    assert len(data.action) == len(repair["actions"]) - start
    assert data.recovery.all()
    assert int(data.action[0]) == 2 and data.actor_mask[0]
    assert data.valid_durations[0].any()


# An enemy and two pipes, the second taller (the hand-made chained_obstacles
# layout before the chained families were composed from sections).
TWO_PIPES = {
    "world_width": 512,
    "mario": [20, 204],
    "platforms": [[0, 220, 512, 20], [178, 178, 28, 42], [317, 162, 32, 58]],
    "platform_kinds": ["ground", "pipe", "pipe"],
    "enemies": [
        [97, 206, 97, 97, 0],
        {"x": 386, "y": 206, "patrol_min": 374, "patrol_max": 414, "speed": 0.484},
    ],
    "coins": [[132, 190, 10, 10], [220, 176, 10, 10], [356, 156, 10, 10], [430, 190, 10, 10]],
    "goal": [482, 200, 16, 20],
    "reward_goal_distance_shaping": 2.0,
    "goal_requires_support": True,
}


def test_short_second_pipe_jump_gets_certified_longer_hold_and_complete_recovery():
    scenario = TWO_PIPES
    # Stomp, mount and leave the first pipe, then a 6-frame jump at the taller second pipe.
    runs = [(1, 26), (2, 20), (1, 46), (2, 20), (1, 46), (2, 6), (1, 160)]
    actions = [action for action, count in runs for _ in range(count)]
    repairs = repair_policy_actions(scenario, actions)
    assert_completed(scenario, repairs)
    assert any(
        r["recovery_reason"] == "duration" and r["supervision_start_frame"] == 158 for r in repairs
    )


def test_stomp_repairs_cover_later_interceptions_without_losing_first_departure():
    case = sample("enemy_stomp", "medium", 259)
    runs = [
        (2, 12),
        (1, 20),
        (2, 1),
        (1, 17),
        (3, 1),
        (4, 15),
        (3, 18),
        (1, 1),
        (2, 12),
        (1, 19),
        (3, 1),
        (4, 16),
        (3, 18),
        (1, 4),
        (2, 10),
        (1, 155),
    ]
    actions = [action for action, count in runs for _ in range(count)]
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    assert len(repairs) == 3
    starts = [r["supervision_start_frame"] for r in repairs]
    assert 0 in starts and max(starts) >= 85
    for repair in repairs:
        start = repair["supervision_start_frame"]
        assert repair["actions"][:start] == actions[:start]


def test_plant_retry_repairs_preserve_first_mistake_with_bounded_capacity():
    case = sample("piranha_avoidance", "medium", 824)
    runs = [(1, 8), (2, 9), (1, 21), (2, 2), (1, 18), (3, 16), (1, 11), (2, 9), (1, 9)]
    actions = [action for action, count in runs for _ in range(count)]
    repairs = repair_policy_actions(case.scenario, actions)
    assert_completed(case.scenario, repairs)
    starts = [r["supervision_start_frame"] for r in repairs]
    assert len(repairs) == 3 and 8 in starts and max(starts) >= 57
    pair = repair_policy_actions(case.scenario, actions, max_repairs=2)
    assert len(pair) == 2 and pair[0]["supervision_start_frame"] == 8
    assert pair[1]["supervision_start_frame"] >= 57


@pytest.mark.parametrize("left", [False, True])
def test_next_platform_retains_blocking_pipe_at_wall_contact(left):
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario=dict(
                world_width=400,
                mario=[90 if not left else 130, 204],
                platforms=[[0, 220, 400, 20], [100, 170, 30, 50], [260, 160, 30, 60]],
                goal=[20 if left else 360, 200, 16, 20],
                task_direction=-1 if left else 1,
            )
        )
        env._terrain_left = left
        features = geometry_features(env)["state"]
        assert features[FEATURE_NAMES.index("next_platform_dx")] == 0
        assert features[FEATURE_NAMES.index("next_platform_dy")] == pytest.approx(-50 / 240)
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
    monkeypatch.setattr(
        demonstrations, "build_balanced_demonstrations", lambda *args, **kwargs: data
    )

    def update(*args, recovery_records, **kwargs):
        recovery_records.append({"from_policy": True})
        return {"episodes": 1}, None

    def collect(records, *args, **kwargs):
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
    repairs = repair_policy_actions(case.scenario, POST_STOMP_STALL)
    case = replace(case, oracle=next(r for r in repairs if r["recovery_reason"] == "stall"))
    config = tiny_config(walk_duration_primitives=False)
    single = collect_demonstrations([(11, case)], config, SupportVision)
    paired = collect_demonstrations([(11, case), (11, case)], config, SupportVision)
    n = len(single.action)
    assert single.action[-1] == 1 and single.actor_mask[-1]
    assert torch.equal(paired.next_c[n - 1], single.next_c[-1])
    assert not torch.equal(paired.next_c[n - 1], paired.c[n])


# Easy stair failure: two successful jumps, landing on the second step at frame
# 80, ready for the final riser. The policy then walks against the wall or waits.
STAIR_ARRIVAL = [2] * 28 + [1] * 16 + [2] * 18 + [1] * 19
ARRIVAL = len(STAIR_ARRIVAL)
# A 10-frame hop released in flight lands back on the floor at frame 33; the
# teacher then walks seven frames before climbing the stairs.
HOP_LANDING = 33
STAIR_AFTER_HOP = (
    [2] * 10
    + [1] * (HOP_LANDING + 7 - 10)
    + [2] * 26
    + [1] * 16
    + [2] * 18
    + [1] * 19
    + [2] * 16
    + [1] * 60
)


@pytest.mark.parametrize(
    "tail,reason",
    [
        ([1], "landing_recovery"),
        ([1] * 120, "pause_recovery"),
        ([0] * 120, "pause_recovery"),
        # A one-frame hop against the wall fails to mount the last step.
        ([1] * 12 + [2] + [1] * 120, "retry_recovery"),
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
        r
        for r in repairs
        if r["recovery_reason"] == reason and r["supervision_start_frame"] >= ARRIVAL
    )
    start = repair["supervision_start_frame"]
    assert start >= ARRIVAL
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
        assert all(r["supervision_start_frame"] >= ARRIVAL for r in repairs)
        assert start >= ARRIVAL + 17
    if reason == "retry_recovery":
        pause = next(r for r in repairs if r["recovery_reason"] == "pause_recovery")
        assert pause["supervision_start_frame"] >= start + 16


def test_stair_repairs_respect_a_smaller_budget_and_zero_budget():
    case = sample("stair_climb", "easy", 60)
    actions = STAIR_ARRIVAL + [1] * 120
    repairs = repair_policy_actions(case.scenario, actions, max_repairs=1)
    assert_completed(case.scenario, repairs)
    assert len(repairs) == 1
    assert repairs[0]["supervision_start_frame"] == ARRIVAL
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


def test_stair_landing_after_air_release_is_a_fresh_actor_decision():
    from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
    from retroagi.stages.block_smb.primitive_execution import BlockSMBPrimitiveExecutor

    case = sample("stair_climb", "easy", 60)
    case = replace(case, oracle={**case.oracle, "actions": STAIR_AFTER_HOP})
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
            support = "ground" if env.mario["on_ground"] else "air"
            if executor.resolve_landing(support_override=support):
                break
            execution = executor.execute(
                2,
                support_override=support,
                enemy_contact_override=False,
            )
            env.step(execution.action)
        else:
            pytest.fail("The first stair jump did not land")
        assert frame == HOP_LANDING
        restarted = executor.execute(2, support_override="ground", enemy_contact_override=False)
        assert restarted.action == 2 and restarted.started
        # The teacher chose to walk here after releasing in flight. Those
        # frames are decisions; the executor also permits an immediate jump.
        assert data.action[frame : frame + 2].tolist() == [1, 1]
        assert data.actor_mask[frame : frame + 2].all()
        assert not data.forced_release[frame : frame + 2].any()
        jump = frame + 7
        assert data.action[frame:jump].tolist() == [1] * 7
        assert data.action[jump] == 2 and data.actor_mask[jump]
    finally:
        env.close()


def test_stair_release_mask_migrates_caches_without_crossing_episode_boundaries():
    from dataclasses import fields

    from retroagi.stages.block_smb.demonstrations import without_walk_commitments
    from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES

    case = sample("stair_climb", "easy", 60)
    actions = STAIR_AFTER_HOP
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
    hop = slice(HOP_LANDING - 1, HOP_LANDING + 2)
    separated = replace(legacy, **{f.name: getattr(legacy, f.name)[hop] for f in fields(legacy)})
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
    # A wasted early hop, then a walk into the plant's pipe until timeout.
    actions = [0] * 8 + [2] + [1] * 90
    repairs = repair_policy_actions(case.scenario, actions, max_repairs=1)
    assert len(repairs) == 1
    assert repairs[0]["supervision_start_frame"] > 20
    assert_completed(case.scenario, repairs)
    start = repairs[0]["supervision_start_frame"]
    assert repairs[0]["actions"][:start] == actions[:start]
