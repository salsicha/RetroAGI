"""The batched teaching path must preserve the live task and decision contract."""

import pytest
import torch

from retroagi.stages.block_smb.demonstrations import (
    collect_demonstrations,
    demonstration_sample_weights,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from scripts.block_smb_family_learning import samples
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_committed_gap_keeps_its_takeoff_goal():
    case = samples("pit_leap", 13, "train", 1)[0]
    data = collect_demonstrations([(0, case)], tiny_config(), StaticBlockVision)
    jumping = data.motor_action == 2
    assert jumping.any()
    assert torch.all(data.goal[jumping] == data.goal[jumping][0])
    assert data.goal[jumping][0, 0] == 1


def test_stomp_duration_labels_require_actual_stomp_credit():
    case = samples("stomp_mount", 13, "train", 1)[0]
    data = collect_demonstrations([(0, case)], tiny_config(), StaticBlockVision)
    allowed = data.valid_durations[0].nonzero().flatten() + 1
    assert len(allowed) > 0
    env = MarioScenarioEnv()
    try:
        for hold in allowed.tolist():
            env.reset(scenario=case.scenario)
            for frame in range(80):
                _, _, done, truncated, info = env.step(2 if frame < hold else 1)
                if done or truncated:
                    break
            assert env._stomp_credited and env._goal_credited
    finally:
        env.close()


def test_sampling_balances_decisions_instead_of_held_button_frames():
    case = samples("tall_pipe_jump", 13, "train", 1)[0]
    data = collect_demonstrations([(0, case)], tiny_config(), StaticBlockVision)
    weights = demonstration_sample_weights(data)
    walk = weights[data.actor_mask & (data.action == 1)].sum()
    jump = weights[data.actor_mask & (data.action == 2)].sum()
    assert torch.allclose(walk, jump)
    assert weights.sum().item() == pytest.approx(1)
    assert all(data.valid_durations.any(dim=-1))


def test_batched_forcing_preserves_each_actor_distribution_and_motor_intent():
    from retroagi.stages.block_smb.train import make_block_smb_model

    model = make_block_smb_model(tiny_config()).eval()
    agent = model.agent
    from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC

    a = torch.zeros(2, BLOCK_SMB_SPEC.seq_len_a, dtype=torch.long)
    b = torch.zeros(2, BLOCK_SMB_SPEC.seq_len_b, dtype=torch.long)
    c = torch.zeros(2, BLOCK_SMB_SPEC.seq_len_c)
    with torch.no_grad():
        expected = agent(a, b, c)[0]
        forced = agent(a, b, c, forced_action=torch.tensor([1, 2]))[0]
    assert torch.equal(agent.last_unforced_logits_a, expected)
    assert forced[:, -1].argmax(-1).tolist() == [1, 2]
    with pytest.raises(ValueError, match="one action per batch"):
        agent(a, b, c, forced_action=torch.tensor([1]))


def test_new_numeric_paths_are_zero_residual_for_existing_checkpoints():
    from retroagi.core.models import LEVEL_B_PRIMITIVE_ALLOWED_MISSING_PREFIXES
    from retroagi.stages.block_smb.train import make_block_smb_model

    model = make_block_smb_model(tiny_config()).eval()
    previous = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith(("agent.action_state_head.", "agent.duration_state_head."))
    }
    loaded = model.load_state_dict(previous, strict=False)
    assert loaded.missing_keys
    assert all(
        k.startswith(LEVEL_B_PRIMITIVE_ALLOWED_MISSING_PREFIXES) for k in loaded.missing_keys
    )
    for head in (model.agent.action_state_head, model.agent.duration_state_head):
        assert torch.count_nonzero(head[-1].weight) == 0
        assert torch.count_nonzero(head[-1].bias) == 0


def test_production_training_runs_bootstrap_and_balanced_rehearsal(monkeypatch):
    from dataclasses import replace

    from retroagi.stages.block_smb import demonstrations
    from retroagi.stages.block_smb.train import train_and_evaluate_block_smb

    base = tiny_config(generated_scenarios=0)
    config = replace(
        base,
        ablation=replace(base.ablation, recurrent_state_enabled=False),
        numeric_policy_learning_rate=0.003,
        demonstration_bootstrap_updates=2,
        demonstration_rehearsal_updates=2,
        mastery_gated_schedule=True,
    )
    data = collect_demonstrations(
        [(0, samples("flat_run", 7, "train", 1)[0])], config, StaticBlockVision
    )
    monkeypatch.setattr(demonstrations, "build_balanced_demonstrations", lambda *args: data)
    result = train_and_evaluate_block_smb(config, vision_factory=StaticBlockVision)
    assert result["history"][0]["demonstration_rehearsal_updates"] == 2
    assert result["history"][0]["episodes"] == 1
    assert result["history"][0]["demonstration_rehearsal_loss"] >= 0


def test_motion_observations_expose_patrol_direction_and_keep_legacy_layout():
    import copy

    import numpy as np

    from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage

    sample = samples("stomp_mount", 101, "train", 3)[2]
    values = []
    for direction in (-1, 1):
        scenario = copy.deepcopy(sample.scenario)
        scenario["enemies"][0][-1] = direction
        stage = BlockSMBStage(
            scenario=scenario,
            vision=StaticBlockVision(),
            observation_config=BlockSMBObservationConfig(motion_observations=True),
        )
        try:
            obs = stage.reset()
            batch = stage.encode_observation(obs)
            start, end = batch.metadata["vision_fusion"]["c_state"]
            assert end - start == 35
            values.append(
                (stage.last_info["state_vec"].copy(), stage.state_features(stage.last_info))
            )
            stage.observation_config = BlockSMBObservationConfig()
            legacy = stage.encode_observation(obs)
            lo, hi = legacy.metadata["vision_fusion"]["c_state"]
            assert hi - lo == 27
            torch.testing.assert_close(legacy.src_c[:, :hi], batch.src_c[:, :hi])
        finally:
            stage.env.close()
    np.testing.assert_array_equal(values[0][0], values[1][0])
    assert values[0][1][27] < 0 < values[1][1][27]


def test_checkpoint_rejects_a_different_observation_layout(tmp_path):
    from retroagi.stages.block_smb.train import (
        make_block_smb_model,
        make_block_smb_optimizer,
        restore_block_smb_checkpoint,
        save_block_smb_checkpoint,
    )

    config = tiny_config()
    model = make_block_smb_model(config)
    optimizer = make_block_smb_optimizer(model, config)
    path = tmp_path / "policy.pth"
    save_block_smb_checkpoint(
        path, model, optimizer, epoch=0, global_step=0, config=config, metrics={}
    )
    with pytest.raises(ValueError, match="motion-observation layout"):
        restore_block_smb_checkpoint(path, model, motion_observations=True)


def test_local_stomp_bounce_clears_goal_and_receives_no_actor_credit():
    from dataclasses import replace

    case = samples("enemy_stomp", 101, "train", 1)[0]
    # Use a local traversal family with identical physics to exercise the
    # collision reward path (ordinary hops need not emit stomp geometry).
    case = replace(case, family="enemy_hop")
    data = collect_demonstrations([(0, case)], tiny_config(), StaticBlockVision)
    recovery = (~data.actor_mask) & (data.motor_action == 1) & (data.c[:, 13] == 0)
    assert recovery.any()
    assert torch.all(data.goal[recovery] == 0)


def test_frozen_vision_cache_preserves_fresh_symbolic_state():
    from retroagi.stages.block_smb.adapter import BlockSMBStage

    class CountingVision(StaticBlockVision):
        calls = 0

        def encode(self, observation):
            self.calls += 1
            return super().encode(observation)

    vision = CountingVision()
    stage = BlockSMBStage(scenario=samples("flat_run", 7, "train", 1)[0].scenario, vision=vision)
    try:
        obs = stage.reset()
        before = stage.encode_observation(obs)
        info = dict(stage.last_info)
        info["state_vec"] = info["state_vec"].copy()
        info["state_vec"][0] += 0.1
        after = stage.encode_observation(obs, info)
        assert vision.calls == 1
        start, _ = after.metadata["vision_fusion"]["c_state"]
        assert after.src_c[0, start] != before.src_c[0, start]
        stage.reset()
        stage.encode_observation(obs)
        assert vision.calls == 2
    finally:
        stage.env.close()


def test_platform_hop_goal_requires_supported_landing():
    case = samples("platform_hop", 99173, "validation", 9)[3]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        import pygame

        airborne_goal_contact = False
        for frame in range(80):
            _, _, done, truncated, info = env.step(2 if frame < 10 else 1)
            m = env.mario
            if not m["on_ground"] and pygame.Rect(m["x"], m["y"], m["w"], m["h"]).colliderect(
                env.goal
            ):
                airborne_goal_contact = True
                assert not env._goal_credited and not done
            if done or truncated:
                break
        assert airborne_goal_contact
        assert env._goal_credited and env.mario["on_ground"]
        env.reset(scenario=case.scenario)
        for action in case.oracle["actions"]:
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert env._goal_credited and env.mario["on_ground"]
        assert env.mario["_platform"]["rect"].top == 198
    finally:
        env.close()


def test_bridge_exit_demonstrations_keep_committed_exit_goal():
    for case in samples("wait_timing", 101, "train", 3):
        data = collect_demonstrations([(0, case)], tiny_config(), StaticBlockVision)
        last_wait = int((data.action == 0).nonzero()[-1])
        assert (data.action[last_wait + 1 :] == 1).all()
        assert not data.goal[last_wait + 1 :].any()


def test_native_greedy_sampling_matches_two_pass_conditioning():
    from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
    from retroagi.stages.block_smb.train import make_block_smb_model

    torch.manual_seed(19)
    model = make_block_smb_model(tiny_config()).eval()
    a = torch.zeros(1, BLOCK_SMB_SPEC.seq_len_a, dtype=torch.long)
    b = torch.zeros(1, BLOCK_SMB_SPEC.seq_len_b, dtype=torch.long)
    c = torch.rand(1, BLOCK_SMB_SPEC.seq_len_c)
    with torch.no_grad():
        native = model(a, b, c, policy_action_mode="greedy")
        native_duration = model.last_motor_primitives.hold_duration_logits.clone()
        native_action = model.last_selected_action_id
        model.agent.supports_action_sampling = False
        baseline = model(a, b, c, policy_action_mode="greedy")
    assert native_action == model.last_selected_action_id
    for actual, expected in zip(native, baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        native_duration, model.last_motor_primitives.hold_duration_logits, rtol=0, atol=0
    )


def test_wait_and_walk_supervision_matches_committed_duration():
    data = collect_demonstrations(
        [(0, samples("wait_timing", 101, "train", 1)[0])], tiny_config(), StaticBlockVision
    )
    first_walk = int((data.action == 1).nonzero()[0])
    assert data.actor_mask[0]
    assert not data.actor_mask[1:first_walk].any()
    assert (data.duration[:first_walk] == data.duration[0]).all()
    assert int(data.duration[0] + 1) * 4 >= first_walk
    torch.testing.assert_close(
        data.next_c[:first_walk], data.next_c[first_walk - 1].expand(first_walk, -1)
    )


def test_cached_steady_migration_matches_fresh_collection(monkeypatch):
    from dataclasses import fields

    from retroagi.stages.block_smb import demonstrations

    cases = [(0, samples("wait_timing", 101, "train", 1)[0])]
    align = demonstrations.align_steady_demonstrations
    expected = collect_demonstrations(cases, tiny_config(), StaticBlockVision)
    with monkeypatch.context() as patch:
        patch.setattr(demonstrations, "align_steady_demonstrations", lambda data, *args, **kwargs: data)
        legacy = collect_demonstrations(cases, tiny_config(), StaticBlockVision)
    migrated = align(legacy, [0])
    for field in fields(expected):
        torch.testing.assert_close(getattr(migrated, field.name), getattr(expected, field.name))


def test_varied_stomp_demonstrations_cover_different_takeoff_states():
    from retroagi.stages.block_smb.demonstrations import varied_demonstration
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

    sample = sample_block_smb_monte_carlo_scenario(
        family="enemy_stomp", seed=101, split="train", sample_index=2, difficulty="hard"
    )
    starts = set()
    for seed in range(12):
        variant = varied_demonstration(sample, seed)
        if variant is not None:
            actions = variant.oracle["actions"]
            starts.add(actions.index(2))
    assert len(starts) >= 3


@pytest.mark.parametrize("walk_primitives", [True, False])
def test_batched_evaluation_matches_sequential_physics_frames(walk_primitives):
    from dataclasses import replace

    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import (
        block_smb_policy_scenario,
        collect_trajectory,
        make_block_smb_model,
    )
    from scripts.block_smb_batched_evaluation import evaluate_batched

    config = tiny_config(
        evaluation_max_steps=80,
        ranked_candidate_search=False,
        walk_duration_primitives=walk_primitives,
    )
    config = replace(config, ablation=replace(config.ablation, recurrent_state_enabled=False))
    torch.manual_seed(101)
    model = make_block_smb_model(config).eval()
    with torch.no_grad():
        model.agent.fc_out_A.weight.zero_()
        model.agent.fc_out_A.bias.fill_(-20)
        model.agent.fc_out_A.bias[1] = 0
        model.agent.fc_out_A.bias[2] = 0
        head = torch.nn.Linear(model.agent.action_state_head[0].in_features, 6)
        head.weight.zero_()
        head.bias.zero_()
        head.weight[2, 12] = 200
        head.bias[2] = -36
        model.agent.action_state_head = head
    cases = [
        samples(f, 99173, "validation", 1)[0]
        for f in ("stomp_mount", "tall_pipe_jump", "platform_hop")
    ]
    result = evaluate_batched(model, cases, config, StaticBlockVision, return_actions=True)
    for sample, actual in zip(cases, result["actions"]):
        stage = BlockSMBStage(
            scenario=block_smb_policy_scenario(sample.scenario, True), vision=StaticBlockVision()
        )
        try:
            with torch.no_grad():
                trajectory = collect_trajectory(
                    model,
                    stage,
                    sample.scenario_id,
                    rollout_steps=80,
                    seed=sample.sample_seed % (2**31),
                    deterministic=True,
                    device=torch.device("cpu"),
                    ablation=config.ablation,
                    walk_duration_primitives=config.walk_duration_primitives,
                )
            assert actual == [t.action for t in trajectory.transitions]
        finally:
            stage.env.close()


def test_fixed_commitment_learning_ignores_continuation_duration_labels():
    import copy
    from dataclasses import replace

    from retroagi.stages.block_smb.demonstrations import fit_demonstrations
    from retroagi.stages.block_smb.train import make_block_smb_model

    data = collect_demonstrations(
        [(0, samples("tall_pipe_jump", 13, "train", 1)[0])], tiny_config(), StaticBlockVision
    )
    changed = replace(data, valid_durations=data.valid_durations.clone())
    changed.valid_durations[~data.actor_mask] = torch.roll(
        changed.valid_durations[~data.actor_mask], 8, dims=-1
    )
    first = make_block_smb_model(tiny_config())
    second = copy.deepcopy(first)
    for model, batch in ((first, data), (second, changed)):
        torch.manual_seed(101)
        fit_demonstrations(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            batch,
            steps=1,
            seed=3,
            decision_durations_only=True,
        )
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_walk_replanning_preserves_jump_and_wait_commitments():
    from retroagi.core.actions import SMBParameterizedPrimitiveExecutor

    executor = SMBParameterizedPrimitiveExecutor(walk_primitives=False, default_hold_frames=8)
    walk = executor.execute(1, support_override="ground")
    assert walk.action == 1 and executor.committed_action is None
    jump = executor.execute(2, support_override="ground")
    assert jump.started and executor.committed_action == 2
    executor.reset()
    wait = executor.execute(0, support_override="ground")
    assert wait.started and executor.committed_action == 0


def test_frame_walk_cache_migration_matches_fresh_collection():
    from dataclasses import fields, replace

    from retroagi.stages.block_smb.demonstrations import without_walk_commitments

    config = tiny_config()
    cases = [(0, samples("tall_pipe_jump", 13, "train", 1)[0])]
    cached = collect_demonstrations(cases, config, StaticBlockVision)
    migrated = without_walk_commitments(cached)
    fresh = collect_demonstrations(
        cases, replace(config, walk_duration_primitives=False), StaticBlockVision
    )
    for field in fields(fresh):
        assert torch.equal(getattr(fresh, field.name), getattr(migrated, field.name))
    ground_walk = ((migrated.motor_action == 1) | (migrated.motor_action == 3)) & (
        migrated.c[:, 16] > 0.5
    )
    assert migrated.actor_mask[ground_walk].all()


def test_robust_gap_route_has_margin_at_its_takeoff():
    from retroagi.stages.block_smb.demonstrations import varied_demonstration
    from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds

    sample = samples("single_gap", 303, "train", 3)[2]
    route = varied_demonstration(sample, 303, robust=True)
    assert route is not None
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        for action in route.oracle["actions"]:
            if action == 2 and env.mario["on_ground"]:
                holds = safe_jump_holds(env, local_objective(env), 1)
                assert len(holds) >= 4
                break
            env.step(action)
        else:
            pytest.fail("No jump in the robust route")
    finally:
        env.close()


def test_demonstrated_action_sets_accept_alternatives_without_merging_goals():
    from types import SimpleNamespace

    from retroagi.stages.block_smb.demonstrations import demonstrated_action_sets

    data = SimpleNamespace(
        a=torch.zeros(4, 2),
        b=torch.zeros(4, 2),
        c=torch.zeros(4, 3),
        goal=torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
        action=torch.tensor([1, 2, 3, 4]),
        actor_mask=torch.tensor([True, True, True, False]),
    )
    allowed = demonstrated_action_sets(data)
    assert allowed[0].nonzero().flatten().tolist() == [1, 2]
    assert torch.equal(allowed[0], allowed[1])
    assert allowed[2].nonzero().flatten().tolist() == [3]
    assert not allowed[:3, 4].any()


def test_priority_sampling_preserves_family_action_balance():
    from retroagi.stages.block_smb.demonstrations import priority_sample_weights

    groups = torch.tensor([0, 0, 1, 1, 7, 7])
    base = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    weights = priority_sample_weights(base, groups, torch.tensor([1.0, 9.0, 2.0, 1.0, 3.0, 1.0]))
    assert weights[1] > weights[0]
    assert torch.allclose(
        torch.bincount(groups, weights=weights), torch.bincount(groups, weights=base)
    )


def test_pipe_above_continuous_floor_is_not_a_gap():
    from retroagi.stages.block_smb.local_traversal import local_objective

    sample = samples("full_smb_opening_proxy", 99173, "validation", 3)[1]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        pipe = env.platforms[1]
        env.mario.update(
            x=float(pipe["rect"].right - 3),
            y=float(pipe["rect"].top - env.mario["h"]),
            on_ground=True,
            _platform=pipe,
        )
        objective = local_objective(env)
        assert objective.kind != "gap"
        assert objective.kind in ("mount", "enemy")
    finally:
        env.close()


def test_duration_margin_prefers_safe_interiors_and_preserves_holes():
    from retroagi.stages.block_smb.demonstrations import interior_duration_targets

    allowed = torch.tensor(
        [
            [False, True, True, True, False, True, False],
            [False, False, True, False, False, False, False],
        ]
    )
    targets = interior_duration_targets(allowed)
    assert torch.allclose(targets.sum(-1), torch.ones(2))
    assert not targets[~allowed].any()
    assert targets[0, 2] > targets[0, 1] == targets[0, 3]
    assert targets[1, 2] == 1


def test_grouped_evaluation_preserves_per_family_denominators():
    from types import SimpleNamespace

    from scripts.block_smb_joint_learning import group_family_evaluation

    cases = [
        SimpleNamespace(family=f, difficulty_bin="hard", scenario_id=f"{f}_{i}")
        for f, n in [("one", 10), ("two", 2)]
        for i in range(n)
    ]
    result = group_family_evaluation(cases, {"failures": [{"scenario": "one_0"}]})
    assert result["one"]["counts"]["hard"] == [9, 10]
    assert result["two"]["counts"]["hard"] == [2, 2]
    assert result["one"]["passed"] and result["two"]["passed"]


def test_jump_labels_reject_an_unrecoverable_next_enemy_landing():
    from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds

    sample = samples("enemy_patrol", 99173, "validation", 30)[13]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        for _ in range(15):
            env.step(1)
        objective = local_objective(env)
        before = (env.mario["x"], env.mario["y"], env.steps)
        assert 13 in safe_jump_holds(env, objective, 1, verify_recovery=False)
        safe = safe_jump_holds(env, objective, 1)
        assert safe and 13 not in safe
        assert before == (env.mario["x"], env.mario["y"], env.steps)
    finally:
        env.close()


def test_failure_practice_keeps_other_families_and_action_balance():
    from types import SimpleNamespace

    data = SimpleNamespace(
        family=torch.tensor([0, 0, 1, 1]),
        actor_mask=torch.ones(4, dtype=torch.bool),
        action=torch.tensor([1, 2, 1, 2]),
        # The demonstration dataclass guarantees these via __post_init__
        # defaults (absent phase = all-routine, absent carry = zero); the
        # stand-in must honor the same contract.
        phase=torch.zeros(4, dtype=torch.long),
        carry_progress=torch.zeros(4),
    )
    weights = demonstration_sample_weights(data, {1: 5})
    assert weights[:2].sum() == 1
    assert weights[2:].sum() == 5
    assert weights[0] == weights[1] and weights[2] == weights[3]


def test_shared_practice_survives_a_new_pass_and_decays_gradually():
    from scripts.block_smb_joint_learning import BLOCK_SMB_MC_FAMILIES, retained_practice_weights

    family = "enemy_patrol"
    index = BLOCK_SMB_MC_FAMILIES.index(family)
    streaks = {}
    weights = {index: 5.0}
    passing = {family: {"passed": True}}
    for _ in range(3):
        weights = retained_practice_weights(weights, passing, streaks, 5.0, 3)
        assert weights[index] == 5.0
    weights = retained_practice_weights(weights, passing, streaks, 5.0, 3)
    assert weights[index] == 4.0
    weights = retained_practice_weights(weights, {family: {"passed": False}}, streaks, 5.0, 3)
    assert weights[index] == 5.0 and streaks[index] == 0


def test_production_mastery_cannot_hide_a_failed_difficulty_in_the_average():
    from retroagi.stages.block_smb.train import (
        initial_block_smb_mastery_state,
        update_block_smb_mastery_state,
    )

    evaluation = {
        "families": {"tall_pipe_jump": {"success_rate": 0.95}},
        "difficulty_bins": {
            f"tall_pipe_jump:{d}": {"success_rate": r}
            for d, r in [("easy", 1.0), ("medium", 1.0), ("hard", 0.8)]
        },
    }
    state = update_block_smb_mastery_state(
        initial_block_smb_mastery_state(), evaluation, family_pass_rate_gate=0.9
    )
    assert not state["tall_pipe_jump"]["mastered"]
    assert state["tall_pipe_jump"]["mastered_evals"] == 0
