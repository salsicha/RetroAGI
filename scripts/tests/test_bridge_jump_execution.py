"""Bridge labels must be executable by the policy, with matching goals."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.stages.block_smb.demonstrations import collect_demonstrations, varied_demonstration
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.primitive_execution import BlockSMBPrimitiveExecutor
from retroagi.stages.block_smb.skills import requested_block_smb_skill_goal
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(params=["bridge_mount", "bridge_dismount"])
def bridge_case(request):
    return sample_block_smb_monte_carlo_scenario(
        family=request.param,
        split="train",
        seed=101,
        sample_index=0,
        difficulty="easy",
        validate_reachability=False,
    )


def motor(index):
    logits = torch.full((1, 1, 16), -100.0)
    logits[..., index] = 100
    return SimpleNamespace(hold_duration_logits=logits, duration_bin_values=torch.arange(1, 17))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_duration_labels_reproduce_landings_through_policy_executor(bridge_case, difficulty):
    case = sample_block_smb_monte_carlo_scenario(
        family=bridge_case.family,
        split="validation",
        seed=101,
        sample_index=0,
        difficulty=difficulty,
        validate_reachability=False,
    )
    actions = case.oracle["actions"]
    departure = actions.index(2)
    hold = actions.count(2)
    index = NES_JUMP_FRAMES.index(hold)
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=case.scenario)
        executor = BlockSMBPrimitiveExecutor(env, adaptive_duration=False)
        raw = motor(index)
        for frame in range(320):
            intent = executor.committed_action
            if intent is None:
                intent = 0 if frame < departure else 2
            mapped = executor.motor_parameters(intent, raw)
            execution = executor.execute(
                intent,
                motor_primitives=mapped,
                support_override="ground" if env.mario["on_ground"] else "air",
            )
            _, _, done, truncated, _ = env.step(execution.action)
            if done or truncated:
                break
        assert env._goal_credited
        assert raw.duration_bin_values.tolist() == list(range(1, 17))
    finally:
        env.close()


def test_demonstrations_preserve_bridge_goal_and_have_successful_alternate(bridge_case):
    data = collect_demonstrations([(0, bridge_case)], tiny_config(), StaticBlockVision)
    goal = requested_block_smb_skill_goal(bridge_case.scenario)
    assert torch.all(data.goal == goal)
    assert data.actor_mask[data.action == 0].all()
    alternative = varied_demonstration(bridge_case, 101, robust=True)
    assert alternative is not None
    if bridge_case.family == "bridge_mount":
        # Dismount windows never certify enough holds for a deeper departure.
        assert alternative.oracle["actions"] != bridge_case.oracle["actions"]


def test_batched_and_sequential_bridge_goals_and_waits_match(bridge_case):
    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import (
        block_smb_policy_scenario,
        collect_trajectory,
        make_block_smb_model,
    )
    from scripts.block_smb_batched_evaluation import evaluate_batched

    config = tiny_config(evaluation_max_steps=12, ranked_candidate_search=False)
    config = replace(config, ablation=replace(config.ablation, recurrent_state_enabled=False))
    model = make_block_smb_model(config).eval()
    goals = []
    original = model.forward

    def forward(*args, **kwargs):
        goals.append(kwargs["skill_goal"].clone())
        result = original(*args, **kwargs)
        model.last_policy_logits_a[..., :6] = -100
        model.last_policy_logits_a[..., 0] = 100
        model.last_selected_action_id = 0
        return result

    model.forward = forward
    result = evaluate_batched(model, [bridge_case], config, StaticBlockVision, return_actions=True)
    batched_goals = goals.copy()
    goals.clear()
    stage = BlockSMBStage(
        scenario=block_smb_policy_scenario(bridge_case.scenario, True), vision=StaticBlockVision()
    )
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                model,
                stage,
                bridge_case.scenario_id,
                rollout_steps=12,
                seed=101,
                deterministic=True,
                device=torch.device("cpu"),
                ablation=config.ablation,
            )
        assert result["actions"][0] == [t.action for t in trajectory.transitions]
        expected = requested_block_smb_skill_goal(bridge_case.scenario)
        assert all(torch.equal(g, expected) for g in batched_goals + goals)
    finally:
        stage.env.close()


def test_training_keeps_physical_duration_labels_through_release(bridge_case):
    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import collect_trajectory, make_block_smb_model

    model = make_block_smb_model(tiny_config()).eval()
    stage = BlockSMBStage(scenario=bridge_case.scenario, vision=StaticBlockVision())
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                model,
                stage,
                bridge_case.scenario_id,
                rollout_steps=320,
                seed=101,
                deterministic=True,
                device=torch.device("cpu"),
                demonstration_actions=bridge_case.oracle["actions"],
            )
        assert trajectory.success
        jump_rows = [t for t in trajectory.transitions if t.info.get("primitive_valid_hold_frames")]
        assert jump_rows and any(t.action == 1 for t in jump_rows)
        for step in jump_rows:
            assert step.duration_bin_values.tolist() == list(NES_JUMP_FRAMES)
            # Bridge departures are coached toward the longest certified hold.
            assert step.info["primitive_valid_hold_frames"] == [step.info["primitive_target_hold"]]
            assert step.info["primitive_outcome_target"] == step.info["primitive_target_hold"] / 32
        waits = [t for t in trajectory.transitions if t.action == 0]
        assert all("primitive_outcome_target" not in t.info for t in waits)
    finally:
        stage.env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_teacher_departs_from_the_robust_window(bridge_case, difficulty):
    # The window opens with only the longest hold certified; departing on
    # that thin edge left a frame or two of timing margin.
    from retroagi.core.smb_coaching import training_target
    from retroagi.stages.block_smb.bridge_curriculum import (
        bridge_jump_allowed,
        bridge_jump_oracle,
        bridge_takeoff_window,
    )
    from retroagi.stages.block_smb.local_traversal import safe_jump_holds

    case = sample_block_smb_monte_carlo_scenario(
        family=bridge_case.family,
        split="train",
        seed=101,
        sample_index=1,
        difficulty=difficulty,
        validate_reachability=False,
    )
    env = MarioScenarioEnv()

    def certify():
        return safe_jump_holds(env, training_target(env), 1)

    try:
        departures = []
        for variant in range(4):
            actions = bridge_jump_oracle(case.scenario, variant=variant)
            departure = actions.index(2)
            departures.append(departure)
            env.reset(scenario=case.scenario)
            for action in actions[: departure - 1]:
                env.step(action)
            if variant == 0:
                assert not bridge_jump_allowed(*bridge_takeoff_window(env, certify))
            env.step(actions[departure - 1])
            now, later = bridge_takeoff_window(env, certify)
            assert bridge_jump_allowed(now, later), variant
            # The longest certified hold tolerates departure-timing drift.
            assert actions.count(2) == max(now)
            for action in actions[departure:]:
                _, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            assert env._goal_credited, variant
        assert departures[:3] == sorted(departures[:3])
    finally:
        env.close()


def test_hard_dismount_alternatives_do_not_require_five_safe_holds():
    from retroagi.stages.block_smb.bridge_curriculum import bridge_jump_oracle

    case = sample_block_smb_monte_carlo_scenario(
        family="bridge_dismount",
        split="validation",
        seed=50000,
        sample_index=710,
        difficulty="hard",
        validate_reachability=False,
    )
    env = MarioScenarioEnv()
    try:
        for variant in (1, 2, 3):
            actions = bridge_jump_oracle(case.scenario, variant=variant)
            env.reset(scenario=case.scenario)
            for action in actions:
                _, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            assert env._goal_credited, variant
    finally:
        env.close()


def test_bridge_demonstrations_accept_only_the_longest_hold(bridge_case):
    data = collect_demonstrations([(0, bridge_case)], tiny_config(), StaticBlockVision)
    jumps = data.motor_action == 2
    assert jumps.any()
    assert (data.valid_durations[jumps].sum(-1) == 1).all()
    assert data.valid_durations[jumps][:, NES_JUMP_FRAMES.index(32)].all()


def test_bridge_labels_wait_through_the_thin_opening_edge(bridge_case):
    from retroagi.core.smb_coaching import training_target
    from retroagi.stages.block_smb.bridge_curriculum import bridge_takeoff_actions
    from retroagi.stages.block_smb.local_traversal import ROBUST_TAKEOFF_HOLDS, safe_jump_holds
    from retroagi.stages.block_smb.tactics import TACTIC_STANCES, tactic_label

    env = MarioScenarioEnv()

    def certify():
        return safe_jump_holds(env, training_target(env), 1)

    try:
        env.reset(scenario=bridge_case.scenario)
        edge = robust = None
        for frame in range(240):
            holds = certify()
            labels = bridge_takeoff_actions(env, certify)
            stance = TACTIC_STANCES[tactic_label(env, family=bridge_case.family)]
            if holds and edge is None:
                edge = frame
                # First certified frame: wait, the window is still widening.
                assert len(holds) < ROBUST_TAKEOFF_HOLDS
                assert labels == [True, False, False, False, False, False]
                assert stance == "hold_area"
            if labels[2]:
                robust = frame
                assert stance == "advance"
                break
            env.step(0)
        assert edge is not None and robust is not None and robust > edge
    finally:
        env.close()


@pytest.mark.parametrize("hold", [8, 32])
def test_unreachable_bridge_jump_corrects_action_without_inventing_duration(bridge_case, hold):
    from retroagi.stages.block_smb.adapter import BlockSMBStage
    from retroagi.stages.block_smb.train import collect_trajectory, make_block_smb_model

    model = make_block_smb_model(tiny_config()).eval()
    stage = BlockSMBStage(scenario=bridge_case.scenario, vision=StaticBlockVision())
    try:
        with torch.no_grad():
            trajectory = collect_trajectory(
                model,
                stage,
                bridge_case.scenario_id,
                rollout_steps=160,
                seed=101,
                deterministic=True,
                device=torch.device("cpu"),
                demonstration_actions=[2] * hold + [1] * 96,
            )
        assert not trajectory.success
        start = trajectory.transitions[0]
        assert start.info["primitive_unreachable"]
        assert start.info["jump_overreach"]
        assert start.info["jump_overreach_action"] == 2
        assert not any("primitive_target_hold" in t.info for t in trajectory.transitions)
        assert not any("primitive_outcome_target" in t.info for t in trajectory.transitions)
    finally:
        stage.env.close()


def test_nes_fallback_coaching_uses_physical_hold_limit():
    from retroagi.stages.block_smb.train import jump_overreach, sign_coached_hold

    assert sign_coached_hold(28, 40, 90, max_hold=32) == 29
    assert sign_coached_hold(32, 40, 90, max_hold=32) == 32
    assert not jump_overreach(28, 40, 90, max_hold=32)
    assert jump_overreach(32, 40, 90, max_hold=32)


@pytest.mark.parametrize("seed", [101, 20260908])
def test_production_builder_covers_bridge_and_plant_variants_in_every_difficulty(monkeypatch, seed):
    from collections import defaultdict

    from retroagi.stages.block_smb import demonstrations, monte_carlo

    monkeypatch.setattr(
        monte_carlo,
        "BLOCK_SMB_MC_FAMILIES",
        ("bridge_mount", "bridge_dismount", "piranha_avoidance"),
    )

    def sample(**kwargs):
        return SimpleNamespace(
            scenario=(
                {}
                if kwargs["family"] == "piranha_avoidance"
                else {"bridge_jump_task": kwargs["family"]}
            ),
            difficulty=kwargs["difficulty"],
            index=kwargs["sample_index"],
            variant=0,
        )

    def alternate(case, route_seed, **kwargs):
        return SimpleNamespace(**{**vars(case), "variant": 1 + route_seed % 3})

    def arrivals(case, index):
        return [SimpleNamespace(**{**vars(case), "variant": "arrival"})]

    from retroagi.stages.block_smb import piranha_tactics

    monkeypatch.setattr(monte_carlo, "sample_block_smb_monte_carlo_scenario", sample)
    monkeypatch.setattr(demonstrations, "varied_demonstration", alternate)
    monkeypatch.setattr(piranha_tactics, "arrival_demonstrations", arrivals)
    monkeypatch.setattr(
        demonstrations, "collect_demonstrations", lambda cases, *args, **kwargs: cases
    )
    config = SimpleNamespace(
        seed=seed,
        log_path=None,
        demonstration_layouts_per_family=6,
        demonstration_robust_routes=True,
        demonstration_varied_routes=True,
        demonstration_enemy_wait_routes=False,
    )
    groups = defaultdict(set)
    for family, case in demonstrations.build_balanced_demonstrations(config, None):
        groups[family, case.difficulty, case.index].add(case.variant)
    assert len(groups) == 18
    plant = monte_carlo.BLOCK_SMB_MC_FAMILIES.index("piranha_avoidance")
    for (family, _difficulty, _index), variants in groups.items():
        # Plant layouts also teach corrections from learner arrival states.
        expected = {0, 1, 2, 3} | ({"arrival"} if family == plant else set())
        assert variants == expected
