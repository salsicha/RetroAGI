"""Temporal decisions must be observable, executable, and jointly learned."""

from copy import deepcopy
from functools import lru_cache

import pytest
import torch

from retroagi.core.models import TACTIC_STANCES
from retroagi.core.smb_enemy_history import EnemyObservationHistory
from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage
from retroagi.stages.block_smb.demonstrations import collect_demonstrations, fit_demonstrations
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.piranha import plant_oracle
from retroagi.stages.block_smb.piranha_tactics import (
    fresh_retraction,
    tactical_choice,
    timed_safe_holds,
)
from retroagi.stages.block_smb.policy_recovery import combine_demonstrations, repair_policy_actions
from retroagi.stages.block_smb.primitive_execution import (
    BlockSMBPrimitiveExecutor,
    teacher_route_reachable,
)
from retroagi.stages.block_smb.train import (
    collect_trajectory,
    compute_block_smb_losses,
    make_block_smb_model,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@lru_cache(None)
def timed_sample(difficulty="hard"):
    for seed in range(20):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            split="train",
            difficulty=difficulty,
            seed=seed,
            sample_index=0,
        )
        if sample.parameters["crossing_mode"] == "timed":
            return sample
    raise AssertionError("No temporal practice in the family")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_wait_then_cross_uses_observed_retraction_and_real_executor(difficulty):
    sample = timed_sample(difficulty)
    env = MarioScenarioEnv()
    try:
        for phase in (0, 25, 90, 130):
            scenario = deepcopy(sample.scenario)
            scenario["enemies"][0]["phase"] = phase
            actions = plant_oracle(scenario)
            assert actions and 0 in actions and 2 in actions
            env.reset(scenario=scenario)
            assert teacher_route_reachable(env, actions)
            history = EnemyObservationHistory()
            for action in actions:
                features = history.observe(env, env.steps)
                if action == 2 and env.mario["on_ground"]:
                    assert fresh_retraction(features)
                    assert timed_safe_holds(env, features)
                _, _, done, _, info = env.step(action)
                assert not info["death"] and info["reward_terms"]["enemy_stomp"] == 0
                if done:
                    break
            assert env._goal_credited
        # Keeping the plant exposed removes the route: waiting is necessary.
        env.reset(scenario=sample.scenario)
        env.enemies[0]["conservative_envelope"] = True
        from retroagi.stages.block_smb.piranha_tactics import timed_suffix

        assert timed_suffix(env, max_frames=180) is None
    finally:
        env.close()


def test_timing_targets_do_not_read_hidden_cycle_and_unknown_empty_pipe_waits():
    env = MarioScenarioEnv()
    sample = timed_sample()
    try:
        env.reset(scenario=sample.scenario)
        history = EnemyObservationHistory()
        for action in sample.oracle["actions"]:
            features = history.observe(env, env.steps)
            if action == 2 and env.mario["on_ground"]:
                break
            env.step(action)
        before = snapshot_env_state(env)
        reference = tactical_choice(env, features)
        assert reference[0:2] == ("advance", 2)
        # Same observed state, arbitrarily different privileged timers.
        for phase, hidden in [(0, 48), (50, 64), (120, 55)]:
            env.enemies[0].update(plant_tick=phase, hidden_frames=hidden, rise_frames=20)
            mutated = snapshot_env_state(env)
            assert tactical_choice(env, features) == reference
            assert snapshot_env_state(env) == mutated
        restore_env_state(env, before)
        unknown = EnemyObservationHistory().observe(env, env.steps)
        assert not fresh_retraction(unknown)
        assert tactical_choice(env, unknown)[0:2] == ("hold_area", 0)
    finally:
        env.close()


def test_temporal_wait_reobserves_every_frame():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=timed_sample().scenario)
        executor = BlockSMBPrimitiveExecutor(env, steady_primitives=True)
        for _ in range(5):
            execution = executor.execute(0, support_override="ground")
            assert execution.hold_frames == 1 and execution.released
            env.step(execution.action)
    finally:
        env.close()


def test_demonstrations_jointly_train_tactics_skill_and_primitive():
    torch.manual_seed(4)
    config = tiny_config(
        motion_observations=True,
        hazard_observations=True,
        adaptive_duration_control=False,
        walk_duration_primitives=False,
    )
    sample = timed_sample()
    data = collect_demonstrations(
        [(BLOCK_SMB_MC_FAMILIES.index(sample.family), sample)], config, StaticBlockVision
    )
    assert {TACTIC_STANCES.index("advance"), TACTIC_STANCES.index("hold_area")} <= set(
        data.tactic.tolist()
    )
    assert (data.tactic[~data.actor_mask] == -1).all()
    assert (data.duration[data.action == 0] == 0).all()
    assert data.actor_mask[data.action == 0].all()
    merged = combine_demonstrations([data, data])
    assert torch.equal(merged.tactic, torch.cat([data.tactic, data.tactic]))
    model = make_block_smb_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_demonstrations(
        model,
        optimizer,
        data,
        steps=4,
        batch_size=64,
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
    assert model.last_demonstration_metrics["tactic_loss"] > 0
    assert model.tactics_network.stance_context.weight.abs().sum() > 0
    assert model.tactics_network.feature_position_gain.abs() > 0
    # The semantic stance now has a causal path to A's action logits.
    args = (data.a[:1], data.b[:1], data.c[:1])
    choices = []
    for stance in ("advance", "hold_area"):
        index = TACTIC_STANCES.index(stance)

        def force_stance(module, inputs, output):
            forced = torch.full_like(output, -30)
            forced[:, index] = 30
            return forced

        hook = model.tactics_network.stance_head.register_forward_hook(force_stance)
        try:
            with torch.no_grad():
                model(*args, skill_goal=data.goal[:1], critic_feedback_enabled=False)
                choices.append(model.last_policy_logits_a.clone())
        finally:
            hook.remove()
    assert not torch.equal(*choices)


def test_online_tactic_loss_is_decision_masked_and_reaches_the_stance_head():
    torch.manual_seed(5)
    config = tiny_config(
        motion_observations=True,
        hazard_observations=True,
        adaptive_duration_control=False,
        walk_duration_primitives=False,
    )
    model = make_block_smb_model(config)
    sample = timed_sample("easy")
    stage = BlockSMBStage(
        scenario=sample.scenario,
        vision=StaticBlockVision(),
        observation_config=BlockSMBObservationConfig(
            motion_observations=True, hazard_observations=True
        ),
    )
    try:
        trajectory = collect_trajectory(
            model,
            stage,
            sample.scenario_id,
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
        assert {TACTIC_STANCES.index("advance"), TACTIC_STANCES.index("hold_area")} <= {
            t.tactic_target for t in labels
        }
        assert len(labels) < len(trajectory.transitions)
        losses = compute_block_smb_losses(
            model, trajectory.transitions, config, torch.device("cpu")
        )
        assert losses["tactic_supervised_steps"].item() == len(labels)
        (gradient,) = torch.autograd.grad(
            losses["loss_tactic"], model.tactics_network.stance_head.weight
        )
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    finally:
        stage.env.close()


def test_premature_departure_gets_a_successful_waiting_recovery():
    sample = timed_sample()
    actions = list(sample.oracle["actions"])
    wait = actions.index(0)
    failed = actions[:wait] + [2] * 16 + [1] * 80
    repairs = repair_policy_actions(sample.scenario, failed, max_repairs=2)
    assert repairs and repairs[0]["recovery_reason"] == "tactic"
    env = MarioScenarioEnv()
    try:
        for repair in repairs:
            start = repair["supervision_start_frame"]
            assert repair["actions"][:start] == failed[:start]
            assert 0 in repair["actions"][start:]
            env.reset(scenario=sample.scenario)
            assert teacher_route_reachable(env, repair["actions"])
    finally:
        env.close()


def test_evaluation_reports_temporal_and_clearance_crossings_separately(monkeypatch):
    from retroagi.stages.block_smb import train
    from retroagi.stages.block_smb.monte_carlo import BlockSMBMonteCarloSampleSet

    samples = {}
    for seed in range(20):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            difficulty="hard",
            split="validation",
            seed=seed,
            sample_index=0,
        )
        samples[sample.parameters["crossing_mode"]] = sample
        if len(samples) == 2:
            break
    assert len(samples) == 2
    cases = BlockSMBMonteCarloSampleSet(
        schema_version=sample.schema_version,
        distribution_id=sample.distribution_id,
        split="validation",
        seed=0,
        samples=tuple(samples.values()),
    )
    monkeypatch.setattr(train, "sample_block_smb_monte_carlo_split", lambda **kwargs: cases)
    config = tiny_config(
        motion_observations=True,
        hazard_observations=True,
        evaluation_max_steps=12,
        adaptive_duration_control=False,
        walk_duration_primitives=False,
    )
    evaluation = train.evaluate_block_smb_monte_carlo(
        make_block_smb_model(config),
        config,
        split="validation",
        sample_count=2,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
    )
    assert set(evaluation["piranha_crossing_modes"]) == {"timed:hard", "clearance:hard"}
    assert all(v["sample_count"] == 1 for v in evaluation["piranha_crossing_modes"].values())
    assert evaluation["piranha_tactics"]["decisions"] > 0
    assert 0 <= evaluation["piranha_tactics"]["accuracy"] <= 1
