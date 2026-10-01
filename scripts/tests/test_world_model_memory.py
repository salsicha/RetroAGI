"""The world-model LSTM can carry episodic hazard memory for the actor."""

from dataclasses import replace

import pytest
import torch

from retroagi.core.architectures import make_agent_world_model_critic
from retroagi.core.smb_enemy_history import HAZARD_MEMORY_NAMES
from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.demonstrations import (
    collect_demonstrations,
    fit_demonstrations,
    refresh_demonstration_memory,
)
from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
from retroagi.stages.block_smb.policy_recovery import combine_demonstrations
from retroagi.stages.block_smb.train import (
    BlockSMBAblationConfig,
    collect_trajectory,
    compute_block_smb_losses,
    make_block_smb_model,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_piranha_tactics import timed_sample


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def memory_config(**overrides):
    values = dict(
        adaptive_duration_control=False,
        walk_duration_primitives=False,
        architecture_config={"world_model_memory_dim": len(HAZARD_MEMORY_NAMES)},
        world_model_memory_weight=1.0,
        memory_refresh_interval=4,
        ablation=BlockSMBAblationConfig(recurrent_state_enabled=True),
        demonstration_bootstrap_updates=8,
    )
    values.update(overrides)
    return tiny_config(**values)


@pytest.fixture(scope="module")
def timed_demonstrations():
    sample = timed_sample()
    family = BLOCK_SMB_MC_FAMILIES.index(sample.family)
    return collect_demonstrations([(family, sample)], memory_config(), StaticBlockVision)


def test_memory_world_model_reads_observation_slots_and_reports_memory():
    plain = make_block_smb_model(tiny_config())
    memory = make_block_smb_model(memory_config())
    assert plain.world_model.memory_head is None
    assert plain.world_model.observation_encoder is None
    assert memory.world_model.memory_head.out_features == len(HAZARD_MEMORY_NAMES)
    assert memory.world_model.lstm.input_size == plain.world_model.lstm.input_size + 32

    config = memory_config()
    stage = BlockSMBStage(scenario=timed_sample().scenario, vision=StaticBlockVision())
    try:
        batch = stage.encode_observation(stage.reset(seed=0))
    finally:
        stage.env.close()
    memory.eval()
    state = memory.initial_world_model_state(1, torch.device("cpu"))
    memory(batch.src_a, batch.src_b, batch.src_c, world_model_state=state)
    assert memory.last_memory_prediction.shape == (1, len(HAZARD_MEMORY_NAMES))
    plain.eval()
    plain(batch.src_a, batch.src_b, batch.src_c)
    assert plain.last_memory_prediction is None
    assert config.architecture_config["world_model_memory_dim"] == 1


def test_architecture_rejects_negative_memory_dim():
    from scripts.tests.test_architectures import tiny_stage

    with pytest.raises(ValueError, match="world_model_memory_dim"):
        make_agent_world_model_critic(tiny_stage(), {"world_model_memory_dim": -1})
    model = make_agent_world_model_critic(tiny_stage(), {"world_model_memory_dim": 2})
    assert model.world_model.memory_head.out_features == 2


def test_demonstrations_record_episode_clock_and_observable_memory(timed_demonstrations):
    data = timed_demonstrations
    assert int(data.frame_index[0]) == 0
    assert torch.equal(data.frame_index, torch.arange(len(data.action)))
    assert data.memory_target.shape == (len(data.action), len(HAZARD_MEMORY_NAMES))
    # Peak exposure only grows within an episode, and a timed plant is seen.
    assert torch.all(data.memory_target[1:] >= data.memory_target[:-1])
    assert float(data.memory_target.max()) > 0
    assert data.memory_state.shape == (len(data.action), 0)


def test_refreshed_states_match_sequential_episode_replay(timed_demonstrations):
    data = replace(timed_demonstrations)
    # Present the rows as two episodes of different lengths.
    split = 70
    data.frame_index = torch.cat((torch.arange(split), torch.arange(len(data.action) - split)))
    torch.manual_seed(5)
    model = make_block_smb_model(memory_config()).eval()
    refresh_demonstration_memory(model, data)
    batched = data.memory_state.clone()
    refresh_demonstration_memory(model, data, chunk_size=1)
    assert torch.allclose(batched, data.memory_state, atol=1e-5)

    hidden = model.world_model.hidden_size
    with torch.no_grad():
        for start, end in ((0, split), (split, len(data.action))):
            state = model.initial_world_model_state(1, torch.device("cpu"))
            for row in range(start, end):
                stored = batched[row]
                assert torch.allclose(stored[:hidden], state.hidden[0, 0], atol=1e-5), row
                assert torch.allclose(stored[hidden : 2 * hidden], state.cell[0, 0], atol=1e-5)
                stance = state.stance if state.stance is not None else torch.zeros(1, 8, 4)
                assert torch.allclose(stored[2 * hidden :], stance.flatten(), atol=1e-5), row
                state = model(
                    data.a[row : row + 1],
                    data.b[row : row + 1],
                    data.c[row : row + 1],
                    skill_goal=data.goal[row : row + 1],
                    forced_action=data.motor_action[row : row + 1],
                    critic_feedback_enabled=False,
                    world_model_state=state,
                    return_world_model_state=True,
                )[-1]
    # Episode starts enter with an empty memory.
    assert torch.count_nonzero(batched[0]) == 0
    assert torch.count_nonzero(batched[split]) == 0
    assert torch.count_nonzero(batched[split + 1]) > 0


def test_fitting_trains_lstm_memory_through_the_actor(timed_demonstrations):
    data = replace(timed_demonstrations)
    torch.manual_seed(7)
    model = make_block_smb_model(memory_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit_demonstrations(
        model,
        optimizer,
        data,
        steps=4,
        batch_size=32,
        decision_durations_only=True,
        walk_durations=False,
        memory_weight=1.0,
        memory_refresh_interval=2,
    )
    assert model.last_demonstration_metrics["memory_loss"] > 0
    # LSTM hidden and cell state, then the eight-entry stance history.
    assert data.memory_state.shape[1] == 2 * model.world_model.hidden_size + 8 * 4
    assert data.world_model_inputs.shape == (len(data.action), 52)
    reached = {
        name.split(".")[1] if name.startswith("world_model.") else name.split(".")[0]
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    }
    assert {"memory_head", "observation_encoder", "lstm", "world_model_actor_context"} <= reached

    # Without refreshed states the rows fall back to an empty memory.
    torch.manual_seed(7)
    model = make_block_smb_model(memory_config())
    fit_demonstrations(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        replace(timed_demonstrations),
        steps=2,
        batch_size=32,
        decision_durations_only=True,
        walk_durations=False,
        memory_weight=1.0,
        memory_refresh_interval=0,
    )
    assert model.last_demonstration_metrics["memory_loss"] == 0


def test_combined_demonstrations_drop_stale_carried_states(timed_demonstrations):
    data = replace(timed_demonstrations)
    data.memory_state = torch.ones((len(data.action), 64))
    combined = combine_demonstrations([data, data])
    assert combined.memory_state.shape == (2 * len(data.action), 0)
    assert torch.equal(combined.memory_target[: len(data.action)], data.memory_target)
    assert int((combined.frame_index == 0).sum()) == 2


def test_rollouts_supervise_the_carried_memory():
    config = memory_config()
    torch.manual_seed(11)
    model = make_block_smb_model(config)
    sample = timed_sample()
    stage = BlockSMBStage(
        scenario=sample.scenario,
        vision=StaticBlockVision(),
    )
    try:
        trajectory = collect_trajectory(
            model,
            stage,
            sample.scenario_id,
            rollout_steps=24,
            seed=0,
            deterministic=False,
            device=torch.device("cpu"),
            ablation=config.ablation,
            adaptive_duration_control=False,
            walk_duration_primitives=False,
        )
    finally:
        stage.env.close()
    assert all(t.memory_prediction is not None for t in trajectory.transitions)
    assert all(t.memory_target.shape == (1,) for t in trajectory.transitions)
    losses = compute_block_smb_losses(model, trajectory.transitions, config, torch.device("cpu"))
    assert float(losses["loss_world_model_memory"].detach()) > 0
    losses["loss_total"].backward()
    assert model.world_model.memory_head.weight.grad.abs().sum() > 0


def test_recurrent_demonstrations_require_refreshed_memory():
    with pytest.raises(ValueError, match="recurrent_state_enabled"):
        memory_config(memory_refresh_interval=0)
    with pytest.raises(ValueError, match="recurrent_state_enabled"):
        memory_config(architecture_config={})
    with pytest.raises(ValueError, match="non-negative"):
        memory_config(world_model_memory_weight=-1.0)
    config = memory_config()
    assert config.ablation.recurrent_state_enabled and config.memory_refresh_interval == 4


def test_recipe_is_valid_and_learning_tools_stay_feedforward(tmp_path):
    import json
    from pathlib import Path

    from retroagi.stages.block_smb.cli import _normalize_config_values
    from retroagi.stages.block_smb.train import BlockSMBTrainingConfig
    from scripts.block_smb_family_learning import feedforward_recipe

    recipe = json.loads(Path("scripts/configs/block_smb_full_volume.json").read_text())
    BlockSMBTrainingConfig(
        **_normalize_config_values(
            {
                **recipe,
                "checkpoint_path": tmp_path / "policy.pth",
                "log_path": tmp_path / "events.jsonl",
            }
        )
    )
    values = feedforward_recipe(recipe)
    assert "world_model_memory_dim" not in values["architecture_config"]
    assert values["ablation"]["recurrent_state_enabled"] is False
    assert values["world_model_memory_weight"] == 0 and values["memory_refresh_interval"] == 0
    assert recipe["architecture_config"] is not values["architecture_config"]


def test_production_training_fits_and_rehearses_lstm_memory(monkeypatch, timed_demonstrations):
    from retroagi.stages.block_smb import demonstrations
    from retroagi.stages.block_smb.train import train_and_evaluate_block_smb

    config = memory_config(
        generated_scenarios=0,
        numeric_policy_learning_rate=0.003,
        demonstration_bootstrap_updates=2,
        demonstration_rehearsal_updates=2,
        mastery_gated_schedule=True,
    )
    monkeypatch.setattr(
        demonstrations,
        "build_balanced_demonstrations",
        lambda *args, **kwargs: replace(timed_demonstrations),
    )
    result = train_and_evaluate_block_smb(config, vision_factory=StaticBlockVision)
    epoch = result["history"][0]
    assert epoch["demonstration_rehearsal_updates"] == 2
    assert epoch["demonstration_memory_loss"] > 0
    assert "loss_world_model_memory" in epoch


def test_stance_history_records_distinct_tactics():
    from retroagi.core.models import advance_stance_history

    history = torch.zeros(1, 3, 4)
    advance = torch.tensor([[0.9, 0.0, 0.1, 0.0]])
    hold = torch.tensor([[0.1, 0.0, 0.9, 0.0]])
    history = advance_stance_history(history, advance)
    history = advance_stance_history(history, advance * 0.5 + hold * 0.5 - 0.01 * hold)
    # A repeated stance refreshes the latest entry instead of filling history.
    assert history[0, :2].abs().sum() == 0
    history = advance_stance_history(history, hold)
    assert history[0, 1].argmax() == 0 and history[0, 2].argmax() == 2
    assert history[0, 0].abs().sum() == 0


def test_carried_state_holds_stance_history_and_learned_goals_replace_the_script():
    from retroagi.core.models import STRATEGY_OBJECTIVES, skill_goal_objective
    from retroagi.core.skills import skill_goal_encoding

    goals = torch.cat((skill_goal_encoding("enemy_clear"), torch.zeros(1, 6)))
    assert skill_goal_objective(goals).tolist() == [3, len(STRATEGY_OBJECTIVES) - 1]

    torch.manual_seed(2)
    model = make_block_smb_model(memory_config()).eval()
    stage = BlockSMBStage(scenario=timed_sample().scenario, vision=StaticBlockVision())
    try:
        batch = stage.encode_observation(stage.reset(seed=0))
    finally:
        stage.env.close()
    with torch.no_grad():
        state = model(batch.src_a, batch.src_b, batch.src_c, return_world_model_state=True)[-1]
        assert state.stance.shape == (1, 8, 4) and state.stance.abs().sum() > 0
        # Each forward carries the history it received forward.
        again = model(
            batch.src_a,
            batch.src_b,
            batch.src_c,
            world_model_state=state,
            return_world_model_state=True,
        )[-1]
        assert torch.allclose(again.stance[:, :-1], state.stance[:, :-1]) or torch.allclose(
            again.stance[:, :-2], state.stance[:, 1:-1]
        )
        objective = model.last_objective_logits.argmax(-1)
        predicted_goal = model.objective_skill_goals[objective]
        scripted = model(
            batch.src_a,
            batch.src_b,
            batch.src_c,
            skill_goal=predicted_goal,
            world_model_state=state,
        )[4]
        model.learned_skill_goals = True
        learned = model(
            batch.src_a,
            batch.src_b,
            batch.src_c,
            skill_goal=skill_goal_encoding("retreat_recover"),
            world_model_state=state,
        )[4]
    assert torch.allclose(scripted, learned)


def test_unrolled_state_matches_refresh_and_trains_the_lstm(timed_demonstrations):
    from retroagi.stages.block_smb.demonstrations import (
        carried_memory_state,
        unrolled_memory_state,
    )

    data = replace(timed_demonstrations)
    torch.manual_seed(4)
    model = make_block_smb_model(memory_config()).eval()
    refresh_demonstration_memory(model, data)
    ids = torch.tensor([0, 3, 40, len(data.action) - 1])
    stored = carried_memory_state(model, data, ids, torch.device("cpu"))
    unrolled, predictions, targets = unrolled_memory_state(model, data, ids, torch.device("cpu"), 6)
    # Fresh states: replaying stored inputs reproduces the stored state.
    assert torch.allclose(unrolled.hidden, stored.hidden, atol=1e-5)
    assert torch.allclose(unrolled.cell, stored.cell, atol=1e-5)
    assert torch.equal(unrolled.stance, stored.stance)
    # Row 0 starts an episode and row 3 has only three earlier rows.
    assert [len(p) for p in predictions] == [2, 2, 2, 3, 3, 3]
    assert all(len(p) == len(t) for p, t in zip(predictions, targets))
    unrolled.hidden.sum().backward()
    assert model.world_model.lstm.weight_hh_l0.grad.abs().sum() > 0
    assert model.world_model.observation_encoder[0].weight.grad.abs().sum() > 0


def test_the_unroll_adds_policy_gradients_through_the_carried_state(timed_demonstrations):
    def lstm_gradient(unroll):
        torch.manual_seed(7)
        model = make_block_smb_model(memory_config())
        data = replace(timed_demonstrations)
        fit_demonstrations(
            model,
            torch.optim.SGD(model.parameters(), lr=0.0),
            data,
            steps=1,
            batch_size=32,
            decision_durations_only=True,
            walk_durations=False,
            memory_weight=0.0,
            memory_refresh_interval=1,
            memory_unroll=unroll,
        )
        return model.world_model.lstm.weight_hh_l0.grad

    # Identical fresh states and samples; only the carried-state path differs.
    assert (lstm_gradient(4) - lstm_gradient(0)).abs().sum() > 0


def test_fitting_learns_the_strategy_objective(timed_demonstrations):
    data = replace(timed_demonstrations)
    torch.manual_seed(9)
    model = make_block_smb_model(memory_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    head = model.strategy_network.objective_head
    before = [p.detach().clone() for p in head.parameters()]
    fit_demonstrations(
        model,
        optimizer,
        data,
        steps=3,
        batch_size=32,
        decision_durations_only=True,
        walk_durations=False,
        memory_weight=1.0,
        memory_refresh_interval=2,
        memory_unroll=2,
        strategy_loss_weight=0.5,
    )
    assert model.last_demonstration_metrics["strategy_loss"] > 0
    assert any(not torch.equal(b, p) for b, p in zip(before, head.parameters()))


def test_replayed_prefixes_are_unsupervised_context():
    from retroagi.stages.block_smb.demonstrations import demonstration_sample_weights
    from retroagi.stages.block_smb.piranha_tactics import overshoot_demonstration

    sample = overshoot_demonstration(timed_sample())
    start = sample.oracle["supervision_start_frame"]
    assert start > 0
    family = BLOCK_SMB_MC_FAMILIES.index(sample.family)
    data = collect_demonstrations([(family, sample)], memory_config(), StaticBlockVision)
    assert int(data.context.sum()) == start and bool(data.context[:start].all())
    assert not data.actor_mask[data.context].any()
    assert (data.tactic[data.context] == -1).all()
    assert (demonstration_sample_weights(data)[data.context] == 0).all()
    assert data.actor_mask[~data.context].any()
    # Memory is observed from the first replayed frame, as a policy would.
    assert int(data.frame_index[0]) == 0
    assert torch.equal(
        data.memory_target[: start + 1], data.memory_target[: start + 1].cummax(0)[0]
    )


def test_warm_start_adds_memory_to_a_feedforward_checkpoint(tmp_path):
    from retroagi.stages.block_smb.train import (
        restore_block_smb_checkpoint,
        save_block_smb_checkpoint,
    )

    plain_config = tiny_config()
    torch.manual_seed(12)
    plain = make_block_smb_model(plain_config).eval()
    path = tmp_path / "plain.pth"
    save_block_smb_checkpoint(
        path,
        plain,
        torch.optim.Adam(plain.parameters()),
        epoch=1,
        global_step=1,
        config=plain_config,
        metrics={},
    )
    config = memory_config()
    memory = make_block_smb_model(config).eval()
    with pytest.raises(ValueError, match="architecture config"):
        restore_block_smb_checkpoint(
            path,
            memory,
            architecture_name=config.architecture_name,
            architecture_config=config.architecture_config,
            restore_rng=False,
        )
    restore_block_smb_checkpoint(
        path,
        memory,
        architecture_name=config.architecture_name,
        architecture_config=config.architecture_config,
        restore_rng=False,
        migrate_world_model_memory=True,
    )
    weight = memory.world_model.lstm.weight_ih_l0
    old = plain.world_model.lstm.weight_ih_l0
    assert torch.equal(weight[:, : old.size(1)], old)
    assert weight[:, old.size(1) :].abs().sum() == 0
    stage = BlockSMBStage(
        scenario=timed_sample().scenario,
        vision=StaticBlockVision(),
    )
    try:
        batch = stage.encode_observation(stage.reset(seed=0))
    finally:
        stage.env.close()
    with torch.no_grad():
        state = plain.initial_world_model_state(1, torch.device("cpu"))
        expected = plain(batch.src_a, batch.src_b, batch.src_c, world_model_state=state)
        actual = memory(batch.src_a, batch.src_b, batch.src_c, world_model_state=state)
    # The migrated model starts with exactly the checkpoint's behavior.
    assert torch.allclose(expected[4], actual[4], atol=1e-6)
    assert torch.allclose(expected[1], actual[1], atol=1e-6)


def test_batched_evaluation_carries_each_levels_state():
    from retroagi.stages.block_smb.train import block_smb_policy_scenario
    from scripts.block_smb_batched_evaluation import evaluate_batched
    from scripts.block_smb_family_learning import samples

    config = memory_config(evaluation_max_steps=60, ranked_candidate_search=False)
    torch.manual_seed(21)
    model = make_block_smb_model(config).eval()
    cases = [timed_sample(), samples("flat_run", 99173, "validation", 1)[0]]
    result = evaluate_batched(model, cases, config, StaticBlockVision, return_actions=True)
    errors = []
    decisions = [0, 0]
    for sample, actions in zip(cases, result["actions"]):
        stage = BlockSMBStage(
            scenario=block_smb_policy_scenario(sample.scenario, True),
            vision=StaticBlockVision(),
        )
        try:
            with torch.no_grad():
                trajectory = collect_trajectory(
                    model,
                    stage,
                    sample.scenario_id,
                    rollout_steps=60,
                    seed=sample.sample_seed % (2**31),
                    deterministic=True,
                    device=torch.device("cpu"),
                    ablation=config.ablation,
                    adaptive_duration_control=False,
                    walk_duration_primitives=False,
                )
        finally:
            stage.env.close()
        assert actions == [t.action for t in trajectory.transitions]
        errors += [
            float((t.memory_prediction.reshape(-1) - t.memory_target).abs().mean())
            for t in trajectory.transitions
        ]
        for t in trajectory.transitions:
            decisions[0] += 1
            decisions[1] += int(int(t.objective_logits.argmax()) == t.objective_target)
    assert result["memory_mean_absolute_error"] == pytest.approx(
        sum(errors) / len(errors), abs=1e-4
    )
    assert result["strategy_accuracy"] == pytest.approx(decisions[1] / decisions[0])


@pytest.mark.parametrize("family", ["stomp_recovery", "platform_chain"])
def test_repositioning_families_receive_tactical_targets(family):
    from retroagi.core.models import TACTIC_STANCES
    from retroagi.stages.block_smb.env import MarioScenarioEnv
    from retroagi.stages.block_smb.tactics import tactic_label
    from scripts.block_smb_family_learning import samples

    cases = [(BLOCK_SMB_MC_FAMILIES.index(family), s) for s in samples(family, 13, "train", 3)]
    data = collect_demonstrations(cases, tiny_config(), StaticBlockVision)
    from retroagi.stages.block_smb.policy_recovery import RECOVERY_FAMILIES

    labelled = data.tactic[data.actor_mask & (data.tactic >= 0)]
    # Routes head straight for the goal, so demonstrations teach advance.
    assert len(labelled) > 0 and set(labelled.tolist()) == {TACTIC_STANCES.index("advance")}
    assert family in RECOVERY_FAMILIES
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=cases[0][1].scenario)
        online = tactic_label(env, family=family)
    finally:
        env.close()
    # Stomp recovery has no certified online teacher; platform chains do.
    assert (online == -1) == (family == "stomp_recovery")


def test_runtime_contract_and_components_accept_the_strategy_objective(tmp_path):
    from retroagi.core.smb_runtime import SMBRuntimeContract, attach_runtime

    contract = SMBRuntimeContract.from_block_config(
        {"learned_skill_goals": True, "ablation": {"recurrent_state_enabled": True}}
    )
    model = make_block_smb_model(memory_config())
    attach_runtime(model, contract.manifest())
    assert model.learned_skill_goals and model.smb_runtime_contract.recurrent_state
    attach_runtime(model, SMBRuntimeContract().manifest())
    assert not model.learned_skill_goals


def test_enemy_wait_routes_hold_for_a_moving_enemy_then_complete():
    from retroagi.core.models import TACTIC_STANCES
    from retroagi.stages.block_smb.demonstrations import ENEMY_WAIT_FRAMES, enemy_wait_demonstration
    from retroagi.stages.block_smb.monte_carlo import validate_block_smb_monte_carlo_oracle
    from scripts.block_smb_family_learning import samples

    sample, waiting = next(
        (s, w)
        for s in samples("enemy_stomp", 13, "train", 6)
        if (w := enemy_wait_demonstration(s, 3)) is not None
    )
    route = waiting.oracle["actions"]
    assert validate_block_smb_monte_carlo_oracle(sample.scenario, route)["reachable"]
    start = next(i for i, (a, b) in enumerate(zip(route, sample.oracle["actions"])) if a != b)
    held = next(i for i in range(start, len(route)) if route[i] != 0) - start
    assert held in ENEMY_WAIT_FRAMES
    family = BLOCK_SMB_MC_FAMILIES.index("enemy_stomp")
    data = collect_demonstrations([(family, waiting)], tiny_config(), StaticBlockVision)
    assert (
        data.tactic[data.actor_mask & (data.action == 0)] == TACTIC_STANCES.index("hold_area")
    ).all()
    assert (data.actor_mask & (data.action == 0)).any()
    # Avoidance layouts and stationary enemies get no wait routes.
    assert enemy_wait_demonstration(samples("enemy_patrol", 13, "train", 1)[0], 3) is None
    assert enemy_wait_demonstration(samples("enemy_hop", 13, "train", 1)[0], 3) is None


def test_strategy_loss_trains_only_the_objective_reader(timed_demonstrations):
    import copy

    torch.manual_seed(15)
    base = make_block_smb_model(memory_config())
    fitted = []
    for weight in (0.0, 0.5):
        model = copy.deepcopy(base)
        fit_demonstrations(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            replace(timed_demonstrations),
            steps=3,
            batch_size=32,
            decision_durations_only=True,
            walk_durations=False,
            memory_weight=1.0,
            memory_refresh_interval=2,
            memory_unroll=2,
            strategy_loss_weight=weight,
        )
        fitted.append(dict(model.named_parameters()))
    changed = {name for name, value in fitted[0].items() if not torch.equal(value, fitted[1][name])}
    assert changed and all(name.startswith("strategy_network.objective_head.") for name in changed)
