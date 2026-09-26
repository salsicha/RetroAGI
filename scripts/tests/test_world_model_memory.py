"""The world-model LSTM can carry episodic hazard memory for the actor."""

from dataclasses import replace

import pytest
import torch

from retroagi.core.architectures import make_agent_world_model_critic
from retroagi.core.smb_enemy_history import HAZARD_MEMORY_NAMES
from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig, BlockSMBStage
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
        motion_observations=True,
        hazard_observations=True,
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
    assert plain.carry_stance_history
    assert memory.world_model.memory_head.out_features == len(HAZARD_MEMORY_NAMES)
    assert memory.world_model.lstm.input_size == plain.world_model.lstm.input_size + 32
    # Episode context lives in the LSTM; the stance history stays reset.
    assert not memory.carry_stance_history

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
                stored = batched[row].view(2, 1, hidden)
                assert torch.allclose(stored[0], state.hidden[:, 0], atol=1e-5), row
                assert torch.allclose(stored[1], state.cell[:, 0], atol=1e-5), row
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
    assert data.memory_state.shape[1] == 2 * model.world_model.hidden_size
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
        observation_config=BlockSMBObservationConfig(
            motion_observations=True, hazard_observations=True
        ),
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

    recipe = json.loads(Path("scripts/configs/block_smb_full_volume_revision2.json").read_text())
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
        lambda *args: replace(timed_demonstrations),
    )
    result = train_and_evaluate_block_smb(config, vision_factory=StaticBlockVision)
    epoch = result["history"][0]
    assert epoch["demonstration_rehearsal_updates"] == 2
    assert epoch["demonstration_memory_loss"] > 0
    assert "loss_world_model_memory" in epoch
