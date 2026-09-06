"""Behavioral regressions for action credit, local scoring, and successful replay."""

from unittest.mock import patch

import pytest
import torch

from retroagi.core.actions import SMBParameterizedPrimitiveExecutor
from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.train import (
    BlockSMBSuccessReplay,
    _action_from_model,
    block_smb_evaluation_target,
    collect_trajectory,
    make_block_smb_model,
    train_block_smb_epoch,
)
from scripts.tests import test_core_models as model_test_helpers
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config
from scripts.tests.test_tall_pipe_traversal import PhaseIntentPolicy, pipe_sample, rollout


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_sampled_action_drives_motor_and_world_model_despite_critic_rejection():
    model, agent, inputs = model_test_helpers.TestRankedCandidateSearch()._make_model(
        (-1.0,), (0.99,)
    )
    with patch.object(torch.distributions.Categorical, "sample", return_value=torch.tensor([2])):
        outputs = model(*inputs, policy_action_mode="sample")
    assert model.last_selected_action_id == 2
    assert agent.forced_actions == [None, 2]
    # Preserve the distribution actually sampled, not forced candidate logits.
    assert outputs[4][0, -1].argmax().item() == 1
    assert model.last_action_refinement.iterations == 1


def test_local_mount_can_progress_even_when_final_goal_distance_increases():
    helper = model_test_helpers.TestDeterministicCriticGates()
    model = helper._make_model()
    model.deterministic_critic_slots.update(position_x=0, position_y=1)
    model.action_evaluation_target = torch.tensor([[0.6, 0.4, 0.05, 0.01]])
    current = [0.5, 0.8, 0, 0.3, 0, 0, 0, 0]
    mounted = [0.6, 0.4, 0, 0.5, 0, 0, 0, 0]
    retreat = [0.3, 0.8, 0, 0.2, 0, 0, 0, 0]
    assert helper._evaluate(model, current, mounted).would_progress.item()
    assert not helper._evaluate(model, current, retreat).would_progress.item()


def test_committed_frames_have_no_action_choice_credit():
    sample = pipe_sample()
    stage = BlockSMBStage(scenario=sample.scenario, vision=StaticBlockVision())
    model = make_block_smb_model(tiny_config())
    controller = SMBParameterizedPrimitiveExecutor()
    try:
        obs = stage.reset(seed=0)
        batch = stage.encode_observation(obs)
        controller.execute(2, support_override="ground")
        result = _action_from_model(
            model,
            batch,
            deterministic=False,
            tau=1,
            primitive_executor=controller,
            support_override="ground",
            enemy_contact_override=False,
        )
        assert result[0] == 2
        assert result[1].item() == 0  # No fabricated probability for held buttons.
        assert result[2].item() == 0
    finally:
        stage.env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tall_pipe_coaching_certifies_successful_collision_holds(difficulty):
    sample = pipe_sample(difficulty)
    trajectory = rollout(sample, PhaseIntentPolicy())
    assert trajectory.success
    certified = [t for t in trajectory.transitions if t.info.get("primitive_valid_hold_frames")]
    assert certified
    assert any(t.info.get("primitive_target_phase") == "mount" for t in certified)
    assert not any(t.info.get("jump_overreach") for t in certified)


def test_saved_success_can_be_reexecuted_with_a_different_model():
    sample = pipe_sample()
    solved = rollout(sample, PhaseIntentPolicy())
    buffer = BlockSMBSuccessReplay()
    buffer.add(solved, sample.family, sample.scenario_id, sample.scenario)
    record = buffer.sample_scenarios(1)[0]
    assert record["actions"] == tuple(t.action for t in solved.transitions)
    model = make_block_smb_model(tiny_config())
    stage = BlockSMBStage(scenario=record["scenario"], vision=StaticBlockVision())
    try:
        demo = collect_trajectory(
            model,
            stage,
            sample.scenario_id,
            rollout_steps=160,
            seed=0,
            deterministic=True,
            device=torch.device("cpu"),
            demonstration_actions=record["actions"],
        )
        assert demo.success
        assert all(t.log_prob.item() == 0 for t in demo.transitions)
        assert [t.action for t in demo.transitions] == list(record["actions"])
    finally:
        stage.env.close()


@pytest.mark.parametrize("demo_succeeds", [True, False])
def test_failed_practice_only_admits_revalidated_success(demo_succeeds):
    sample = pipe_sample()
    solved = rollout(sample, PhaseIntentPolicy())
    failed = rollout(sample, PhaseIntentPolicy(finish=False))
    buffer = BlockSMBSuccessReplay()
    buffer.add(solved, sample.family, sample.scenario_id, sample.scenario)
    config = tiny_config(episodes_per_epoch=1, success_replay_rehearsals_per_epoch=1)
    model = make_block_smb_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    with patch(
        "retroagi.stages.block_smb.train.collect_trajectory",
        side_effect=[failed, failed, solved if demo_succeeds else failed],
    ) as collect:
        with patch(
            "retroagi.stages.block_smb.train.compute_block_smb_losses",
            side_effect=lambda *a, **kw: {"loss_total": next(model.parameters()).square().mean()},
        ):
            metrics, _ = train_block_smb_epoch(
                model,
                optimizer,
                [(sample.scenario_id, sample.scenario)],
                config,
                0,
                device=torch.device("cpu"),
                vision_factory=StaticBlockVision,
                success_replay=buffer,
            )
    assert collect.call_args_list[-1].kwargs["demonstration_actions"]
    assert metrics["success_rehearsal_success_rate"] == 0
    key = "retention_demonstrations" if demo_succeeds else "retention_demonstration_rejections"
    assert metrics[key] == 1


def test_policy_credit_is_the_probability_of_the_actual_sample():
    sample = pipe_sample()
    stage = BlockSMBStage(scenario=sample.scenario, vision=StaticBlockVision())
    model = make_block_smb_model(tiny_config())
    try:
        observation = stage.reset(seed=0)
        batch = stage.encode_observation(observation)
        with patch.object(
            torch.distributions.Categorical, "sample", return_value=torch.tensor([2])
        ):
            result = _action_from_model(model, batch, deterministic=False, tau=1)
        assert result[0] == 2
        logits = result[4][-1][0, -1, :6]
        assert torch.allclose(result[1], torch.log_softmax(logits, -1)[2])
        (-result[1]).backward()
        assert model.agent.fc_out_A.weight.grad.abs().sum() > 0
    finally:
        stage.env.close()


def test_finished_bridge_targets_goal_instead_of_the_entire_shore():
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

    sample = sample_block_smb_monte_carlo_scenario(
        split="train", seed=1, sample_index=0, family="bridge_wait", difficulty="easy"
    )
    stage = BlockSMBStage(scenario=sample.scenario, vision=StaticBlockVision())
    try:
        stage.reset(seed=0)
        final = block_smb_evaluation_target(stage.env, phase="finish", bridge=True)
        assert torch.equal(final, block_smb_evaluation_target(stage.env))
        assert not torch.equal(
            final, block_smb_evaluation_target(stage.env, phase="board", bridge=True)
        )
    finally:
        stage.env.close()
