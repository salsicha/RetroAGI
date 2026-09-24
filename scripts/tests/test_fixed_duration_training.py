"""Train duration choices at the observations where the executor consumes them."""

import pytest
import torch

from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.train import (
    collect_trajectory,
    compute_block_smb_losses,
    make_block_smb_model,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.mark.parametrize("adaptive", [False, True])
def test_duration_gradients_follow_executor_decision_times(adaptive):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    config = tiny_config(adaptive_duration_control=adaptive)
    model = make_block_smb_model(config)
    sample = sample_block_smb_monte_carlo_scenario(
        split="train", seed=3, sample_index=0, family="stomp_mount", difficulty="easy"
    )
    stage = BlockSMBStage(scenario=sample.scenario, vision=StaticBlockVision())
    try:
        trajectory = collect_trajectory(
            model,
            stage,
            sample.scenario_id,
            rollout_steps=40,
            seed=0,
            deterministic=True,
            use_oracle_actions=True,
            adaptive_duration_control=False,
            device=torch.device("cpu"),
        )
        supervised = [
            step
            for step in trajectory.transitions
            if step.info.get("primitive_outcome_target") is not None
            and step.hold_duration_logits is not None
        ]
        starts = [step for step in supervised if step.info["primitive_frame_index"] == 0]
        assert starts and len(supervised) > len(starts)
        # Independent graph leaves distinguish a takeoff choice from later
        # predictions even when both originate in the same learned head.
        for step in supervised:
            step.hold_duration_logits = torch.zeros_like(
                step.hold_duration_logits, requires_grad=True
            )
        losses = compute_block_smb_losses(
            model, trajectory.transitions, config, torch.device("cpu")
        )
        expected = len(supervised) if adaptive else len(starts)
        assert losses["primitive_outcome_supervised_steps"].item() == expected
        gradients = torch.autograd.grad(
            losses["loss_primitive_outcome"],
            [step.hold_duration_logits for step in supervised],
            allow_unused=True,
        )
        for step, gradient in zip(supervised, gradients):
            if adaptive or step.info["primitive_frame_index"] == 0:
                assert gradient is not None and torch.isfinite(gradient).all()
            else:
                assert gradient is None

        # Outcome prediction still learns from every committed frame.
        assert all(step.info.get("primitive_outcome_batch") is not None for step in supervised)
    finally:
        stage.env.close()
        torch.set_num_threads(previous)
