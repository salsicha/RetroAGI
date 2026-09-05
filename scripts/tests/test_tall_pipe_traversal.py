"""Regression coverage for local jump targets and complete pipe traversal."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from retroagi.stages.block_smb.adapter import BlockSMBStage
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    sample_block_smb_monte_carlo_split,
)
from retroagi.stages.block_smb.pipe_traversal import TallPipeTraversal, training_rollout_steps
from retroagi.stages.block_smb.train import (
    BlockSMBSuccessReplay,
    _add_monte_carlo_rollup,
    _finalize_monte_carlo_rollups,
    collect_trajectory,
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
    train_block_smb_epoch,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pipe_sample(difficulty="easy", split="train"):
    return sample_block_smb_monte_carlo_scenario(
        split=split, seed=3, sample_index=0, family="tall_pipe_jump", difficulty=difficulty
    )


class PhaseIntentPolicy(torch.nn.Module):
    """Test intents through the real executor/physics, without oracle bypasses."""

    def __init__(self, *, finish=True):
        super().__init__()
        self.finish = finish
        self.goals = []
        holds = torch.full((1, 1, 16), -30.0)
        holds[..., -1] = 30.0
        self.last_motor_primitives = SimpleNamespace(
            hold_duration_logits=holds, duration_bin_values=torch.arange(1, 17)
        )

    def forward(self, a, b, c, **kwargs):
        goal = kwargs.get("skill_goal")
        self.goals.append(goal.clone() if goal is not None else None)
        action = 1 if self.finish and goal is not None and not goal.any() else 2
        logits = torch.full((1, a.shape[1], 6), -30.0)
        logits[..., action] = 30.0
        return a.float(), c.clone(), torch.zeros_like(c), a.float(), logits, b, b, None


def rollout(sample, model, steps=120, **kwargs):
    stage = BlockSMBStage(scenario=sample.scenario, vision=StaticBlockVision())
    try:
        with torch.no_grad():
            return collect_trajectory(
                model,
                stage,
                sample.scenario_id,
                rollout_steps=steps,
                seed=0,
                deterministic=True,
                device=torch.device("cpu"),
                **kwargs,
            )
    finally:
        stage.env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_mount_target_survives_phase_change_and_finish_remains_learned(difficulty):
    sample = pipe_sample(difficulty)
    policy = PhaseIntentPolicy()
    trajectory = rollout(sample, policy)
    assert trajectory.success
    assert len(trajectory.transitions) > 60
    mount_jumps = [
        span
        for span in trajectory.spans
        if span.command.get("primitive") == "jump"
        and span.state_after["y"] < span.state_before["y"] - 50
        and span.termination_reason == "success"
    ]
    assert mount_jumps
    for span in mount_jumps:
        start = trajectory.transitions[span.start_frame]
        assert start.info["primitive_target_phase"] == "mount"
        assert start.info["primitive_target_x"] == 195.0
        assert not start.info.get("jump_overreach", False)
        assert start.info["primitive_target_hold"] == span.command["held_frames"]
    assert policy.goals[0].any()
    assert not policy.goals[-1].any()
    assert any(t.info["pipe_mounted"] for t in trajectory.transitions)
    assert trajectory.transitions[-1].info["skill_phase"] == "finish"

    # The rollout does not force RIGHT after mounting: a policy that keeps
    # requesting jumps still overshoots and fails to touch the real goal.
    repeating = rollout(sample, PhaseIntentPolicy(finish=False))
    assert not repeating.success
    assert any(t.info["pipe_mounted"] for t in repeating.transitions)
    finish_jumps = [
        t for t in repeating.transitions if t.info.get("primitive_target_phase") == "finish"
    ]
    assert finish_jumps
    assert all(
        t.info["primitive_target_x"] == sample.parameters["goal_x"] + 8 for t in finish_jumps
    )
    assert any(t.info["primitive_target_hold"] < 16 for t in finish_jumps)


def test_mount_requires_actual_pipe_support_and_rearms_after_retreat():
    sample = pipe_sample()
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        traversal = TallPipeTraversal.from_stage(sample.scenario, env)
        env.mario.update(x=185.0, y=traversal.top - env.mario["h"], on_ground=False)
        assert not traversal.observe(env)  # Passing over the top is not mounting.
        env.mario.update(y=204.0, on_ground=True)
        assert not traversal.observe(env)  # A ground landing is not mounting.
        env.mario.update(y=traversal.top - env.mario["h"])
        assert traversal.observe(env)
        assert traversal.phase == "finish"
        env.mario.update(x=130.0, y=204.0)
        assert not traversal.observe(env)
        assert traversal.phase == "mount"
        assert traversal.mounted  # Historical diagnostic remains credited.
    finally:
        env.close()


def test_training_floor_does_not_extend_evaluation_or_other_families():
    sample = pipe_sample()
    assert training_rollout_steps(60, sample.scenario) == 160
    assert training_rollout_steps(200, sample.scenario) == 200
    assert training_rollout_steps(60, {}) == 60
    assert not rollout(sample, PhaseIntentPolicy(), steps=60).success
    assert rollout(
        sample, PhaseIntentPolicy(), steps=training_rollout_steps(60, sample.scenario)
    ).success


def test_training_and_rehearsal_both_receive_full_budget():
    sample = pipe_sample()
    successful = rollout(sample, PhaseIntentPolicy())
    replay = BlockSMBSuccessReplay()
    replay.add(successful, sample.family, sample.scenario_id, sample.scenario)
    config = tiny_config(rollout_steps=60, success_replay_rehearsals_per_epoch=1)
    model = make_block_smb_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    # Isolate scheduling from optimization; the real-physics rollout above
    # supplies a complete successful episode for both collection paths.
    with patch(
        "retroagi.stages.block_smb.train.collect_trajectory", return_value=successful
    ) as collect:
        with patch(
            "retroagi.stages.block_smb.train.compute_block_smb_losses",
            side_effect=lambda *args, **kwargs: {
                "loss_total": next(model.parameters()).square().mean()
            },
        ):
            metrics, _ = train_block_smb_epoch(
                model,
                optimizer,
                [(sample.scenario_id, sample.scenario)],
                config,
                epoch=0,
                device=torch.device("cpu"),
                vision_factory=StaticBlockVision,
                success_replay=replay,
            )
    assert [call.kwargs["rollout_steps"] for call in collect.call_args_list] == [160, 160]
    assert metrics["training_rollout_budget_extensions"] == 2
    assert metrics["training_rollout_steps_max"] == 160


def test_evaluation_distinguishes_mounting_from_finishing():
    samples = replace(
        sample_block_smb_monte_carlo_split(split="validation", seed=3, sample_count=0),
        samples=(pipe_sample(split="validation"),),
    )
    config = tiny_config(evaluation_max_steps=120, evaluation_episodes=2)
    for finish in (False, True):
        with patch(
            "retroagi.stages.block_smb.train.sample_block_smb_monte_carlo_parameter_sweep",
            return_value=samples,
        ):
            result = evaluate_block_smb_monte_carlo(
                PhaseIntentPolicy(finish=finish),
                config,
                split="validation",
                sample_count=1,
                device=torch.device("cpu"),
                vision_factory=StaticBlockVision,
                stratified_repeats_per_difficulty=3,
            )
        family = result["families"]["tall_pipe_jump"]
        assert family["success_rate"] == float(finish)
        assert family["pipe_metrics"] == {
            "episodes": 2,
            "mount_successes": 2,
            "finish_after_mount_successes": 2 if finish else 0,
            "mount_success_rate": 1.0,
            "finish_after_mount_success_rate": float(finish),
        }
        assert (
            result["difficulty_bins"]["tall_pipe_jump:easy"]["pipe_metrics"]
            == family["pipe_metrics"]
        )


def test_conditional_finish_metric_uses_counts_and_no_mount_is_undefined():
    rollups = {}
    for counts in ((2, 0, 0), (2, 1, 1)):
        episodes, mounts, finishes = counts
        _add_monte_carlo_rollup(
            rollups,
            "tall_pipe_jump",
            {
                "pipe_metrics": {
                    "episodes": episodes,
                    "mount_successes": mounts,
                    "finish_after_mount_successes": finishes,
                },
            },
            [],
        )
        if not mounts:
            assert (
                _finalize_monte_carlo_rollups(rollups)["tall_pipe_jump"]["pipe_metrics"][
                    "finish_after_mount_success_rate"
                ]
                is None
            )
    metrics = _finalize_monte_carlo_rollups(rollups)["tall_pipe_jump"]["pipe_metrics"]
    assert metrics["mount_success_rate"] == 0.25
    assert metrics["finish_after_mount_success_rate"] == 1.0
