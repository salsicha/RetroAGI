"""Actual policy timing/recovery and error-adaptive practice regressions."""

from types import SimpleNamespace

import pytest
import torch

from retroagi.core.smb_learning import block_stage, collect_reactive_case, make_model, primitive
from retroagi.stages.block_smb.demonstrations import adaptive_group_weights, priority_sample_weights
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case


def stage_for(family, index=10000, difficulty="easy"):
    return block_stage(
        sample_nes_case(
            family=family, split="validation", seed=20260908, index=index, difficulty=difficulty
        )
    )


def install_policy(monkeypatch, stage, *, jump_frame, hold):
    def forward(*args, **kwargs):
        logits = torch.full((1, 6), -20.0)
        logits[0, 1 if stage.env.steps < jump_frame else 2] = 20
        return SimpleNamespace(logits=logits, motor_primitives=primitive(hold))

    monkeypatch.setattr("retroagi.core.smb_learning._policy_action_logits_and_state", forward)


def test_early_pipe_takeoff_gets_its_actual_longer_duration(monkeypatch):
    stage = stage_for("tall_pipe_jump")
    install_policy(monkeypatch, stage, jump_frame=50, hold=9)
    try:
        rows, result = collect_reactive_case(
            make_model(hidden_dim=8), stage, policy_rollout="takeoff"
        )
        assert result["success"]
        assert result["policy_frames"] == 50
        assert rows[0][4] == 2 and rows[0][6] >= 10
        assert not rows[0][10][9]
        assert result["policy_diagnostics"]["unsafe_holds"] == 1
    finally:
        stage.env.close()


def test_impossible_stomp_mount_takeoff_teaches_approach(monkeypatch):
    stage = stage_for("stomp_mount", 10011, "hard")
    install_policy(monkeypatch, stage, jump_frame=3, hold=15)
    try:
        rows, result = collect_reactive_case(
            make_model(hidden_dim=8), stage, policy_rollout="takeoff"
        )
        assert result["success"]
        assert rows[0][4] == 1 and rows[0][7]
        assert result["policy_diagnostics"]["unsafe_takeoffs"] == 1
        assert result["policy_frames"] == 3
    finally:
        stage.env.close()


def test_failed_pipe_landing_teaches_recovery(monkeypatch):
    stage = stage_for("tall_pipe_jump")
    install_policy(monkeypatch, stage, jump_frame=50, hold=9)
    try:
        rows, result = collect_reactive_case(make_model(hidden_dim=8), stage, policy_rollout="miss")
        assert result["success"], result
        assert result["actual_miss_recovery"]["target"] == "mount"
        assert any(row[4] == 2 and row[7] for row in rows)
    finally:
        stage.env.close()


def test_walk_against_pipe_gets_a_jump_recovery(monkeypatch):
    stage = stage_for("tall_pipe_jump")
    install_policy(monkeypatch, stage, jump_frame=1000, hold=9)
    try:
        rows, result = collect_reactive_case(make_model(hidden_dim=8), stage, policy_rollout="miss")
        assert result["success"]
        assert result["actual_miss_recovery"]["reason"] == "wall_stall"
        assert rows[0][4] == 2
    finally:
        stage.env.close()


def test_adaptive_replay_increases_braking_and_preserves_family_retention():
    # Family 0: 100 ride-brake rows, 100 exit-walk rows. Family 1: ordinary walk.
    groups = torch.tensor([31] * 100 + [36] * 100 + [43] * 100)
    base = torch.full((300,), 0.005)
    base[200:] = 0.01
    counts = torch.bincount(groups).float()
    errors = torch.zeros_like(counts)
    errors[31] = 1
    weights = adaptive_group_weights(base, groups, counts, errors)
    assert weights[:100].sum() > base[:100].sum()
    assert weights[:200].sum() == pytest.approx(1.0, abs=1e-5)
    assert weights[200:].sum() == pytest.approx(1.0, abs=1e-5)
    assert bool((weights >= 0.35 * base).all())
    priorities = torch.linspace(0.05, 5, 300)
    weighted = priority_sample_weights(weights, groups, priorities)
    for group in groups.unique():
        assert weighted[groups == group].sum() == pytest.approx(
            float(weights[groups == group].sum()), abs=1e-5
        )
    # A corrected group gives its allocation back; stale failure priority is not permanent.
    errors.zero_()
    assert torch.allclose(adaptive_group_weights(base, groups, counts, errors), base)


def test_tiny_correction_group_cannot_get_full_error_boost():
    groups = torch.tensor([17] * 2 + [31] * 100 + [36] * 100)
    base = torch.cat((torch.full((2,), 0.25), torch.full((200,), 0.005)))
    counts = torch.bincount(groups).float()
    errors = torch.ones_like(counts)
    weights = adaptive_group_weights(base, groups, counts, errors)
    tiny_ratio = weights[:2].sum() / base[:2].sum()
    supported_ratio = weights[2:102].sum() / base[2:102].sum()
    assert supported_ratio > tiny_ratio


def test_compound_miss_collection_continues_after_a_successful_stomp(monkeypatch):
    from retroagi.core.smb_coaching import coach_choice, training_target

    stage = stage_for("chained_obstacles", 10006)
    model = make_model(hidden_dim=8)

    def forward(*args, **kwargs):
        action, hold, _ = coach_choice(model, stage.env)
        if training_target(stage.env).kind == "mount" and stage.env.mario["on_ground"]:
            action, hold = 2, 0  # execute a real too-short jump at the later obstacle
        logits = torch.full((1, 6), -20.0)
        logits[0, action] = 20
        return SimpleNamespace(logits=logits, motor_primitives=primitive(hold))

    monkeypatch.setattr("retroagi.core.smb_learning._policy_action_logits_and_state", forward)
    try:
        rows, result = collect_reactive_case(model, stage, policy_rollout="miss")
        assert result["success"], result
        assert stage.env._stomp_credited
        assert result["actual_miss_recovery"]["target"] == "mount"
        assert result["policy_diagnostics"]["jump_proposals"] >= 2
        assert any(row[4] == 2 and row[7] for row in rows)
    finally:
        stage.env.close()


@pytest.mark.parametrize("mode", ["decision", "brake"])
def test_bridge_collects_correction_at_actual_policy_braking_state(monkeypatch, mode):
    from retroagi.core.smb_coaching import coach_choice

    stage = stage_for("moving_bridge")
    model = make_model(hidden_dim=8)

    def forward(*args, **kwargs):
        action, hold, _ = coach_choice(model, stage.env)
        if stage.env._bridge_boarded:
            action = 1
        logits = torch.full((1, 6), -20.0)
        logits[0, action] = 20
        return SimpleNamespace(logits=logits, motor_primitives=primitive(hold))

    monkeypatch.setattr("retroagi.core.smb_learning._policy_action_logits_and_state", forward)
    try:
        rows, result = collect_reactive_case(model, stage, policy_rollout=mode)
        assert result["success"], result
        assert result["policy_diagnostics"]["missed_brakes"] == 1
        assert rows[0][4] == 3 and rows[0][7]
        assert result["policy_frames"] > 0
    finally:
        stage.env.close()


def test_validation_reports_partial_family_scores_without_coaching(monkeypatch):
    from scripts import smb_composable_training as training

    model = torch.nn.Linear(1, 1)
    cases = [
        SimpleNamespace(family="flat_run", difficulty_bin="easy", scenario_id=str(i))
        for i in range(12)
    ]
    monkeypatch.setattr(
        training,
        "block_stage",
        lambda *a, **k: SimpleNamespace(env=SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(
        training, "playback", lambda *a, **k: dict(success=True, death=False, frames=10, actions=[])
    )
    events = []
    result = training.evaluate(model, cases, None, log=events.append)
    assert len(result["results"]) == 12
    assert [e["completed"] for e in events] == [10, 12]
    assert events[-1]["family_successes"] == 12
