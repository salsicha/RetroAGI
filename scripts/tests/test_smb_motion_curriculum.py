"""Regression coverage for occlusion, bridge decisions and real policy misses."""

from types import SimpleNamespace

import pygame
import pytest
import torch

from retroagi.core.smb_learning import (
    block_stage,
    collect_reactive_case,
    make_model,
    primitive,
    rows_to_data,
)
from retroagi.core.smb_runtime import make_smb_executor
from retroagi.core.smb_scene import platform_relative_motion
from retroagi.core.smb_supervision import labels_to_vision
from retroagi.core.smb_tracking import PerceivedSMBScene, update_motion_tracks
from retroagi.stages.block_smb.demonstrations import (
    demonstration_groups,
    demonstration_sample_weights,
    priority_sample_weights,
)
from retroagi.stages.block_smb.nes_curriculum import canonical_families, sample_nes_case
from scripts.tests.test_smb_sensorimotor_repairs import pixels


def test_motion_survives_camera_landmark_loss_and_short_body_occlusion():
    tracks = []
    for frame in range(6):
        boxes = [] if frame == 3 else [pygame.Rect(90 + frame, 220, 40, 8)]
        estimates, tracks = update_motion_tracks(
            tracks, boxes, 60 + 2 * frame, frame=frame, camera_delta=0, camera_known=frame < 2
        )
        if frame == 3:
            assert estimates == {}  # no invented invisible support
        elif frame >= 2:
            estimate = next(iter(estimates.values()))
            assert estimate["relative_vx"] == -1
            assert estimate["relative_age"] == 0
            assert estimate["vx"] == 1
            assert estimate["motion_age"] == frame - 1
    estimates, _ = update_motion_tracks(
        tracks, [pygame.Rect(97, 220, 40, 8)], 74, frame=7, camera_delta=0, camera_known=False
    )
    assert next(iter(estimates.values()))["vx"] is None


@pytest.mark.parametrize("scroll", [0, 2, -2])
def test_platform_relative_motion_and_carry_without_static_landmarks(scroll):
    tracker = PerceivedSMBScene()
    for frame in range(6):
        geometry = tracker.observe(
            labels_to_vision(pixels(x=110 + (1 - scroll) * frame, bridge=90 + (1 - scroll) * frame))
        )
    assert platform_relative_motion(geometry) == (0, True, 0)
    assert geometry["scene"].mario["vx"] == 0
    assert geometry["availability"][2] == 1


def test_long_bridge_wait_reconsiders_without_phase_change():
    executor = make_smb_executor(make_model(hidden_dim=8))
    batch = SimpleNamespace(
        metadata={
            "smb_geometry": {"support": "ground", "objective": SimpleNamespace(kind="bridge_wait")}
        }
    )
    assert executor.execute(0, batch=batch, motor_primitives=primitive(15)).action == 0
    executor.prepare(batch)
    assert executor.committed_action is None
    assert executor.execute(1, batch=batch, motor_primitives=primitive(15)).action == 1


def test_critical_bridge_phases_keep_equal_mass_under_priorities():
    rows = []
    for phase, count in [(0, 1000), (1, 50), (3, 2), (4, 30), (5, 1)]:
        for _ in range(count):
            rows.append(
                (
                    torch.zeros(1, 8),
                    torch.zeros(1, 16),
                    torch.zeros(1, 64),
                    torch.zeros(1, 8),
                    1,
                    1,
                    0,
                    True,
                    torch.zeros(1, 64),
                    0,
                    [True] * 16,
                    phase,
                )
            )
    data = rows_to_data(rows)
    weights = demonstration_sample_weights(data)
    prioritized = priority_sample_weights(
        weights, demonstration_groups(data), torch.linspace(0.05, 5, len(rows))
    )
    for phase in data.phase.unique():
        assert weights[data.phase == phase].sum() == pytest.approx(0.2, abs=1e-5)
        assert prioritized[data.phase == phase].sum() == pytest.approx(0.2, abs=1e-5)


def test_wait_timing_is_one_canonical_distribution():
    assert canonical_families(["wait_timing", "moving_bridge", "bridge_wait"]) == [
        "bridge_wait",
        "moving_bridge",
    ]
    cases = [
        sample_nes_case(family=f, split="validation", seed=20260908, index=10000, difficulty="easy")
        for f in ("wait_timing", "bridge_wait")
    ]
    assert cases[0].family == cases[1].family == "bridge_wait"
    assert cases[0].scenario_id == cases[1].scenario_id
    assert cases[0].oracle == cases[1].oracle


@pytest.mark.parametrize("mode", ["takeoff", "miss"])
def test_actual_policy_takeoff_and_miss_recovery(monkeypatch, mode):
    sample = sample_nes_case(
        family="enemy_stomp", split="validation", seed=20260908, index=10000, difficulty="easy"
    )
    stage = block_stage(sample)
    model = make_model(hidden_dim=8)

    def proposal(*args, **kwargs):
        logits = torch.full((1, 6), -20.0)
        logits[0, 1 if stage.env.steps < 24 else 2] = 20
        return SimpleNamespace(logits=logits, motor_primitives=primitive(8))

    monkeypatch.setattr("retroagi.core.smb_learning._policy_action_logits_and_state", proposal)
    try:
        rows, result = collect_reactive_case(model, stage, policy_rollout=mode)
        assert result["success"], result
        assert result["policy_frames"] >= 24
        assert all(row[10][row[6]] for row in rows)
        if mode == "miss":
            assert result["actual_miss_recovery"]["vx"] > 0
            decisions = [row[4] for row in rows if row[7]]
            assert decisions.index(3) < decisions.index(4)
        else:
            assert rows[0][4] == 2 and rows[0][6] in range(1, 7)
    finally:
        stage.env.close()


def test_narrow_bridge_coaching_brakes_instead_of_sliding_off():
    sample = sample_nes_case(
        family="moving_bridge", split="validation", seed=20260908, index=10000, difficulty="easy"
    )
    stage = block_stage(sample)
    try:
        rows, result = collect_reactive_case(make_model(hidden_dim=8), stage)
        assert result["success"]
        assert any(row[4] == 3 and row[7] for row in rows)
        assert {3, 5} <= {row[11] for row in rows if row[7]}
    finally:
        stage.env.close()


def test_occluded_relative_estimate_reaches_model_with_age_and_expires():
    from retroagi.core.smb_scene import CanonicalSMBProjector
    from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC

    tracker = PerceivedSMBScene()
    projector = CanonicalSMBProjector(BLOCK_SMB_SPEC)
    for frame in range(8):
        vision = labels_to_vision(pixels(x=60 + frame, bridge=110 if frame < 2 else None))
        geometry = tracker.observe(vision)
        batch = projector.project(
            vision,
            torch.zeros(1, 35),
            {"smb_geometry": geometry},
            availability=geometry["availability"],
        )
        if 2 <= frame <= 5:
            assert batch.src_c[0, 59] == pytest.approx(-1 / 3)
            assert batch.src_c[0, 60] == 0
            assert batch.src_c[0, 61] == pytest.approx((frame - 1) / 5)
        if frame >= 6:
            assert batch.src_c[0, 59] == 0
            assert batch.src_c[0, 61] == 1
        if frame >= 2:
            assert not any(p.get("moving") for p in geometry["scene"].platforms)
