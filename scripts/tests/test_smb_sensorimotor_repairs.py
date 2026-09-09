"""Physical and pixel regressions for the four September curriculum repairs."""

from types import SimpleNamespace

import numpy as np
import pytest

from retroagi.core.smb_coaching import safe_jump_indices, training_target
from retroagi.core.smb_learning import block_stage, make_model, primitive
from retroagi.core.smb_runtime import make_smb_executor
from retroagi.core.smb_supervision import labels_to_vision
from retroagi.core.smb_tracking import PerceivedSMBScene
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case


def pixels(*, x=60, bottom=220, enemy=None, bridge=None):
    labels = np.zeros((240, 256), dtype=np.uint8)
    labels[220:] = 2
    labels[bottom - 12 : bottom, x : x + 10] = 1
    if enemy is not None:
        labels[208:220, enemy : enemy + 12] = 5
    if bridge is not None:
        labels[220:227, max(0, bridge) : bridge + 120] = 6
    return labels


def test_descending_near_floor_does_not_consume_nes_jump_press():
    tracker = PerceivedSMBScene()
    executor = make_smb_executor(make_model(hidden_dim=8, provider="perceived"))
    for bottom in (212, 219, 220, 220):
        geometry = tracker.observe(labels_to_vision(pixels(bottom=bottom)))
        batch = SimpleNamespace(metadata={"smb_geometry": geometry})
        execution = executor.execute(2, batch=batch, motor_primitives=primitive(4))
        if bottom == 219:
            assert geometry["support"] == "air"
            assert execution.action == 1
            assert executor.committed_action is None
    assert geometry["support"] == "ground"
    assert execution.action == 2


def test_classification_fringe_cannot_be_a_landing_platform():
    tracker = PerceivedSMBScene()
    image = pixels(bottom=161)
    image[161, 59:71] = 2
    tracker.observe(labels_to_vision(image))
    geometry = tracker.observe(labels_to_vision(image))
    assert geometry["support"] == "air"


def test_stomp_survives_overshoot_and_brief_target_occlusion():
    tracker = PerceivedSMBScene()
    tracker.observe(labels_to_vision(pixels(x=40, enemy=76)), objective_kind="stomp")
    geometry = tracker.observe(labels_to_vision(pixels(x=99, enemy=76)), objective_kind="stomp")
    assert geometry["objective"].kind == "stomp"
    assert geometry["objective"].direction == -1
    missing = tracker.observe(labels_to_vision(pixels(x=99)), objective_kind="stomp")
    assert missing["objective"].kind == "stomp"
    assert missing["objective"].direction == -1


def test_stomp_completion_observes_bounce_after_enemy_disappears():
    tracker = PerceivedSMBScene()
    for bottom, enemy in [(195, 76), (203, 76), (210, None), (205, None)]:
        geometry = tracker.observe(
            labels_to_vision(pixels(x=76, bottom=bottom, enemy=enemy)), objective_kind="stomp"
        )
    assert geometry["bouncing"]
    assert tracker.stomp_complete
    for _ in range(2):
        geometry = tracker.observe(labels_to_vision(pixels(x=100)), objective_kind="stomp")
    assert geometry["objective"].kind == "finish"


def test_camera_tracking_ignores_bridge_occluding_floor_and_clipped_width():
    tracker = PerceivedSMBScene()
    velocities = []
    for frame in range(9):
        image = pixels(x=210, bridge=-20 + frame // 2)
        # Fixed gap boundary remains visible below the moving occluder.
        image[227:, 95:180] = 0
        geometry = tracker.observe(labels_to_vision(image))
        if frame:
            assert geometry["scroll"] == 0
            assert geometry["scene"].mario["vx"] == 0
            platform = next(p for p in geometry["scene"].platforms if p.get("moving"))
            velocities.append(platform["move_speed"] * platform["move_dir"])
    assert velocities[-1] == pytest.approx(0.5)
    assert geometry["availability"][5] == 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [101, 202, 20260908])
def test_platform_hop_has_safe_immediate_jump_without_runup(seed, difficulty):
    sample = sample_nes_case(
        family="platform_hop", split="validation", seed=seed, index=10000, difficulty=difficulty
    )
    assert sample.oracle["actions"][0] == 2
    stage = block_stage(sample)
    try:
        stage.reset()
        assert stage.env.mario["on_ground"]
        assert len(safe_jump_indices(make_model(hidden_dim=8), stage.env, 2)) >= 2
    finally:
        stage.env.close()


def test_teacher_targets_live_enemy_behind_player():
    sample = sample_nes_case(
        family="enemy_stomp", split="validation", seed=20260908, index=10000, difficulty="easy"
    )
    stage = block_stage(sample)
    try:
        stage.reset()
        stage.env.mario["x"] = stage.env.enemies[0]["x"] + 35
        target = training_target(stage.env)
        assert target.kind == "stomp"
        assert target.direction == -1
    finally:
        stage.env.close()


def test_supported_body_height_jitter_does_not_lose_takeoff_or_gap_goal():
    tracker = PerceivedSMBScene()
    for height in (12, 13, 12):
        image = pixels(x=82)
        image[220:, 94:149] = 0
        image[208 : 208 + height, 82:92] = 1
        geometry = tracker.observe(labels_to_vision(image))
        assert geometry["support"] == "ground"
        assert geometry["objective"].kind == "gap"


def test_platform_carry_is_not_mario_locomotion_velocity():
    tracker = PerceivedSMBScene()
    for frame in range(5):
        image = pixels(x=110 + frame, bridge=90 + frame)
        image[227:, 95:210] = 0
        geometry = tracker.observe(labels_to_vision(image))
    assert geometry["support"] == "ground"
    assert geometry["scene"].mario["vx"] == pytest.approx(0)
    assert geometry["availability"][2] == 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_overshoot_recovery_coaching_completes_required_stomp(difficulty):
    from retroagi.core.smb_learning import collect_stomp_recovery_case

    sample = sample_nes_case(
        family="enemy_stomp", split="validation", seed=20260908, index=10000, difficulty=difficulty
    )
    stage = block_stage(sample)
    original = stage.scenario
    try:
        rows, result = collect_stomp_recovery_case(make_model(hidden_dim=8), stage, family=8)
        assert result["success"]
        assert any(row[4] == 4 and row[7] for row in rows)
        assert stage.scenario is original
    finally:
        stage.env.close()
