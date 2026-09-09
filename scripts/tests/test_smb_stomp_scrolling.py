"""Stomp timing must remain observable without static camera landmarks."""

from types import SimpleNamespace

import pytest
import torch

from retroagi.core.smb_coaching import safe_jump_indices, stomp_takeoff_choice
from retroagi.core.smb_components import SMBComponentContract
from retroagi.core.smb_learning import (
    block_stage,
    collect_reactive_case,
    collect_stomp_recovery_case,
    make_model,
)
from retroagi.core.smb_scene import SCENE_ENCODER, CanonicalSMBProjector, enemy_relative_motion
from retroagi.core.smb_supervision import labels_to_vision
from retroagi.core.smb_tracking import PerceivedSMBScene
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.block_smb.geometry_expert import snapshot_env_state
from retroagi.stages.block_smb.nes_curriculum import sample_nes_case
from scripts.tests.test_smb_sensorimotor_repairs import pixels


@pytest.mark.parametrize("scroll_speed", [0, 2, 3, -2])
def test_relative_velocity_cancels_scroll_over_featureless_floor(scroll_speed):
    tracker = PerceivedSMBScene()
    projector = CanonicalSMBProjector(BLOCK_SMB_SPEC)
    for frame in range(6):
        vision = labels_to_vision(
            pixels(
                x=60 + (2 - scroll_speed) * frame,
                enemy=120 + (1 - scroll_speed) * frame,
            )
        )
        geometry = tracker.observe(vision, objective_kind="stomp")
        batch = projector.project(
            vision,
            torch.zeros(1, 35),
            {"smb_geometry": geometry},
            availability=geometry["availability"],
        )
        assert batch.src_c.shape == (1, 64)
        assert batch.metadata["smb_scene_encoder"] == SCENE_ENCODER
        assert not geometry["availability"][2]  # absolute motion is unobservable
        assert batch.src_c[0, 14] == 0  # no false world velocity from screen displacement
        assert batch.src_c[0, 17] == 0.5  # its derived facing is also unavailable
        assert batch.src_c[0, 63] == bool(frame)
        if frame:
            assert enemy_relative_motion(geometry) == (-1, True)
            assert batch.src_c[0, 62] == pytest.approx(-1 / 3)


def test_relative_motion_missing_target_and_reset_are_explicitly_unknown():
    tracker = PerceivedSMBScene()
    for x in (60, 62):
        geometry = tracker.observe(labels_to_vision(pixels(x=x, enemy=120)))
    assert enemy_relative_motion(geometry) == (-2, True)
    missing = tracker.observe(labels_to_vision(pixels(x=64)))
    assert enemy_relative_motion(missing) == (-2, False)
    reappeared = tracker.observe(labels_to_vision(pixels(x=66, enemy=120)))
    assert enemy_relative_motion(reappeared) == (-2, True)
    tracker.reset()
    first = tracker.observe(labels_to_vision(pixels(x=60, enemy=120)))
    assert enemy_relative_motion(first) == (0, False)


def test_required_stomp_target_follows_enemy_during_airborne_camera_lock():
    tracker = PerceivedSMBScene()
    tracker.observe(labels_to_vision(pixels(x=100, enemy=160)), objective_kind="stomp")
    for frame in range(1, 6):
        geometry = tracker.observe(
            labels_to_vision(pixels(x=100, bottom=220 - 3 * frame, enemy=160 - 2 * frame)),
            objective_kind="stomp",
        )
        assert geometry["support"] == "air"
        assert geometry["objective"].left == 160 - 2 * frame
        assert geometry["objective"].kind == "stomp"


def test_relative_motion_oracle_has_same_meaning_including_passive_carry():
    scene = SimpleNamespace(
        mario=dict(x=60, w=10, vx=2, _platform=dict(moving=True, move_speed=0.5, move_dir=1)),
        enemies=[dict(x=120, w=12, speed=1, direction=-1, relative_vx=-3.5)],
    )
    assert enemy_relative_motion(dict(scene=scene, observation_provider="oracle")) == (-3.5, True)
    assert enemy_relative_motion(dict(scene=scene, observation_provider="perceived")) == (
        -3.5,
        True,
    )


def test_old_scene_encoder_cannot_be_silently_swapped():
    with pytest.raises(ValueError, match="scene interface"):
        SMBComponentContract(scene_encoder="canonical_semantic_v2")


def stomp_sample(difficulty="easy"):
    return sample_nes_case(
        family="enemy_stomp", split="validation", seed=20260908, index=10000, difficulty=difficulty
    )


def test_failed_takeoff_receives_short_physical_hold_without_mutating_state():
    stage = block_stage(stomp_sample())
    model = make_model(hidden_dim=8)
    try:
        stage.reset()
        for _ in range(24):
            stage.env.step(1)
        saved = snapshot_env_state(stage.env)
        valid = safe_jump_indices(model, stage.env, 2)
        assert valid == [1, 2, 3, 4, 5, 6]
        action, duration, allowed, delayed = stomp_takeoff_choice(
            model,
            stage.env,
            (1, 0, list(range(16))),
            proposal=2,
        )
        assert action == 2 and duration in valid and allowed == valid and not delayed
        assert duration != 8  # the failed policy held A for sixteen frames
        assert snapshot_env_state(stage.env) == saved
    finally:
        stage.env.close()


def test_nearby_takeoff_coaching_teaches_shorter_holds_on_later_jumps():
    stage = block_stage(stomp_sample())
    model = make_model(hidden_dim=8)
    try:
        early, first = collect_reactive_case(model, stage)
        later, second = collect_reactive_case(model, stage, takeoff_delay=9)
        assert first["success"] and second["success"]
        a = next((i, r[6]) for i, r in enumerate(early) if r[7] and r[4] == 2)
        b = next((i, r[6]) for i, r in enumerate(later) if r[7] and r[4] == 2)
        assert b[0] == a[0] + 9 and b[1] < a[1]
        assert all(r[10][r[6]] for r in later)
    finally:
        stage.env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_momentum_recovery_teaches_braking_before_left_jump(difficulty):
    stage = block_stage(stomp_sample(difficulty))
    original = stage.scenario
    try:
        rows, result = collect_stomp_recovery_case(make_model(hidden_dim=8), stage, family=8)
        assert result["success"]
        decisions = [r[4] for r in rows if r[7]]
        assert decisions.index(3) < decisions.index(4)
        assert stage.scenario is original
    finally:
        stage.env.close()


def test_clipped_enemy_motion_uses_shared_visible_edge():
    tracker = PerceivedSMBScene()
    for frame in range(3):
        image = pixels(x=60)
        image[208:220, : 12 - 2 * frame] = 5
        geometry = tracker.observe(labels_to_vision(image))
        if frame:
            assert enemy_relative_motion(geometry) == (-2, True)


def test_runtime_rejects_observations_with_old_scene_meanings():
    from retroagi.stages.full_smb.train import _smb_forward_kwargs

    model = make_model(hidden_dim=8)
    stage = block_stage(stomp_sample())
    try:
        batch = stage.encode_observation(stage.reset())
        batch.metadata["smb_scene_encoder"] = "canonical_semantic_v2"
        with pytest.raises(ValueError, match="scene encoder"):
            _smb_forward_kwargs(model, batch, True)
    finally:
        stage.env.close()


def test_full_pixel_adapter_receives_same_relative_motion_without_ram():
    from retroagi.core.smb_learning import runtime
    from retroagi.stages.full_smb.adapter import FullSMBStage
    from scripts.tests.test_smb_transfer_contract import RAMEnv

    class Vision:
        frame = 0

        def encode(self, observation):
            image = pixels(x=100, enemy=160 - 2 * self.frame)
            self.frame += 1
            return labels_to_vision(image)

    stage = FullSMBStage(env=RAMEnv(), vision=Vision(), task_objective="stomp")
    stage.configure_policy_runtime(runtime("perceived"))
    try:
        observation = stage.reset()

        def forbidden():
            raise AssertionError("Pixel motion must not read RAM")

        stage.env.get_ram = forbidden
        stage.encode_observation(observation)
        # A new physical frame is required; re-encoding a frame is cached.
        stage._geometry_frame += 1
        batch = stage.encode_observation(observation)
        assert batch.src_c[0, 62] == pytest.approx(-2 / 3)
        assert batch.src_c[0, 63] == 1
    finally:
        stage.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_scrolling_curriculum_routes_preserve_physics_and_use_pixel_motion(difficulty):
    from retroagi.core.smb_learning import collect_stomp_scroll_case
    from retroagi.core.smb_scene import block_oracle_scene
    from retroagi.core.smb_supervision import OracleSceneVision

    vision = OracleSceneVision(lambda: block_oracle_scene(stage.env)["scene"], "cpu")
    stage = block_stage(stomp_sample(difficulty), vision=vision)
    original = stage.scenario
    try:
        rows, result = collect_stomp_scroll_case(
            make_model(hidden_dim=8, provider="perceived"),
            stage,
            family=8,
        )
        assert result["success"]
        assert stage.scenario is original
        assert stage.env.camera_x > 0
        assert any(r[7] and r[4] == 2 and r[2][0, 63] == 1 for r in rows)
    finally:
        stage.env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_stomp_motion_has_no_invisible_midfloor_reversals(difficulty):
    sample = stomp_sample(difficulty)
    assert sample.distribution_id == "block_smb_nes_land_v4"
    assert sample.parameters["family_revision"] == 3
    assert sample.parameters["patrol_halfwidth"] is None
    enemy = sample.scenario["enemies"][0]
    assert enemy[2:4] == [0, sample.scenario["world_width"]]
    stage = block_stage(sample)
    try:
        stage.reset()
        direction = stage.env.enemies[0]["direction"]
        for _ in range(90):
            stage.env.step(0)
            assert stage.env.enemies[0]["direction"] == direction
    finally:
        stage.env.close()


def test_generator_and_coaching_cover_fast_enemies_in_both_directions():
    directions = set()
    for index in (10000, 10002):
        sample = sample_nes_case(
            family="enemy_stomp",
            split="validation",
            seed=20260908,
            index=index,
            difficulty="hard",
        )
        directions.add(sample.parameters["enemy_initial_direction"])
        stage = block_stage(sample)
        try:
            rows, result = collect_reactive_case(make_model(hidden_dim=8), stage)
            assert result["success"]
            assert any(r[7] and r[4] == 2 for r in rows)
        finally:
            stage.env.close()
    assert directions == {-1, 1}
