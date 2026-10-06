"""Visual controller safety and actual closed-loop motion, independent of weights."""

from collections import deque

import pytest

from retroagi.core.smb_executor import ActionPlan, SMBExecutor
from retroagi.core.smb_physics import NESPlayerMotion
from retroagi.core.smb_scene_labels import (
    BlockView,
    EnemyView,
    MarioView,
    SceneObservation,
    Surface,
    scene_from_labels,
)
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_trajectory import Track, VisualTracks, plan_flight, predict
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv


def tunnel_scene(enemy=60):
    return SceneObservation(
        MarioView((19, 208, 29, 220), False, "ground", True),
        enemies=(EnemyView((enemy, 200, enemy + 16, 220), "other"),),
        blocks=(BlockView((64, 40, 215, 190), "brick"),),
        surfaces=(Surface(8, 248, 220, False), Surface(64, 215, 40, False)),
    )


def test_identical_destination_needs_a_longer_initial_hold_to_clear_the_monster():
    scene = tunnel_scene()
    tracks = VisualTracks([Track(scene.enemies[0].box, "other", [(-0.53, 0)] * 32)])
    short = predict(
        scene,
        tracks.tracks,
        NESPlayerMotion(facing=-1),
        scene.mario.box,
        (61, 220),
        1,
        18,
        grounded=True,
    )
    assert not short.safe and short.reason == "hazard"
    flight = plan_flight(scene, tracks, 0, SkillToken("jump", 37, 0), ActionPlan(2, 18))
    assert flight is not None
    assert flight.hold > 18 and flight.prediction.reached


def test_full_body_clearance_blocks_a_jump_under_a_low_ceiling():
    scene = SceneObservation(
        MarioView((40, 208, 50, 220), True, "ground", True),
        blocks=(BlockView((20, 175, 120, 195), "brick"),),
        surfaces=(Surface(8, 248, 220, False), Surface(20, 120, 175, False)),
    )
    assert (
        plan_flight(scene, VisualTracks(), 0, SkillToken("jump", 40, 0), ActionPlan(2, 32)) is None
    )


def test_scroll_does_not_create_enemy_velocity_and_ambiguous_tracks_are_unknown():
    tracks = VisualTracks()
    for i in range(4):
        tracks.observe(tunnel_scene(61 - i * 2), 2)
    assert tracks.ready and tracks.tracks[0].velocity == (0, 0)
    tracks.observe(tunnel_scene(61), None)
    assert not tracks.ready


def test_reversing_hazard_discards_the_old_direction_before_the_next_prediction():
    tracks = VisualTracks()
    for x in range(70, 60, -1):
        tracks.observe(tunnel_scene(x), 0)
    assert tracks.tracks[0].velocity == (-1, 0)
    tracks.observe(tunnel_scene(62), 0)
    assert tracks.tracks[0].velocity == (1, 0)
    assert not tracks.ready


def test_unknown_takeoff_speed_never_bypasses_the_clearance_check():
    scene = SceneObservation(
        MarioView((40, 208, 50, 220), True, "ground", True),
        blocks=(BlockView((20, 175, 120, 195), "brick"),),
        surfaces=(Surface(8, 248, 220, False),),
    )
    feedback = SpatialFeedback()
    feedback.observe(scene)
    assert feedback.begin(SkillToken("jump", 40, 0), scene, ActionPlan(2, 32)).action == 6
    assert feedback.status == "observing_motion"


def test_ground_contact_corrects_platform_pixels_merged_into_marios_feet():
    scene = SceneObservation(
        MarioView((92, 168, 102, 187), True, "ground", True),
        surfaces=(Surface(8, 104, 180, False), Surface(123, 248, 220, False)),
    )
    flight = plan_flight(scene, VisualTracks(), 0, SkillToken("jump", 47, 40), ActionPlan(2, 18))
    assert flight is not None and flight.goal == (144, 220)
    assert flight.prediction.safe and flight.prediction.reached


def test_raised_ground_face_needs_clearance_even_without_a_block_detection():
    scene = SceneObservation(
        MarioView((95, 208, 105, 220), True, "ground", True),
        surfaces=(
            Surface(8, 40, 220, False),
            Surface(40, 88, 166, False),
            Surface(88, 208, 220, False),
        ),
    )
    flight = plan_flight(scene, VisualTracks(), 0, SkillToken("jump", -17, -54), ActionPlan(4, 24))
    assert flight is not None and flight.hold > 15
    assert flight.prediction.reached


def test_landing_just_before_a_lethal_hazard_arrives_is_rejected():
    scene = tunnel_scene()
    tracks = [Track(scene.enemies[0].box, "other", [(-0.53, 0)] * 32)]
    prediction = predict(
        scene, tracks, NESPlayerMotion(facing=-1), scene.mario.box, (49, 220), 1, 2, grounded=True
    )
    assert not prediction.safe


def test_lost_vision_releases_buttons_and_reports_the_failed_execution():
    scene = tunnel_scene()
    tracks = VisualTracks([Track(scene.enemies[0].box, "other", [(-0.53, 0)] * 32)])
    flight = plan_flight(scene, tracks, 0, SkillToken("jump", 37, 0), ActionPlan(2, 18))
    assert flight.press(None) == 0
    assert flight.done and flight.status == "lost_observation"


def test_missing_mario_cannot_start_an_unchecked_jump():
    feedback = SpatialFeedback()
    scene = SceneObservation(MarioView(None, True, "air", False))
    plan = feedback.begin(SkillToken("jump", 40, 0), scene, ActionPlan(2, 18))
    executor = SMBExecutor()
    executor.start(plan, flight=feedback.flight)
    assert executor.press(scene) == 0 and feedback.status == "lost_observation"


def test_new_flight_release_edge_does_not_advance_its_takeoff_model():
    scene = tunnel_scene()
    tracks = VisualTracks([Track(scene.enemies[0].box, "other", [(-0.53, 0)] * 32)])
    flight = plan_flight(scene, tracks, 0, SkillToken("jump", 37, 0), ActionPlan(2, 18))
    executor = SMBExecutor(last_button=2)
    executor.start(ActionPlan(2, flight.hold), flight=flight)
    assert executor.press(scene) == 1
    assert flight.elapsed == executor.pressed == 0
    assert executor.press(scene) == 2
    assert flight.elapsed == executor.pressed == 1


def test_uncertified_destination_is_reported_and_does_not_launch():
    scene = tunnel_scene()
    feedback = SpatialFeedback(speed=0, velocities=deque([0, 0]))
    feedback.tracks = VisualTracks([Track(scene.enemies[0].box, "other", [(-0.53, 0)] * 32)])
    plan = feedback.begin(SkillToken("jump", 200, -180), scene, ActionPlan(2, 32))
    assert plan.action == 6 and feedback.flight is None
    assert feedback.status == "no_safe_trajectory"


@pytest.mark.parametrize("guarded", [False, True])
def test_actual_monster_crossing_is_one_jump_and_survives_with_visual_feedback(guarded):
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 414,
                "mario": [67, 208],
                "platforms": [[0, 220, 414, 20], [112, 40, 151, 150]],
                "enemies": [
                    {
                        "kind": "monster",
                        "x": 127,
                        "y": 200,
                        "patrol_min": 44,
                        "patrol_max": 295,
                        "speed": 0.53,
                        "direction": -1,
                    }
                ],
            }
        )
        feedback = SpatialFeedback()
        for _ in range(32):
            feedback.observe(scene_from_labels(env.scene_labels()))
            env.step(0)
        executor = SMBExecutor()
        airborne = False
        buttons = []
        info = {}
        for frame in range(100):
            scene = scene_from_labels(env.scene_labels())
            feedback.observe(scene)
            if guarded:
                if executor.idle or executor.finished or executor.reconsider:
                    executor.end("recheck")
                    plan = feedback.begin(SkillToken("jump", 37, 0), scene, ActionPlan(2, 18))
                    executor.start(plan, flight=feedback.flight)
                button = executor.press(scene)
            else:
                button = 2 if frame < 18 else 1
            buttons.append(button)
            _, _, done, _, info = env.step(button)
            airborne |= not env.mario["on_ground"]
            if done or (airborne and env.mario["on_ground"]):
                break
        if guarded:
            assert not info["death"] and airborne and env.mario["on_ground"]
            assert env.mario["x"] >= env.enemies[0]["x"] + env.enemies[0]["w"]
            jumps = [i for i, b in enumerate(buttons) if b in (2, 4, 5)]
            assert 18 < len(jumps) <= 32
            assert jumps == list(range(jumps[0], jumps[-1] + 1))
            assert executor.pressed > executor.plan.frames  # coast belongs to this same flight
        else:
            assert info["death"]
    finally:
        env.close()
