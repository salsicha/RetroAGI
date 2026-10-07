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


def walker_scene(*positions):
    return SceneObservation(
        MarioView((40, 208, 50, 220), True, "ground", True),
        enemies=tuple(EnemyView((x, 210, x + 10, 220), "walker") for x in positions),
        surfaces=(Surface(8, 248, 220, False),),
    )


def test_target_identity_survives_scroll_and_reversal_but_not_ambiguity():
    tracks = VisualTracks()
    tracks.observe(walker_scene(70), 0)
    target = tracks.tracks[0]
    tracks.observe(walker_scene(69), 2)
    assert tracks.tracks[0] is target and target.velocity == (1, 0)
    tracks.observe(walker_scene(68), 0)
    assert tracks.tracks[0] is target and target.velocity == (-1, 0)
    tracks.observe(walker_scene(67, 69), 0)
    assert all(t is not target for t in tracks.tracks)
    assert not tracks.ready


def test_stomp_target_cannot_transfer_to_a_neighbour_or_count_a_floor_landing():
    scene = walker_scene(65)
    target = Track(scene.enemies[0].box, "walker", [(0, 0)] * 4)
    # Same geometry is not the same identity after an ambiguous/lost track.
    neighbour = Track(target.box, "walker", [(0, 0)] * 4)
    prediction = predict(
        scene,
        [neighbour],
        NESPlayerMotion(),
        scene.mario.box,
        (70, 210),
        1,
        12,
        grounded=True,
        target=target,
    )
    assert not prediction.reached
    floor = predict(
        walker_scene(),
        [],
        NESPlayerMotion(),
        scene.mario.box,
        (70, 220),
        1,
        12,
        grounded=True,
        target=target,
    )
    assert floor.safe and not floor.reached


def test_losing_the_bound_target_reports_loss_without_retargeting():
    scene = walker_scene(65)
    tracks = VisualTracks([Track(scene.enemies[0].box, "walker", [(0, 0)] * 4)])
    flight = plan_flight(scene, tracks, 0, SkillToken("jump", 25, -10), ActionPlan(2, 12))
    assert flight is not None and flight.target is tracks.tracks[0]
    target = flight.target
    tracks.observe(walker_scene(), 0)
    flight.press(walker_scene())
    assert flight.status == "lost_target" and flight.target is target
    assert not flight.prediction.reached


def test_destination_beyond_an_enemy_remains_a_bypass():
    scene = walker_scene(65)
    tracks = VisualTracks([Track(scene.enemies[0].box, "walker", [(0, 0)] * 4)])
    flight = plan_flight(scene, tracks, 2.5, SkillToken("jump", 80, 0), ActionPlan(2, 18))
    assert flight is not None and flight.target is None
    assert flight.prediction.reason == "landed"


@pytest.mark.parametrize("distance", [35, 40, 45])
@pytest.mark.parametrize("tracking", [False, True])
def test_reversing_stomp_target_is_intercepted_in_the_original_jump(distance, tracking):
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 340,
                "mario": [40, 208],
                "platforms": [[0, 220, 340, 20]],
                "enemies": [[65, 206, 37, 68.6, 0.6, 1]],
                "goal_on_stomp": True,
                "goal": [63, 186, 16, 20],
            }
        )
        feedback = SpatialFeedback()
        for _ in range(3):
            feedback.observe(scene_from_labels(env.scene_labels()))
            env.step(0)
        scene = scene_from_labels(env.scene_labels())
        feedback.observe(scene)
        flight = plan_flight(
            scene, feedback.tracks, 0, SkillToken("jump", distance, -14), ActionPlan(2, 12)
        )
        assert flight is not None and flight.target is feedback.tracks.tracks[0]
        feedback.flight = flight
        original_goal = flight.goal
        if not tracking:
            flight.target = None  # Counterfactual: keep the old fixed contact point.
        executor = SMBExecutor()
        executor.start(ActionPlan(2, flight.hold), flight=flight)
        buttons, directions = [], []
        for frame in range(90):
            if frame:
                scene = scene_from_labels(env.scene_labels())
                feedback.observe(scene)
            if executor.finished:
                break
            buttons.append(executor.press(scene))
            _, _, done, _, _ = env.step(buttons[-1])
            directions.append(env.enemies[0]["direction"])
            if done or (frame > 0 and env.mario["on_ground"]):
                break
        assert 1 in directions and -1 in directions
        assert env._goal_credited == tracking
        if tracking:
            assert flight.goal != original_goal
            jumping = [i for i, button in enumerate(buttons) if button in (2, 4, 5)]
            assert jumping == list(range(len(jumping)))
            assert executor.flight is flight
    finally:
        env.close()


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


def test_floating_ledge_does_not_extend_down_to_an_adjacent_lower_step():
    from retroagi.core.smb_trajectory import geometry

    scene = SceneObservation(
        MarioView((70, 208, 80, 220), True, "ground", True),
        blocks=(BlockView((79, 160, 119, 170), "brick"),),
        surfaces=(
            Surface(50, 119, 220, False),
            Surface(79, 119, 160, False),
            Surface(119, 147, 190, False),
            Surface(147, 248, 160, False),
        ),
    )
    solids, _ = geometry(scene, [])
    assert ((79, 160, 119, 170), (0, 0)) in solids
    assert ((79, 160, 119, 190), (0, 0)) not in solids
    flight = plan_flight(scene, VisualTracks(), 0, SkillToken("jump", 40, -30))
    assert flight is not None and flight.prediction.reached
