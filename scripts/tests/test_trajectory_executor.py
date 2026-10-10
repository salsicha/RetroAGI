"""The executor attempts destinations independently of terrain/hazard safety."""

from dataclasses import replace

import pytest

from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import (
    BlockView,
    EnemyView,
    MarioView,
    SceneObservation,
    Surface,
)
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_trajectory import Track, VisualTracks, plan_flight
from retroagi.core.tokens import SkillToken


def plain_scene():
    return SceneObservation(
        MarioView((40, 208, 50, 220), True, "ground", True), surfaces=(Surface(8, 248, 220, False),)
    )


def test_obstacles_hazards_and_missing_support_cannot_veto_or_change_a_jump():
    scene = plain_scene()
    blocked = replace(
        scene,
        surfaces=(),
        blocks=(BlockView((50, 120, 200, 230), "brick"),),
        enemies=(EnemyView((50, 200, 66, 220), "other"),),
    )
    plans = []
    for picture in (scene, blocked):
        tracks = VisualTracks([Track((50, 200, 66, 220), "other")])
        flight = plan_flight(picture, tracks, 0, SkillToken("jump", 60, 0))
        plans.append((flight.hold, flight.direction))
        assert flight.press(picture) in (2, 4, 5)
    assert plans[0] == plans[1]


@pytest.mark.parametrize("goal", [SkillToken("jump", 256, -240), SkillToken("jump", -200, -100)])
def test_unreachable_destinations_still_take_off_immediately(goal):
    flight = plan_flight(plain_scene(), VisualTracks(), 0, goal)
    assert flight.prediction.steps
    assert flight.press(plain_scene()) in (2, 4, 5)


def test_losing_vision_releases_buttons_and_reports_failure():
    flight = plan_flight(plain_scene(), VisualTracks(), 0, SkillToken("jump", 40, 0))
    assert flight.press(SceneObservation(MarioView(None, False, "air", False))) == 0
    assert flight.done and flight.status == "lost_observation"


def test_new_jump_has_a_physical_release_edge():
    flight = plan_flight(plain_scene(), VisualTracks(), 0, SkillToken("jump", 40, 0))
    feedback = SpatialFeedback()
    feedback.motion_ready = True
    feedback.velocities.extend([0, 0])
    feedback.speed = 0
    plan = feedback.begin(SkillToken("jump", 40, 0), plain_scene())
    executor = SMBExecutor(last_button=2)
    executor.start(plan, flight=flight)
    assert executor.press(plain_scene()) not in (2, 4, 5)
    assert flight.elapsed == 0
    executor.press(plain_scene())
    assert flight.elapsed == 1


def test_moving_goal_tracks_only_the_requested_object(monkeypatch):
    # In-flight re-aiming exists for when re-prediction is on (Full SMB play).
    from retroagi.core import smb_trajectory

    monkeypatch.setattr(smb_trajectory, "REPLAN_IN_FLIGHT", True)
    scene = plain_scene()
    target = Track((80, 190, 90, 200), "walker")
    flight = plan_flight(scene, VisualTracks([target]), 0, SkillToken("jump", 40, -30))
    assert flight.target is target
    target.box = (83, 190, 93, 200)
    flight.press(scene)
    assert flight.goal == (88, 190)
    neighbour = Track((83, 190, 93, 200), "walker")
    flight.tracks.tracks = [neighbour]
    flight.press(scene)
    assert flight.target is None and flight.goal == (88, 190)


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


def tunnel_scene(enemy=60):
    return SceneObservation(
        MarioView((19, 208, 29, 220), False, "ground", True),
        enemies=(EnemyView((enemy, 200, enemy + 16, 220), "other"),),
        blocks=(BlockView((64, 40, 215, 190), "brick"),),
        surfaces=(Surface(8, 248, 220, False), Surface(64, 215, 40, False)),
    )


def walker_scene(*positions):
    return SceneObservation(
        MarioView((40, 208, 50, 220), True, "ground", True),
        enemies=tuple(EnemyView((x, 210, x + 10, 220), "walker") for x in positions),
        surfaces=(Surface(8, 248, 220, False),),
    )


def test_with_re_prediction_off_a_flight_plays_its_takeoff_plan():
    # Block SMB training: the hold and steering are set at takeoff; a moving
    # target does not change the buttons.
    scene = plain_scene()
    target = Track((80, 190, 90, 200), "walker")
    flight = plan_flight(scene, VisualTracks([target]), 0, SkillToken("jump", 40, -30))
    planned = [step[2] for step in flight.prediction.steps]
    target.box = (95, 190, 105, 200)
    pressed = [flight.press(scene) for _ in range(len(planned) + 3)]
    assert pressed[: len(planned)] == planned
    assert pressed[len(planned) :] == [1, 1, 1]  # then keep steering, button released
    assert flight.released
