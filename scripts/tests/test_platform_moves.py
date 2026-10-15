"""Moves onto moving platforms, and how long actions last."""

from collections import deque
from types import SimpleNamespace

import numpy as np

from retroagi.core.smb_scene_labels import MarioView, SceneObservation, Surface
from retroagi.core.smb_trajectory import Track, VisualTracks


def test_boarding_request_moves_onto_platform_instead_of_braking_back_to_shore():
    from retroagi.core.smb_ground_control import GroundMove
    from retroagi.core.smb_physics import NESPlayerMotion
    from retroagi.core.smb_spatial_feedback import SpatialFeedback

    scene = SceneObservation(
        MarioView((81, 208, 91, 220), True, "moving_platform", True),
        surfaces=(Surface(8, 85, 220, False),),
        moving_platforms=((90, 220, 190, 232),),
    )
    track = Track(scene.moving_platforms[0], "platform", [(-2.1, 0)] * 4)
    feedback = SpatialFeedback(
        tracks=VisualTracks([track]),
        velocities=deque([0, 0]),
        motion=NESPlayerMotion(x_speed=0, moving=1),
    )
    move = GroundMove(feedback, 14, 0, 220)
    assert move.press(scene) == 1
    assert not move.done  # Partial contact is not completion.
    assert move.target is track


def test_duration_audit_counts_completed_actions_and_censors_final_cutoff():
    from retroagi.stages.block_smb.layered_train import action_duration_summary

    episode = SimpleNamespace(
        frames=241, decision_frames=np.array([0, 7, 72, 240]), last_action_ended=False
    )
    assert action_duration_summary([episode]) == {
        "completed": 3,
        "max_frames": 168,
        "over_64_frames": 2,
    }
    # The last action counts only when it ended before the episode did.
    episode.last_action_ended = True
    assert action_duration_summary([episode])["completed"] == 4
