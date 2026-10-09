"""Observed player feedback and simulator collision geometry remain separate."""

import pytest

from retroagi.core.smb_collision import stomp_contact, walker_body
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken


def test_stomp_requires_physical_overlap_and_post_floor_downward_velocity():
    enemy = walker_body((89, 210, 99, 220))
    assert enemy == (89, 210, 99, 216)
    assert walker_body((89.1, 210, 99.10000000001, 220))[3] == 216
    assert not stomp_contact((80, 208, 90, 220), enemy, 0)
    assert not stomp_contact((81, 206, 91, 218), enemy, 4)
    assert stomp_contact((81, 204, 91, 216), enemy, 4)
    assert not stomp_contact((79, 204, 89, 216), enemy, 4)
    assert walker_body((89, 204, 105, 220)) == (89, 204, 105, 220)


def test_landing_reconciles_small_speed_error_that_changes_jump_physics():
    from collections import deque

    from retroagi.core.smb_physics import NESPlayerMotion
    from retroagi.core.smb_scene_labels import MarioView, SceneObservation

    feedback = SpatialFeedback(
        previous=SceneObservation(MarioView((40, 206, 50, 218), True, "air", False)),
        motion=NESPlayerMotion(x_speed=30),
        motion_ready=True,
        velocities=deque([1, 2, 1], maxlen=4),
    )
    feedback.observe(SceneObservation(MarioView((42, 208, 52, 220), True, "ground", True)))
    assert feedback.motion.x_speed == 24


def test_skill_token_still_rejects_fractional_and_out_of_range_destinations():
    for x, y in [(0.5, 0), (0, 0.5), (257, 0), (0, 241)]:
        with pytest.raises(ValueError, match="integer pixels"):
            SkillToken("jump", x, y)
