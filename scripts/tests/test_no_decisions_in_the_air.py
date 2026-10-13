"""Decisions are made with Mario standing (or on an enemy he has just
stomped): an action does not end, and an episode does not start, while he is
in the air."""

import pytest

from retroagi.stages.block_smb.controller_teacher import SkillToken
from retroagi.stages.block_smb.teacher_actions import _trial_of

from test_teacher_actions import Decision


@pytest.mark.timeout(120)
def test_the_watched_frames_last_until_mario_has_landed():
    # landing_enemy starts Mario above the floor; he lands after its 12
    # watched frames.
    with Decision("landing_enemy") as d:
        assert d.env.mario["on_ground"]
        assert len(d.env.watched_screens) > d.scenario["watch_frames"]


@pytest.mark.timeout(120)
def test_a_run_onto_a_lower_floor_ends_when_mario_lands_there():
    with Decision("action_descend_back") as d:
        env, m = d.env, d.env.mario
        step = next(p["rect"] for p in env.platforms if p["rect"].top == m["y"] + m["h"])
        floor = max(p["rect"].top for p in env.platforms)
        drop = floor - step.top
        middle = m["x"] + m["w"] / 2
        run = SkillToken("run", round(step.left - middle) - 12, drop)
        result = _trial_of(env, d.state, run)
        assert result.end_dy == drop


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "family, difficulty, q",
    [
        # Mario starts above the floor (decided in the air at frame 12).
        ("landing_enemy", "medium", 0.5),
        # Walking off a low step, vision's contact flickered on at the edge
        # (an action ended there, and the next decision was in the air).
        ("action_descend_back", "easy", 0.0),
    ],
)
def test_the_teacher_decides_standing_or_on_a_stomped_enemy(family, difficulty, q):
    from retroagi.stages.block_smb.family_atlas import atlas_layout, play_teacher

    scenario, why = atlas_layout(family, difficulty, q)
    assert scenario is not None, why
    played = play_teacher(scenario, family, difficulty)
    assert played["won"]
    assert [d["place"] for d in played["decisions"] if d["place"] == "air"] == []
