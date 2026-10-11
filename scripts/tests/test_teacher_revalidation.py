"""Layouts the teacher lost in its 2026-10-11 re-validation, won since."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.teacher_replay import replay


def layout(family, index, difficulty):
    return sample_block_smb_monte_carlo_scenario(
        split="train", seed=2026100909, sample_index=index, family=family, difficulty=difficulty
    ).scenario


@pytest.mark.timeout(300)
def test_a_running_start_jumps_at_once_over_a_pit_too_wide_for_a_standing_jump():
    # Mario starts running; a run to the edge would end at rest, from where
    # no jump clears the 64-pixel pit.
    result = replay(layout("pit_leap", 6015, "hard"), family="pit_leap")
    assert result["won"] and result["commands"][0]["skill"]["mode"] == "jump"


@pytest.mark.timeout(300)
def test_a_stomp_is_not_made_where_its_bounce_lands_against_a_walker():
    result = replay(layout("enemy_hop", 6007, "medium"), family="enemy_hop")
    assert result["won"]


@pytest.mark.timeout(300)
def test_a_walker_on_a_raised_platform_is_watched_and_stomped_at_once():
    scenario = layout("action_stomp_up", 10007, "medium")
    assert scenario["watch_frames"] > 0
    result = replay(scenario, family="action_stomp_up")
    assert result["won"] and len(result["commands"]) == 1


@pytest.mark.timeout(300)
@pytest.mark.parametrize("index,difficulty", [(3011, "hard"), (3013, "easy")])
def test_a_stomps_bounce_is_judged_as_the_next_command_plays_it(index, difficulty):
    # With no buttons the bounce stomped the arriving walker too and looked
    # clear; the next command, steering in the air, came down beside it.
    result = replay(layout("enemy_hop", index, difficulty), family="enemy_hop")
    assert result["won"]


@pytest.mark.timeout(600)
def test_an_action_lesson_the_teacher_cannot_win_is_redrawn():
    from collections import Counter

    rejected = Counter()
    sample = sample_block_smb_monte_carlo_scenario(
        split="train",
        seed=2026100909,
        sample_index=3008,
        family="action_stomp_up",
        difficulty="medium",
        rejection_counter=rejected,
    )
    assert rejected["teacher_cannot_aim"] >= 1
    assert replay(sample.scenario, family="action_stomp_up")["won"]
