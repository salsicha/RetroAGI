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
def test_an_action_lesson_the_teacher_cannot_win_in_one_action_is_redrawn():
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
    # The first draw the teacher lost; since run-ups keep their speed it wins
    # it, but in two actions: an action lesson is one action.
    assert rejected["teacher_loses"] + rejected["two_actions"] >= 1
    played = replay(sample.scenario, family="action_stomp_up")
    assert played["won"] and len(played["commands"]) == 1


@pytest.mark.timeout(600)
def test_a_bypass_whose_room_closes_before_the_landing_is_redrawn():
    from collections import Counter

    # A standing enemy, then a walker coming on: by the landing the room
    # between them was 2 pixels.
    rejected = Counter()
    sample = sample_block_smb_monte_carlo_scenario(
        split="train",
        seed=2026100909,
        sample_index=9001,
        family="skill_enemy_bypass",
        difficulty="easy",
        rejection_counter=rejected,
    )
    assert rejected["teacher_loses"] >= 1
    assert replay(sample.scenario, family="skill_enemy_bypass")["won"]


@pytest.mark.timeout(600)
def test_every_skill_layout_with_an_enemy_is_watched_first():
    from retroagi.stages.block_smb.skill_families import WATCH_FRAMES

    for family in ("enemy_hop", "landing_enemy", "stomp_mount", "enemy_patrol"):
        assert layout(family, 0, "medium")["watch_frames"] == WATCH_FRAMES, family
    assert not layout("flat_run", 0, "medium").get("watch_frames")


# ── Lost in the 2026-10-14 re-check ──────────────────────────────────────────


def practice_layout(family, index, difficulty, rejected=None):
    return sample_block_smb_monte_carlo_scenario(
        split="train",
        seed=2026101201,
        sample_index=index,
        family=family,
        difficulty=difficulty,
        rejection_counter=rejected,
    ).scenario


@pytest.mark.timeout(900)
def test_the_teacher_waits_for_a_run_up_rather_than_strand_mario_at_a_wide_pit():
    # A 56-pixel pit no jump from rest crosses, a walker pacing the landing:
    # walking to the edge stranded Mario there until the episode stalled. He
    # waits where he starts, then runs up keeping his speed into the jump.
    result = replay(practice_layout("single_gap", 19, "hard"), family="single_gap")
    assert result["won"]
    skills = [c["skill"] for c in result["commands"]]
    assert skills[0]["mode"] == "hold"
    move = next(s for s in skills if s["mode"] != "hold")
    assert move["mode"] == "run" and move["carry"]


@pytest.mark.timeout(600)
def test_a_one_jump_lesson_only_one_jump_height_wins_is_redrawn():
    from collections import Counter

    from retroagi.stages.block_smb.monte_carlo import _winning_holds

    # A walker standing on a small raised platform: only one height of jump
    # stomped it, and the lesson was lost when vision placed it 4 rows lower.
    rejected = Counter()
    scenario = practice_layout("action_stomp_up_back", 3014, "medium", rejected)
    assert rejected["one_height"] >= 1
    assert len(_winning_holds("action_stomp_up_back", scenario)) >= 2
