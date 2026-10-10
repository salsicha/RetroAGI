"""The spatial teacher labels the working command with the most margin.

Among the commands that make the same move, it teaches the one that keeps
Mario farthest from enemies and lands him farthest from a platform's edges.
When several commands give the very same jump, it takes the middle of the
range of destinations that work.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from types import SimpleNamespace

import pytest

from retroagi.stages.block_smb.controller_teacher import Result, _command_room, _rank, _works
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.teacher_replay import replay


def sample(family, index):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family=family, difficulty="medium"
    ).scenario


def start(scenario):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        m = env.mario
        enemies = [(e["x"], e["x"] + e["w"]) for e in env.enemies]
        return m["x"] + m["w"] / 2, [p["rect"] for p in env.platforms], enemies
    finally:
        env.close()


def test_margin_is_the_tighter_of_the_enemy_gap_and_the_edge_margin():
    assert Result(enemy_gap=30.0, edge_margin=12.0).margin == 12.0
    assert Result(enemy_gap=5.0, edge_margin=40.0).margin == 5.0
    assert Result(enemy_gap=7.0).margin == 7.0  # not standing: no edge margin


def test_room_is_largest_in_the_middle_of_the_working_destinations():
    # 48 stomps the enemy (another move); 50 to 96 all pass it the same way.
    tested = [(48, False), (50, True), (64, True), (72, True), (96, True), (32, False)]
    found = [(SimpleNamespace(x=x), SimpleNamespace()) for x, works in tested if works]
    _command_room(found, tested)
    room = {g.x: r.room for g, r in found}
    # The range runs from halfway between 48 and 50 to the last tested, 96.
    assert room == {50: 1.0, 64: 15.0, 72: 23.0, 96: 0.0}
    # With equal margins (the very same jump), the middle ranks first.
    same_jump = [Result(enemy_gap=0.0, frames=53) for _ in found]
    for result, (_, r) in zip(same_jump, found):
        result.room = r.room
    best = max(zip(found, same_jump), key=lambda v: _rank(v[0][0], v[1]))
    assert best[0][0].x == 72
    # A margin larger by 2 pixels or more still comes first.
    wider = Result(enemy_gap=2.0, frames=53)
    assert _rank(None, wider) > _rank(None, best[1])


def test_a_shorter_hop_is_not_a_version_of_a_longer_one():
    outcome = (False, 0, frozenset(), (0, 0), (True,))
    chosen = Result(safe=True, clearance=40, progress=40, outcome=outcome)
    short = Result(safe=True, clearance=90, progress=10, enemy_gap=90.0, outcome=outcome)
    longer = Result(safe=True, clearance=30, progress=44, enemy_gap=30.0, outcome=outcome)
    assert _works(short, chosen)  # the same platform and sides...
    assert not _works(short, chosen, hop=True)  # ...but a hop must keep its progress
    assert _works(longer, chosen, hop=True)
    passed = Result(safe=True, clearance=40, progress=60, outcome=outcome[:4] + ((False,),))
    assert not _works(passed, chosen)  # ending past the enemy is another move


@pytest.mark.timeout(240)
@pytest.mark.parametrize("index", [0, 1])
def test_a_bypass_jump_is_taught_far_from_the_destinations_that_stomp(index):
    scenario = sample("skill_enemy_bypass", index)
    cx, _, enemies = start(scenario)
    result = replay(scenario, family="skill_enemy_bypass")
    assert result["won"]
    jump = result["commands"][0]["skill"]
    assert jump["mode"] == "jump"
    # Destinations up to about the enemy's far side stomp it; the label is
    # well past them.
    far_side = enemies[0][1] - cx
    assert jump["x"] >= far_side + 16, (jump, enemies, cx)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("index", [0, 1])
def test_a_platform_jump_lands_in_the_middle_of_the_platform(index):
    scenario = sample("action_platform_up", index)
    cx, rects, _ = start(scenario)
    result = replay(scenario, family="action_platform_up")
    assert result["won"]
    jump = result["commands"][0]["skill"]
    platform = next(r for r in rects if r.top < 220)
    assert abs(cx + jump["x"] - platform.centerx) <= 4, (jump, platform, cx)


# ── Stomp or pass: the strategy decides, and the teacher follows it ───────────


def test_lessons_that_must_stomp_are_played_for_points():
    must_stomp = ("enemy_stomp", "stomp_mount", "action_stomp", "action_stomp_back")
    passing = ("skill_enemy_bypass", "skill_enemy_bypass_back", "landing_enemy", "enemy_hop")
    for family in must_stomp:
        assert sample(family, 0)["strategy"] == "max_points", family
    for family in passing:
        assert sample(family, 0).get("strategy", "speed_run") == "speed_run", family


def train_sample(family, index, difficulty):
    return sample_block_smb_monte_carlo_scenario(
        split="train", seed=0, sample_index=index, family=family, difficulty=difficulty
    ).scenario


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "family,index,difficulty",
    [("landing_enemy", 8, "hard"), ("landing_enemy", 9, "hard"), ("enemy_hop", 0, "medium")],
)
def test_under_speed_run_the_teacher_passes_an_enemy_it_once_stomped(family, index, difficulty):
    scenario = train_sample(family, index, difficulty)
    coins = len(scenario.get("coins", []))
    result = replay(scenario, family=family)
    assert result["won"]
    # No kill: the points are at most the coins.
    commands = [c["skill"] for c in result["commands"]]
    assert result["points"] <= coins and not any(c["y"] == -10 for c in commands), commands


@pytest.mark.timeout(300)
def test_under_max_points_the_teacher_stomps():
    scenario = train_sample("enemy_stomp", 0, "medium")
    result = replay(scenario, family="enemy_stomp")
    assert result["won"] and result["points"] >= 1


@pytest.mark.timeout(600)
def test_landing_enemy_layouts_that_look_alike_get_the_same_first_move():
    # The walker patrols the whole visible floor, so no hidden turnaround
    # decides between waiting, jumping over and stomping.
    first = []
    for index in range(6):
        result = replay(train_sample("landing_enemy", index, "hard"), family="landing_enemy")
        assert result["won"]
        on_ground = result["commands"][1]["skill"]  # the first after landing
        first.append(on_ground["mode"])
    assert first.count(max(set(first), key=first.count)) >= 5, first


def test_a_jump_moved_to_the_same_place_stays_within_the_command_range():
    from retroagi.core.tokens import SkillToken
    from retroagi.stages.block_smb.controller_teacher import _shifted

    assert _shifted(SkillToken("jump", 72, 0), 16) == SkillToken("jump", 56, 0)
    assert _shifted(SkillToken("jump", -250, -10), 12) == SkillToken("jump", -256, -10)
    assert _shifted(SkillToken("jump", 250, 0), -12) == SkillToken("jump", 256, 0)
