"""The spatial teacher labels the working command with the most margin.

Among the commands that make the same move, it teaches the one that keeps
Mario farthest from every enemy and pit edge, before, during and after a
jump, and lands him farthest from a platform's edges. A jump is labelled
where it lands.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.stages.block_smb.controller_teacher import Result, _distance_key, _rank, _works
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


def test_threats_are_compared_from_the_closest_up():
    # Both start 2 pixels from a pit edge (shared); the second lands farther
    # from the next pit, so it ranks first although their closest threat ties.
    near = Result(distances={("pit", 100, 220): 2.0, ("pit", 140, 220): 6.0})
    far = Result(distances={("pit", 100, 220): 2.0, ("pit", 140, 220): 20.0})
    assert _distance_key(far) > _distance_key(near)
    assert near.margin == far.margin == 2.0
    # Distances within 2 pixels tie; beyond 64 pixels they no longer count.
    assert _distance_key(Result(distances={"a": 30.0})) == _distance_key(
        Result(distances={"a": 31.0})
    )
    assert _distance_key(Result(distances={"a": 70.0})) == _distance_key(
        Result(distances={"a": 90.0})
    )


def test_a_jump_is_labelled_where_it_lands():
    from types import SimpleNamespace

    # Equal room: the destination nearest where Mario lands ranks first, not
    # one far beyond his reach that gives the same jump.
    landing = Result(distances={"a": 20.0}, frames=50, end_dx=51.0)
    near, beyond = (
        (SimpleNamespace(x=52), Result(**vars(landing))),
        (SimpleNamespace(x=72), Result(**vars(landing))),
    )
    near[1].aim_error, beyond[1].aim_error = abs(52 - 51.0), abs(72 - 51.0)
    assert _rank(*near) > _rank(*beyond)


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
@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_bypass_jump_lands_between_the_enemy_and_what_follows_it(index):
    scenario = sample("skill_enemy_bypass", index)
    cx, _, _ = start(scenario)
    result = replay(scenario, family="skill_enemy_bypass")
    assert result["won"] and result["points"] == 0  # passed, not stomped
    jump = result["commands"][0]["skill"]
    left, _, width, _ = scenario["goal"]
    assert jump["mode"] == "jump" and left <= cx + jump["x"] <= left + width, (jump, scenario)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_platform_jump_keeps_away_from_the_platform_edges(index):
    scenario = sample("action_platform_up", index)
    cx, rects, _ = start(scenario)
    result = replay(scenario, family="action_platform_up")
    assert result["won"]
    jump = result["commands"][0]["skill"]
    platform = next(r for r in rects if r.top < 220)
    landing = cx + jump["x"]
    assert min(landing - platform.left, platform.right - landing) >= 8, (jump, platform, cx)


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


def test_landing_enemy_walkers_patrol_the_floor_they_stand_on():
    # No hidden turnaround: each walker turns only at a visible floor edge.
    for index in range(4):
        scenario = train_sample("landing_enemy", index, "hard")
        floors = scenario["platforms"]
        for x, y, low, high, *_ in scenario["enemies"]:
            assert any(low == fx and high == fx + fw - 10 for fx, _, fw, _ in floors), scenario


def test_a_jump_moved_to_the_same_place_stays_within_the_command_range():
    from retroagi.core.tokens import SkillToken
    from retroagi.stages.block_smb.controller_teacher import _shifted

    assert _shifted(SkillToken("jump", 72, 0), 16) == SkillToken("jump", 56, 0)
    assert _shifted(SkillToken("jump", -250, -10), 12) == SkillToken("jump", -256, -10)
    assert _shifted(SkillToken("jump", 250, 0), -12) == SkillToken("jump", 256, 0)
