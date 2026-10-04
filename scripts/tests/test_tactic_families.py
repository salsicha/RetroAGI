"""Explicit tactics per layout, the route and retreat families, and the clones."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.tokens import TACTICS
from retroagi.stages.block_smb import tactic_schedule
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.tactic_families import (
    CHOICE_FAMILIES,
    CLONE_FAMILIES,
    LOW_CHOICE_FAMILIES,
    STRATEGY_FAMILIES,
    TACTIC_FAMILIES,
)
from retroagi.stages.block_smb.tactic_schedule import segment
from retroagi.stages.block_smb.teacher_tokens import (
    _first_plan,
    episode_teacher,
    teacher_plan,
    teacher_skill,
    teacher_tactic,
)


def _flat(tactics, *, platforms=None, enemies=(), mario=(40, 208)):
    return {
        "world_width": 400,
        "mario": list(mario),
        "platforms": platforms or [[0, 220, 400, 20]],
        "enemies": list(enemies),
        "goal": [300, 200, 16, 20],
        "tactics": tactics,
    }


def _run(env, action, frames):
    info = {}
    for _ in range(frames):
        _, _, done, _, info = env.step(action)
        if done:
            break
    return info


def _sample(family, index=0, difficulty="easy"):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=7, sample_index=index, family=family, difficulty=difficulty
    ).scenario


def test_a_forbidden_platform_ends_the_episode_as_a_loss():
    env = MarioScenarioEnv()
    env.reset(
        scenario=_flat(
            [segment("advance", 1, forbidden=[1])],
            platforms=[[0, 220, 100, 20], [100, 220, 300, 20]],
        )
    )
    info = _run(env, 1, 200)
    assert info["off_route"] and info["terminated"] and not env._goal_credited


def test_leaving_a_hold_area_early_loses_and_holding_it_then_advancing_wins():
    schedule = [segment("hold_area", 1, area=16, frames=40), segment("advance", 1)]
    env = MarioScenarioEnv()
    env.reset(scenario=_flat(schedule))
    assert _run(env, 1, 120)["off_route"] and not env._goal_credited
    env.reset(scenario=_flat(schedule))
    _run(env, 0, 40)
    assert tactic_schedule.current(env)["stance"] == "advance"
    _run(env, 1, 400)
    assert env._goal_credited


def test_the_goal_counts_only_after_the_retreat():
    schedule = [segment("retreat", -1, reach_x=60), segment("advance", 1)]
    env = MarioScenarioEnv()
    env.reset(scenario=_flat(schedule, mario=(120, 208)))
    info = _run(env, 1, 400)
    assert not env._goal_credited and not info["terminated"]
    env.reset(scenario=_flat(schedule, mario=(120, 208)))
    _run(env, 3, 80)
    assert tactic_schedule.current(env)["stance"] == "advance"
    _run(env, 1, 400)
    assert env._goal_credited


def test_segments_end_in_order_and_routes_follow_the_platform_underfoot():
    platforms = [[0, 220, 400, 20], [80, 196, 40, 24], [120, 172, 40, 48]]
    schedule = [segment("alternate_route", 1, route=[1, 2], on=2), segment("advance", 1)]
    env = MarioScenarioEnv()
    env.reset(scenario=_flat(schedule, platforms=platforms))
    assert tactic_schedule.next_route_platform(env) == 1
    with pytest.raises(ValueError):
        tactic_schedule.check_schedule([segment("advance", 1, on=0)], 1, 0)
    with pytest.raises(ValueError):
        segment("charge", 1)


def test_the_monster_cannot_be_stomped_is_the_other_kind_and_wakes_on_screen():
    monster = {
        "kind": "monster",
        "x": 120,
        "y": 200,
        "patrol_min": 0,
        "patrol_max": 400,
        "speed": 0.6,
        "direction": -1,
    }
    env = MarioScenarioEnv()
    env.reset(scenario=_flat([segment("advance", 1)], enemies=[monster], mario=(116, 160)))
    info = _run(env, 0, 60)  # falls onto its back
    assert info["death"]
    assert set(env.instance_meta()[1].values()) == {"other"}
    far = dict(monster, x=380)
    env.reset(scenario=_flat([segment("advance", 1)], enemies=[far]))
    _run(env, 0, 30)
    assert env.enemies[0]["x"] == 380  # off screen: it has not woken
    env.close()


def test_every_family_states_its_tactics_and_every_tactic_has_families():
    from retroagi.stages.block_smb.monte_carlo import _family_tactics

    stances = set()
    kinds = set()
    for family in BLOCK_SMB_MC_FAMILIES:
        if family in TACTIC_FAMILIES:
            continue  # made by tactic_families, with their own tactics
        scenario = {"mario": [20, 200], "goal": [200, 200, 16, 20]}
        for seg in _family_tactics(family, scenario):
            stances.add(seg["stance"])
            kinds.add(seg["kind"])
    for family in (
        "dead_end_retreat",
        "monster_retreat",
        "choice_alternate_route",
        "choice_hold_area",
    ):
        for seg in _sample(family)["tactics"]:
            stances.add(seg["stance"])
            kinds.add(seg["kind"])
    # Hold area also comes from the moving-platform, plant and monster segments.
    assert stances == set(TACTICS) and {"bridge", "plant", "monster"} <= kinds


def test_clones_play_the_same_layouts_and_only_their_tactics_differ():
    for group in (CHOICE_FAMILIES, LOW_CHOICE_FAMILIES):
        layouts = {family: _sample(family, index=3) for family in group}
        first = layouts[group[0]]
        for family, scenario in layouts.items():
            assert scenario["platforms"] == first["platforms"]
            assert scenario["mario"] == first["mario"]
        schedules = [str(scenario["tactics"]) for scenario in layouts.values()]
        assert len(set(schedules)) == len(group)


def test_in_the_clones_the_tactic_decides_the_first_skill():
    first = {}
    for family in CHOICE_FAMILIES + LOW_CHOICE_FAMILIES:
        scenario = _sample(family, index=3)
        env = MarioScenarioEnv()
        env.reset(scenario=scenario, seed=0)
        teacher = episode_teacher(scenario)
        teacher.observe_frame(env)
        scene = scene_from_labels(env.scene_labels())
        tactic = teacher_tactic(env, teacher)
        skill = teacher_skill(env, scene, teacher, tactic)
        plan, _ = teacher_plan(env, teacher)
        first[family] = (tactic.stance, skill.kind, plan.action if plan else None)
        env.close()
    assert first["choice_advance"][:2] == ("advance", "jump_gap")
    assert first["choice_alternate_route"][:2] == ("alternate_route", "climb")
    assert first["choice_hold_area"] == ("hold_area", "wait", 0)
    assert first["choice_retreat"][:2] == ("retreat", "retreat")
    assert first["low_choice_advance"][:2] == ("advance", "jump_gap")
    assert first["low_choice_alternate_route"][:2] == ("alternate_route", "descend")


def test_the_monster_family_needs_backing_off_jumping_in_the_tunnel_fails():
    from retroagi.stages.block_smb.local_traversal import LocalObjective, safe_jump_holds

    scenario = _sample("monster_retreat", index=1, difficulty="medium")
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    env.render = lambda: None
    index = next(i for i, e in enumerate(env.enemies) if e.get("kind") == "monster")
    certified = False
    for _ in range(80):  # walk at it inside the tunnel, trying every jump
        monster = env.enemies[index]
        objective = LocalObjective(
            "enemy",
            monster["x"],
            monster["x"] + monster["w"] + 24,
            monster["y"],
            enemy_index=index,
        )
        certified |= bool(env.mario["on_ground"] and safe_jump_holds(env, objective, 1))
        _, _, done, _, _ = env.step(1)
        if done:
            break
    assert not certified
    assert tactic_schedule.current(env)["kind"] == "monster"
    env.close()


def test_composed_scenes_change_tactics_along_the_way():
    for family in ("chained_obstacles", "tactics_bridge_then_gap"):
        tactics = _sample(family)["tactics"]
        assert len(tactics) >= 3
        assert len({(seg["stance"], seg["kind"]) for seg in tactics}) >= 2


def test_the_tactic_learner_leaves_out_the_clones():
    from retroagi.stages.block_smb.layered_train import learner_families

    families = learner_families("tactic", BLOCK_SMB_MC_FAMILIES)
    assert not set(CLONE_FAMILIES) & set(families)
    assert set(CLONE_FAMILIES) <= set(learner_families("skill", BLOCK_SMB_MC_FAMILIES))


def test_only_the_tactic_learner_trains_on_the_strategy_courses():
    from retroagi.stages.block_smb.layered_train import learner_families

    for learner in ("action", "skill"):
        assert not set(STRATEGY_FAMILIES) & set(learner_families(learner, BLOCK_SMB_MC_FAMILIES))
    assert set(STRATEGY_FAMILIES) <= set(learner_families("tactic", BLOCK_SMB_MC_FAMILIES))


def test_the_action_learner_trains_only_on_advance_families():
    from retroagi.stages.block_smb.layered_train import learner_families
    from retroagi.stages.block_smb.monte_carlo import ADVANCE_FAMILIES

    families = learner_families("action", BLOCK_SMB_MC_FAMILIES)
    assert set(families) == set(ADVANCE_FAMILIES) and len(families) == 16
    assert not set(families) & set(TACTIC_FAMILIES)
    for family in ("moving_bridge", "piranha_avoidance", "retreat_recovery", "stomp_recovery"):
        assert family not in families
    for family in families:
        assert {seg["stance"] for seg in _sample(family)["tactics"]} == {"advance"}


def test_the_skill_learner_leaves_out_courses_and_composed_scenes():
    from retroagi.stages.block_smb.layered_train import learner_families
    from retroagi.stages.block_smb.tactic_families import COMPOSED_RECIPES

    families = learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    assert not set(families) & (set(STRATEGY_FAMILIES) | set(COMPOSED_RECIPES))
    assert set(CLONE_FAMILIES) <= set(families) and len(families) == 34


def _labels_along_the_route(family):
    """The teacher's tactic and skill at the start of each stretch of its route."""
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=0, family=family, difficulty="easy"
    )
    env = MarioScenarioEnv()
    env.reset(scenario=sample.scenario, seed=0)
    teacher = episode_teacher(sample.scenario)
    route, made, t = list(sample.oracle["actions"]), [], 0
    while t < len(route) and not env._goal_credited:
        teacher.observe_frame(env)
        plan = _first_plan(route[t:])
        tactic = teacher_tactic(env, teacher)
        skill = teacher_skill(env, scene_from_labels(env.scene_labels()), teacher, tactic)
        made.append((plan.action, tactic, skill))
        for action in route[t : t + plan.frames]:
            env.step(action)
        t += plan.frames
    env.close()
    return made


@pytest.mark.parametrize("family", ["retreat_recovery", "stomp_recovery"])
def test_going_back_to_the_left_is_never_labelled_advance(family):
    made = _labels_along_the_route(family)
    left = [(tactic, skill) for action, tactic, skill in made if action in (3, 4)]
    assert left
    for tactic, skill in left:
        assert tactic.stance == "retreat" and tactic.direction == -1
        assert skill.kind != "advance"


# ── Strategy courses ──────────────────────────────────────────────────────────

_COURSES: dict = {}


def _course(family):
    """A strategy course layout (cached: each sample plays all three strategies)."""
    if family not in _COURSES:
        _COURSES[family] = _sample(family, index=5, difficulty="medium")
    return _COURSES[family]


def _played(scenario, actions):
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    env.render = lambda: None
    info, total = {}, 0.0
    for action in actions:
        _, reward, done, _, info = env.step(action)
        total += reward
        if done:
            break
    coins = sum(coin["collected"] for coin in env.coins)
    env.close()
    return info, total, coins


@pytest.mark.timeout(600)
def test_strategy_courses_share_layouts_and_each_takes_its_best_route():
    from retroagi.stages.block_smb.tactic_families import STRATEGY_ORDER

    courses = {family: _course(family) for family in STRATEGY_FAMILIES}
    first = courses["speed_run_course"]
    for family, scenario in courses.items():
        strategy = STRATEGY_FAMILIES[family]
        assert scenario["platforms"] == first["platforms"]
        assert scenario["strategy"] == strategy
        results = scenario["route_results"]
        assert results == first["route_results"]  # every combination was played
        best = min(results, key=lambda c: STRATEGY_ORDER[strategy](*results[c]))
        assert scenario["strategy_route"] == best
    fastest = first["route_results"][first["strategy_route"]][0]
    assert first["strategy_objective"] == {"deadline": int(fastest * 1.1)}
    assert set(courses["max_coins_course"]["strategy_objective"]) == {"coins"}
    assert "strategy_objective" not in courses["careful_course"]


def test_finishing_after_the_deadline_misses_the_objective():
    scenario = _flat([segment("advance", 1)])
    scenario.update(strategy="speed_run", strategy_objective={"deadline": 60})
    info, _, _ = _played(scenario, [1] * 400)  # walking to x=300 takes longer
    assert info["objective_missed"] and info["reward_terms"]["goal"] == 0
    scenario["strategy_objective"] = {"deadline": 400}
    info, _, _ = _played(scenario, [1] * 400)
    assert not info["objective_missed"] and info["reward_terms"]["goal"] > 50


def test_reaching_the_goal_without_the_coins_misses_the_objective():
    scenario = _flat([segment("advance", 1)])
    scenario.update(
        coins=[[150, 150, 10, 10]],  # above Mario's head: walking on misses it
        strategy="max_coins",
        strategy_objective={"coins": 1},
    )
    info, _, _ = _played(scenario, [1] * 400)
    assert info["objective_missed"] and info["terminated"] and info["reward_terms"]["goal"] == 0
    scenario["coins"] = [[150, 200, 10, 10]]  # on the way: collected
    info, total, _ = _played(scenario, [1] * 400)
    assert not info["objective_missed"] and info["reward_terms"]["goal"] == 50


@pytest.mark.timeout(600)
def test_the_teacher_gives_the_course_strategy_and_other_layouts_a_speed_run():
    from retroagi.stages.block_smb.teacher_tokens import teacher_strategy

    assert teacher_strategy(episode_teacher(_course("max_coins_course"))).kind == "max_coins"
    assert teacher_strategy(episode_teacher(_sample("flat_run"))).kind == "speed_run"
