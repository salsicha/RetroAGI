"""The tactic layer's families: one scene per tactic, under each of the two strategies."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.core.tokens import STRATEGIES, TACTICS
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.strategy_families import (
    STRATEGY_ORDER,
    STRATEGY_SCENE_FAMILIES,
    STRATEGY_TACTIC_FAMILIES,
)
from retroagi.stages.block_smb.tactic_schedule import segment
from retroagi.stages.block_smb.teacher_tokens import (
    _first_plan,
    episode_teacher,
    teacher_strategy,
    teacher_tactic,
)

_SAMPLES: dict = {}


def _sample(family, index=0, difficulty="medium"):
    """A layout of a family (cached: siblings play the same layouts)."""
    key = (family, index, difficulty)
    if key not in _SAMPLES:
        _SAMPLES[key] = sample_block_smb_monte_carlo_scenario(
            split="validation", seed=7, sample_index=index, family=family, difficulty=difficulty
        )
    return _SAMPLES[key]


def _flat(**extra):
    scenario = {
        "world_width": 400,
        "mario": [40, 208],
        "platforms": [[0, 220, 400, 20]],
        "goal": [300, 200, 16, 20],
        "tactics": [segment("advance", 1)],
    }
    scenario.update(extra)
    return scenario


def _played(scenario, actions):
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    env.render = lambda: None
    info, terms = {}, {}
    for action in actions:
        _, _, done, _, info = env.step(action)
        for name, value in info["reward_terms"].items():
            terms[name] = terms.get(name, 0.0) + value
        if done:
            break
    points = env.points()
    env.close()
    return info, terms, points


def test_there_are_two_strategies_and_fourteen_families_one_per_tactic_and_strategy():
    from retroagi.stages.block_smb.layered_train import learner_families
    from retroagi.stages.block_smb.tactic_families import MULTI_TACTIC_FAMILIES

    assert STRATEGIES == ("speed_run", "max_points")
    assert len(STRATEGY_TACTIC_FAMILIES) == 14
    assert set(STRATEGY_TACTIC_FAMILIES.values()) == {(s, t) for s in STRATEGIES for t in TACTICS}
    assert set(STRATEGY_TACTIC_FAMILIES) <= set(BLOCK_SMB_MC_FAMILIES)
    families = learner_families("tactic", BLOCK_SMB_MC_FAMILIES)
    compositions = {f for f in BLOCK_SMB_MC_FAMILIES if f.startswith("tactics_")}
    assert len(compositions) == 4
    assert set(families) == set(STRATEGY_TACTIC_FAMILIES) | set(STRATEGY_SCENE_FAMILIES) | compositions | set(
        MULTI_TACTIC_FAMILIES
    ) | {
        "mixed_section",
        "chained_obstacles",
        "chained_enemy_gauntlet",
        "full_smb_opening_proxy",
        "full_smb_blocks_proxy",
        "full_smb_rows_proxy",
        "pen_pipe_climb",
        "upper_route",
        "lower_route",
        "dead_end_retreat",
        "monster_retreat",
        "moving_bridge",
        "wait_timing",
        "piranha_avoidance",
        "choice_alternate_route",
        "low_choice_alternate_route",
        "low_choice_advance",
    }
    assert len(families) == 49


@pytest.mark.timeout(600)
@pytest.mark.parametrize("tactic", ["advance", "climb_backward", "hold_ground"])
def test_siblings_share_layouts_and_each_strategy_takes_its_best_route(tactic):
    fast = _sample(f"speed_run_{tactic}").scenario
    rich = _sample(f"max_points_{tactic}").scenario
    for key in ("platforms", "coins", "enemies", "goal", "route_results"):
        assert fast[key] == rich[key]  # the same layout, every combination played
    assert fast["coins"] and _sample(f"speed_run_{tactic}").parameters["detours"]
    results = fast["route_results"]
    for scenario in (fast, rich):
        order = STRATEGY_ORDER[scenario["strategy"]]
        assert scenario["strategy_route"] == min(results, key=lambda c: order(*results[c]))
    fastest, richest = results[fast["strategy_route"]], results[rich["strategy_route"]]
    reference = fast["strategy_reference"]
    assert reference["execution"] == "spatial" and reference["vision_sha256"]
    assert fast["strategy_objective"] == {"deadline": int(reference["frames"] * 1.1)}
    assert richest[1] > fastest[1]  # max points gathers more than speed run
    assert fastest[1] < rich["strategy_objective"]["points"] <= richest[1]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "family", [f for f in STRATEGY_TACTIC_FAMILIES if f.startswith("speed_run_")]
)
def test_the_teacher_wins_with_the_strategys_objective_and_teaches_the_scenes_tactic(family):
    _, tactic = STRATEGY_TACTIC_FAMILIES[family]
    sample = _sample(family, index=1, difficulty="easy")
    scenario = sample.scenario
    assert teacher_strategy(episode_teacher(scenario)).kind == "speed_run"
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    env.render = lambda: None
    teacher = episode_teacher(scenario)
    route, t, labels = list(sample.oracle["actions"]), 0, set()
    while t < len(route) and not env._goal_credited:
        teacher.observe_frame(env)
        plan = _first_plan(route[t:])
        labels.add(teacher_tactic(env, teacher).stance)
        for action in route[t : t + plan.frames]:
            env.step(action)
        t += plan.frames
    assert env._goal_credited and not env._objective_missed, family
    assert tactic in labels, (family, labels)
    env.close()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_the_hold_scene_shows_what_mario_waits_for_and_the_teacher_holds_first(seed):
    # User, 2026-10-23: "Hold should hold for things like plant or moving
    # bridge platform". The scene had been the advance scene with a timed
    # hold nothing in the picture asked for.
    import random

    from retroagi.stages.block_smb.monte_carlo import _finish_layout
    from retroagi.stages.block_smb.strategy_families import LIFT_AWAY, strategy_scene

    scenario, parameters = strategy_scene(random.Random(seed), "medium", "hold_ground")
    advance, _ = strategy_scene(random.Random(seed), "medium", "advance")
    assert scenario["platforms"] != advance["platforms"]
    _finish_layout("speed_run_hold_ground", scenario, parameters)
    scenario["tactics"] = scenario.pop("route_tactics")["coins_up=skip"]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario, seed=0)
        env.render = lambda: None
        if parameters["hold_on"] == "plant":
            plant = env.enemies[0]
            assert plant["kind"] == "piranha_plant" and plant["timed_crossing"]
            assert plant["h"] == plant["plant_height"]  # fully up in the first picture
        else:
            lift = next(p for p in env.platforms if p["moving"])
            assert lift["rect"].x >= env.platforms[0]["rect"].right + LIFT_AWAY
            assert lift["move_dir"] == 1  # moving away
        teacher = episode_teacher(scenario)
        teacher.observe_frame(env)
        assert teacher_tactic(env, teacher).stance == "hold_ground"
    finally:
        env.close()


def test_the_hold_scene_waits_for_a_plant_or_a_moving_platform():
    import random

    from retroagi.stages.block_smb.strategy_families import strategy_scene

    kinds = {strategy_scene(random.Random(s), "medium", "hold_ground")[1]["hold_on"] for s in (0, 1, 2, 5)}
    assert kinds == {"plant", "bridge"}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_block_goomba_max_points_stomps_it_clear_of_the_block_and_speed_run_goes_over(difficulty):
    # User, 2026-10-27: "one family (max points) where mario waits for the
    # goomba to be clear of the block and then stomps it. and another (speed
    # run) where mario simply ignores the goomba and jumps over the block".
    made = {
        family: sample_block_smb_monte_carlo_scenario(
            split="validation", seed=7, sample_index=0, family=family, difficulty=difficulty,
            calibrate_deadline=False,
        )
        for family in ('speed_run_block_goomba', 'max_points_block_goomba')
    }
    fast, rich = made["speed_run_block_goomba"], made["max_points_block_goomba"]
    assert fast.scenario["platforms"] == rich.scenario["platforms"]  # the same layout
    assert fast.scenario["strategy_route"].startswith("goomba=over")
    assert rich.scenario["strategy_route"].startswith("goomba=stomp")
    block_x, top = fast.parameters["block_x"], 220 - fast.parameters["block_bottom_up"] - 16
    for sample, stomped, over in ((fast, False, True), (rich, True, False)):
        env = MarioScenarioEnv()
        env.reset(scenario=sample.scenario, seed=0)
        env.render = lambda: None
        went_over = False
        for action in sample.oracle["actions"]:
            env.step(action)
            middle = env.mario["x"] + env.mario["w"] / 2
            if block_x <= middle <= block_x + 16 * sample.parameters["blocks"] and env.mario["y"] + env.mario["h"] <= top:
                went_over = True
            if env._goal_credited:
                break
        assert env._goal_credited
        assert env.enemies[0]["dead"] == stomped and went_over == over
        env.close()


def test_a_hold_can_end_when_a_walker_is_out_from_under_a_block():
    from types import SimpleNamespace

    from retroagi.stages.block_smb.tactic_schedule import _over

    seg = segment("hold_area", 1, area=8, enemy_clear_of=[0, 100])
    walker = {"x": 95, "w": 10, "dead": False}
    env = SimpleNamespace(enemies=[walker], mario={"x": 40, "w": 10, "on_ground": True})
    assert not _over(env, seg)  # its far edge at 105, still under
    walker["x"] = 90
    assert _over(env, seg)
    walker.update(x=150, dead=True)
    assert _over(env, seg)
    back = segment("hold_area", -1, area=8, enemy_clear_of=[0, 100])
    walker.update(x=99, dead=False)
    assert not _over(env, back) and (walker.update(x=101) or _over(env, back))


@pytest.mark.timeout(900)
def test_block_goomba_arriving_running_max_points_stops_short_of_the_block():
    # User, 2026-10-28: in 1-1 Mario comes to the first Goomba running.
    made = {
        family: next(
            sample
            for index in range(8)
            if (
                sample := sample_block_smb_monte_carlo_scenario(
                    split="validation", seed=7, sample_index=index, family=family, difficulty="medium",
                    calibrate_deadline=False,
                )
            ).parameters["arrival"]
            == "running"
        )
        for family in ('speed_run_block_goomba', 'max_points_block_goomba')
    }
    fast, rich = made["speed_run_block_goomba"], made["max_points_block_goomba"]
    assert fast.scenario["platforms"] == rich.scenario["platforms"]
    assert fast.scenario["strategy_route"].startswith("goomba=over")
    stances = [seg["stance"] for seg in rich.scenario["tactics"]]
    assert stances == ["advance", "hold_area", "advance", "advance"]
    block = rich.parameters["block_x"]
    env = MarioScenarioEnv()
    env.reset(scenario=rich.scenario, seed=0)
    env.render = lambda: None
    held, fastest = [], 0.0
    for action in rich.oracle["actions"]:
        env.step(action)
        if env._tactic_index == 0:
            fastest = max(fastest, env.mario["vx"])
        if env._tactic_index == 1:
            held.append(env.mario["x"])
        if env._goal_credited:
            break
    assert env._goal_credited and env.enemies[0]["dead"]
    assert fastest >= 2.0  # it ran in
    assert held and max(held) + 10 < block  # and stopped short of the block
    env.close()


@pytest.mark.timeout(900)
def test_block_goomba_as_in_1_1_has_a_row_past_the_block_with_the_goomba_under_it():
    # User, 2026-10-29: in 1-1 the first Goomba starts under the brick row
    # past the first question block.
    sample = next(
        s
        for index in range(12)
        if (
            s := sample_block_smb_monte_carlo_scenario(
                split="validation", seed=21, sample_index=index, family="max_points_block_goomba",
                difficulty="medium", calibrate_deadline=False,
            )
        ).parameters["second_row"]
    )
    block, row = sample.scenario["platforms"][1], sample.scenario["platforms"][2]
    assert row[1] == block[1] and 48 <= row[0] - (block[0] + block[2]) <= 64
    assert row[2] == 16 * sample.parameters["second_row"]
    assert row[0] < sample.parameters["goomba_x"] < row[0] + row[2]
    assert sample.scenario["strategy_route"].startswith("goomba=stomp")


@pytest.mark.timeout(1800)
def test_pen_from_pipe_max_points_stomps_the_walker_and_speed_run_passes_it():
    # User, 2026-10-29: in 1-1 most speed runs jumped down from a pipe into the
    # pen beyond it beside its Goomba and were caught.
    made = {
        family: sample_block_smb_monte_carlo_scenario(
            split="validation", seed=21, sample_index=1, family=family, difficulty="easy",
            calibrate_deadline=False,
        )
        for family in ("speed_run_pen_from_pipe", "max_points_pen_from_pipe")
    }
    fast, rich = made["speed_run_pen_from_pipe"], made["max_points_pen_from_pipe"]
    assert fast.scenario["platforms"] == rich.scenario["platforms"]  # the same layout
    first, second = fast.scenario["platforms"][1], fast.scenario["platforms"][2]
    assert first[3] <= second[3] <= 64  # the second pipe as tall or taller
    # Mario starts on the first pipe.
    assert first[0] <= fast.scenario["mario"][0] <= first[0] + first[2]
    assert fast.scenario["strategy_route"].startswith("pen=pass")
    assert rich.scenario["strategy_route"].startswith("pen=stomp")
    for sample, stomped in ((fast, False), (rich, True)):
        env = MarioScenarioEnv()
        env.reset(scenario=sample.scenario, seed=0)
        env.render = lambda: None
        for action in sample.oracle["actions"]:
            env.step(action)
            if env._goal_credited:
                break
        assert env._goal_credited and env.enemies[0]["dead"] == stomped
        env.close()


def test_finishing_after_the_deadline_misses_the_objective():
    scenario = _flat(strategy="speed_run", strategy_objective={"deadline": 60})
    info, _, _ = _played(scenario, [1] * 400)  # walking to x=300 takes longer
    assert info["objective_missed"] and info["reward_terms"]["goal"] == 0
    scenario["strategy_objective"] = {"deadline": 400}
    info, _, _ = _played(scenario, [1] * 400)
    assert not info["objective_missed"] and info["reward_terms"]["goal"] > 50


@pytest.mark.parametrize("tactic,seed", [("descend_forward", 15), ("descend_backward", 31)])
def test_shallow_descent_spawns_on_the_ledge_without_a_collision_shortcut(tactic, seed):
    import random

    from retroagi.stages.block_smb.monte_carlo import _finish_layout
    from retroagi.stages.block_smb.strategy_families import strategy_scene

    scenario, parameters = strategy_scene(random.Random(seed), "easy", tactic)
    assert parameters["drop_height"] == 8
    _finish_layout(f"speed_run_{tactic}", scenario, parameters)
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        assert env.mario["y"] + env.mario["h"] == 212
        x = env.mario["x"]
        env.step(4)
        assert abs(env.mario["x"] - x) <= 3
        assert not env._goal_credited
    finally:
        env.close()


def test_reaching_the_goal_without_the_points_misses_the_objective():
    scenario = _flat(
        coins=[[150, 150, 10, 10]],  # above Mario's head: walking on misses it
        strategy="max_points",
        strategy_objective={"points": 1},
    )
    info, _, points = _played(scenario, [1] * 400)
    assert points == 0 and info["objective_missed"] and info["reward_terms"]["goal"] == 0
    scenario["coins"] = [[150, 209, 10, 10]]  # on the way: collected
    info, _, points = _played(scenario, [1] * 400)
    assert points == 1 and not info["objective_missed"] and info["reward_terms"]["goal"] == 50


def test_speed_run_is_paid_for_time_and_max_points_for_coins_and_kills():
    coin = [[150, 209, 10, 10]]
    _, fast, _ = _played(_flat(coins=coin, strategy="speed_run"), [1] * 400)
    _, rich, _ = _played(_flat(coins=coin, strategy="max_points"), [1] * 400)
    assert fast["coin"] == 0 and rich["coin"] == 25
    assert fast["frame_penalty"] == pytest.approx(5 * rich["frame_penalty"])


@pytest.mark.timeout(300)
def test_a_segment_naming_an_enemy_sends_the_teacher_back_to_stomp_it():
    from retroagi.stages.block_smb.compose import route_actions

    scenario = _flat(
        mario=[104, 204],
        enemies=[[40, 206, 24, 80, 0.4, 1]],  # patrolling behind Mario
        goal=[340, 200, 16, 20],
        tactics=[segment("retreat", -1, stomp=0, past_enemy=0), segment("advance", 1)],
        frame_budget=600,
    )
    route = route_actions(scenario)
    assert route
    _, terms, points = _played(scenario, route)
    assert terms["enemy_stomp"] > 0 and points == 1


def test_speed_run_can_select_a_faster_bypass_while_max_points_selects_stomps(monkeypatch):
    from retroagi.stages.block_smb import strategy_families as module

    routes = {
        "detour=skip|motion=default": (20, 1, [1] * 20),
        "detour=skip|motion=bypass": (12, 0, [2] * 12),
    }
    monkeypatch.setattr(module, "_played_routes", lambda scenario: routes)
    for strategy, motion in (("speed_run", "bypass"), ("max_points", "default")):
        scenario = {"route_tactics": {"detour=skip": [segment("advance")]}}
        route = module.strategy_route(f"{strategy}_advance", scenario)
        assert route == routes[f"detour=skip|motion={motion}"][2]
        assert scenario["prefer_enemy_bypass"] == (motion == "bypass")


def test_teacher_route_cache_does_not_reuse_routes_from_different_code(monkeypatch, tmp_path):
    from retroagi.stages.block_smb import strategy_families as module

    version = ["old"]
    calls = []
    monkeypatch.setattr(module, "_PLAYED", {})
    monkeypatch.setattr(module, "_route_cache", lambda: tmp_path)
    monkeypatch.setattr(module, "code_fingerprint", lambda: version[0])
    monkeypatch.setattr(module, "_points_collected", lambda scenario, route: 0)

    def route(scenario):
        calls.append(version[0])
        return [1] if version[0] == "old" else [1, 1]

    monkeypatch.setattr(module, "route_actions", route)
    scenario = {"route_tactics": {"direct": []}}
    old = module._played_routes(scenario)
    module._PLAYED.clear()  # reuse the disk cache in a fresh worker
    assert module._played_routes(scenario) == old
    assert len(calls) == 2
    version[0] = "new"
    assert module._played_routes(scenario) != old
    assert calls == ["old", "old", "new", "new"]
