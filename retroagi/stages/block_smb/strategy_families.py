"""The tactic layer's families: one scene per tactic, played under each strategy.

There are seven scenes, one per tactic (tokens.TACTICS), and each is played under
both strategies (tokens.STRATEGIES): 14 families, named <strategy>_<tactic>
(speed_run_advance, max_points_climb_backward, ...). The two families of a
scene play the very same layouts (a layout depends only on the sample's seed)
and differ only in the strategy switch, what the episode pays and wins, and
the route the teacher takes.

A scene's main structure needs its tactic to reach the goal:

| Scene | Main structure |
|---|---|
| advance | a floor to the right, sometimes cut by a pit to jump |
| retreat | the goal back to the left |
| climb_forward | the goal on a step ahead |
| climb_backward | the goal on a step behind |
| descend_forward | Mario on a ledge, the goal on the floor ahead |
| descend_backward | Mario on a ledge, the goal on the floor behind |
| hold_ground | hold the starting spot for 64 frames, then traverse to the right |

Forward is right. The backward scenes fit in one screen: the camera never
scrolls back. Difficulty sets the pit, step and drop sizes (as in
action_families) and how many enemies walk the way.

Coins and enemies are inserted at random:

- on the way: coins at Mario's height, and enemies walking toward him, which
  every route meets;
- off the way, one or two detours, each one Mario may take or skip: coins on
  a floating platform above the way (climb onto it), coins behind Mario (go
  back for them), or an enemy patrolling behind him (go back and stomp it).

The teacher plays every combination of taking and skipping the detours and
measures each: frames to the goal, and points (coins collected plus enemies
killed). Each strategy takes its best:

- speed run: the fewest frames; it wins only within 10% of that time;
- max points: the most points, then the fewest frames; it wins only with at
  least three quarters of the points its route gathers beyond speed run's.

A layout is kept only if max points' best route gathers more points than
speed run's. What each strategy is paid for: env.STRATEGY_REWARDS.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

from retroagi.core.tokens import STRATEGIES, TACTICS

from .action_families import DROPS, GAPS, STEPS, STOMP_SPEEDS
from .compose import FLOOR, route_actions
from .layout_cache import code_fingerprint
from .tactic_schedule import segment

STRATEGY_TACTIC_FAMILIES = {
    f"{strategy}_{tactic}": (strategy, tactic) for strategy in STRATEGIES for tactic in TACTICS
}
# A speed run wins within this much of its teacher's (the fastest) time.
SPEED_SLACK = 1.1
# What each strategy minimises over a layout's route combinations, given
# (frames, points).
STRATEGY_ORDER = {
    "speed_run": lambda frames, points: (frames,),
    "max_points": lambda frames, points: (-points, frames),
}
SCREEN = 256
MARIO_WIDTH = 10
ENEMY_HEIGHT = 14
COIN_SIZE = 10
PLATFORM_WIDTH = 48
# Enemies walking the way, by difficulty: (fewest, most).
WAY_ENEMIES = {"easy": (0, 1), "medium": (1, 1), "hard": (1, 2)}
DETOURS = ("coins_up", "coins_behind", "enemy_behind")


def _coin(x: int, ground: int) -> list:
    """A coin at Mario's height, standing on ``ground``: walking through takes it."""
    return [x, ground - 11, COIN_SIZE, COIN_SIZE]


def _main(rng, difficulty: str, tactic: str) -> dict:
    """The scene's main structure: platforms, goal, Mario's start, the ground
    he starts on, the stretch of it where things are inserted (``way``), and
    the stretch behind him (``behind``)."""
    if tactic in ("advance", "climb_forward", "descend_forward"):
        x0 = 104  # room behind Mario, still on screen (the camera keeps him a third across)
        behind = (24, x0 - 24)
        start = x0 + MARIO_WIDTH + 32
        end = start + rng.randint(176, 240)
        ground = FLOOR
        if tactic == "advance":
            pit = rng.randint(*GAPS[difficulty]) if rng.random() < 0.5 else 0
            far = end + 24 + pit
            world = far + 96
            platforms = (
                [[0, FLOOR, end + 24, 20], [far, FLOOR, 96, 20]] if pit else [[0, FLOOR, world, 20]]
            )
            goal = [far + 32, FLOOR - 20, 16, 20]
            params = {"pit_width": pit}
        elif tactic == "climb_forward":
            height = rng.randint(*STEPS[difficulty])
            step = end + 24
            world = step + 96
            platforms = [[0, FLOOR, world, 20], [step, FLOOR - height, 48, height]]
            goal = [step + 8, FLOOR - height - 20, 32, 20]
            params = {"step_height": height}
        else:
            drop = rng.randint(*DROPS[difficulty])
            edge = end + 24
            world = edge + 160
            ground = FLOOR - drop
            platforms = [[0, FLOOR, world, 20], [0, ground, edge, drop]]
            goal = [edge + 8, FLOOR - 20, 120, 20]
            params = {"drop_height": drop}
        return {
            "side": 1,
            "x0": x0,
            "ground": ground,
            "way": (start, end),
            "behind": behind,
            "world": world,
            "platforms": platforms,
            "goal": goal,
            "params": params,
        }
    # Backward: the whole scene on one screen, Mario on its right.
    x0 = 184
    behind = (x0 + MARIO_WIDTH + 24, SCREEN - 8)
    ground = FLOOR
    platforms = [[0, FLOOR, SCREEN, 20]]
    if tactic == "retreat":
        goal = [8, FLOOR - 20, 16, 20]
        way = (48, x0 - 24)
        params = {}
    elif tactic == "climb_backward":
        height = rng.randint(*STEPS[difficulty])
        platforms.append([16, FLOOR - height, 48, height])
        goal = [24, FLOOR - height - 20, 32, 20]
        way = (88, x0 - 24)
        params = {"step_height": height}
    else:
        drop = rng.randint(*DROPS[difficulty])
        edge = 104
        ground = FLOOR - drop
        platforms.append([edge, ground, SCREEN - edge, drop])
        goal = [edge - 96, FLOOR - 20, 88, 20]
        way = (edge + 16, x0 - 24)
        params = {"drop_height": drop}
    return {
        "side": -1,
        "x0": x0,
        "ground": ground,
        "way": way,
        "behind": behind,
        "world": SCREEN,
        "platforms": platforms,
        "goal": goal,
        "params": params,
    }


def strategy_scene(rng, difficulty: str, tactic: str) -> tuple[dict, dict]:
    """A layout of a tactic's scene (the same for both strategies).

    Returns (scenario, parameters). The scenario's schedule skips every
    detour; scenario["route_tactics"] holds one schedule for every
    combination of taking and skipping them (keyed "name=take|name=skip").
    """
    if tactic not in TACTICS:
        raise ValueError(f"unknown tactic {tactic!r}")
    main = _main(rng, difficulty, "advance" if tactic == "hold_ground" else tactic)
    side, ground = main["side"], main["ground"]
    (lo, hi), (back_lo, back_hi) = main["way"], main["behind"]
    platforms = list(main["platforms"])
    coins, enemies = [], []

    # On the way: coins, and enemies walking toward Mario.
    way_coins = rng.randint(0, 3)
    for _ in range(way_coins):
        coins.append(_coin(rng.randint(lo, hi - COIN_SIZE), ground))
    fewest, most = WAY_ENEMIES[difficulty]
    if hi - lo < 96:
        most = min(most, 1)
    way_enemies = rng.randint(fewest, most)
    for _ in range(way_enemies):
        x = rng.randint(lo, hi - 12)
        speed = round(rng.uniform(*STOMP_SPEEDS[difficulty]), 2)
        enemies.append([x, ground - ENEMY_HEIGHT, lo, hi, speed, -side])

    # Off the way: one or two detours, at most one behind Mario.
    kinds = [k for k in DETOURS if k != "coins_up" or hi - lo >= PLATFORM_WIDTH + 16]
    if rng.choice((1, 2)) == 2 and "coins_up" in kinds:
        chosen = ["coins_up", rng.choice(("coins_behind", "enemy_behind"))]
    else:
        chosen = [rng.choice(kinds)]
    detours = []
    for kind in chosen:
        if kind == "coins_up":
            px = rng.randint(lo, hi - PLATFORM_WIDTH)
            top = ground - rng.randint(32, 40)
            index = len(platforms)
            platforms.append([px, top, PLATFORM_WIDTH, 10])
            for k in range(rng.choice((2, 3))):
                coins.append([px + 2 + 16 * k, top - 11, COIN_SIZE, COIN_SIZE])
            entry = px - 48 if side > 0 else px + PLATFORM_WIDTH + 40
            detours.append(
                {
                    "name": kind,
                    "where": "ahead",
                    "platform": index,
                    "entry": entry,
                    "segments": [segment("alternate_route", side, route=[index], on=index)],
                }
            )
        elif kind == "coins_behind":
            count = min(rng.randint(1, 3), (back_hi - back_lo) // 16)
            xs = (
                [back_lo + 16 * k for k in range(count)]
                if side > 0
                else [back_hi - COIN_SIZE - 16 * k for k in range(count)]
            )
            coins.extend(_coin(x, ground) for x in xs)
            reach = min(xs) if side > 0 else max(xs)
            detours.append(
                {
                    "name": kind,
                    "where": "behind",
                    "segments": [segment("retreat", -side, reach_x=reach)],
                }
            )
        else:
            index = len(enemies)
            x = rng.randint(back_lo, back_hi - 12)
            speed = round(rng.uniform(0.3, 0.6), 2)
            enemies.append([x, ground - ENEMY_HEIGHT, back_lo, back_hi, speed, rng.choice((-1, 1))])
            detours.append(
                {
                    "name": kind,
                    "where": "behind",
                    "segments": [segment("retreat", -side, stomp=index, past_enemy=index)],
                }
            )

    schedules = {}
    for taken in itertools.product((False, True), repeat=len(detours)):
        key = "|".join(f"{d['name']}={'take' if t else 'skip'}" for d, t in zip(detours, taken))
        schedules[key] = _schedule(side, detours, dict(zip(chosen, taken)))
        if tactic == "hold_ground":
            # A fixed interval is observable through tactic age. Both strategies
            # must preserve this spot before choosing their onward route.
            schedules[key].insert(0, segment("hold_area", side, area=1, frames=64))
    world = main["world"]
    scenario = {
        "world_width": world,
        # Leave one pixel above the intended support before spawn settling.
        # On an eight-pixel ledge, exact contact also put the lower floor
        # inside the settling window; its earlier list position embedded
        # Mario in the ledge and allowed a sideways collision to skip the level.
        "mario": [main["x0"], ground - 17],
        "platforms": platforms,
        "coins": coins,
        "enemies": enemies,
        "goal": main["goal"],
        "goal_requires_support": True,
        "reward_goal_distance_shaping": 2.0,
        "tactics": schedules[next(iter(schedules))],
        "route_tactics": schedules,
        "frame_budget": int(200 + world + 150 * len(detours)),
    }
    params = {
        **main["params"],
        "way_coins": way_coins,
        "way_enemies": way_enemies,
        "detours": chosen,
        "difficulty_bin": difficulty,
    }
    return scenario, params


def _schedule(side: int, detours: list, taken: dict) -> list:
    """The plan for one combination: go back for the detour behind Mario if
    taken, then forward, climbing onto the floating platform if taken; a
    skipped platform is kept off."""
    avoid = [d["platform"] for d in detours if "platform" in d and not taken[d["name"]]]
    segments = []
    for d in detours:
        if d["where"] == "behind" and taken[d["name"]]:
            segments += d["segments"]
    for d in detours:
        if d["where"] == "ahead" and taken[d["name"]]:
            segments.append(segment("advance", side, avoid=avoid, reach_x=d["entry"]))
            segments += d["segments"]
    segments.append(segment("advance", side, avoid=avoid))
    return segments


def _points_collected(scenario: dict, route: list[int]) -> int:
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        for action in route:
            if env.step(action)[2]:
                break
        return env.points()
    finally:
        env.close()


# Each layout's route combinations, as played by the teacher. The two
# strategy siblings play the same layouts, so the results are kept in memory
# and on disk, where every worker process finds them (RETROAGI_ROUTE_CACHE,
# else a temporary folder).
_PLAYED: dict = {}


def _route_cache() -> Path:
    import os
    import tempfile

    return Path(os.environ.get("RETROAGI_ROUTE_CACHE") or tempfile.gettempdir()) / (
        "retroagi_strategy_routes"
    )


def _played_routes(scenario: dict) -> dict:
    """{combination: (frames, points, route)} for every combination the teacher finishes."""
    import hashlib
    import json
    import os

    key = (
        code_fingerprint()
        + ":"
        + json.dumps(
            {k: v for k, v in scenario.items() if k not in ("strategy", "strategy_objective")},
            sort_keys=True,
            default=str,
        )
    )
    if key in _PLAYED:
        return _PLAYED[key]
    path = _route_cache() / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    try:
        played = {c: tuple(v) for c, v in json.loads(path.read_text()).items()}
    except (OSError, ValueError):
        played = {}
        for combination, schedule in scenario["route_tactics"].items():
            for bypass in (False, True):
                trial = copy.deepcopy({k: v for k, v in scenario.items() if k != "route_tactics"})
                trial["tactics"] = schedule
                trial["prefer_enemy_bypass"] = bypass
                trial.pop("strategy", None)
                trial.pop("strategy_objective", None)
                route = route_actions(trial)
                if route:
                    key_name = combination + ("|motion=bypass" if bypass else "|motion=default")
                    played[key_name] = (len(route), _points_collected(trial, route), route)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            spare = path.with_suffix(f".{os.getpid()}.tmp")
            spare.write_text(json.dumps(played))
            os.replace(spare, path)  # whole files only, for the other workers
        except OSError:
            pass
    if len(_PLAYED) > 64:
        _PLAYED.clear()
    _PLAYED[key] = played
    return played


def strategy_route(family: str, scenario: dict) -> list[int]:
    """The teacher's route for a layout of one of these families ([] when the
    layout is not kept): the combination best for the family's strategy, with
    its plan and the strategy's objective set on ``scenario``."""
    played = _played_routes(scenario)
    choices = scenario.pop("route_tactics")
    if not played:
        return []

    def best(strategy):
        return min(played, key=lambda c: STRATEGY_ORDER[strategy](*played[c][:2]))

    fastest, richest = best("speed_run"), best("max_points")
    few, many = played[fastest][1], played[richest][1]
    if many <= few:
        return []  # no detour gained points over the fastest route here
    strategy = STRATEGY_TACTIC_FAMILIES[family][0]
    chosen = best(strategy)
    base, motion = chosen.rsplit("|motion=", 1)
    scenario["tactics"] = choices[base]
    scenario["prefer_enemy_bypass"] = motion == "bypass"
    scenario["strategy_route"] = chosen
    # What the teacher measured: frames and points.
    scenario["route_results"] = {c: list(result[:2]) for c, result in played.items()}
    if strategy == "speed_run":
        scenario["strategy_objective"] = {"deadline": int(played[chosen][0] * SPEED_SLACK)}
    else:
        scenario["strategy_objective"] = {"points": few + -(-3 * (many - few) // 4)}
    return list(played[chosen][2])
