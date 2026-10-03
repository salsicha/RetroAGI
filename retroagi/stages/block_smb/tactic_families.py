"""Families built for the tactic and skill layers, and for the strategy token.

Four groups, all with explicit tactics (tactic_schedule):

- **The scene decides the tactic** (tactic, skill and action layers):
  upper_route and lower_route (alternate route), dead_end_retreat (retreat out
  of a dead end, then an alternate route) and monster_retreat (keep away from
  a monster that cannot be stomped, then jump over it).
- **Composed scenes** (compose.py), whose tactics change along the way: the
  chained and sequence families, rebuilt from sections.
- **Clones** (skill and action layers): sibling families that play the very
  same layouts (the layout depends only on the sample's seed) and differ only
  in their tactics, so the right skill is decided by the tactic token. A
  sibling loses its episode when Mario does not follow its tactic (a
  forbidden platform, leaving a hold area early; the goal counts only once the
  retreat is done). Every sibling's route is checked on each layout, so a
  layout one sibling cannot finish is redrawn for all of them. The clones are
  left out of tactic-layer training: their tactic is not decided by the scene.
- **Strategy courses**: composed scenes only, played by three siblings, one
  per strategy (tokens.STRATEGIES), on the very same layouts. Each course has
  two or three sections that offer two routes (compose: coin_detour,
  hazard_bypass, lift_shortcut) among other sections. The teacher plays every
  combination of routes and each strategy takes the best for its objective:
  - speed run: the fewest frames; paid a time bonus for finishing early
    (MarioScenarioEnv.STRATEGY_REWARDS); wins only within 10% of that time;
  - max coins: the most coins, then the fewest frames; paid more per coin;
    wins only with most of the coins it gathers beyond speed run's;
  - careful: the fewest enemies passed, then the fewest frames; paid nothing
    for time and charged more for dying; wins by finishing.
  A layout is kept only if max coins' best route gathers more coins than speed
  run's.
"""

from __future__ import annotations

import copy
import random

from .compose import FLOOR, compose, route_actions
from .tactic_schedule import segment

# The scene decides the tactic.
SCENE_TACTIC_FAMILIES = ("upper_route", "lower_route", "dead_end_retreat", "monster_retreat")
# Rebuilt from sections, with tactics that change along the way.
COMPOSED_RECIPES = {
    "chained_obstacles": ["enemy", "pipe", "upper_route", "enemy"],
    "chained_enemy_gauntlet": ["enemy", "gap", "monster", "enemy"],
    "full_smb_opening_proxy": ["enemy", "pipe", ("plant", {"timed": True}), "pipe"],
    "mixed_section": None,  # drawn per layout (mixed_recipe)
    "tactics_bridge_sequence": ["bridge", ("plant", {"timed": True})],
    "tactics_obstacle_sequence": ["enemy", "upper_route", ("plant", {"timed": True}), "pipe"],
    "tactics_bridge_then_gap": ["bridge", "gap", "stairs", "dead_end"],
    "tactics_mixed_sequence": ["enemy", "lower_route", "monster", "gap"],
}
STANDALONE = {
    "upper_route": ["upper_route"],
    "lower_route": ["lower_route"],
    "dead_end_retreat": [("dead_end", {"inside": True})],
    "monster_retreat": [("monster", {"inside": True})],
}
# Clones: siblings on the same layouts that differ only in their tactics.
CHOICE_FAMILIES = ("choice_advance", "choice_alternate_route", "choice_hold_area", "choice_retreat")
LOW_CHOICE_FAMILIES = ("low_choice_advance", "low_choice_alternate_route")
CLONE_FAMILIES = CHOICE_FAMILIES + LOW_CHOICE_FAMILIES
# Strategy courses: siblings on the same composed layouts, one per strategy.
STRATEGY_FAMILIES = {
    "speed_run_course": "speed_run",
    "max_coins_course": "max_coins",
    "careful_course": "careful",
}
TACTIC_FAMILIES = (
    *SCENE_TACTIC_FAMILIES,
    *COMPOSED_RECIPES,
    *CLONE_FAMILIES,
    *STRATEGY_FAMILIES,
)
NEW_FAMILIES = (*SCENE_TACTIC_FAMILIES, *CLONE_FAMILIES, *STRATEGY_FAMILIES)
# A speed run wins within this much of its teacher's (the fastest) time.
SPEED_SLACK = 1.1
# Sections that offer two routes, and the other sections of a course.
ROUTE_SECTIONS = ("coin_detour", "hazard_bypass", "lift_shortcut")
COURSE_SECTIONS = ("enemy", "gap", "pipe", "stairs", "plant", "monster")
# What each strategy minimises over a layout's route combinations, given
# (frames, coins collected, enemies passed).
STRATEGY_ORDER = {
    "speed_run": lambda frames, coins, hazards: (frames,),
    "max_coins": lambda frames, coins, hazards: (-coins, frames),
    "careful": lambda frames, coins, hazards: (hazards, frames),
}

SIMPLE_SECTIONS = ("enemy", "gap", "pipe", "stairs")
SPECIAL_SECTIONS = ("plant", "upper_route", "lower_route", "monster", "dead_end", "bridge")


def mixed_recipe(rng: random.Random) -> list:
    """Two or three simple sections and two different special ones, shuffled."""
    parts = rng.sample(SIMPLE_SECTIONS, rng.choice((2, 3))) + rng.sample(SPECIAL_SECTIONS, 2)
    rng.shuffle(parts)
    return parts


def course_recipe(rng: random.Random) -> list:
    """Two or three sections that offer routes (at most one moving platform)
    and one or two other sections, shuffled."""
    choices = rng.sample(ROUTE_SECTIONS, rng.choice((2, 3)))
    parts = choices + rng.sample(COURSE_SECTIONS, rng.choice((1, 2)))
    rng.shuffle(parts)
    return parts


def _choice_layout(rng: random.Random, difficulty: str) -> tuple[dict, dict]:
    """A floor with a jumpable pit ahead, a raised path above it, room behind.

    Platforms: 0 the floor before the pit, 1 the floor after it, 2-4 the
    raised path (2 over the near floor, 3 above the pit's near edge, 4 over
    the far floor).
    """
    tier = ("easy", "medium", "hard").index(difficulty)
    pit_x = rng.randint(150, 170)
    pit = (34, 40, 46)[tier] + rng.randint(-2, 2)
    far = pit_x + pit
    width = far + 160
    spawn = rng.randint(60, 76)
    scenario = {
        "world_width": width,
        "mario": [spawn, FLOOR - 16],
        "platforms": [
            [0, FLOOR, pit_x, 20],
            [far, FLOOR, width - far, 20],
            [pit_x - 84, 176, 40, 10],
            [pit_x - 28, 132, 44, 10],
            [far + 20, 172, 40, 10],
        ],
        "coins": [[pit_x - 16, 112, 10, 10]],
        "goal": [width - 32, FLOOR - 20, 16, 20],
        "goal_requires_support": True,
        "reward_goal_distance_shaping": 2.0,
        "frame_budget": 360,
    }
    back = spawn - rng.randint(32, 48)
    hold = rng.randint(48, 96)
    raised = [2, 3, 4]
    schedules = {
        "choice_advance": [segment("advance", 1, forbidden=raised)],
        "choice_alternate_route": [
            segment("alternate_route", 1, route=raised, forbidden=[1], on=4),
            segment("advance", 1),
        ],
        "choice_hold_area": [
            segment("hold_area", 1, area=16, frames=hold),
            segment("advance", 1, forbidden=raised),
        ],
        "choice_retreat": [
            segment("retreat", -1, reach_x=back),
            segment("advance", 1, forbidden=raised),
        ],
    }
    params = {"pit_x": pit_x, "pit_width": pit, "spawn_x": spawn, "hold_frames": hold}
    params["retreat_to"] = back
    return scenario, {"schedules": schedules, **params}


def _low_choice_layout(rng: random.Random, difficulty: str) -> tuple[dict, dict]:
    """A raised ledge path with jumpable gaps over a walkable lower floor.

    Platforms: 0 the starting ledge, 1 the floating ledge between the gaps,
    2 the far ledge (the goal is on it), 3 the lower floor, 4 a step from it.
    """
    tier = ("easy", "medium", "hard").index(difficulty)
    edge = rng.randint(120, 140)
    first = (28, 34, 40)[tier] + rng.randint(-2, 2)
    second = (28, 34, 40)[tier] + rng.randint(-2, 2)
    middle = edge + first
    far = middle + 40 + second
    width = far + 120
    scenario = {
        "world_width": width,
        "mario": [rng.randint(40, 60), 160 - 16],
        "platforms": [
            [0, 160, edge, 60],
            [middle, 160, 40, 10],
            [far, 160, 120, 60],
            [edge, FLOOR, far - edge, 20],
            [far - 28, 190, 28, 30],
        ],
        "coins": [[middle + 15, 130, 10, 10]],
        "goal": [width - 32, 140, 16, 20],
        "goal_requires_support": True,
        "reward_goal_distance_shaping": 2.0,
        "frame_budget": 320,
    }
    schedules = {
        "low_choice_advance": [segment("advance", 1, route=[1, 2], forbidden=[3, 4])],
        "low_choice_alternate_route": [
            segment("alternate_route", 1, route=[3, 4, 2], forbidden=[1], on=2),
            segment("advance", 1),
        ],
    }
    return scenario, {"schedules": schedules, "edge": edge, "gaps": [first, second]}


def tactic_family_scenario(family: str, rng: random.Random, difficulty: str):
    """(scenario, parameters, route placeholder) for a family of this module.

    The route is the teacher's, worked out on the finished layout (family_route).
    """
    if family in CHOICE_FAMILIES:
        scenario, params = _choice_layout(rng, difficulty)
    elif family in LOW_CHOICE_FAMILIES:
        scenario, params = _low_choice_layout(rng, difficulty)
    elif family in STRATEGY_FAMILIES:
        scenario, params = compose(rng, difficulty, course_recipe(rng))
        scenario["strategy"] = STRATEGY_FAMILIES[family]
    else:
        recipe = STANDALONE.get(family) or COMPOSED_RECIPES[family] or mixed_recipe(rng)
        scenario, params = compose(rng, difficulty, recipe)
    if family in CLONE_FAMILIES:
        scenario["sibling_tactics"] = params.pop("schedules")
        scenario["tactics"] = scenario["sibling_tactics"][family]
    params["difficulty_bin"] = difficulty
    return scenario, params, []


def _coins_collected(scenario: dict, route: list[int]) -> int:
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        for action in route:
            if env.step(action)[2]:
                break
        return sum(coin["collected"] for coin in env.coins)
    finally:
        env.close()


# Each layout's route combinations, as played by the teacher: siblings made in
# the same process share them (the layout depends only on the sample's seed).
_PLAYED: dict = {}


def _played_routes(scenario: dict) -> dict:
    """{combination: (frames, coins collected, enemies passed, route)} for every
    route combination the teacher finishes."""
    import json

    key = json.dumps(
        {k: v for k, v in scenario.items() if k not in ("strategy", "strategy_objective")},
        sort_keys=True,
        default=str,
    )
    if key not in _PLAYED:
        played = {}
        for combination, schedule in scenario["route_tactics"].items():
            trial = {
                k: v for k, v in scenario.items() if k not in ("route_tactics", "route_hazards")
            }
            trial = copy.deepcopy(trial)
            trial["tactics"] = schedule
            trial.pop("strategy_objective", None)
            route = route_actions(trial)
            if route:
                hazards = scenario["route_hazards"][combination]
                played[combination] = (len(route), _coins_collected(trial, route), hazards, route)
        if len(_PLAYED) > 64:
            _PLAYED.clear()
        _PLAYED[key] = played
    return _PLAYED[key]


def _strategy_route(family: str, scenario: dict) -> list[int]:
    """The teacher's route for one strategy course: the route combination best
    for the course's strategy, with the strategy's objective set on ``scenario``."""
    played = _played_routes(scenario)
    choices = scenario.pop("route_tactics")
    scenario.pop("route_hazards")
    if not played:
        return []

    def best(strategy):
        return min(played, key=lambda c: STRATEGY_ORDER[strategy](*played[c][:3]))

    fastest, richest = best("speed_run"), best("max_coins")
    few, many = played[fastest][1], played[richest][1]
    if many <= few:
        return []  # coins gained nothing over the fastest route here
    strategy = STRATEGY_FAMILIES[family]
    chosen = best(strategy)
    scenario["tactics"] = choices[chosen]
    scenario["strategy_route"] = chosen
    # What the teacher measured: frames, coins collected, enemies passed.
    scenario["route_results"] = {c: list(result[:3]) for c, result in played.items()}
    if strategy == "speed_run":
        scenario["strategy_objective"] = {"deadline": int(played[chosen][0] * SPEED_SLACK)}
    elif strategy == "max_coins":
        scenario["strategy_objective"] = {"coins": few + -(-3 * (many - few) // 4)}
    return played[chosen][3]


def family_route(family: str, scenario: dict) -> list[int]:
    """The teacher's route for a finished layout ([] when it cannot finish).

    A clone's layout must be finished under every sibling's tactics, so all
    siblings accept or redraw the same layouts. A strategy course's layout
    must also show each strategy's route best for its own reward.
    """
    if family in STRATEGY_FAMILIES:
        return _strategy_route(family, scenario)
    siblings = scenario.pop("sibling_tactics", None)
    if siblings:
        for name, schedule in siblings.items():
            if name == family:
                continue
            trial = copy.deepcopy(scenario)
            trial["tactics"] = schedule
            if not route_actions(trial):
                return []
    return route_actions(scenario)
