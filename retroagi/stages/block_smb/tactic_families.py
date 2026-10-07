"""Families with an explicit plan (tactic_schedule), in four groups.

All four groups supply skill training except the tactic-only compositions
whose names start with `tactics_`. Those compositions and the strategy
families train the tactic layer:

- **The scene decides the tactic**: upper_route and lower_route (an alternate route), dead_end_retreat (retreat out of a dead
  end, then an alternate route) and monster_retreat (keep away from a monster
  that cannot be stomped, then jump over it).
- **Composed scenes** (compose.py), whose tactics change along the way: the
  chained and sequence families, rebuilt from sections.
- **Clones**: sibling families that play the very same layouts (the layout
  depends only on the sample's seed) and differ only in their plans. They
  teach the skill layer different destination choices in the same scene.
  A sibling loses its episode when Mario does not follow its plan (a
  forbidden platform, leaving a hold area early; the goal counts only once the
  retreat is done). Every sibling's route is checked on each layout, so a
  layout one sibling cannot finish is redrawn for all of them.
- **Strategy families** (strategy_families): the tactic layer's families,
  one scene per tactic played under each strategy.
"""

from __future__ import annotations

import copy
import random

from .compose import FLOOR, compose, route_actions
from .strategy_families import STRATEGY_TACTIC_FAMILIES, strategy_route, strategy_scene
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
TACTIC_FAMILIES = (
    *SCENE_TACTIC_FAMILIES,
    *COMPOSED_RECIPES,
    *CLONE_FAMILIES,
    *STRATEGY_TACTIC_FAMILIES,
)
NEW_FAMILIES = (*SCENE_TACTIC_FAMILIES, *CLONE_FAMILIES, *STRATEGY_TACTIC_FAMILIES)

SIMPLE_SECTIONS = ("enemy", "gap", "pipe", "stairs")
SPECIAL_SECTIONS = ("plant", "upper_route", "lower_route", "monster", "dead_end", "bridge")


def mixed_recipe(rng: random.Random) -> list:
    """Two or three simple sections and two different special ones, shuffled."""
    parts = rng.sample(SIMPLE_SECTIONS, rng.choice((2, 3))) + rng.sample(SPECIAL_SECTIONS, 2)
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
    elif family in STRATEGY_TACTIC_FAMILIES:
        strategy, tactic = STRATEGY_TACTIC_FAMILIES[family]
        scenario, params = strategy_scene(rng, difficulty, tactic)
        scenario["strategy"] = strategy
    else:
        recipe = STANDALONE.get(family) or COMPOSED_RECIPES[family] or mixed_recipe(rng)
        scenario, params = compose(rng, difficulty, recipe)
    if family in CLONE_FAMILIES:
        scenario["sibling_tactics"] = params.pop("schedules")
        scenario["tactics"] = scenario["sibling_tactics"][family]
    params["difficulty_bin"] = difficulty
    return scenario, params, []


def family_route(family: str, scenario: dict) -> list[int]:
    """The teacher's route for a finished layout ([] when it cannot finish).

    A clone's layout must be finished under every sibling's tactics, so all
    siblings accept or redraw the same layouts. A strategy family's layout
    must show each strategy's route best for its own reward
    (strategy_families.strategy_route).
    """
    if family in STRATEGY_TACTIC_FAMILIES:
        return strategy_route(family, scenario)
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
