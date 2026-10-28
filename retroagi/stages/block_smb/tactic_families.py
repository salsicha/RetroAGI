"""Scenario generators and curriculum ownership for tactics.

Tactic training includes complete routes, strategy objectives, waiting/proceeding
choices and every family whose teacher changes tactics within an episode.
Sibling choice layouts share geometry but differ in their required routes;
choice_hold_area, a local fixed-tactic skill, has layouts of its own
(_hold_layout).
The route skill families end on one destination support instead of completing
an entire route. Generator membership (TACTIC_FAMILIES) is distinct from
learner membership (TACTIC_TRAINING_FAMILIES).
"""

from __future__ import annotations

import copy
import random

from .compose import FLOOR, compose, route_actions
from .strategy_families import STRATEGY_FAMILIES, strategy_route, strategy_scene
from .tactic_schedule import segment

# The scene decides the tactic.
SCENE_TACTIC_FAMILIES = ("upper_route", "lower_route", "dead_end_retreat", "monster_retreat", "pen_pipe_climb")
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
    # Full SMB 1-1's opening (user, 2026-10-25): blocks overhead, walkers
    # between pipes and dropping from a high row, then a pit.
    "full_smb_blocks_proxy": ["blocks", "pipe_pen", "high_row", "gap"],
    # 1-1 past its fourth pipe (user, 2026-10-28): a pit, a low row with a
    # Goomba walking off it, the high row with another, a wider pit.
    "full_smb_rows_proxy": ["gap", "low_row", "high_row", "gap"],
}
STANDALONE = {
    "upper_route": ["upper_route"],
    "lower_route": ["lower_route"],
    "dead_end_retreat": [("dead_end", {"inside": True})],
    "monster_retreat": [("monster", {"inside": True})],
    # Started in a pen between pipes, before one too tall to climb from a
    # standstill, walkers behind (user, 2026-10-28).
    "pen_pipe_climb": [("pipe_pen", {"inside": True})],
}
# Clones: siblings on the same layouts that differ only in their tactics.
CHOICE_FAMILIES = ("choice_advance", "choice_hold_area", "choice_retreat")
LOW_CHOICE_FAMILIES = ("low_choice_advance",)
CLONE_FAMILIES = ("choice_advance", "choice_retreat")
# This complete route is selected for visible upper-path coins under max_points.
# Its individual maneuvers live in skill_families.ROUTE_SKILL_FAMILIES.
ROUTE_CHOICE_FAMILIES = ("choice_alternate_route", "low_choice_alternate_route")
TACTIC_FAMILIES = (
    *SCENE_TACTIC_FAMILIES,
    *COMPOSED_RECIPES,
    *CLONE_FAMILIES,
    *LOW_CHOICE_FAMILIES,
    "choice_hold_area",
    *ROUTE_CHOICE_FAMILIES,
    *STRATEGY_FAMILIES,
)
NEW_FAMILIES = (
    *SCENE_TACTIC_FAMILIES,
    *CLONE_FAMILIES,
    *LOW_CHOICE_FAMILIES,
    "choice_hold_area",
    *ROUTE_CHOICE_FAMILIES,
    *STRATEGY_FAMILIES,
)

# Waiting versus proceeding is a tactic decision.
TIMING_TACTIC_FAMILIES = ("moving_bridge", "wait_timing", "piranha_avoidance")
# A skill episode follows one supplied tactic throughout. These legacy names
# conceal switches in teacher_tactic (including bridge phases and a final
# approach/retreat after a landing), even with a single written schedule segment.
# (choice_advance, enemy_gap and enemy_patrol left on 2026-10-11: once jumps
# stopped landing beyond the goal or a route platform, neither the reference
# route nor the teacher switched in any of 36 layouts each.)
MULTI_TACTIC_FAMILIES = (
    "choice_retreat",
    "stair_climb",
    "platform_chain",
    "retreat_recovery",
    "tall_pipe_jump",
    "bridge_wait",
    "bridge_mount",
    "bridge_dismount",
    "stair_gap",
    "enemy_on_platform",
)
TACTIC_TRAINING_FAMILIES = (
    *SCENE_TACTIC_FAMILIES,
    *ROUTE_CHOICE_FAMILIES,
    *LOW_CHOICE_FAMILIES,
    *COMPOSED_RECIPES,
    *TIMING_TACTIC_FAMILIES,
    *STRATEGY_FAMILIES,
    *MULTI_TACTIC_FAMILIES,
)

SIMPLE_SECTIONS = ("enemy", "gap", "pipe", "stairs")
SPECIAL_SECTIONS = ("plant", "upper_route", "lower_route", "monster", "dead_end", "bridge")


def mixed_recipe(rng: random.Random) -> list:
    """Two or three simple sections and two different special ones, shuffled."""
    parts = rng.sample(SIMPLE_SECTIONS, rng.choice((2, 3))) + rng.sample(SPECIAL_SECTIONS, 2)
    rng.shuffle(parts)
    return parts


def _choice_layout(rng: random.Random, difficulty: str, vary: bool = False) -> tuple[dict, dict]:
    """A floor with a jumpable pit ahead, a raised path above it, room behind.

    Platforms: 0 the floor before the pit, 1 the floor after it, 2-4 the
    raised path (2 over the near floor, 3 above the pit's near edge, 4 over
    the far floor). With ``vary``, the raised platforms' places, widths and
    heights also vary (by up to 6, 8 and 4 pixels), so the moves between
    them are different distances in each scene.
    """
    tier = ("easy", "medium", "hard").index(difficulty)
    pit_x = rng.randint(150, 170)
    pit = (34, 40, 46)[tier] + rng.randint(-2, 2)
    far = pit_x + pit
    width = far + 160
    spawn = rng.randint(60, 76)

    def jiggle(x, top, wide):
        if not vary:
            return [x, top, wide, 10]
        # Small enough that each move stays within a jump from a standstill
        # (the teacher's run-ups stop before a jump).
        return [
            x + rng.randint(-6, 6),
            top + rng.randint(-4, 4),
            wide + rng.randint(-8, 8),
            10,
        ]

    scenario = {
        "world_width": width,
        "mario": [spawn, FLOOR - 16],
        "platforms": [
            [0, FLOOR, pit_x, 20],
            [far, FLOOR, width - far, 20],
            jiggle(pit_x - 84, 176, 40),
            jiggle(pit_x - 28, 132, 44),
            jiggle(far + 20, 172, 40),
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
        "choice_retreat": [
            segment("retreat", -1, reach_x=back),
            segment("advance", 1, forbidden=raised),
        ],
    }
    params = {"pit_x": pit_x, "pit_width": pit, "spawn_x": spawn, "hold_frames": hold}
    params["retreat_to"] = back
    return scenario, {"schedules": schedules, **params}


def _hold_layout(rng: random.Random, difficulty: str) -> tuple[dict, dict]:
    """Holding while walkers come at Mario (choice_hold_area), with pits and
    obstacles about that a jump over a walker must not go into (user,
    2026-10-15): on each side of Mario, at a random distance, the floor ends
    at a pit, or a pipe stands, or the floor goes on; sometimes a brick row
    hangs low over him. Each walker, at its own speed, walks the floor
    between them, turning at the pit's lip or the pipe, so it comes at Mario
    again and again. Mario holds where he is and, when a walker comes within
    its trigger distance, jumps over it and lands behind it (the teacher's
    rule: controller_teacher._dodge_in_area); a jump over it into the pit or
    the pipe, or up into the bricks, is what he must learn not to make.

    Platforms: 0 Mario's floor, then the floor beyond a pit, pipes and the
    brick row, as drawn. Layouts the teacher cannot hold through are redrawn
    (monte_carlo.TEACHER_ROUTED).
    """
    from .threat_variety import ENEMY_HEIGHT, ENEMY_WIDTH

    tier = ("easy", "medium", "hard").index(difficulty)
    spawn = rng.randint(100, 124)  # Mario's left edge
    # How far the floor runs on each side of Mario before a pit or a pipe:
    # from just enough for a walker and a short jump over it (a long jump
    # lands in the pit or hits the pipe) to well beyond any jump.
    rooms = {side: rng.randint(32, 80) for side in (-1, 1)}
    # Easy layouts keep one side's floor open; harder ones may close both.
    choices = ("pit", "pipe", "open")
    ends = {side: rng.choice(choices) for side in (-1, 1)}
    if tier == 0 and "open" not in ends.values():
        ends[rng.choice((-1, 1))] = "open"
    left, right = spawn - rooms[-1], spawn + 10 + rooms[1]
    width = right + 160
    floor_left = left if ends[-1] == "pit" else 0
    floor_right = right if ends[1] == "pit" else width
    platforms = [[floor_left, FLOOR, floor_right - floor_left, 20]]
    kinds = ["ground"]
    # Where the walkers turn: the lip of a pit, the face of a pipe, or (open
    # floor) as far again from Mario.
    turn = {-1: left, 1: right}
    for side in (-1, 1):
        edge = turn[side]
        if ends[side] == "pit":
            pit = (28, 36, 44)[tier] + rng.randint(-4, 4)
            if side < 0 and edge - pit >= 24:
                platforms.append([0, FLOOR, edge - pit, 20])
                kinds.append("ground")
            elif side > 0:
                platforms.append([edge + pit, FLOOR, width - edge - pit, 20])
                kinds.append("ground")
        elif ends[side] == "pipe":
            height = rng.randint(24, 44)
            x = edge - 32 if side < 0 else edge
            platforms.append([x, FLOOR - height, 32, height])
            kinds.append("pipe")
        else:
            turn[side] = max(8, edge - 40) if side < 0 else min(width - 8, edge + 40)
    if rng.random() < (0.2, 0.3, 0.4)[tier]:
        # A brick row low over Mario: high jumps hit it.
        low = rng.randint(46, 62)
        wide = rng.randint(32, 64)
        x = spawn + 5 - wide // 2 + rng.randint(-24, 24)
        platforms.append([x, FLOOR - low - 16, wide, 16])
        kinds.append("brick")
    walkers = 1 if tier == 0 else rng.randint(1, 2)
    # Each walker its own speed, so each needs its own trigger.
    speeds = [round((0.4, 0.5, 0.6)[tier] + rng.uniform(-0.1, 0.25), 3) for _ in range(walkers)]
    lo, hi = turn[-1], turn[1] - ENEMY_WIDTH
    first = rng.choice((-1, 1))
    enemies = []
    for speed, side in zip(speeds, (first, -first)):
        if side > 0:
            x = rng.randint(spawn + 10 + 20, max(spawn + 30, hi))
        else:
            x = rng.randint(min(lo, spawn - 20 - ENEMY_WIDTH), spawn - 20 - ENEMY_WIDTH)
        enemies.append([x, FLOOR - ENEMY_HEIGHT, lo, hi, speed, -side])
    hold = rng.randint(120, 200)
    scenario = {
        "world_width": width,
        "mario": [spawn, FLOOR - 16],
        "platforms": platforms,
        "platform_kinds": kinds,
        "enemies": enemies,
        "coins": [],
        # Holding until the end, anywhere on Mario's floor between the pits
        # and pipes, completes the lesson.
        "goal": [left, FLOOR - 20, right - left, 20],
        "goal_requires_support": True,
        "frame_budget": hold + 40,
        "tactics": [segment("hold_area", 1, dodge=True, frames=hold)],
    }
    params = {
        "hold_frames": hold,
        "walker_speeds": speeds,
        "walkers": walkers,
        "ends": [ends[-1], ends[1]],
        "rooms": [rooms[-1], rooms[1]],
    }
    return scenario, params


def _low_choice_layout(
    rng: random.Random, difficulty: str, vary: bool = False
) -> tuple[dict, dict]:
    """A raised ledge path with jumpable gaps over a walkable lower floor.

    Platforms: 0 the starting ledge, 1 the floating ledge between the gaps,
    2 the far ledge (the goal is on it), 3 the lower floor, 4 a step from it.
    With ``vary``, the gaps (by up to 6 pixels more), the floating ledge's
    width (28 to 60), the step's width and height and the ledge's edge vary
    more, so the moves are different distances in each scene.
    """
    tier = ("easy", "medium", "hard").index(difficulty)
    spread = 6 if vary else 2
    edge = rng.randint(104, 152) if vary else rng.randint(120, 140)
    # With ``vary``, the gaps are up to 6 pixels narrower as well as wider,
    # so that most landings stay within a standing jump's reach.
    narrower = 6 if vary else 0
    first = (28, 34, 40)[tier] - narrower + rng.randint(-spread, spread)
    second = (28, 34, 40)[tier] - narrower + rng.randint(-spread, spread)
    if vary:
        from .threat_variety import best_jump_room, standing_reach

        # The floating ledge is the landing room past the first gap: drawn so
        # that the jump keeping the most room from both gaps (covering the
        # gap and the ledge) is spread over every jump that clears the gap.
        ledge = best_jump_room(rng, 20, 60, standing_reach(0) - first)
    else:
        ledge = 40
    step = rng.randint(20, 40) if vary else 28
    if vary and rng.random() < 0.6:
        # Under the floating ledge only one jump climbs the step (the ledge
        # blocks higher ones), so most layouts leave 24 pixels of floor
        # before the step clear of the ledge, where every climbing jump fits.
        step = max(20, min(step, second - 24))
    step_top = 190 + (rng.randint(-6, 6) if vary else 0)
    if second - step < 24:
        # The lower floor before the step lies under the floating ledge (no
        # takeoff with Mario's head clear of it). The jumps that clear the
        # step without hitting the ledge's underside (50 pixels above the
        # floor) rise 31 pixels (a 4-frame hold); the next a jump can be
        # commanded with rises 39. So the step is at most 28 pixels high:
        # the teacher's 2 pixels above its top, and 1 to spare.
        step_top = max(step_top, 192)
    middle = edge + first
    far = middle + ledge + second
    width = far + 120
    scenario = {
        "world_width": width,
        "mario": [rng.randint(40, 60), 160 - 16],
        "platforms": [
            [0, 160, edge, 60],
            [middle, 160, ledge, 10],
            [far, 160, 120, 60],
            [edge, FLOOR, far - edge, 20],
            [far - step, step_top, step, FLOOR - step_top],
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
    if family == "choice_alternate_route":
        scenario, params = _choice_layout(rng, difficulty)
        params.pop("schedules")
        # Unlike a supplied-tactic clone, route choice needs an observable
        # reason. The upper-path coin and max_points switch provide it.
        scenario["strategy"] = "max_points"
        scenario["strategy_objective"] = {"points": len(scenario["coins"])}
        scenario["tactics"] = [
            segment("alternate_route", 1, route=[2, 3, 4], forbidden=[1], on=4),
            segment("advance", 1),
        ]
    elif family == "low_choice_alternate_route":
        scenario, params = _low_choice_layout(rng, difficulty)
        schedules = params.pop("schedules")
        scenario["tactics"] = schedules[family]
        # Visible points on the lower path justify choosing it under max_points.
        edge = scenario["platforms"][3][0]
        scenario["coins"] = [[edge + 22, FLOOR - 20, 10, 10]]
        scenario["strategy"] = "max_points"
        scenario["strategy_objective"] = {"points": len(scenario["coins"])}
    elif family == "choice_hold_area":
        # Its own layouts, not its siblings': holding while walkers come at
        # Mario, among pits and obstacles (_hold_layout).
        scenario, params = _hold_layout(rng, difficulty)
    elif family in CHOICE_FAMILIES:
        scenario, params = _choice_layout(rng, difficulty)
        if family == "choice_advance":
            far = scenario["platforms"][1][0]
            scenario["goal"] = [far, FLOOR - 20, 24, 20]
    elif family in LOW_CHOICE_FAMILIES:
        scenario, params = _low_choice_layout(rng, difficulty)
        scenario["tactics"] = params.pop("schedules")[family]
        scenario["strategy"] = "speed_run"
    elif family in STRATEGY_FAMILIES:
        strategy, tactic = STRATEGY_FAMILIES[family]
        scenario, params = strategy_scene(rng, difficulty, tactic)
        scenario["strategy"] = strategy
    else:
        recipe = STANDALONE.get(family) or COMPOSED_RECIPES[family] or mixed_recipe(rng)
        scenario, params = compose(rng, difficulty, recipe)
    if family in CLONE_FAMILIES:
        scenario["sibling_tactics"] = {
            k: v for k, v in params.pop("schedules").items() if k in CLONE_FAMILIES
        }
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
    if family in STRATEGY_FAMILIES:
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
