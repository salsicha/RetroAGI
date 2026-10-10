"""Dedicated destination-selection scenes, in addition to every legacy family.

A bypass must land beyond a live enemy: stomping is an explicit failure.
The alternate route's local maneuvers finish at their first destination;
choosing and sequencing the full route belongs to the tactic curriculum.
"""

import math

from .action_families import FLOOR, STANDING, _mirror
from .tactic_schedule import segment
from .threat_variety import follow_up, reachable_span, standing_reach, start_distance

ROUTE_SKILL_FAMILIES = (
    "skill_overhead_climb",
    "skill_raised_climb",
    "skill_gap_descent",
    "skill_ledge_descent",
)
LOW_ROUTE_SKILL_FAMILIES = (
    "skill_lower_descent",
    "skill_lower_step_climb",
    "skill_lower_exit_climb",
)
UPPER_GAP_SKILL_FAMILIES = ("skill_upper_gap_entry", "skill_upper_gap_exit")
# Pre-episode frames the agent watches before its first decision in the enemy
# families (env "watch_frames"): its memory steps every 4 frames, so by the
# first decision it has seen the enemy move, and its forecast of where the
# enemy will be when the next action ends can tell a standing enemy from a
# walking one, and which way it walks.
WATCH_FRAMES = 12
SKILL_FAMILIES = (
    "skill_enemy_bypass",
    "skill_enemy_bypass_back",
    *ROUTE_SKILL_FAMILIES,
    *LOW_ROUTE_SKILL_FAMILIES,
    *UPPER_GAP_SKILL_FAMILIES,
)


def _route_skill_scenario(family, rng, difficulty):
    from .tactic_families import _choice_layout

    scenario, params = _choice_layout(rng, difficulty, vary=True)
    for key in ("schedules", "hold_frames", "retreat_to"):
        params.pop(key)
    source, target = {
        "skill_overhead_climb": (0, 2),
        "skill_raised_climb": (2, 3),
        "skill_gap_descent": (3, 4),
        "skill_ledge_descent": (4, 1),
    }[family]
    platforms = scenario["platforms"]
    if source:
        x, top, width, _ = platforms[source]
        scenario["mario"] = [x + rng.randint(4, max(4, width - 16)), top - 16]
    else:
        # On the floor, from right below the raised platform's near edge to
        # 48 pixels before it.
        scenario["mario"][0] = max(16, platforms[target][0] - 10 - start_distance(rng, 48))
    left, top, width, _ = platforms[target]
    if family == "skill_ledge_descent":
        # Finish on the floor beyond this ledge, not at the far level goal;
        # a pit or an enemy on that floor bounds the landing.
        left = platforms[source][0] + platforms[source][2] + 1
        cx = scenario["mario"][0] + 5
        drop = FLOOR - platforms[source][1]
        room = reachable_span(rng, 24, 80, 2 * (standing_reach(-drop) - 3 - (left - cx)))
        _, room, _ = follow_up(rng, scenario, 1, left, 1, window=(room, room))
        width = room
    scenario.update(
        coins=[],
        goal=[left, top - 20, width, 20],
        goal_requires_support=True,
        frame_budget=240,
        strategy="speed_run",
        tactics=[segment("advance", 1, route=[target])],
    )
    params.update(
        spawn_x=scenario["mario"][0],
        source_platform=source,
        target_platform=target,
        difficulty_bin=difficulty,
    )
    return scenario, params, []


def skill_family_scenario(family, rng, difficulty):
    if family in ROUTE_SKILL_FAMILIES:
        return _route_skill_scenario(family, rng, difficulty)
    if family in LOW_ROUTE_SKILL_FAMILIES + UPPER_GAP_SKILL_FAMILIES:
        from .tactic_families import _low_choice_layout

        scenario, params = _low_choice_layout(rng, difficulty, vary=True)
        params.pop("schedules")
        source, target = {
            "skill_lower_descent": (0, 3),
            "skill_lower_step_climb": (3, 4),
            "skill_lower_exit_climb": (4, 2),
            "skill_upper_gap_entry": (0, 1),
            "skill_upper_gap_exit": (1, 2),
        }[family]
        sx, sy, sw, _ = scenario["platforms"][source]
        if source:
            scenario["mario"] = [sx + rng.randint(3, max(3, min(sw - 16, 24))), sy - 16]
        else:
            # On the starting ledge, from right at its edge to 80 pixels back.
            scenario["mario"][0] = max(16, sx + sw - 10 - start_distance(rng, 80))
        x, y, w, _ = scenario["platforms"][target]
        if family in ("skill_upper_gap_exit", "skill_lower_exit_climb"):
            # The far ledge ends at a pit or carries an enemy, at a varied
            # distance: the landing is bounded on both sides, its middle
            # usually within a standing jump from the source's far edge.
            cx = sx + sw - 5 if family == "skill_upper_gap_exit" else scenario["mario"][0] + 5
            reach = standing_reach(sy - y) - 3 - (x - cx)
            room = reachable_span(rng, 24, 80, 2 * reach)
            _, w, _ = follow_up(rng, scenario, target, x, 1, window=(room, room))
        scenario.update(
            coins=[],
            goal=[x, y - 20, w, 20],
            tactics=[
                segment(
                    "advance",
                    1,
                    route=[target],
                    forbidden=[3, 4] if family in UPPER_GAP_SKILL_FAMILIES else [],
                )
            ],
            frame_budget=240,
        )
        params.update(source_platform=source, target_platform=target, difficulty_bin=difficulty)
        return scenario, params, []
    side = -1 if family.endswith("back") else 1
    speed = rng.choice({"easy": (0.0,), "medium": (0.2, 0.4), "hard": (0.4, 0.6)}[difficulty])
    walks = rng.choice((-1, 1)) if speed else -1
    # From Mario's front to the enemy: right next to it up to 27 pixels, but
    # not so close that an enemy walking toward him reaches him while he
    # watches it before his first decision. The landing room starts past the
    # enemy, 19 pixels plus this distance from Mario's middle: farther than
    # 27, it would begin beyond a jump from a standstill (50 pixels, 4 to
    # spare), and no single jump could land in it.
    distance = start_distance(rng, int(standing_reach(0)) - 19 - 4)
    if walks < 0:
        distance = max(distance, math.ceil(speed * WATCH_FRAMES) + 4)
    start = rng.randint(40, 90)
    enemy = start + 10 + distance
    scenario = {
        "world_width": 256,
        "mario": [start, STANDING],
        "platforms": [[0, FLOOR, 256, 20]],
        "enemies": [[enemy, FLOOR - 14, 0, 256, speed, walks]],
        "goal_requires_support": True,
        "action_jump_direction": 1,
        "strategy": "speed_run",
        "prefer_enemy_bypass": True,
        "watch_frames": WATCH_FRAMES,
    }
    # A second pit or enemy past the first bounds the landing (farther when
    # the first walks away, into the landing); the goal is the floor between.
    window = (48, 96) if walks > 0 else (28, 72)
    kind, room, detail = follow_up(rng, scenario, 0, enemy + 14, 1, window=window)
    scenario["goal"] = [enemy + 14, STANDING, room, 20]
    if side < 0:
        scenario = _mirror(scenario)
    scenario["tactics"] = [segment("advance" if side > 0 else "retreat", side, keep_alive=[0])]
    return (
        scenario,
        {
            "mario_x": start,
            "enemy_distance": distance,
            "enemy_speed": speed,
            "enemy_direction": walks,
            "then": kind,
            "landing_room": room,
            "then_size": detail,
            "difficulty_bin": difficulty,
        },
        [2 if side > 0 else 4],
    )
