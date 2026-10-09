"""Dedicated destination-selection scenes, in addition to every legacy family.

A bypass must land beyond a live enemy: stomping is an explicit failure.
The alternate route's local maneuvers finish at their first destination;
choosing and sequencing the full route belongs to the tactic curriculum.
"""

from .action_families import FLOOR, STANDING, _mirror
from .tactic_schedule import segment

ROUTE_SKILL_FAMILIES = (
    "skill_overhead_climb",
    "skill_raised_climb",
    "skill_gap_descent",
    "skill_ledge_descent",
)
SKILL_FAMILIES = ("skill_enemy_bypass", "skill_enemy_bypass_back", *ROUTE_SKILL_FAMILIES)


def _route_skill_scenario(family, rng, difficulty):
    from .tactic_families import _choice_layout

    scenario, params = _choice_layout(rng, difficulty)
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
        scenario["mario"] = [x + rng.randint(4, width - 16), top - 16]
    left, top, width, _ = platforms[target]
    if family == "skill_ledge_descent":
        # Finish on the floor beyond this ledge, not at the far level goal.
        right = left + width
        left = platforms[source][0] + platforms[source][2] + 1
        width = right - left
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
    side = -1 if family.endswith("back") else 1
    distance = rng.randint(24, 32)
    speed = rng.choice({"easy": (0.0,), "medium": (0.2, 0.4), "hard": (0.4, 0.6)}[difficulty])
    start = 80
    enemy = start + 10 + distance
    scenario = {
        "world_width": 256,
        "mario": [start, STANDING],
        "platforms": [[0, FLOOR, 256, 20]],
        "enemies": [[enemy, FLOOR - 14, 0, 256, speed, -1]],
        "goal": [enemy + 14, STANDING, 256 - enemy - 14, 20],
        "goal_requires_support": True,
        "action_jump_direction": 1,
        "strategy": "speed_run",
        "prefer_enemy_bypass": True,
    }
    if side < 0:
        scenario = _mirror(scenario)
    scenario["tactics"] = [segment("advance" if side > 0 else "retreat", side, keep_alive=[0])]
    return (
        scenario,
        {"enemy_distance": distance, "enemy_speed": speed, "difficulty_bin": difficulty},
        [2 if side > 0 else 4],
    )
