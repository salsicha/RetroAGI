"""Dedicated destination-selection scenes, in addition to every legacy family.

A bypass must land beyond a live enemy: stomping is an explicit failure.
These are skill scenes, not additions to the isolated action curriculum.
"""

from .action_families import FLOOR, STANDING, _mirror
from .tactic_schedule import segment

SKILL_FAMILIES = ("skill_enemy_bypass", "skill_enemy_bypass_back")


def skill_family_scenario(family, rng, difficulty):
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
