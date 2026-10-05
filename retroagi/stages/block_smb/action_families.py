"""The action layer's families: scenes that each need a single action.

Each scene asks for one skill, the one the teacher labels at every decision of
its route, and nothing else:

| Family | The one action | Parameters (every value is swept) |
|---|---|---|
| action_walk | walk right to the goal | distance |
| action_walk_back | walk back left to the goal behind | distance |
| action_jump_gap | jump a pit from a standstill at its edge | pit width, distance to the edge |
| action_climb | jump up onto a step from a standstill | step height, distance to the step |
| action_descend | walk off a ledge down to the floor | drop height, distance to the edge |
| action_stomp | land on an enemy walking toward Mario | its distance, its speed |
| action_wait | stand still on a moving platform that carries Mario to the goal | how far it travels, its speed |

Difficulty splits each family's main parameter into three ranges. The ranges
stay inside what a jump from a standstill reaches (about 50 pixels forward and
66 up), so every combination is a scene the action can finish; the sampler
still checks the teacher's route and leaves out any that it can't.

Each generator returns (scenario, parameters, authored actions); the teacher's
route under the layout's tactics replaces the authored actions
(monte_carlo._generate_family_scenario).
"""

from typing import Any

from .tactic_schedule import segment

FLOOR = 220
MARIO_WIDTH = 10
# Generators place Mario for a 16-pixel-tall body (monte_carlo._finish_layout).
STANDING = FLOOR - 20

ACTION_FAMILIES = (
    "action_walk",
    "action_walk_back",
    "action_jump_gap",
    "action_climb",
    "action_descend",
    "action_stomp",
    "action_wait",
)

# The main parameter's range at each difficulty.
DISTANCES = {"easy": (24, 80), "medium": (81, 140), "hard": (141, 200)}
GAPS = {"easy": (8, 20), "medium": (21, 30), "hard": (31, 40)}
STEPS = {"easy": (8, 24), "medium": (25, 40), "hard": (41, 56)}
DROPS = {"easy": (8, 32), "medium": (33, 56), "hard": (57, 80)}
STOMP_SPEEDS = {"easy": (0.4, 0.53), "medium": (0.54, 0.66), "hard": (0.67, 0.8)}
TRAVELS = {"easy": (48, 64), "medium": (65, 88), "hard": (89, 112)}


def _walk(rng, difficulty: str, back: bool) -> tuple[dict, dict, list]:
    distance = rng.randint(*DISTANCES[difficulty])
    start = 40 + distance if back else 40
    goal = 40 if back else 40 + distance
    scenario = {
        "world_width": 40 + distance + 64,
        "mario": [start, STANDING],
        "platforms": [[0, FLOOR, 40 + distance + 64, 20]],
        "goal": [goal, STANDING, 16, 20],
    }
    return scenario, {"distance": distance, "difficulty_bin": difficulty}, [3 if back else 1]


def action_walk(rng, difficulty: str):
    return _walk(rng, difficulty, back=False)


def action_walk_back(rng, difficulty: str):
    return _walk(rng, difficulty, back=True)


def action_jump_gap(rng, difficulty: str):
    width = rng.randint(*GAPS[difficulty])
    edge = 120
    distance = rng.randint(0, 4)  # from Mario's front to the edge
    scenario = {
        "world_width": edge + width + 120,
        "mario": [edge - MARIO_WIDTH - distance, STANDING],
        "platforms": [[0, FLOOR, edge, 20], [edge + width, FLOOR, 120, 20]],
        "goal": [edge + width, STANDING, 48, 20],
        "goal_requires_support": True,
    }
    parameters = {"gap_width": width, "edge_distance": distance, "difficulty_bin": difficulty}
    return scenario, parameters, [2]


def action_climb(rng, difficulty: str):
    height = rng.randint(*STEPS[difficulty])
    step = 120
    distance = rng.randint(0, 8)  # from Mario's front to the step
    scenario = {
        "world_width": step + 48 + 80,
        "mario": [step - MARIO_WIDTH - distance, STANDING],
        "platforms": [[0, FLOOR, step + 128, 20], [step, FLOOR - height, 48, height]],
        "goal": [step + 8, FLOOR - height - 20, 32, 20],
        "goal_requires_support": True,
    }
    parameters = {"step_height": height, "step_distance": distance, "difficulty_bin": difficulty}
    return scenario, parameters, [2]


def action_descend(rng, difficulty: str):
    height = rng.randint(*DROPS[difficulty])
    edge = 120
    distance = rng.randint(0, 24)  # from Mario's front to the ledge's edge
    # The goal covers the floor where a walk off the ledge lands, however high.
    scenario = {
        "world_width": edge + 160,
        "mario": [edge - MARIO_WIDTH - distance, FLOOR - height - 20],
        "platforms": [[0, FLOOR, edge + 160, 20], [0, FLOOR - height, edge, height]],
        "goal": [edge + 8, STANDING, 120, 20],
        "goal_requires_support": True,
    }
    parameters = {"drop_height": height, "edge_distance": distance, "difficulty_bin": difficulty}
    return scenario, parameters, [1]


def action_stomp(rng, difficulty: str):
    distance = rng.randint(24, 72)  # from Mario's front to the enemy
    speed = round(rng.uniform(*STOMP_SPEEDS[difficulty]), 3)
    mario_x = 60
    enemy_x = mario_x + MARIO_WIDTH + distance
    world = enemy_x + 120
    scenario = {
        "world_width": world,
        "mario": [mario_x, STANDING],
        "platforms": [[0, FLOOR, world, 20]],
        # Walking left, toward Mario, over the whole floor.
        "enemies": [[enemy_x, 206, 0, world, speed, -1]],
        "goal": [enemy_x - 2, 186, 16, 20],
        "goal_on_stomp": True,
    }
    parameters = {"enemy_distance": distance, "enemy_speed": speed, "difficulty_bin": difficulty}
    return scenario, parameters, [2]


def action_wait(rng, difficulty: str):
    travel = rng.randint(*TRAVELS[difficulty])
    speed = round(rng.uniform(0.5, 1.0), 3)
    shore, width, offset = 40, 48, 16  # Mario stands 16 pixels into the platform
    start = shore
    end = start + travel
    scenario = {
        "world_width": end + width + 80,
        "mario": [start + offset, STANDING],
        # The left shore, then a pit; the platform starts against the shore.
        "platforms": [
            [0, FLOOR, shore, 20],
            {"x": start, "y": FLOOR, "w": width, "h": 10, "moving": [start, end, speed]},
        ],
        # Where Mario is when the platform reaches the end of its travel.
        "goal": [end + offset - 3, STANDING, MARIO_WIDTH + 6, 20],
        "goal_requires_support": True,
        "tactics": [segment("hold_area", 1)],
    }
    parameters = {"travel": travel, "platform_speed": speed, "difficulty_bin": difficulty}
    return scenario, parameters, [0]


GENERATORS = {
    "action_walk": action_walk,
    "action_walk_back": action_walk_back,
    "action_jump_gap": action_jump_gap,
    "action_climb": action_climb,
    "action_descend": action_descend,
    "action_stomp": action_stomp,
    "action_wait": action_wait,
}


def action_family_scenario(family: str, rng, difficulty: str) -> tuple[dict, Any, list]:
    return GENERATORS[family](rng, difficulty)
