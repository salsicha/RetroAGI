"""The action layer's families: scenes that each need a single action.

Each scene asks for one isolated maneuver. The teacher supplies spatial skill
commands to the action learner. The tactic labels below are used when these
scenes train skill. The additional mirrored families at the end cover raised
platforms/enemies and immediate jumps down to a lower landing.

| Family | The one action | Tactic | Parameters (every value is swept) |
|---|---|---|---|
| action_walk | walk right to the goal | advance | distance |
| action_walk_back | walk back left to the goal behind | retreat | distance |
| action_jump_gap | jump a pit from a standstill at its edge | advance | pit width, distance to the edge |
| action_climb | jump up onto a step ahead from a standstill | climb_forward | step height, distance to the step |
| action_climb_back | jump up onto a step behind from a standstill | climb_backward | step height, distance to the step |
| action_descend | walk off a ledge ahead down to the floor | descend_forward | drop height, distance to the edge |
| action_descend_back | walk off a ledge behind down to the floor | descend_backward | drop height, distance to the edge |
| action_stomp | land on an enemy walking toward Mario | advance | its distance, its speed |
| action_wait | stand still on a moving platform that carries Mario to the goal | hold_ground | how far it travels, its speed |

The scenes going back to the left fit in one screen: the camera never
scrolls back, so the whole scene must be in view from the start.

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
    "action_climb_back",
    "action_descend",
    "action_descend_back",
    "action_stomp",
    "action_wait",
    "action_jump_gap_back",
    "action_platform_up",
    "action_platform_up_back",
    "action_stomp_back",
    "action_stomp_up",
    "action_stomp_up_back",
    "action_jump_down",
    "action_jump_down_back",
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


def action_climb_back(rng, difficulty: str):
    height = rng.randint(*STEPS[difficulty])
    step, width = 40, 48
    distance = rng.randint(0, 8)  # from Mario's back to the step
    world = step + width + 120
    scenario = {
        "world_width": world,
        "mario": [step + width + distance, STANDING],
        "platforms": [[0, FLOOR, world, 20], [step, FLOOR - height, width, height]],
        "goal": [step + 8, FLOOR - height - 20, 32, 20],
        "goal_requires_support": True,
    }
    parameters = {"step_height": height, "step_distance": distance, "difficulty_bin": difficulty}
    return scenario, parameters, [4]


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


def action_descend_back(rng, difficulty: str):
    height = rng.randint(*DROPS[difficulty])
    edge, world = 136, 256
    distance = rng.randint(0, 24)  # from Mario's back to the ledge's edge
    # The goal covers the floor where a walk off the ledge lands, however high.
    scenario = {
        "world_width": world,
        "mario": [edge + distance, FLOOR - height - 20],
        "platforms": [[0, FLOOR, world, 20], [edge, FLOOR - height, world - edge, height]],
        "goal": [edge - 128, STANDING, 120, 20],
        "goal_requires_support": True,
    }
    parameters = {"drop_height": height, "edge_distance": distance, "difficulty_bin": difficulty}
    return scenario, parameters, [3]


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
        "tactics": [segment("hold_area", 1, area=1)],
    }
    parameters = {"travel": travel, "platform_speed": speed, "difficulty_bin": difficulty}
    return scenario, parameters, [0]


def _mirror(scenario):
    """Reflect a one-screen maneuver, including enemy motion and its goal."""
    import copy

    reflected = copy.deepcopy(scenario)
    world = scenario["world_width"]
    reflected["mario"][0] = world - scenario["mario"][0] - MARIO_WIDTH
    for platform in reflected["platforms"]:
        platform[0] = world - platform[0] - platform[2]
    goal = reflected["goal"]
    goal[0] = world - goal[0] - goal[2]
    for enemy in reflected.get("enemies", []):
        enemy[0] = world - enemy[0] - 10
        enemy[2], enemy[3] = world - enemy[3], world - enemy[2]
        enemy[5] = -enemy[5]
    reflected["action_jump_direction"] = -1
    return reflected


def action_jump_gap_back(rng, difficulty):
    scenario, params, _ = action_jump_gap(rng, difficulty)
    # Fit backward landings entirely in view.
    scenario["world_width"] = 256
    scenario["platforms"][-1][2] = 256 - scenario["platforms"][-1][0]
    return _mirror(scenario), params, [4]


def action_stomp_back(rng, difficulty):
    scenario, params, _ = action_stomp(rng, difficulty)
    scenario["world_width"] = 256
    scenario["platforms"][0][2] = 256
    scenario["enemies"][0][3] = 256
    return _mirror(scenario), params, [4]


def _raised(rng, difficulty, *, enemy=False, back=False):
    height = rng.randint(*({"easy": (8, 16), "medium": (17, 28), "hard": (29, 40)}[difficulty]))
    distance = rng.randint(0, 4)
    step, width = 120, 88
    top = FLOOR - height
    scenario = {
        "world_width": 256,
        "mario": [step - MARIO_WIDTH - distance, STANDING],
        "platforms": [[0, FLOOR, 256, 20], [step, top, width, 8]],
        "goal": [step + 4, top - 20, width - 8, 20],
        "goal_requires_support": True,
        "action_jump_direction": 1,
    }
    params = {"height": height, "distance": distance, "difficulty_bin": difficulty}
    if enemy:
        speed = rng.choice((0.0, 0.4, 0.8))
        scenario["enemies"] = [[step + 24, top - 14, step, step + width, speed, -1]]
        scenario["goal_on_stomp"] = True
        params["enemy_speed"] = speed
    return (_mirror(scenario) if back else scenario), params, [4 if back else 2]


def action_platform_up(rng, difficulty):
    return _raised(rng, difficulty)


def action_platform_up_back(rng, difficulty):
    return _raised(rng, difficulty, back=True)


def action_stomp_up(rng, difficulty):
    return _raised(rng, difficulty, enemy=True)


def action_stomp_up_back(rng, difficulty):
    return _raised(rng, difficulty, enemy=True, back=True)


def _jump_down(rng, difficulty, back=False):
    drop = rng.randint(*DROPS[difficulty])
    gap = rng.randint(8, 24)
    edge = 104
    scenario = {
        "world_width": 256,
        "mario": [edge - MARIO_WIDTH - 2, FLOOR - drop - 20],
        "platforms": [[0, FLOOR - drop, edge, 12], [edge + gap, FLOOR, 256 - edge - gap, 20]],
        "goal": [edge + gap, STANDING, 256 - edge - gap, 20],
        "goal_requires_support": True,
        "action_jump_direction": 1,
    }
    params = {"drop_height": drop, "gap_width": gap, "difficulty_bin": difficulty}
    return (_mirror(scenario) if back else scenario), params, [4 if back else 2]


def action_jump_down(rng, difficulty):
    return _jump_down(rng, difficulty)


def action_jump_down_back(rng, difficulty):
    return _jump_down(rng, difficulty, back=True)


GENERATORS = {
    "action_walk": action_walk,
    "action_walk_back": action_walk_back,
    "action_jump_gap": action_jump_gap,
    "action_climb": action_climb,
    "action_climb_back": action_climb_back,
    "action_descend": action_descend,
    "action_descend_back": action_descend_back,
    "action_stomp": action_stomp,
    "action_wait": action_wait,
    "action_jump_gap_back": action_jump_gap_back,
    "action_platform_up": action_platform_up,
    "action_platform_up_back": action_platform_up_back,
    "action_stomp_back": action_stomp_back,
    "action_stomp_up": action_stomp_up,
    "action_stomp_up_back": action_stomp_up_back,
    "action_jump_down": action_jump_down,
    "action_jump_down_back": action_jump_down_back,
}


def action_family_scenario(family: str, rng, difficulty: str) -> tuple[dict, Any, list]:
    return GENERATORS[family](rng, difficulty)
