"""Generated practice for situations exposed by frozen Full SMB evaluation.

These are independently sampled Block layouts, not copies of emulator levels or
held-out action traces. They keep Block physics and the existing observation
contract; NES duration calibration remains a separate transfer requirement.
"""

TRANSFER_FAILURE_FAMILIES = ("stair_gap", "landing_enemy", "enemy_on_platform")

TRANSFER_FAILURE_SCHEMAS = {
    "stair_gap": {
        "family_revision": [1, 1],
        "step_height": [24, 24],
        "gap_width": [26, 46],
        "takeoff_width": [36, 52],
        "landing_width": [36, 52],
        "goal": "climb, jump from the top across a gap, descend, and finish",
    },
    "landing_enemy": {
        "family_revision": [1, 1],
        "spawn_drop": [12, 28],
        "enemy_distance": [54, 90],
        "enemy_speed": [0.2, 0.6],
        "goal": "finish a descent, clear an approaching enemy, and reach the exit",
    },
    "enemy_on_platform": {
        "family_revision": [1, 1],
        "platform_height": [28, 50],
        "platform_width": [80, 104],
        "enemy_offset": [29, 55],
        "enemy_speed": [0.2, 0.6],
        "goal": "mount an occupied platform, clear its enemy, and finish",
    },
}


def transfer_failure_scenario(family, rng, difficulty):
    """Build the geometry; certify the complete route in the normal sampler."""
    tier = ("easy", "medium", "hard").index(difficulty)
    params = {"family_revision": 1, "difficulty_bin": difficulty}
    if family == "stair_gap":
        takeoff_width = (48, 40, 36)[tier] + rng.randint(0, 4)
        gap = (28, 36, 44)[tier] + rng.randint(-2, 2)
        landing_width = (48, 40, 36)[tier] + rng.randint(0, 4)
        edge = 128 + takeoff_width
        far = edge + gap
        world_width = far + landing_width + 160
        scenario = {
            "world_width": world_width,
            "mario": [rng.randint(24, 40), 204],
            "platforms": [
                [0, 220, 64, 20],
                [64, 196, 32, 44],
                [96, 172, 32, 68],
                [128, 148, takeoff_width, 92],
                [far, 148, landing_width, 92],
                [far + landing_width, 172, 32, 68],
                [far + landing_width + 32, 196, 32, 44],
                [far + landing_width + 64, 220, 96, 20],
            ],
            "goal": [world_width - 32, 204, 16, 16],
        }
        params.update(
            step_height=24,
            gap_width=gap,
            gap_x=edge,
            takeoff_width=takeoff_width,
            landing_width=landing_width,
        )
    elif family == "landing_enemy":
        spawn_x = rng.randint(40, 56)
        drop = (24, 18, 12)[tier] + rng.randint(0, 4)
        distance = (86, 70, 58)[tier] + rng.randint(-4, 4)
        enemy_x = spawn_x + distance
        speed = (0.2, 0.4, 0.6)[tier]
        scenario = {
            "world_width": 288,
            # More than eight pixels above support avoids the spawn-grounding
            # snap. The policy must first finish this real airborne descent.
            "mario": [spawn_x, 204 - drop],
            "platforms": [[0, 220, 288, 20]],
            "enemies": [[enemy_x, 206, enemy_x - 16, enemy_x + 16, speed, -1]],
            "goal": [260, 204, 16, 16],
        }
        params.update(spawn_drop=drop, enemy_distance=distance, enemy_speed=speed)
    elif family == "enemy_on_platform":
        height = (30, 40, 48)[tier] + rng.randint(-2, 2)
        left = rng.randint(112, 124)
        width = (104, 92, 80)[tier]
        offset = (52, 42, 32)[tier] + rng.randint(-3, 3)
        enemy_x = left + offset
        speed = (0.2, 0.4, 0.6)[tier]
        scenario = {
            "world_width": 352,
            "mario": [rng.randint(24, 40), 204],
            "platforms": [[0, 220, 352, 20], [left, 220 - height, width, height]],
            "enemies": [[enemy_x, 206 - height, enemy_x - 12, enemy_x + 12, speed, -1]],
            "goal": [324, 204, 16, 16],
        }
        params.update(
            platform_height=height, platform_width=width, enemy_offset=offset, enemy_speed=speed
        )
    else:
        raise ValueError(f"Unknown transfer failure family: {family}")
    scenario["goal_requires_support"] = True
    from .local_traversal import terrain_oracle

    return scenario, params, terrain_oracle(scenario)
