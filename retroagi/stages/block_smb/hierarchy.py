"""Skill prerequisites and physically continuous tactical/strategic families.

Family names and teacher plans are curriculum metadata, never policy inputs.
Every layer learns on basic skills; these later tasks teach their composition.
"""

TACTICS_SEQUENCE_FAMILIES = ("tactics_bridge_sequence", "tactics_obstacle_sequence")
STRATEGY_SEQUENCE_FAMILIES = ("strategy_bridge_then_gap", "strategy_mixed_sequence")
HIERARCHY_FAMILIES = TACTICS_SEQUENCE_FAMILIES + STRATEGY_SEQUENCE_FAMILIES

FAMILY_PREREQUISITES = {
    "chained_obstacles": ("enemy_hop", "tall_pipe_jump", "enemy_patrol"),
    "chained_enemy_gauntlet": ("enemy_hop", "single_gap", "enemy_patrol", "tall_pipe_jump"),
    "mixed_section": ("chained_obstacles", "chained_enemy_gauntlet"),
    "full_smb_opening_proxy": ("chained_obstacles", "tall_pipe_jump"),
    "tactics_bridge_sequence": ("wait_timing", "bridge_mount", "bridge_dismount"),
    "tactics_obstacle_sequence": ("enemy_hop", "tall_pipe_jump", "enemy_patrol"),
    "strategy_bridge_then_gap": ("tactics_bridge_sequence", "single_gap", "stair_climb"),
    "strategy_mixed_sequence": ("tactics_obstacle_sequence", "single_gap"),
}


def eligible_families(families, mastery):
    """Unlock after held-out prerequisite mastery, retaining unlocked practice."""
    eligible = set(families) - FAMILY_PREREQUISITES.keys()
    eligible.update(name for name in families if mastery.get(name, {}).get("hierarchy_unlocked"))
    for _ in range(len(FAMILY_PREREQUISITES)):
        for name, prerequisites in FAMILY_PREREQUISITES.items():
            if name in families and all(
                p in eligible and mastery.get(p, {}).get("mastered", False) for p in prerequisites
            ):
                eligible.add(name)
    return tuple(name for name in families if name in eligible)


def hierarchy_status(families, mastery):
    active = set(eligible_families(families, mastery))
    return {
        "active_families": sorted(active),
        "locked_families": {
            name: list(FAMILY_PREREQUISITES[name]) for name in families if name not in active
        },
        "tactics_families": [f for f in TACTICS_SEQUENCE_FAMILIES if f in active],
        "strategy_families": [f for f in STRATEGY_SEQUENCE_FAMILIES if f in active],
    }


def bridge_training_active(env):
    """A bridge followed by terrain stops requesting waits once it is crossed."""
    return bool(env._require_bridge_before_goal) and not (
        getattr(env, "_bridge_then_terrain", False) and env._bridge_crossed
    )


def hierarchy_scenario(family, rng, difficulty):
    from .local_traversal import terrain_oracle
    from .monte_carlo import _chained_enemy_gauntlet, _chained_obstacles, _wait_timing

    if family == "tactics_bridge_sequence":
        scenario, params, actions = _wait_timing(rng, difficulty)
        sequence = ["wait_pass", "board", "ride", "exit"]
    elif family == "tactics_obstacle_sequence":
        scenario, params, actions = _chained_obstacles(rng, difficulty)
        sequence = ["enemy_clear", "mount_platform", "mount_platform", "enemy_clear"]
    elif family == "strategy_mixed_sequence":
        scenario, params, actions = _chained_enemy_gauntlet(rng, difficulty)
        sequence = ["enemy_clear", "clear_gap", "enemy_clear", "mount_platform"]
    elif family == "strategy_bridge_then_gap":
        scenario, params, _ = _wait_timing(rng, difficulty)
        # Retain the bridge's verified shores, then append a gap and a step.
        # This is one world and one recurrent episode, with no reset at a boundary.
        gap = {"easy": 32, "medium": 42, "hard": 50}[difficulty] + rng.randint(-2, 2)
        landing = 380 + gap
        rise = {"easy": 20, "medium": 28, "hard": 36}[difficulty]
        scenario["world_width"] = landing + 180
        scenario["platforms"].extend(
            [[landing, 220, 180, 20], [landing + 75, 220 - rise, 105, rise + 20]]
        )
        scenario["goal"] = [landing + 140, 200 - rise, 16, 20]
        scenario["bridge_then_terrain"] = True
        scenario["goal_requires_support"] = True
        actions = terrain_oracle(scenario, max_steps=320)
        params.update(followup_gap_width=gap, followup_rise=rise)
        sequence = ["wait_pass", "board", "ride", "exit", "clear_gap", "mount_platform"]
    else:
        raise ValueError(f"Unknown hierarchy family {family!r}")
    params.update(
        hierarchy_level="tactics" if family in TACTICS_SEQUENCE_FAMILIES else "strategy",
        skill_sequence=sequence,
        prerequisites=list(FAMILY_PREREQUISITES[family]),
    )
    return scenario, params, actions
