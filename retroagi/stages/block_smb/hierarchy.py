"""Skill prerequisites and physically continuous tactical/strategic families.

Family names and teacher plans are curriculum metadata, never policy inputs.
Every layer learns on basic skills; these later tasks teach their composition.
The sequence families are composed from sections (tactic_families.COMPOSED_RECIPES).
"""

TACTICS_SEQUENCE_FAMILIES = ("tactics_bridge_sequence", "tactics_obstacle_sequence")
STRATEGY_SEQUENCE_FAMILIES = ("tactics_bridge_then_gap", "tactics_mixed_sequence")
HIERARCHY_FAMILIES = TACTICS_SEQUENCE_FAMILIES + STRATEGY_SEQUENCE_FAMILIES

FAMILY_PREREQUISITES = {
    "chained_obstacles": ("enemy_hop", "tall_pipe_jump", "enemy_patrol"),
    "chained_enemy_gauntlet": ("enemy_hop", "single_gap", "enemy_patrol", "tall_pipe_jump"),
    "mixed_section": ("chained_obstacles", "chained_enemy_gauntlet"),
    "full_smb_opening_proxy": ("chained_obstacles", "tall_pipe_jump"),
    "tactics_bridge_sequence": ("wait_timing", "bridge_mount", "bridge_dismount"),
    "tactics_obstacle_sequence": ("enemy_hop", "tall_pipe_jump", "enemy_patrol"),
    "tactics_bridge_then_gap": ("tactics_bridge_sequence", "single_gap", "stair_climb"),
    "tactics_mixed_sequence": ("tactics_obstacle_sequence", "single_gap"),
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
    """A bridge followed by terrain stops requesting waits once it is crossed.

    A moving platform Mario may ride or not (optional_bridge) is taught only
    in a moving-platform segment of the layout's tactics, until crossed.
    """
    if getattr(env, "_optional_bridge", False):
        from . import tactic_schedule

        return tactic_schedule.current(env)["kind"] == "bridge" and not env._bridge_crossed
    return bool(env._require_bridge_before_goal) and not (
        getattr(env, "_bridge_then_terrain", False) and env._bridge_crossed
    )
