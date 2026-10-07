"""Family prerequisites and physically continuous tactical/strategic families.

Family names and teacher plans are curriculum metadata, never policy inputs.
These later tasks teach the composition of basic moves.
The sequence families are composed from sections (tactic_families.COMPOSED_RECIPES).
"""

TACTICS_SEQUENCE_FAMILIES = ("tactics_bridge_sequence", "tactics_obstacle_sequence")
STRATEGY_SEQUENCE_FAMILIES = ("tactics_bridge_then_gap", "tactics_mixed_sequence")
HIERARCHY_FAMILIES = TACTICS_SEQUENCE_FAMILIES + STRATEGY_SEQUENCE_FAMILIES

# These compositions train tactic selection and termination, using frozen skills.
TACTIC_ONLY_FAMILIES = HIERARCHY_FAMILIES

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
