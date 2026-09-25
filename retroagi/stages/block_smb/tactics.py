"""Training-only tactical supervision shared by obstacle families.

Successful demonstration actions provide route-consistent targets. Autonomous
rollouts are labeled at their actual decision states using physical safety
checks, never the old demonstration's frame index. Uncertain states are masked.
These labels do not select or override the policy's action at playback.
"""

from retroagi.core.models import TACTIC_STANCES

BRIDGE_FAMILIES = frozenset(
    ("bridge_wait", "wait_timing", "moving_bridge", "bridge_mount", "bridge_dismount")
)
OBSTACLE_FAMILIES = frozenset(
    (
        "enemy_stomp",
        "enemy_patrol",
        "enemy_on_platform",
        "landing_enemy",
        "tall_pipe_jump",
        "stair_climb",
        "stair_gap",
        "retreat_recovery",
        "chained_obstacles",
        "chained_enemy_gauntlet",
        "mixed_section",
        "full_smb_opening_proxy",
    )
)
TACTICAL_FAMILIES = BRIDGE_FAMILIES | OBSTACLE_FAMILIES | {"piranha_avoidance"}


def goal_direction(env):
    return (
        -1 if env.goal is not None and env.goal.centerx < env.mario["x"] + env.mario["w"] / 2 else 1
    )


def action_stance(env, action):
    """Direction is relative to the goal: going left need not be retreat."""
    if action == 0:
        return "hold_area"
    if action in (1, 2, 3, 4):
        direction = 1 if action in (1, 2) else -1
        return "advance" if direction == goal_direction(env) else "retreat"
    return None  # A vertical jump alone does not identify tactical intent.


def safe_ground_motion(env, action, frames=4):
    """Certify a short positioning step without stepping off support or dying."""
    from retroagi.core.smb_coaching import probe_state

    with probe_state(env):
        for _ in range(frames):
            _, _, done, truncated, info = env.step(action)
            if info["death"] or truncated or not env.mario["on_ground"]:
                return False
            if done:
                return bool(env._goal_credited)
        return True


def tactic_label(env, history=None, action=None, *, family, phase=None):
    """Label only grounded, uncommitted decisions in the tactical curriculum.

    The caller masks executor commitments and landing-release frames. `action`
    must come from a successful demonstration, never from the learner itself.
    """
    if family not in TACTICAL_FAMILIES or not env.mario["on_ground"]:
        return -1
    if family == "piranha_avoidance":
        from .piranha_tactics import tactic_label as plant_label

        return plant_label(env, history, action)
    if action is not None:
        # Riding/braking can require directional corrections while retaining
        # the same tactical intent. Successful route variants may depart at
        # different safe times; each route supplies its own consistent label.
        stance = (
            "hold_area"
            if phase in ("wait", "ride") and action not in (2, 4)
            else action_stance(env, action)
        )
        return TACTIC_STANCES.index(stance) if stance is not None else -1

    if family in ("bridge_wait", "wait_timing", "moving_bridge"):
        from .bridge_traversal import bridge_phase

        current = bridge_phase(env, True)
        stance = "hold_area" if current in ("wait", "ride") else "advance"
        return TACTIC_STANCES.index(stance)

    from retroagi.core.smb_coaching import training_target

    from .local_traversal import local_target_distance, safe_jump_holds, support_edge_distance

    target = training_target(env)
    direction = goal_direction(env) if target.kind in ("finish", "retreat") else target.direction
    forward = 1 if direction > 0 else 3
    backward = 3 if direction > 0 else 1
    distance = local_target_distance(env, target)
    if family in ("bridge_mount", "bridge_dismount"):
        if env._goal_credited:
            return -1
        # A geometrically reachable departure is an advance; otherwise hold
        # only when the current moving support remains safe while waiting.
        if safe_jump_holds(env, target, direction):
            return TACTIC_STANCES.index(action_stance(env, forward))
        return TACTIC_STANCES.index("hold_area") if safe_ground_motion(env, 0) else -1

    if target.kind in ("finish", "retreat"):
        return (
            TACTIC_STANCES.index(action_stance(env, forward))
            if safe_ground_motion(env, forward)
            else -1
        )
    near = distance < 65 or support_edge_distance(env, direction) < 24
    if near and safe_jump_holds(env, target, direction):
        return TACTIC_STANCES.index(action_stance(env, forward))
    # At a wall, moving away creates a new takeoff opportunity. Certify that
    # the fallback is supported; never label a retreat into a pit as safe.
    if target.kind == "mount" and distance <= 1:
        return (
            TACTIC_STANCES.index(action_stance(env, backward))
            if safe_ground_motion(env, backward)
            else -1
        )
    if safe_ground_motion(env, forward):
        return TACTIC_STANCES.index(action_stance(env, forward))
    if safe_ground_motion(env, 0):
        return TACTIC_STANCES.index("hold_area")
    if safe_ground_motion(env, backward):
        return TACTIC_STANCES.index(action_stance(env, backward))
    return -1


def compatible_actions(env, label):
    """Motor commands consistent with a stance, including braking to hold."""
    allowed = [False] * 6
    if label < 0:
        return allowed
    stance = TACTIC_STANCES[label]
    if stance in ("advance", "retreat"):
        direction = goal_direction(env) * (1 if stance == "advance" else -1)
        for action in (1, 2) if direction > 0 else (3, 4):
            allowed[action] = True
    elif stance == "hold_area":
        allowed[0] = True
        if abs(env.mario["vx"]) > 0.5:
            allowed[3 if env.mario["vx"] > 0 else 1] = True
    return allowed
