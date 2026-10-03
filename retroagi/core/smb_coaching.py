"""Training-only targets and simulator probes for the teachers.

Simulator state supplies labels, never model observations. A probe restores all
state after it plays forward.
"""

from contextlib import contextmanager

from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.local_traversal import (
    LocalObjective,
    local_objective,
    plant_clearance_target,
    retreat_objective,
    route_objective,
)


@contextmanager
def probe_state(env):
    saved = snapshot_env_state(env)
    render = env.__dict__.get("render")
    wait_survival = env._wait_survival
    # Shaping is irrelevant to collision certification; its legacy bridge
    # predictor would otherwise run recursively inside every wait probe.
    env._wait_survival = 0
    env.render = lambda: None
    try:
        yield saved
    finally:
        restore_env_state(env, saved)
        env._wait_survival = wait_survival
        if render is None:
            del env.__dict__["render"]
        else:
            env.render = render


def required_stomp(scene):
    """Required contact remains a target even after the player passes it."""
    m = scene.mario
    candidates = [(i, e) for i, e in enumerate(scene.enemies) if not e.get("dead", False)]
    if not candidates:
        return None
    i, e = min(
        candidates, key=lambda pair: abs(pair[1]["x"] + pair[1]["w"] / 2 - m["x"] - m["w"] / 2)
    )
    direction = 1 if e["x"] + e["w"] / 2 >= m["x"] + m["w"] / 2 else -1
    return LocalObjective(
        "stomp", e["x"], e["x"] + e["w"], e["y"], enemy_index=i, direction=direction
    )


def training_target(env):
    if getattr(env, "_bridge_jump_task", None):
        # The jump task succeeds only on the required collision landing.
        return LocalObjective("finish", env.goal.left, env.goal.right, env.goal.bottom)

    if (env._goal_on_stomp or env._require_stomp_before_goal) and not env._stomp_credited:
        target = required_stomp(env)
        if target is not None:
            return target
    # The layout's tactic segment: its route's next platform, or its retreat line.
    target = route_objective(env) or retreat_objective(env)
    if target is not None:
        return target
    return plant_clearance_target(env, local_objective(env))
