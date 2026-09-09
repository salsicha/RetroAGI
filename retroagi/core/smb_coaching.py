"""Training-only physical coaching for the canonical executor.

Oracle state supplies labels, never model observations. Counterfactual probes
restore all state and require collision outcomes, including bounce recovery.
"""

from contextlib import contextmanager
from types import SimpleNamespace

from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, local_target_distance

COACHING_CONTRACT = "canonical_collision_coaching_v2"


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


def training_target(env):
    from retroagi.core.smb_objectives import required_stomp

    if (env._goal_on_stomp or env._require_stomp_before_goal) and not env._stomp_credited:
        target = required_stomp(env)
        if target is not None:
            return target
    return local_objective(env)


def probe_executor(model):
    # Do not replace the live collector's executor reference on the model.
    from retroagi.core.smb_runtime import make_smb_executor

    previous = getattr(model, "smb_executor", None)
    executor = make_smb_executor(model)
    model.smb_executor = previous
    return executor


def physical_batch(env, *, bouncing=False):
    return SimpleNamespace(
        metadata={
            "smb_geometry": {
                "support": "ground" if env.mario["on_ground"] else "air",
                "enemy_contact": False,
                "bouncing": bouncing,
            }
        }
    )


def safe_jump_indices(model, env, action, *, verify_recovery=True):
    """Certify each physical hold using the same executor as greedy playback."""
    from retroagi.core.smb_learning import primitive

    if not env.mario["on_ground"] or action not in (2, 4, 5):
        return []
    target = training_target(env)
    valid = []
    with probe_state(env) as saved:
        for index in range(16):
            restore_env_state(env, saved)
            executor = probe_executor(model)
            airborne = bouncing = False
            for _ in range(96):
                execution = executor.execute(
                    action,
                    batch=physical_batch(env, bouncing=bouncing),
                    motor_primitives=primitive(index),
                )
                _, _, done, truncated, info = env.step(execution.action)
                airborne |= not env.mario["on_ground"]
                bouncing |= info["reward_terms"]["enemy_stomp"] > 0
                landed = airborne and env.mario["on_ground"]
                if info["death"]:
                    break
                reached = env._goal_credited if env._single_jump_attempt else target.reached(env)
                if reached and (landed or env._goal_credited):
                    recoverable = True
                    if landed and not env._goal_credited and verify_recovery:
                        following = training_target(env)
                        if (
                            following.kind in ("enemy", "stomp")
                            and local_target_distance(env, following) < 50
                        ):
                            recoverable = bool(
                                safe_jump_indices(
                                    model,
                                    env,
                                    2 if following.direction > 0 else 4,
                                    verify_recovery=False,
                                )
                            )
                    if recoverable:
                        valid.append(index)
                    break
                if done or truncated or landed:
                    break
    return valid


def bridge_target(env):
    bridge = next((p for p in env.platforms if p.get("moving")), None)
    if bridge is None or env._bridge_crossed:
        return "finish"
    m, b = env.mario, bridge["rect"]
    if m.get("_platform") is bridge and m["x"] >= b.left + 4:
        return "exit"
    return "board"


def bridge_walk_reached(env, target):
    if target == "exit":
        return bool(env._bridge_crossed)
    bridge = next(p for p in env.platforms if p.get("moving"))
    m, b = env.mario, bridge["rect"]
    return m.get("_platform") is bridge and m["x"] >= b.left + 4 and m["x"] + m["w"] <= b.right - 4


def safe_bridge_wait_indices(env):
    """Actual NES collision/carry/reversal windows, not the legacy predictor.

    Returns whether walking now works, safe departure bins, and safe waiting
    bins for re-observation when the next departure lies beyond this menu.
    """
    target = bridge_target(env)
    if target == "finish":
        return True, [], []
    departures, continuations = [], []
    with probe_state(env) as saved:
        for delay in range(max(NES_JUMP_FRAMES) + 1):
            restore_env_state(env, saved)
            safe = True
            for _ in range(delay):
                _, _, done, truncated, info = env.step(0)
                if done or truncated or info["death"] or not env.mario["on_ground"]:
                    safe = False
                    break
            if not safe:
                continue
            if delay in NES_JUMP_FRAMES:
                continuations.append(NES_JUMP_FRAMES.index(delay))
            for _ in range(48):
                _, _, done, truncated, info = env.step(1)
                if info["death"] or not env.mario["on_ground"]:
                    break
                if bridge_walk_reached(env, target) or env._goal_credited:
                    if delay == 0:
                        return True, [], []
                    departures.append(delay)
                    break
                if done or truncated:
                    break
    # Recheck before a narrow departure falling between two menu values.
    if departures:
        continuations = [i for i in continuations if NES_JUMP_FRAMES[i] <= min(departures)]
    return (
        False,
        [NES_JUMP_FRAMES.index(n) for n in departures if n in NES_JUMP_FRAMES],
        continuations,
    )


def interior_index(indices):
    """Choose the middle of the longest safe run; never cross an unsafe hole."""
    runs = []
    for index in sorted(indices):
        if not runs or index != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(index)
    run = max(runs, key=len)
    return run[len(run) // 2]


def coach_choice(model, env, *, takeoff_distance=50, variant=0):
    if env._require_bridge_before_goal:
        ready, departures, safe = safe_bridge_wait_indices(env)
        if ready:
            return 1, 0, list(range(16))
        if departures:
            index = departures[variant % len(departures)] if variant else interior_index(departures)
            # All departure bins are collision-certified. Earlier rechecks are
            # useful but are not mislabeled as equally good release times.
            return 0, index, departures
        if safe:
            index = max(safe)
            return 0, index, safe
        return 0, 0, [0]
    target = training_target(env)
    direction = target.direction
    action = 2 if direction > 0 else 4
    if (
        env.mario["on_ground"]
        and target.kind not in ("finish", "retreat")
        and (local_target_distance(env, target) < takeoff_distance or env._single_jump_attempt)
    ):
        valid = safe_jump_indices(model, env, action)
        if valid:
            index = valid[variant % len(valid)] if variant else interior_index(valid)
            return action, index, valid
    return (1 if direction > 0 else 3), 0, list(range(16))
