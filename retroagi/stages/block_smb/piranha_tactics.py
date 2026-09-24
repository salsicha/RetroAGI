"""Training-only temporal plant teacher using observed disappearance history.

No live cycle phase or remaining timer is used to choose a departure. The
teacher certifies against the shortest hidden interval and fastest emergence
in the timed family, after actually observing the plant disappear.
"""

from dataclasses import replace

from retroagi.core.models import TACTIC_STANCES
from retroagi.core.smb_enemy_history import EnemyObservationHistory

MIN_HIDDEN_FRAMES = 48


def hold_menu(env):
    from retroagi.core.smb_physics import NES_JUMP_FRAMES, NES_PHYSICS_PROFILE

    return NES_JUMP_FRAMES if env.physics_profile == NES_PHYSICS_PROFILE else tuple(range(1, 17))


def timed_plant(env):
    return next((e for e in env.enemies if e.get("timed_crossing")), None)


def fresh_retraction(history):
    # An initially empty pipe has unknown phase. Require a recent sighting,
    # not merely invisibility or the time elapsed since episode reset.
    return history is not None and history[0] == 0 and 0 < history[5] * 64 <= 8


def staging_x(env):
    plant = timed_plant(env)
    pipe = next(p["rect"] for p in env.platforms if p["rect"].top == plant["pipe_top"])
    return float(pipe.left - 32)


def timed_safe_holds(env, history, direction=1):
    from .geometry_expert import restore_env_state, snapshot_env_state

    plant = timed_plant(env)
    if plant is None or direction != 1 or not env.mario["on_ground"]:
        return []
    if env.mario["x"] >= plant["x"] + plant["w"]:
        return []
    if not fresh_retraction(history):
        return []
    saved = snapshot_env_state(env)
    original_render = env.render
    env.render = lambda: None
    valid = []
    try:
        for hold in hold_menu(env):
            restore_env_state(env, saved)
            probe = timed_plant(env)
            # Bounds are part of the generated family contract, not a peek
            # at this episode's sampled hidden duration or phase. One extra
            # frame covers rounding at disappearance.
            probe.pop("conservative_envelope", None)
            probe.update(
                rise_frames=12, exposed_frames=64, hidden_frames=MIN_HIDDEN_FRAMES, plant_height=80
            )
            probe["plant_tick"] = 2 * 12 + 64 + int(round(history[5] * 64)) + 1
            airborne = False
            for frame in range(96):
                _, _, done, truncated, info = env.step(2 if frame < hold else 1)
                airborne |= not env.mario["on_ground"]
                if info["death"] or truncated:
                    break
                if airborne and env.mario["on_ground"]:
                    if env.mario["x"] >= probe["x"] + probe["w"]:
                        safe = True
                        for _ in range(2):
                            _, _, _, _, landing = env.step(1)
                            safe &= not landing["death"]
                        if safe:
                            valid.append(hold)
                    break
                if done:
                    break
        return valid
    finally:
        restore_env_state(env, saved)
        env.render = original_render


def tactical_choice(env, history):
    """Return stance, motor action and certified holds at a decision state."""
    plant = timed_plant(env)
    if plant is None or env.mario["x"] >= plant["x"] + plant["w"]:
        return "advance", 1, []
    target = staging_x(env)
    if env.mario["x"] < target - 2:
        return "advance", 1, []
    if env.mario["x"] > target + 4:
        return "retreat", 3, []
    if abs(env.mario["vx"]) > 0.5:
        return "hold_area", 0, []
    valid = timed_safe_holds(env, history)
    return ("advance", 2, valid) if valid else ("hold_area", 0, [])


def tactic_label(env, history, action=None):
    """Label decisions only; other families and committed arcs are masked."""
    if not any(e.get("kind") == "piranha_plant" for e in env.enemies):
        return -1
    if not env.mario["on_ground"]:
        return -1
    if timed_plant(env) is not None:
        stance, _, _ = tactical_choice(env, history)
    elif action is not None:
        stance = "hold_area" if action == 0 else "retreat" if action in (3, 4) else "advance"
    else:
        return -1
    return TACTIC_STANCES.index(stance)


def timed_suffix(env, *, max_frames=320, release_state=None, variant=0, observation_history=None):
    from copy import deepcopy

    from .geometry_expert import restore_env_state, snapshot_env_state
    from .policy_recovery import interior_hold
    from .primitive_execution import JumpReleaseState, teacher_route_reachable

    saved = snapshot_env_state(env)
    history = (
        deepcopy(observation_history)
        if observation_history is not None
        else EnemyObservationHistory()
    )
    release = replace(release_state) if release_state is not None else JumpReleaseState()
    remaining = 0
    airborne = not env.mario["on_ground"]
    actions = []
    try:
        for frame in range(max_frames):
            features = history.observe(env, env.steps)
            if release.remaining:
                action = release.action
            elif remaining:
                action = 2
                remaining -= 1
            elif airborne and not env.mario["on_ground"]:
                action = 1
            else:
                airborne = False
                _, action, valid = tactical_choice(env, features)
                if valid:
                    chosen = interior_hold(valid, hold_menu(env))
                    chosen = valid[(valid.index(chosen) + variant) % len(valid)]
                    remaining = chosen - 1
                    airborne = True
            _, _, done, truncated, info = env.step(action)
            release.observe(env, action, info)
            actions.append(action)
            if done or truncated:
                break
        if not env._goal_credited:
            return None
        restore_env_state(env, saved)
        return (
            actions if teacher_route_reachable(env, actions, release_state=release_state) else None
        )
    finally:
        restore_env_state(env, saved)
