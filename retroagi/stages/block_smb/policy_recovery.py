"""The teacher's route from a decision state to the goal. Training only.

Each frame does what the layout's current tactic segment calls for
(_coached_suffix); the teachers take their plans, holds and checks from it.
"""

from dataclasses import replace

from retroagi.core.smb_coaching import training_target
from retroagi.core.smb_physics import NES_JUMP_FRAMES

from .env_state import restore_env_state, snapshot_env_state
from .hierarchy import bridge_training_active
from .local_traversal import (
    JumpReleaseState,
    blocking_wall_distance,
    local_target_distance,
    safe_jump_holds,
    support_edge_distance,
    takeoff_timing_actions,
)

# Frames without progress after which the teacher's route is given up.
STALL_FRAMES = 400


def interior_hold(valid, menu=NES_JUMP_FRAMES):
    runs = []
    for hold in valid:
        if not runs or menu.index(hold) != menu.index(runs[-1][-1]) + 1:
            runs.append([])
        runs[-1].append(hold)
    run = max(runs, key=len)
    return run[len(run) // 2]


def coached_suffix(
    env, *, max_frames=320, release_state=None, observation_history=None, prefix=False
):
    """The teacher's route from here to the goal, following the layout's tactics.

    Plants that can always be cleared are certified against their full height
    (piranha.conservative_suffix); a timed plant is crossed by its teacher.
    """
    if getattr(env, "_action_jump_direction", 0):
        routes = single_jump_routes(env, max_frames=max_frames)
        if not routes:
            return []
        holds = list(routes)
        chosen = interior_hold(holds, menu=tuple(range(33)))
        return routes[chosen]

    from .piranha import conservative_suffix, has_plants
    from .piranha_tactics import timed_plant

    if not prefix and has_plants(env) and timed_plant(env) is None:
        return conservative_suffix(env, max_frames=max_frames, release_state=release_state)
    return _coached_suffix(
        env,
        max_frames=max_frames,
        release_state=release_state,
        observation_history=observation_history,
        prefix=prefix,
    )


def route_wins(env, actions) -> bool:
    """Whether ``actions``, pressed frame by frame from here, reach the goal (state restored)."""
    saved = snapshot_env_state(env)
    try:
        for action in actions:
            if env.step(action)[2]:
                break
        return bool(env._goal_credited)
    finally:
        restore_env_state(env, saved)


def _coached_suffix(
    env,
    *,
    max_frames=320,
    release_state=None,
    hold_variant=0,
    robust_takeoff=False,
    observation_history=None,
    prefix=False,
):
    """Complete from a decision state; never used by policy playback.

    Each grounded frame does what the layout's current tactic segment
    (tactic_schedule) calls for: ride or wait for a moving platform, wait for
    or cross a timed plant (piranha_tactics.tactical_choice), keep away from or
    jump over a monster (monster.monster_choice), hold still, or reach the next
    objective (smb_coaching.training_target: the segment's route platform or
    retreat line, else the nearest obstacle toward the goal). With
    robust_takeoff, jumps over a plant launch only where the online takeoff-timing
    labels allow one, so demonstrations never contradict them.
    """
    from copy import deepcopy

    from retroagi.core.smb_enemy_history import EnemyObservationHistory

    from . import tactic_schedule
    from .monster import monster_choice
    from .piranha_tactics import tactical_choice, timed_plant

    history = None
    if timed_plant(env) is not None:
        history = (
            deepcopy(observation_history)
            if observation_history is not None
            else EnemyObservationHistory()
        )
    actions = []
    remaining = 0
    # A route that starts in the air keeps going toward its target.
    direction = training_target(env).direction
    airborne = False
    retreat = 0
    coasting = False
    release = replace(release_state) if release_state is not None else JumpReleaseState()

    def jump(valid, toward):
        nonlocal remaining, airborne, retreat, direction
        chosen = interior_hold(valid, NES_JUMP_FRAMES)
        if hold_variant:
            chosen = valid[(valid.index(chosen) + hold_variant) % len(valid)]
        remaining = chosen - 1
        airborne = True
        retreat = 0
        direction = toward
        return 2 if toward > 0 else 4

    # A route that makes no progress for this long is given up: no new
    # segment, route platform or furthest point.
    progress, stalled = None, 0
    for _ in range(max_frames):
        mark = (env._tactic_index, env._route_done, int(env._max_x_reached))
        stalled = 0 if mark != progress else stalled + 1
        progress = mark
        if stalled > STALL_FRAMES:
            break
        features = history.observe(env, env.steps) if history is not None else None
        segment = tactic_schedule.current(env)
        plant = timed_plant(env)
        if env.mario["on_ground"]:
            airborne = False
        if release.remaining:
            remaining = 0
            action = release.action
        elif remaining:
            action = 2 if direction > 0 else 4
            remaining -= 1
        elif airborne or not env.mario["on_ground"]:
            action = 1 if direction > 0 else 3
        elif (
            tactic_schedule.in_kind(env, "bridge")
            and bridge_training_active(env)
            and not env._bridge_jump_task
        ):
            from .bridge_traversal import bridge_phase

            phase = bridge_phase(env, True)
            action = 1 if phase in ("approach", "board", "exit", "finish") else 0
            # Arriving at a run, coast down to walking pace before the shore
            # ends: from above 1.5 pixels a frame to below 0.75.
            coasting = phase == "approach" and env.mario["vx"] > (0.75 if coasting else 1.5)
            if coasting:
                action = 0
        elif (
            plant is not None
            and tactic_schedule.in_kind(env, "plant")
            and env.mario["x"] < plant["x"] + plant["w"]
        ):
            _, action, valid = tactical_choice(env, features)
            if valid:
                action = jump(valid, 1)
        elif (choice := monster_choice(env)) is not None:
            _, action, valid = choice
            if valid:
                action = jump(valid, segment["direction"])
        elif segment["kind"] == "plain" and segment["stance"] == "hold_area":
            action = 0
        else:
            target = training_target(env)
            direction = target.direction
            bridge = bool(env._bridge_jump_task)
            distance = local_target_distance(env, target)
            wall = blocking_wall_distance(env, target)
            ready = bridge or distance < 65 or support_edge_distance(env, direction) < 24
            valid = (
                safe_jump_holds(env, target, direction, plant_history=features)
                if ready and (bridge or target.kind not in ("finish", "retreat"))
                else []
            )
            if valid and target.kind == "enemy" and getattr(env, "_prefer_enemy_bypass", False):
                bypass = safe_jump_holds(
                    env, target, direction, plant_history=features, avoid_stomp=True
                )
                if bypass:
                    valid = bypass
            at_plant = (
                target.enemy_index is not None
                and env.enemies[target.enemy_index].get("kind") == "piranha_plant"
            )
            if valid and robust_takeoff and at_plant:
                timing = takeoff_timing_actions(env)
                if timing is not None and not timing[2 if direction > 0 else 4]:
                    valid = []
            if bridge and valid:
                from .bridge_curriculum import bridge_jump_allowed, bridge_takeoff_window

                # Depart from the robust window the online labels teach.
                now, later = bridge_takeoff_window(
                    env, lambda: safe_jump_holds(env, training_target(env), direction)
                )
                if not bridge_jump_allowed(now, later):
                    valid = []
            if valid and bridge:
                # Bridge departures take the longest certified hold, which
                # tolerates departure-timing drift across the window.
                remaining = max(valid) - 1
                airborne = True
                retreat = 0
                action = 2 if direction > 0 else 4
            elif valid:
                action = jump(valid, direction)
            elif bridge:
                action = 0 if abs(env.mario["vx"]) < 1 / 16 else (3 if env.mario["vx"] > 0 else 1)
            elif retreat or (wall is not None and wall <= 1):
                # Some mounts and plant pipes require a run-up. Recheck
                # certification while backing away instead of permanently
                # pushing into the wall. NES walking needs 16 frames to back
                # off about 7 px; 8 frames moved Mario only about 2 px.
                if not retreat:
                    retreat = 16
                retreat -= 1
                action = 3 if direction > 0 else 1
            else:
                action = 1 if direction > 0 else 3
        if prefix and actions and action in (2, 4, 5) and not airborne and not remaining:
            break
        was_airborne = not env.mario["on_ground"]
        before_mark = (env._tactic_index, env._route_done)
        _, _, done, truncated, info = env.step(action)
        release.observe(env, action, info)
        actions.append(action)
        if info["reward_terms"]["enemy_stomp"] > 0:
            remaining = 0
            airborne = True
        if done or truncated:
            break
        if prefix and (
            action == 0
            and env.mario["on_ground"]
            or was_airborne
            and (env.mario["on_ground"] or env.stomped)
            or before_mark != (env._tactic_index, env._route_done)
            or len(actions) >= 64
            and env.mario["on_ground"]
        ):
            break
    if not prefix and not env._goal_credited:
        return None
    return actions


def single_jump_routes(env, *, max_frames=160):
    """Certify one immediate jump and landing, without walking off or retrying.

    Used only by the isolated directional jump families. Replaying a partial
    maneuver in the air only coasts; it cannot launch a second jump.
    """
    from retroagi.core.smb_coaching import probe_state
    from retroagi.core.smb_executor import FRAME_COUNTS

    direction = env._action_jump_direction
    jump_action, coast = (2, 1) if direction > 0 else (4, 3)
    holds = FRAME_COUNTS if env.mario["on_ground"] else (0,)
    routes = {}
    for hold in holds:
        with probe_state(env):
            airborne = not env.mario["on_ground"]
            route = []
            for frame in range(min(max_frames, 160)):
                action = jump_action if frame < hold else coast
                route.append(action)
                _, _, done, _, _ = env.step(action)
                airborne = airborne or not env.mario["on_ground"]
                if env._goal_credited:
                    routes[hold] = route
                    break
                if done or (airborne and env.mario["on_ground"]):
                    break
    return routes
