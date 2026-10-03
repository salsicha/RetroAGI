"""Training-only successful suffixes from the policy's actual approach states."""

from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from retroagi.core.smb_coaching import training_target
from retroagi.core.smb_physics import NES_JUMP_FRAMES

from .env import MarioScenarioEnv
from .geometry_expert import restore_env_state, snapshot_env_state
from .hierarchy import bridge_training_active
from .local_traversal import (
    blocking_wall_distance,
    local_target_distance,
    safe_jump_holds,
    support_edge_distance,
    takeoff_timing_actions,
)
from .monte_carlo import BLOCK_SMB_MC_FAMILIES, block_smb_monte_carlo_metadata
from .primitive_execution import JumpReleaseState, teacher_route_reachable
from .tactics import TACTICAL_FAMILIES
from .transfer_failure_families import TRANSFER_FAILURE_FAMILIES

RECOVERY_FAMILIES = (
    frozenset(
        "bridge_mount bridge_dismount chained_obstacles mixed_section full_smb_opening_proxy "
        "chained_enemy_gauntlet tall_pipe_jump pipe_mount enemy_stomp stair_climb".split()
        + list(TRANSFER_FAILURE_FAMILIES)
    )
    | TACTICAL_FAMILIES
)


def interior_hold(valid, menu=NES_JUMP_FRAMES):
    runs = []
    for hold in valid:
        if not runs or menu.index(hold) != menu.index(runs[-1][-1]) + 1:
            runs.append([])
        runs[-1].append(hold)
    run = max(runs, key=len)
    return run[len(run) // 2]


def coached_suffix(
    env, *, max_frames=320, release_state=None, observation_history=None, replay_check=True
):
    """The teacher's route from here to the goal, following the layout's tactics.

    Plants that can always be cleared are certified against their full height
    (piranha.conservative_suffix); a timed plant is crossed by its teacher.
    With replay_check, the route must also replay through the old jump
    executor (primitive_execution.teacher_route_reachable), as the old trainer
    needs; the four-layer agent's executor plays any route exactly.
    """
    from .piranha import conservative_suffix, has_plants
    from .piranha_tactics import timed_plant

    if has_plants(env) and timed_plant(env) is None:
        return conservative_suffix(
            env, max_frames=max_frames, release_state=release_state, replay_check=replay_check
        )
    return _coached_suffix(
        env,
        max_frames=max_frames,
        release_state=release_state,
        observation_history=observation_history,
        replay_check=replay_check,
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
    replay_check=True,
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

    initial = snapshot_env_state(env)
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

    for _ in range(max_frames):
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
        _, _, done, truncated, info = env.step(action)
        release.observe(env, action, info)
        actions.append(action)
        if info["reward_terms"]["enemy_stomp"] > 0:
            remaining = 0
            airborne = True
        if done or truncated:
            break
    if not env._goal_credited:
        return None
    final = snapshot_env_state(env)
    try:
        restore_env_state(env, initial)
        if not replay_check:
            return actions
        return (
            actions if teacher_route_reachable(env, actions, release_state=release_state) else None
        )
    finally:
        restore_env_state(env, final)


def _bridge_departure_allowed(env, target):
    from .bridge_curriculum import bridge_jump_allowed, bridge_takeoff_window

    now, later = bridge_takeoff_window(
        env, lambda: safe_jump_holds(env, training_target(env), target.direction)
    )
    return bridge_jump_allowed(now, later)


def repair_policy_actions(scenario, actions, *, seed=0, max_repairs=3):
    """Retain completed suffixes only; replay, but never supervise, failed prefixes."""
    env = MarioScenarioEnv()
    repairs = []
    priorities = []
    family = block_smb_monte_carlo_metadata(scenario).get("family")
    stairs = family in ("stair_climb", "stair_gap")
    transfer_failure = family in TRANSFER_FAILURE_FAMILIES
    plants = family == "piranha_avoidance"
    stomp = family == "enemy_stomp"
    revisit_states = plants or stomp
    prioritize_late = stairs or family == "enemy_on_platform" or revisit_states
    captured = set()
    plant_attempt = 0
    stalled = 0
    just_landed = False
    jump_target = None
    release = JumpReleaseState()
    try:
        env.reset(scenario=scenario, seed=seed)
        env.render = lambda: None
        from retroagi.core.smb_enemy_history import EnemyObservationHistory

        from .piranha_tactics import tactical_choice, timed_plant

        history = EnemyObservationHistory()
        for frame, action in enumerate(actions):
            plant_features = history.observe(env, env.steps)
            if (
                env.mario["on_ground"]
                and max_repairs > 0
                and (prioritize_late or len(repairs) < max_repairs)
            ):
                target = training_target(env)
                bridge = bool(env._bridge_jump_task)
                relevant = (
                    bridge
                    or target.kind in ("mount", "stomp")
                    or (transfer_failure and target.kind in ("gap", "enemy"))
                )
                reason = None
                valid = None
                if (
                    bridge_training_active(env)
                    and not env._bridge_jump_task
                    and not release.remaining
                ):
                    from .bridge_traversal import bridge_phase

                    desired = (
                        1
                        if bridge_phase(env, True) in ("approach", "board", "exit", "finish")
                        else 0
                    )
                    if action != desired:
                        reason = "tactic"
                elif (
                    family in ("enemy_patrol", "retreat_recovery")
                    and action in (0, 1, 3)
                    and not release.remaining
                ):
                    from .tactics import compatible_actions, tactic_label

                    label = tactic_label(env, plant_features, family=family)
                    if label >= 0 and not compatible_actions(env, label)[action]:
                        reason = "tactic"
                elif timed_plant(env) is not None and not release.remaining:
                    _, desired, valid = tactical_choice(env, plant_features)
                    held = 0
                    for future in actions[frame:]:
                        if future != action:
                            break
                        held += 1
                    if action != desired or (action == 2 and held not in valid):
                        reason = "tactic" if action != desired else "duration"
                elif relevant and action in (2, 4):
                    valid = safe_jump_holds(env, target, 1 if action == 2 else -1)
                    held = 0
                    for future in actions[frame:]:
                        if future != action:
                            break
                        held += 1
                    if bridge and valid and not _bridge_departure_allowed(env, target):
                        reason = "takeoff"
                    elif held not in valid:
                        reason = "takeoff" if not valid else "duration"
                elif bridge and action == 0:
                    if _bridge_departure_allowed(env, target):
                        reason = "departure_window"
                elif relevant and (stalled >= 3 or just_landed):
                    reason = "stall" if stalled >= 3 else "landing_recovery"
                    if stairs:
                        if just_landed and jump_target is not None and not jump_target.reached(env):
                            reason = "retry_recovery"
                        elif stalled >= 16:
                            reason = "pause_recovery"
                    candidate = (reason, target.kind, target.platform_index, target.enemy_index)
                    # A prolonged wall stall otherwise repeats all sixteen
                    # collision probes on every remaining frame.
                    if candidate not in captured and not safe_jump_holds(
                        env, target, target.direction
                    ):
                        wall = blocking_wall_distance(env, target)
                        # A standing NES jump cannot clear a tall wall; the
                        # coached suffix backs off for a run-up. Each wall
                        # stall is captured once.
                        at_wall = stalled >= 3 and wall is not None and wall <= 1
                        reason = "wall_stall" if at_wall else None
                key = (reason, target.kind, target.platform_index, target.enemy_index)
                if reason == "wall_stall":
                    pass
                elif plants:
                    key += (
                        plant_attempt,
                        int(env.mario["x"] // 8),
                        tuple(e["h"] // 4 for e in env.enemies),
                        round(float(plant_features[5]) * 64) if timed_plant(env) else 0,
                    )
                elif stomp:
                    # Returning to the same enemy from the other side or with
                    # different momentum is a new interception problem. A
                    # family/object-only key suppresses those later repairs.
                    key += (
                        target.direction,
                        int(env.mario["x"] // 8),
                        round(env.mario["vx"] * 2),
                    )
                if reason and key not in captured:
                    saved = snapshot_env_state(env)
                    if prioritize_late or len(repairs) < max_repairs:
                        restore_env_state(env, saved)
                        suffix = coached_suffix(
                            env,
                            release_state=release,
                            observation_history=history,
                        )
                        if suffix is not None:
                            repairs.append(
                                dict(
                                    actions=list(actions[:frame]) + suffix,
                                    supervision_start_frame=frame,
                                    recovery=True,
                                    recovery_reason=reason,
                                )
                            )
                            if prioritize_late:
                                # Keep late arrivals/retries represented: early
                                # mounts must not crowd out the final riser or
                                # the enemy on the platform.
                                priorities.append(
                                    (
                                        frame
                                        if revisit_states
                                        else target.direction * target.center,
                                        {
                                            "retry_recovery": 4,
                                            "landing_recovery": 3,
                                            "pause_recovery": 2,
                                            "stall": 1,
                                            "wall_stall": 1,
                                        }.get(reason, 0),
                                    )
                                )
                                if len(repairs) > max_repairs:
                                    # Keep the first corrected departure as
                                    # well as later recovery states. Otherwise
                                    # repeated plant retries evict the very
                                    # mistake that caused the failed prefix.
                                    discard = min(
                                        range(
                                            1 if revisit_states and max_repairs > 1 else 0,
                                            len(priorities),
                                        ),
                                        key=priorities.__getitem__,
                                    )
                                    priorities.pop(discard)
                                    repairs.pop(discard)
                    restore_env_state(env, saved)
                    captured.add(key)
            before_x = env.mario["x"]
            before_ground = env.mario["on_ground"]
            if plants and before_ground and action in (2, 4) and not release.remaining:
                plant_attempt += 1
            if stairs and before_ground and action in (2, 4):
                jump_target = training_target(env)
            _, _, done, truncated, info = env.step(action)
            release.observe(env, action, info)
            just_landed = not before_ground and env.mario["on_ground"]
            stationary = abs(env.mario["x"] - before_x) < 1
            if stairs:
                stationary &= before_ground and env.mario["on_ground"]
            stalled = stalled + 1 if stationary else 0
            if done or truncated:
                break
        return repairs
    finally:
        env.close()


def _repair_task(record):
    return repair_policy_actions(record["scenario"], record["actions"], seed=record["seed"])


def collect_policy_recovery(records, config, vision_factory, *, pool=None):
    from .demonstrations import collect_demonstrations

    for record in records:
        if block_smb_monte_carlo_metadata(record["scenario"]).get("split") != "train":
            raise ValueError("Policy recovery supervision must come from the train split")
    repairs = map(_repair_task, records) if pool is None else pool.map(_repair_task, records)
    cases = []
    for record, record_repairs in zip(records, repairs):
        family = block_smb_monte_carlo_metadata(record["scenario"])["family"]
        for repair in record_repairs:
            sample = SimpleNamespace(
                scenario=record["scenario"],
                scenario_id=record["scenario_id"],
                sample_seed=record["seed"],
                oracle=repair,
            )
            cases.append((BLOCK_SMB_MC_FAMILIES.index(family), sample))
    if not cases:
        return None
    return collect_demonstrations(cases, config, vision_factory, pool=pool)


def combine_demonstrations(batches):
    from .demonstrations import DemonstrationBatch

    return DemonstrationBatch(
        **{
            field.name: torch.cat([getattr(batch, field.name) for batch in batches])
            for field in fields(DemonstrationBatch)
            # Carried states depend on the weights at refresh; recompute them.
            if field.name not in ("memory_state", "world_model_inputs")
        }
    )
