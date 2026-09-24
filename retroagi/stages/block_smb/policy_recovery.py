"""Training-only successful suffixes from the policy's actual approach states."""

from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from retroagi.core.smb_coaching import training_target

from .env import MarioScenarioEnv
from .geometry_expert import restore_env_state, snapshot_env_state
from .local_traversal import local_target_distance, safe_jump_holds, support_edge_distance
from .monte_carlo import BLOCK_SMB_MC_FAMILIES, block_smb_monte_carlo_metadata
from .primitive_execution import JumpReleaseState, teacher_route_reachable
from .transfer_failure_families import TRANSFER_FAILURE_FAMILIES

RECOVERY_FAMILIES = frozenset(
    "bridge_mount bridge_dismount chained_obstacles mixed_section full_smb_opening_proxy "
    "chained_enemy_gauntlet tall_pipe_jump pipe_mount enemy_stomp stair_climb".split()
    + list(TRANSFER_FAILURE_FAMILIES)
)


def interior_hold(valid, menu=tuple(range(1, 17))):
    runs = []
    for hold in valid:
        if not runs or menu.index(hold) != menu.index(runs[-1][-1]) + 1:
            runs.append([])
        runs[-1].append(hold)
    run = max(runs, key=len)
    return run[len(run) // 2]


def coached_suffix(env, *, closing_window=False, max_frames=320, release_state=None):
    from .piranha import conservative_suffix, has_plants

    if has_plants(env):
        return conservative_suffix(env, max_frames=max_frames, release_state=release_state)
    return _coached_suffix(
        env, closing_window=closing_window, max_frames=max_frames, release_state=release_state
    )


def _coached_suffix(
    env, *, closing_window=False, max_frames=320, release_state=None, hold_variant=0
):
    """Complete from a grounded decision state; never used by policy playback."""
    initial = snapshot_env_state(env)
    actions = []
    remaining = 0
    direction = 1
    airborne = False
    retreat = 0
    release = replace(release_state) if release_state is not None else JumpReleaseState()
    for _ in range(max_frames):
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
        else:
            target = training_target(env)
            direction = target.direction
            bridge = bool(env._bridge_jump_task)
            distance = local_target_distance(env, target)
            ready = bridge or distance < 65 or support_edge_distance(env, direction) < 24
            valid = (
                safe_jump_holds(env, target, direction)
                if ready and (bridge or target.kind not in ("finish", "retreat"))
                else []
            )
            if bridge and valid and closing_window:
                saved = snapshot_env_state(env)
                try:
                    env.step(0)
                    # The last feasible departure also teaches a boundary
                    # that the usual first-safe or interior routes omit.
                    if safe_jump_holds(env, training_target(env), direction):
                        valid = []
                finally:
                    restore_env_state(env, saved)
            if valid:
                from retroagi.core.smb_physics import NES_JUMP_FRAMES, NES_PHYSICS_PROFILE

                menu = (
                    NES_JUMP_FRAMES
                    if env.physics_profile == NES_PHYSICS_PROFILE
                    else tuple(range(1, 17))
                )
                chosen = interior_hold(valid, menu)
                if hold_variant:
                    chosen = valid[(valid.index(chosen) + hold_variant) % len(valid)]
                remaining = chosen - 1
                airborne = True
                retreat = 0
                action = 2 if direction > 0 else 4
            elif bridge:
                action = 0 if abs(env.mario["vx"]) < 1 / 16 else (3 if env.mario["vx"] > 0 else 1)
            elif retreat or (target.kind == "mount" and distance <= 1):
                # Some mounts require a run-up. Recheck certification while
                # backing away instead of permanently pushing into the wall.
                if not retreat:
                    retreat = 8
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
        return (
            actions if teacher_route_reachable(env, actions, release_state=release_state) else None
        )
    finally:
        restore_env_state(env, final)


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
        for frame, action in enumerate(actions):
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
                if relevant and action in (2, 4):
                    valid = safe_jump_holds(env, target, 1 if action == 2 else -1)
                    held = 0
                    for future in actions[frame:]:
                        if future != action:
                            break
                        held += 1
                    if held not in valid:
                        reason = "takeoff" if not valid else "duration"
                elif bridge and action == 0:
                    valid = safe_jump_holds(env, target, 1)
                    if valid:
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
                        reason = None
                key = (reason, target.kind, target.platform_index, target.enemy_index)
                if plants:
                    key += (
                        plant_attempt,
                        int(env.mario["x"] // 8),
                        tuple(e["h"] // 4 for e in env.enemies),
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
                    # A bridge disagreement supplies both the next feasible
                    # departure and the closing boundary of its safe window.
                    for closing in (False, True) if bridge else (False,):
                        if not prioritize_late and len(repairs) >= max_repairs:
                            break
                        restore_env_state(env, saved)
                        suffix = coached_suffix(env, closing_window=closing, release_state=release)
                        if suffix is not None:
                            repairs.append(
                                dict(
                                    actions=list(actions[:frame]) + suffix,
                                    supervision_start_frame=frame,
                                    recovery=True,
                                    recovery_reason=reason,
                                    closing_window=closing,
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


def collect_policy_recovery(records, config, vision_factory):
    from .demonstrations import collect_demonstrations

    cases = []
    for record in records:
        metadata = block_smb_monte_carlo_metadata(record["scenario"])
        if metadata.get("split") != "train":
            raise ValueError("Policy recovery supervision must come from the train split")
        family = metadata["family"]
        for repair in repair_policy_actions(
            record["scenario"], record["actions"], seed=record["seed"]
        ):
            sample = SimpleNamespace(
                scenario=record["scenario"],
                scenario_id=record["scenario_id"],
                sample_seed=record["seed"],
                oracle=repair,
            )
            cases.append((BLOCK_SMB_MC_FAMILIES.index(family), sample))
    if not cases:
        return None
    return collect_demonstrations(cases, config, vision_factory)


def combine_demonstrations(batches):
    from .demonstrations import DemonstrationBatch

    return DemonstrationBatch(
        **{
            field.name: torch.cat([getattr(batch, field.name) for batch in batches])
            for field in fields(DemonstrationBatch)
        }
    )
