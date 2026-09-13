"""Training-only successful suffixes from the policy's actual approach states."""

from dataclasses import fields
from types import SimpleNamespace

import torch

from retroagi.core.smb_coaching import training_target

from .env import MarioScenarioEnv
from .geometry_expert import restore_env_state, snapshot_env_state
from .local_traversal import local_target_distance, safe_jump_holds, support_edge_distance
from .monte_carlo import BLOCK_SMB_MC_FAMILIES, block_smb_monte_carlo_metadata

RECOVERY_FAMILIES = frozenset(
    "bridge_mount bridge_dismount chained_obstacles mixed_section full_smb_opening_proxy "
    "chained_enemy_gauntlet tall_pipe_jump pipe_mount enemy_stomp".split()
)


def interior_hold(valid, menu=tuple(range(1, 17))):
    runs = []
    for hold in valid:
        if not runs or menu.index(hold) != menu.index(runs[-1][-1]) + 1:
            runs.append([])
        runs[-1].append(hold)
    run = max(runs, key=len)
    return run[len(run) // 2]


def coached_suffix(env, *, closing_window=False, max_frames=320):
    """Complete from a grounded decision state; never used by policy playback."""
    actions = []
    remaining = 0
    direction = 1
    airborne = False
    retreat = 0
    for _ in range(max_frames):
        if env.mario["on_ground"]:
            airborne = False
        if remaining:
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
                remaining = interior_hold(valid, menu) - 1
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
        actions.append(action)
        if info["reward_terms"]["enemy_stomp"] > 0:
            remaining = 0
            airborne = True
        if done or truncated:
            break
    return actions if env._goal_credited else None


def repair_policy_actions(scenario, actions, *, seed=0, max_repairs=3):
    """Retain completed suffixes only; replay, but never supervise, failed prefixes."""
    env = MarioScenarioEnv()
    repairs = []
    captured = set()
    stalled = 0
    just_landed = False
    try:
        env.reset(scenario=scenario, seed=seed)
        env.render = lambda: None
        for frame, action in enumerate(actions):
            if env.mario["on_ground"] and len(repairs) < max_repairs:
                target = training_target(env)
                bridge = bool(env._bridge_jump_task)
                relevant = bridge or target.kind in ("mount", "stomp")
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
                    valid = safe_jump_holds(env, target, target.direction)
                    if valid:
                        reason = "stall" if stalled >= 3 else "landing_recovery"
                key = (reason, target.kind, target.platform_index, target.enemy_index)
                if reason and key not in captured:
                    saved = snapshot_env_state(env)
                    # A bridge disagreement supplies both the next feasible
                    # departure and the closing boundary of its safe window.
                    for closing in (False, True) if bridge else (False,):
                        if len(repairs) >= max_repairs:
                            break
                        restore_env_state(env, saved)
                        suffix = coached_suffix(env, closing_window=closing)
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
                    restore_env_state(env, saved)
                    captured.add(key)
            before_x = env.mario["x"]
            before_ground = env.mario["on_ground"]
            _, _, done, truncated, _ = env.step(action)
            just_landed = not before_ground and env.mario["on_ground"]
            stalled = stalled + 1 if abs(env.mario["x"] - before_x) < 1 else 0
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
