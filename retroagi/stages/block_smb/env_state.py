"""Save and restore everything a Block SMB step changes.

The teachers try moves forward in the simulator and then put it back exactly
as it was (snapshot_env_state, restore_env_state). Training only.
"""

import copy
from typing import Any, Mapping

from retroagi.stages.block_smb import tactic_schedule
from retroagi.stages.block_smb.env import MarioScenarioEnv


def snapshot_env_state(env: MarioScenarioEnv) -> dict[str, Any]:
    """Copy every field of the env that ``step`` mutates."""

    mario = dict(env.mario)
    support_index = next(
        (i for i, p in enumerate(env.platforms) if p is mario.get("_platform")), None
    )
    mario["_platform"] = None
    mutable_flags = (
        "_goal_credited",
        "_stomp_credited",
        "_bridge_boarded",
        "_bridge_crossed",
        "_bridge_jump_launched",
        "_attempt_failed",
        "_prev_goal_distance",
        "_episode_energy",
        "_objective_missed",
        *tactic_schedule.STATE_FIELDS,
    )
    return {
        "motion": copy.deepcopy(env.motion),
        "mario": mario,
        "support_index": support_index,
        "goal": env.goal.copy() if env.goal is not None else None,
        "mutable_flags": {name: getattr(env, name) for name in mutable_flags if hasattr(env, name)},
        "platforms": [
            {**{k: v for k, v in plat.items() if k != "rect"}, "rect": plat["rect"].copy()}
            for plat in env.platforms
        ],
        "coins": [
            {"rect": coin["rect"].copy(), "collected": coin["collected"]} for coin in env.coins
        ],
        "power_ups": [
            {"rect": item["rect"].copy(), "collected": item["collected"]} for item in env.power_ups
        ],
        "enemies": [dict(enemy) for enemy in env.enemies],
        "camera_x": env.camera_x,
        "score": env.score,
        "steps": env.steps,
        "world_width": env.world_width,
        "_max_x_reached": env._max_x_reached,
        "_airborne_started_with_jump": env._airborne_started_with_jump,
    }


def restore_env_state(env: MarioScenarioEnv, snapshot: Mapping[str, Any]) -> None:
    """Restore a snapshot produced by :func:`snapshot_env_state`."""

    env.motion = copy.deepcopy(snapshot.get("motion"))
    env.mario = dict(snapshot["mario"])
    env.platforms = [
        {**{k: v for k, v in plat.items() if k != "rect"}, "rect": plat["rect"].copy()}
        for plat in snapshot["platforms"]
    ]
    support_index = snapshot.get("support_index")
    env.mario["_platform"] = env.platforms[support_index] if support_index is not None else None
    env.goal = snapshot["goal"].copy() if snapshot.get("goal") is not None else None
    for name, value in snapshot.get("mutable_flags", {}).items():
        setattr(env, name, value)
    env.coins = [
        {"rect": coin["rect"].copy(), "collected": coin["collected"]} for coin in snapshot["coins"]
    ]
    env.power_ups = [
        {"rect": item["rect"].copy(), "collected": item["collected"]}
        for item in snapshot.get("power_ups", ())
    ]
    env.enemies = [dict(enemy) for enemy in snapshot["enemies"]]
    env.camera_x = snapshot["camera_x"]
    env.score = snapshot["score"]
    env.steps = snapshot["steps"]
    env.world_width = snapshot["world_width"]
    env._max_x_reached = snapshot["_max_x_reached"]
    env._airborne_started_with_jump = snapshot["_airborne_started_with_jump"]
