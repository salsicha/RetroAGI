"""Requalify generated families under NES physics; never reuse legacy labels."""

import copy
from dataclasses import replace

from retroagi.core.smb_physics import NES_PHYSICS_PROFILE
from retroagi.stages.block_smb.bridge_traversal import bridge_oracle
from retroagi.stages.block_smb.local_traversal import terrain_oracle
from retroagi.stages.block_smb.monte_carlo import (
    sample_block_smb_monte_carlo_scenario,
    validate_block_smb_monte_carlo_oracle,
)

DISTRIBUTION = "block_smb_nes_land_v4"
FAMILY_ALIASES = {"wait_timing": "bridge_wait"}


def canonical_families(families):
    return list(dict.fromkeys(FAMILY_ALIASES.get(f, f) for f in families))


def sample_nes_case(*, family, split, seed, index, difficulty="medium", max_rejections=32):
    if not 0 <= max_rejections < 1024 or index < 0:
        raise ValueError("Invalid candidate index or rejection budget")
    requested_family = family
    family = FAMILY_ALIASES.get(family, family)
    reasons = []
    for attempt in range(max_rejections + 1):
        old = sample_block_smb_monte_carlo_scenario(
            split=split,
            seed=seed,
            sample_index=index * 1024 + attempt,
            family=family,
            difficulty=difficulty,
            validate_reachability=False,
        )
        scenario = copy.deepcopy(old.scenario)
        scenario["physics_profile"] = NES_PHYSICS_PROFILE
        scenario["task_direction"] = -1 if family == "retreat_recovery" else 1
        scenario["task_objective"] = "stomp" if family in ("enemy_stomp", "stomp_mount") else None
        scenario["mario"][1] += 4  # preserve the authored feet position with the NES small box
        parameters = dict(old.parameters)
        if requested_family != family:
            scenario["metadata"]["family_alias"] = requested_family
        if family == "enemy_stomp":
            # Short, invisible patrol limits made identical observed motion
            # require incompatible jump holds. Use the visible floor span;
            # no authored turnaround can occur in the middle of a flat approach.
            for enemy in scenario["enemies"]:
                enemy[2], enemy[3] = 0, scenario["world_width"]
            parameters.update(family_revision=3, patrol_halfwidth=None, enemy_motion="floor_span")
            scenario["metadata"]["block_smb_monte_carlo"]["parameters"] = parameters
        if family in ("pit_leap", "platform_hop"):
            # The duration-isolation task begins at a real running takeoff.
            # NES caps a jump initiated from rest at walking horizontal speed.
            scenario["mario_velocity"] = [2.5, 0.0]
        candidates = [list(old.oracle["actions"])]
        if family == "platform_hop":
            from retroagi.core.smb_physics import NES_JUMP_FRAMES

            # This family isolates duration selection at the initial state.
            # Never accept a fallback route that silently adds a run-up.
            routes = [[2] * hold + [1] * (320 - hold) for hold in NES_JUMP_FRAMES]
            routes = [
                route
                for route in routes
                if validate_block_smb_monte_carlo_oracle(scenario, route, max_steps=320)[
                    "reachable"
                ]
            ]
            if not routes:
                reasons.append({"attempt": attempt, "reason": "no_immediate_jump"})
                continue
            candidates = [routes[len(routes) // 2]]
        for method in ("original_verified", "nes_local_search", "nes_bridge_search"):
            if method == "nes_local_search":
                candidates.append(terrain_oracle(scenario, max_steps=320))
            elif method == "nes_bridge_search":
                if family not in ("bridge_wait", "moving_bridge", "wait_timing"):
                    continue
                candidates.append(bridge_oracle(scenario, max_steps=320)[0])
            actions = candidates[-1]
            result = validate_block_smb_monte_carlo_oracle(scenario, actions, max_steps=320)
            if result["reachable"]:
                scenario["metadata"]["block_smb_monte_carlo"]["distribution_id"] = DISTRIBUTION
                scenario_id = f"{DISTRIBUTION}:{split}:{seed}:{family}:{difficulty}:{index}"
                oracle = {
                    **old.oracle,
                    "actions": actions,
                    "action_source": method,
                    "expected_completion_steps": result.get("completion_steps"),
                }
                scenario["metadata"]["block_smb_monte_carlo"].update(
                    scenario_id=scenario_id,
                    sample_index=index,
                    oracle=oracle,
                    reachability=result,
                    physics_profile=NES_PHYSICS_PROFILE,
                )
                return replace(
                    old,
                    parameters=parameters,
                    distribution_id=DISTRIBUTION,
                    scenario_id=scenario_id,
                    sample_index=index,
                    scenario=scenario,
                    oracle=oracle,
                    reachability={
                        **result,
                        "physics_profile": NES_PHYSICS_PROFILE,
                        "rejections": reasons,
                    },
                )
        reasons.append({"attempt": attempt, "reason": "no_verified_route"})
    raise ValueError(
        f"No NES-compatible route: {family}/{difficulty}; rejected {len(reasons)} candidates"
    )
