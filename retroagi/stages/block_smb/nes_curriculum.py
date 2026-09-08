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

DISTRIBUTION = "block_smb_nes_land_v1"


def sample_nes_case(*, family, split, seed, index, difficulty="medium", max_rejections=32):
    if not 0 <= max_rejections < 1024 or index < 0:
        raise ValueError("Invalid candidate index or rejection budget")
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
        scenario["mario"][1] += 4  # preserve the authored feet position with the NES small box
        if family == "pit_leap":
            # The duration-isolation task begins at a real running takeoff.
            # NES caps a jump initiated from rest at walking horizontal speed.
            scenario["mario_velocity"] = [2.5, 0.0]
        candidates = [list(old.oracle["actions"])]
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
