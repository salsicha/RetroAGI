"""HSP2 skill goals for Block SMB: requested goals and hindsight relabels.

The requested goal comes from the scenario family (the scenario IS a skill
request) and conditions the policy during the rollout. The achieved goals are
relabeled afterward from the HSP0 spans using measurable state only —
displacement, elevation change, hazard contact, waiting, and termination
reasons — never from what the attempt was supposed to do.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from retroagi.core.skills import skill_goal_encoding
from retroagi.core.temporal import HierarchicalTransition

from .monte_carlo import block_smb_monte_carlo_metadata

# Which measurable goal each scenario family requests of the skill layer.
BLOCK_SMB_FAMILY_SKILL_GOALS: dict[str, str] = {
    "single_gap": "clear_gap",
    "pit_leap": "clear_gap",
    "platform_hop": "clear_gap",
    "platform_chain": "clear_gap",
    "moving_bridge": "clear_gap",
    "chained_obstacles": "clear_gap",
    "stair_climb": "mount_platform",
    "pipe_mount": "mount_platform",
    "tall_pipe_jump": "mount_platform",
    "wait_timing": "wait_pass",
    "enemy_hop": "enemy_clear",
    "enemy_patrol": "enemy_clear",
    "enemy_gap": "enemy_clear",
    "enemy_stomp": "enemy_clear",
    "stomp_mount": "enemy_clear",
    "chained_enemy_gauntlet": "enemy_clear",
    "retreat_recovery": "retreat_recover",
}

_MIN_GAP_DISPLACEMENT = 24.0
_MIN_MOUNT_RISE = 8.0
_MIN_WAIT_FRAMES = 8
_MIN_RETREAT_DISPLACEMENT = 8.0


def requested_block_smb_skill_goal(
    scenario: Mapping[str, Any] | None,
) -> torch.Tensor | None:
    """Goal encoding requested by the scenario's family, if any."""

    if scenario is None:
        return None
    metadata = block_smb_monte_carlo_metadata(scenario)
    family = ""
    if isinstance(metadata, Mapping):
        family = str(metadata.get("family", "") or "")
    goal_type = BLOCK_SMB_FAMILY_SKILL_GOALS.get(family)
    if goal_type is None:
        return None
    return skill_goal_encoding(goal_type)


def achieved_block_smb_skill_goals(
    spans: Sequence[HierarchicalTransition],
) -> list[dict[str, Any]]:
    """Hindsight relabel: which measurable skill goals did each span achieve?

    Returns dicts with goal_type, magnitude, start_frame, and span id. A span
    can achieve more than one goal (a jump that lands higher up also cleared
    ground). Only verifiable state is consulted.
    """

    achieved: list[dict[str, Any]] = []
    for span in spans:
        if span.level != "motor_primitive":
            continue
        primitive = str(span.command.get("primitive", ""))
        displacement = float(span.outcome.get("displacement", 0.0) or 0.0)
        y_before = span.state_before.get("y")
        y_after = span.state_after.get("y")
        rise = None
        if y_before is not None and y_after is not None:
            rise = float(y_before) - float(y_after)

        def record(goal_type: str, magnitude: float) -> None:
            achieved.append(
                {
                    "goal_type": goal_type,
                    "magnitude": float(magnitude),
                    "start_frame": span.start_frame,
                    "span_id": span.transition_id,
                }
            )

        if primitive == "jump":
            hazard = any(e.get("event") == "hazard_contact" for e in span.events)
            if hazard and span.failure_category != "death":
                # Contact from above without dying: the enemy was cleared by
                # a stomp bounce (side contact would have been a death).
                record("enemy_clear", abs(displacement))
                continue
            if span.termination_reason != "success":
                continue
            if rise is not None and rise >= _MIN_MOUNT_RISE:
                record("mount_platform", rise)
            if abs(displacement) >= _MIN_GAP_DISPLACEMENT:
                record("clear_gap", abs(displacement))
        elif primitive == "wait":
            if (
                span.termination_reason == "success"
                and span.duration >= _MIN_WAIT_FRAMES
            ):
                record("wait_pass", span.duration)
        elif primitive == "run_left":
            if (
                span.termination_reason == "success"
                and displacement <= -_MIN_RETREAT_DISPLACEMENT
            ):
                record("retreat_recover", abs(displacement))
    return achieved
