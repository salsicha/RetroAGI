"""Game-neutral skill-goal contracts for the HSP2 skill layer.

A skill goal is a measurable local objective (doc: Goal-Conditioned Skill
Layer): clear a gap, mount a platform, wait until a hazard passes, clear an
enemy, or retreat and recover. Goals are encoded as a fixed-width vector —
one-hot goal type plus a normalized magnitude (displacement, duration) — so
policies and outcome heads can condition on them uniformly across games.
"""

from __future__ import annotations

import torch

SKILL_GOAL_TYPES = (
    "clear_gap",
    "mount_platform",
    "wait_pass",
    "enemy_clear",
    "retreat_recover",
)

SKILL_GOAL_ENCODING_DIM = len(SKILL_GOAL_TYPES) + 1

# Magnitudes are normalized by these scales before encoding.
_MAGNITUDE_SCALE = {
    "clear_gap": 128.0,
    "mount_platform": 128.0,
    "wait_pass": 64.0,
    "enemy_clear": 128.0,
    "retreat_recover": 128.0,
}


def skill_goal_encoding(goal_type: str, magnitude: float = 0.0) -> torch.Tensor:
    """Encode a skill goal as a [1, SKILL_GOAL_ENCODING_DIM] float tensor."""

    if goal_type not in SKILL_GOAL_TYPES:
        raise ValueError(f"unknown skill goal type {goal_type!r}")
    encoding = torch.zeros(1, SKILL_GOAL_ENCODING_DIM, dtype=torch.float32)
    encoding[0, SKILL_GOAL_TYPES.index(goal_type)] = 1.0
    scale = _MAGNITUDE_SCALE[goal_type]
    encoding[0, -1] = max(-1.0, min(1.0, float(magnitude) / scale))
    return encoding
