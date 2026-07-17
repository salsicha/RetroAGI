"""Action-distribution statistics for detecting policy collapse.

The historical Block SMB failure mode was a policy whose deterministic argmax
collapsed to a single action (RIGHT=15934, everything else 0), discovered only
by counting evaluation actions after an expensive run. These helpers turn any
action-count mapping into entropy/dominance statistics with an explicit
``collapsed`` verdict so trainers can flag the condition per epoch.
"""

import math
from typing import Any, Mapping

DEFAULT_COLLAPSE_SHARE_THRESHOLD = 0.95


def action_distribution_stats(
    action_counts: Mapping[Any, Any],
    *,
    action_count: int,
    collapse_share_threshold: float = DEFAULT_COLLAPSE_SHARE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize an action-count mapping into collapse statistics.

    ``action_counts`` maps action ids (int or str) to counts. Returns entropy
    in nats, entropy normalized by ``log(action_count)`` (1.0 = uniform), the
    dominant action and its share, and ``collapsed`` — True when a single
    action holds at least ``collapse_share_threshold`` of all decisions.
    """

    if action_count <= 1:
        raise ValueError("action_count must be greater than 1")
    if not 0.0 < collapse_share_threshold <= 1.0:
        raise ValueError("collapse_share_threshold must be in (0, 1]")

    counts = [0.0] * action_count
    for raw_action, raw_value in dict(action_counts or {}).items():
        try:
            action = int(raw_action)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if 0 <= action < action_count and value > 0.0:
            counts[action] += value

    total = sum(counts)
    if total <= 0.0:
        return {
            "total_actions": 0,
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "dominant_action": None,
            "dominant_share": 0.0,
            "collapse_share_threshold": float(collapse_share_threshold),
            "collapsed": False,
        }

    shares = [count / total for count in counts]
    entropy = -sum(share * math.log(share) for share in shares if share > 0.0)
    dominant_action = max(range(action_count), key=lambda index: shares[index])
    dominant_share = shares[dominant_action]
    return {
        "total_actions": int(total),
        "entropy_nats": float(entropy),
        "normalized_entropy": float(entropy / math.log(action_count)),
        "dominant_action": int(dominant_action),
        "dominant_share": float(dominant_share),
        "collapse_share_threshold": float(collapse_share_threshold),
        "collapsed": bool(dominant_share >= collapse_share_threshold),
    }
