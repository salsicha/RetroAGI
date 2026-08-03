"""Tests for the HSP0 temporal contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from retroagi.core.temporal import (
    HierarchicalTransition,
    TemporalGoal,
    read_transitions_jsonl,
    reconstruct_episodes,
    transition_from_json_dict,
    write_transitions_jsonl,
)


def _span(episode: str, index: int, start: int, end: int, reason: str = "success"):
    return HierarchicalTransition(
        episode_id=episode,
        level="motor_primitive",
        transition_id=f"{episode}:p{index}",
        start_frame=start,
        end_frame=end,
        termination_reason=reason,
    )


class TestTemporalContracts(unittest.TestCase):
    def test_contracts_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            TemporalGoal(level="galaxy", goal_type="x")
        with self.assertRaises(ValueError):
            TemporalGoal(level="skill", goal_type="")
        with self.assertRaises(ValueError):
            _span("e", 0, 0, 3, reason="finished")  # not a known reason
        with self.assertRaises(ValueError):
            _span("e", 0, 5, 3)  # ends before it starts
        with self.assertRaises(ValueError):
            HierarchicalTransition(
                episode_id="e",
                level="motor_primitive",
                transition_id="e:p0",
                start_frame=0,
                end_frame=1,
                termination_reason="success",
                events=({"event": "warp"},),
            )

    def test_json_round_trip(self):
        span = HierarchicalTransition(
            episode_id="e",
            level="skill",
            transition_id="e:s0",
            start_frame=0,
            end_frame=9,
            termination_reason="timeout",
            goal=TemporalGoal(level="skill", goal_type="clear_gap").to_json_dict(),
            events=({"event": "timeout", "frame": 9},),
            child_ids=("e:p0",),
        )
        restored = transition_from_json_dict(json.loads(json.dumps(span.to_json_dict())))
        self.assertEqual(restored, span)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spans.jsonl"
            write_transitions_jsonl(path, [span])
            self.assertEqual(read_transitions_jsonl(path), [span])

    def test_reconstruction_validates_coverage(self):
        good = [_span("e", 0, 0, 4), _span("e", 1, 5, 9, reason="failure")]
        (report,) = reconstruct_episodes(good)
        self.assertTrue(report.valid)
        self.assertEqual(report.frame_count, 10)
        self.assertEqual(report.end_reason, "failure")

        gap = [_span("g", 0, 0, 3), _span("g", 1, 6, 9)]
        (report,) = reconstruct_episodes(gap)
        self.assertFalse(report.valid)
        self.assertTrue(any("gap" in p for p in report.problems))

        overlap = [_span("o", 0, 0, 5), _span("o", 1, 4, 9)]
        (report,) = reconstruct_episodes(overlap)
        self.assertFalse(report.valid)
        self.assertTrue(any("overlap" in p for p in report.problems))


if __name__ == "__main__":
    unittest.main()
