"""Tests for the HSP3 tactic manager."""

import unittest

import torch

from retroagi.core.skills import SKILL_GOAL_TYPES, skill_goal_encoding
from retroagi.core.tactics import TacticManager, tactic_transition_examples


class TestTacticManager(unittest.TestCase):
    def test_selects_highest_outcome_goal_and_holds_it(self):
        # Outcome stub: enemy_clear scores highest.
        def outcome(state, encoding):
            index = int(encoding.reshape(-1)[: len(SKILL_GOAL_TYPES)].argmax())
            return torch.tensor([2.0 if SKILL_GOAL_TYPES[index] == "enemy_clear" else -2.0])

        manager = TacticManager(reselect_interval=4)
        state = torch.zeros(1, 8)
        goal = manager.maybe_select(state, outcome)
        self.assertEqual(manager.active_goal_type, "enemy_clear")
        # Held between triggers: same object, no re-scoring.
        held = manager.maybe_select(state, outcome)
        self.assertIs(held, goal)
        self.assertEqual(manager.decisions_since_selection, 1)

    def test_reselects_on_span_end_and_interval(self):
        calls = []

        def outcome(state, encoding):
            calls.append(1)
            return torch.tensor([0.0])

        manager = TacticManager(reselect_interval=3)
        state = torch.zeros(1, 8)
        manager.maybe_select(state, outcome)
        first_scoring_calls = len(calls)
        self.assertEqual(first_scoring_calls, len(SKILL_GOAL_TYPES))
        manager.notify_span_end()
        manager.maybe_select(state, outcome)
        self.assertEqual(len(calls), 2 * len(SKILL_GOAL_TYPES))
        # Interval trigger: two quiet decisions, the third re-scores.
        manager.maybe_select(state, outcome)
        manager.maybe_select(state, outcome)
        self.assertEqual(len(calls), 2 * len(SKILL_GOAL_TYPES))
        manager.maybe_select(state, outcome)
        self.assertEqual(len(calls), 3 * len(SKILL_GOAL_TYPES))

    def test_next_skill_prior_breaks_ties_after_span_end(self):
        def outcome(state, encoding):
            return torch.tensor([0.0])

        manager = TacticManager(reselect_interval=100)
        state = torch.zeros(1, 8)
        prior = torch.full((len(SKILL_GOAL_TYPES),), -5.0)
        prior[SKILL_GOAL_TYPES.index("wait_pass")] = 5.0
        manager.notify_span_end()
        manager.maybe_select(state, outcome, next_skill_logits=prior)
        self.assertEqual(manager.active_goal_type, "wait_pass")

    def test_transition_examples_pair_consecutive_goals(self):
        achieved = [
            {"goal_type": "clear_gap", "magnitude": 40.0, "start_frame": 2, "span_id": "a"},
            {"goal_type": "enemy_clear", "magnitude": 30.0, "start_frame": 9, "span_id": "b"},
            {"goal_type": "mount_platform", "magnitude": 20.0, "start_frame": 15, "span_id": "c"},
        ]
        examples = tactic_transition_examples(achieved)
        self.assertEqual(
            examples,
            [
                (2, SKILL_GOAL_TYPES.index("enemy_clear")),
                (9, SKILL_GOAL_TYPES.index("mount_platform")),
            ],
        )
        self.assertEqual(tactic_transition_examples(achieved[:1]), [])

    def test_goal_encoding_dim_matches_head_contract(self):
        encoding = skill_goal_encoding("clear_gap", 32.0)
        self.assertEqual(encoding.shape[-1], len(SKILL_GOAL_TYPES) + 1)


if __name__ == "__main__":
    unittest.main()
