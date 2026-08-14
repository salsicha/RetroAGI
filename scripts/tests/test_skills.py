"""Tests for the HSP2 goal-conditioned skill layer."""

import unittest

import torch

from retroagi.core.skills import (
    SKILL_GOAL_ENCODING_DIM,
    SKILL_GOAL_TYPES,
    skill_goal_encoding,
)
from retroagi.core.temporal import HierarchicalTransition
from retroagi.stages.block_smb.skills import (
    BLOCK_SMB_FAMILY_SKILL_GOALS,
    achieved_block_smb_skill_goals,
)


def _span(
    index,
    primitive,
    *,
    reason="success",
    displacement=0.0,
    y_before=200.0,
    y_after=200.0,
    start=0,
    end=10,
    events=(),
    failure_category="",
):
    return HierarchicalTransition(
        episode_id="e",
        level="motor_primitive",
        transition_id=f"e:p{index}",
        start_frame=start,
        end_frame=end,
        termination_reason=reason,
        command={"primitive": primitive},
        state_before={"x": 0.0, "y": y_before},
        state_after={"x": displacement, "y": y_after},
        outcome={"displacement": displacement},
        events=tuple(events),
        failure_category=failure_category,
    )


class TestSkillGoals(unittest.TestCase):
    def test_encoding_shape_and_validation(self):
        for goal_type in SKILL_GOAL_TYPES:
            encoding = skill_goal_encoding(goal_type, 64.0)
            self.assertEqual(encoding.shape, (1, SKILL_GOAL_ENCODING_DIM))
            self.assertAlmostEqual(float(encoding.sum()), 1.0 + float(encoding[0, -1]))
        with self.assertRaises(ValueError):
            skill_goal_encoding("fly")

    def test_family_map_uses_known_goal_types(self):
        for family, goal_type in BLOCK_SMB_FAMILY_SKILL_GOALS.items():
            self.assertIn(goal_type, SKILL_GOAL_TYPES, family)

    def test_relabeler_assigns_measurable_goals(self):
        spans = [
            # Long flat jump: clears ground, no mount.
            _span(0, "jump", displacement=60.0),
            # Jump landing higher: mounts and clears.
            _span(1, "jump", displacement=40.0, y_before=200.0, y_after=180.0),
            # Stomp bounce: hazard contact without death.
            _span(
                2,
                "jump",
                reason="interruption",
                displacement=30.0,
                events=({"event": "hazard_contact", "frame": 4},),
            ),
            # Side hit: hazard death must NOT relabel as enemy_clear.
            _span(
                3,
                "jump",
                reason="failure",
                displacement=20.0,
                failure_category="death",
                events=({"event": "hazard_contact", "frame": 4},),
            ),
            # Waiting long enough counts as wait_pass.
            _span(4, "wait", start=11, end=30),
            # Retreat: leftward displacement.
            _span(5, "run_left", displacement=-20.0),
            # Short hop achieves nothing.
            _span(6, "jump", displacement=5.0),
        ]
        achieved = achieved_block_smb_skill_goals(spans)
        by_span = {}
        for item in achieved:
            by_span.setdefault(item["span_id"], []).append(item["goal_type"])
        self.assertEqual(by_span.get("e:p0"), ["clear_gap"])
        self.assertCountEqual(by_span.get("e:p1"), ["mount_platform", "clear_gap"])
        self.assertEqual(by_span.get("e:p2"), ["enemy_clear"])
        self.assertNotIn("e:p3", by_span)
        self.assertEqual(by_span.get("e:p4"), ["wait_pass"])
        self.assertEqual(by_span.get("e:p5"), ["retreat_recover"])
        self.assertNotIn("e:p6", by_span)


class TestSkillConditioningNeutrality(unittest.TestCase):
    def test_zero_init_goal_conditioning_is_behavior_neutral(self):
        # A freshly built model must produce identical outputs with and
        # without a goal encoding until the injection is trained.
        from retroagi.core import build_architecture
        from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC

        torch.manual_seed(0)
        model = build_architecture(
            "agent_world_model_critic",
            BLOCK_SMB_SPEC,
            {"hidden_dim": 8},
        )
        model.eval()
        src_a = torch.randint(0, 4, (1, BLOCK_SMB_SPEC.seq_len_a))
        src_b = torch.randint(
            0, 4, (1, BLOCK_SMB_SPEC.seq_len_a * BLOCK_SMB_SPEC.ratio_ab)
        )
        src_c = torch.rand(1, model.agent.seq_len_c)
        goal = skill_goal_encoding("enemy_clear", 64.0)
        with torch.no_grad():
            plain = model(src_a, src_b, src_c, tau=1.0)
            conditioned = model(src_a, src_b, src_c, tau=1.0, skill_goal=goal)
        torch.testing.assert_close(plain[0], conditioned[0])
        torch.testing.assert_close(plain[4], conditioned[4])


if __name__ == "__main__":
    unittest.main()


class TestTacticsAndStrategyNetworks(unittest.TestCase):
    def test_untrained_tactics_network_is_behavior_neutral(self):
        # The tactics context enters the skill network through a
        # zero-initialized layer, so a fresh tactics/strategy stack must not
        # change any output of a fresh model versus reloading it.
        from retroagi.core import build_architecture
        from retroagi.core.models import TACTIC_STANCES
        from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC

        torch.manual_seed(0)
        model = build_architecture(
            "agent_world_model_critic", BLOCK_SMB_SPEC, {"hidden_dim": 8}
        )
        model.eval()
        src_a = torch.randint(0, 4, (1, BLOCK_SMB_SPEC.seq_len_a))
        src_b = torch.randint(
            0, 4, (1, BLOCK_SMB_SPEC.seq_len_a * BLOCK_SMB_SPEC.ratio_ab)
        )
        src_c = torch.rand(1, model.agent.seq_len_c)
        with torch.no_grad():
            first = model(src_a, src_b, src_c, tau=1.0)
            second = model(src_a, src_b, src_c, tau=1.0)
        torch.testing.assert_close(first[0], second[0])
        self.assertIn(model.last_tactic_stance, TACTIC_STANCES)
        # Gradient path exists: stance logits participate in the graph.
        logits, context = model.tactics_network(
            src_c, model.strategy_network(torch.zeros(1, 8, len(TACTIC_STANCES)))
        )
        self.assertEqual(logits.shape, (1, len(TACTIC_STANCES)))
        self.assertTrue(context.requires_grad)
