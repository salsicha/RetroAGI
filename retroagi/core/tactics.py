"""HSP3 tactic manager: select the next skill goal above the controller.

The manager owns one decision (doc: Tactic Manager): which measurable skill
goal the lower layers should pursue right now. It re-selects when the active
skill's span ends, or after a bounded number of control decisions — never
every frame. Selection scores each candidate goal with the model's skill
outcome head (P(goal achievable from here)) plus, right after a finished
span, a learned next-skill prior trained from real successful sequences.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from .skills import SKILL_GOAL_TYPES, skill_goal_encoding

DEFAULT_TACTIC_RESELECT_INTERVAL = 16


class TacticManager:
    """Event-driven skill-goal selection with a bounded candidate set."""

    def __init__(
        self,
        *,
        reselect_interval: int = DEFAULT_TACTIC_RESELECT_INTERVAL,
        initial_goal: torch.Tensor | None = None,
    ) -> None:
        if reselect_interval <= 0:
            raise ValueError("reselect_interval must be positive")
        self.reselect_interval = int(reselect_interval)
        self.active_goal = initial_goal
        self.active_goal_type: str | None = None
        self.decisions_since_selection = 0
        self.span_ended = initial_goal is None
        self.last_scores: dict[str, float] = {}

    def notify_span_end(self) -> None:
        """A skill-level span finished; the next decision re-selects."""

        self.span_ended = True

    def maybe_select(
        self,
        state: torch.Tensor,
        outcome_logit_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        next_skill_logits: torch.Tensor | None = None,
        candidates: Sequence[str] = SKILL_GOAL_TYPES,
    ) -> torch.Tensor | None:
        """Return the goal encoding to condition on for this decision."""

        self.decisions_since_selection += 1
        due = (
            self.active_goal is None
            or self.span_ended
            or self.decisions_since_selection >= self.reselect_interval
        )
        if not due:
            return self.active_goal
        prior = None
        if next_skill_logits is not None:
            prior = torch.softmax(next_skill_logits.reshape(-1).float(), dim=-1)
        best_type: str | None = None
        best_score = float("-inf")
        best_encoding: torch.Tensor | None = None
        self.last_scores = {}
        with torch.no_grad():
            for goal_type in candidates:
                encoding = skill_goal_encoding(goal_type).to(state.device)
                score = float(
                    torch.sigmoid(outcome_logit_fn(state, encoding)).reshape(-1)[0]
                )
                if prior is not None and self.span_ended:
                    score = score + float(prior[SKILL_GOAL_TYPES.index(goal_type)])
                self.last_scores[goal_type] = score
                if score > best_score:
                    best_score = score
                    best_type = goal_type
                    best_encoding = encoding
        self.active_goal = best_encoding
        self.active_goal_type = best_type
        self.decisions_since_selection = 0
        self.span_ended = False
        return self.active_goal


def tactic_transition_examples(
    achieved_goals: Sequence[dict[str, Any]],
) -> list[tuple[int, int]]:
    """(state_frame, next_goal_index) pairs from an episode's achieved goals.

    Each consecutive pair of achieved skills in real play supervises the
    next-skill prior: from the state where the earlier skill began paying
    off, the skill that actually followed is the target.
    """

    ordered = sorted(achieved_goals, key=lambda item: item["start_frame"])
    examples: list[tuple[int, int]] = []
    for earlier, later in zip(ordered, ordered[1:]):
        if later["goal_type"] not in SKILL_GOAL_TYPES:
            continue
        examples.append(
            (
                int(earlier["start_frame"]),
                SKILL_GOAL_TYPES.index(later["goal_type"]),
            )
        )
    return examples
