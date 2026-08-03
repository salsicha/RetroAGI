"""Versioned temporal contracts for hierarchical self-supervised planning.

HSP0 of docs/hierarchical-self-supervised-planning.md: one shared, versioned
format for goals handed down the hierarchy and for the spans of play that
result, so an episode can be reconstructed as frames, primitives, skills,
tactics, or a route from the same log without ambiguity about why any span
ended.

These contracts are game-neutral. Stage adapters translate native
observations, actions, and events into these records; nothing here imports a
game module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

TEMPORAL_SCHEMA_VERSION = "hsp0.1"

# Hierarchy levels, ordered fast to slow (doc: Proposed Temporal Levels).
TEMPORAL_LEVELS = (
    "perception",
    "control",
    "motor_primitive",
    "skill",
    "tactic",
    "route",
    "curriculum",
)

# Why a span ended. A timeout must never be stored as a natural success
# (doc: Episode And Span Boundaries).
TERMINATION_REASONS = (
    "success",
    "failure",
    "interruption",
    "timeout",
    "environment_termination",
    "evaluator_truncation",
)

# Named moments that punctuate play (doc: HSP0 event detectors).
TEMPORAL_EVENTS = (
    "liftoff",
    "release",
    "landing",
    "support_loss",
    "progress_stall",
    "hazard_contact",
    "death",
    "success",
    "timeout",
    "truncation",
)

# Where a record came from; imagined and relabeled data must stay marked.
PROVENANCE_SOURCES = ("real", "scripted", "human", "imagined", "relabeled")


@dataclass
class TemporalGoal:
    """A goal passed to a lower level."""

    level: str
    goal_type: str
    target: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    expected_duration: tuple[int, int] | None = None
    success_condition: str = ""
    failure_condition: str = ""
    interrupt_condition: str = ""
    parent_goal_id: str | None = None
    confidence: float | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        if self.level not in TEMPORAL_LEVELS:
            raise ValueError(f"unknown temporal level {self.level!r}")
        if not self.goal_type:
            raise ValueError("goal_type must be non-empty")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TEMPORAL_SCHEMA_VERSION
        return payload


@dataclass
class HierarchicalTransition:
    """One completed or interrupted span of play at one hierarchy level."""

    episode_id: str
    level: str
    transition_id: str
    start_frame: int
    end_frame: int
    termination_reason: str
    scenario_id: str = ""
    stage: str = ""
    seed: int | None = None
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()
    goal: Mapping[str, Any] | None = None
    command: Mapping[str, Any] = field(default_factory=dict)
    state_before: Mapping[str, Any] = field(default_factory=dict)
    state_after: Mapping[str, Any] = field(default_factory=dict)
    predicted: Mapping[str, Any] = field(default_factory=dict)
    outcome: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    success: bool = False
    failure_category: str = ""
    interruption_source: str = ""
    source: str = "real"
    policy_version: str = ""
    schema_version: str = TEMPORAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.level not in TEMPORAL_LEVELS:
            raise ValueError(f"unknown temporal level {self.level!r}")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError(
                f"unknown termination reason {self.termination_reason!r}"
            )
        if self.source not in PROVENANCE_SOURCES:
            raise ValueError(f"unknown provenance source {self.source!r}")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must not precede start_frame")
        for event in self.events:
            name = event.get("event") if isinstance(event, Mapping) else None
            if name not in TEMPORAL_EVENTS:
                raise ValueError(f"unknown temporal event {name!r}")

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["child_ids"] = list(self.child_ids)
        payload["events"] = [dict(event) for event in self.events]
        return payload


def transition_from_json_dict(payload: Mapping[str, Any]) -> HierarchicalTransition:
    fields = dict(payload)
    fields.pop("duration", None)
    fields["child_ids"] = tuple(fields.get("child_ids", ()) or ())
    fields["events"] = tuple(dict(e) for e in fields.get("events", ()) or ())
    return HierarchicalTransition(**fields)


@dataclass
class EpisodeReconstruction:
    """Validation result for one episode rebuilt from its spans."""

    episode_id: str
    frame_count: int
    span_count: int
    levels: tuple[str, ...]
    end_reason: str
    valid: bool
    problems: tuple[str, ...] = ()


def reconstruct_episodes(
    transitions: Sequence[HierarchicalTransition],
    *,
    coverage_level: str = "motor_primitive",
) -> list[EpisodeReconstruction]:
    """Rebuild episodes from spans and validate coverage and end reasons.

    The HSP0 exit gate: for each episode, the spans of ``coverage_level``
    must tile the episode's frames with no gaps, no overlaps, and no
    ambiguous end reasons.
    """

    by_episode: dict[str, list[HierarchicalTransition]] = {}
    for transition in transitions:
        by_episode.setdefault(transition.episode_id, []).append(transition)

    reports: list[EpisodeReconstruction] = []
    for episode_id, spans in by_episode.items():
        problems: list[str] = []
        cover = sorted(
            (s for s in spans if s.level == coverage_level),
            key=lambda s: s.start_frame,
        )
        if not cover:
            problems.append(f"no {coverage_level} spans")
            frame_count = 0
        else:
            expected = cover[0].start_frame
            if expected != 0:
                problems.append(f"first span starts at frame {expected}, not 0")
            for span in cover:
                if span.start_frame > expected:
                    problems.append(
                        f"gap: frames {expected}..{span.start_frame - 1} uncovered"
                    )
                elif span.start_frame < expected:
                    problems.append(
                        f"overlap at frame {span.start_frame} ({span.transition_id})"
                    )
                expected = max(expected, span.end_frame + 1)
            frame_count = expected
        seen_ids = {s.transition_id for s in spans}
        for span in spans:
            if span.parent_id is not None and span.parent_id not in seen_ids:
                problems.append(
                    f"span {span.transition_id} references missing parent"
                )
        end_reason = cover[-1].termination_reason if cover else ""
        reports.append(
            EpisodeReconstruction(
                episode_id=episode_id,
                frame_count=frame_count,
                span_count=len(spans),
                levels=tuple(sorted({s.level for s in spans})),
                end_reason=end_reason,
                valid=not problems,
                problems=tuple(problems),
            )
        )
    return reports


def write_transitions_jsonl(path: Any, transitions: Sequence[HierarchicalTransition]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(json.dumps(transition.to_json_dict(), sort_keys=True) + "\n")


def read_transitions_jsonl(path: Any) -> list[HierarchicalTransition]:
    transitions: list[HierarchicalTransition] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                transitions.append(transition_from_json_dict(json.loads(line)))
    return transitions
