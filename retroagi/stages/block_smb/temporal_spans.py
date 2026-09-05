"""Build HSP0 temporal spans from Block SMB rollout step records.

Every frame of a rollout is assigned to exactly one motor-primitive-level
span: jump-primitive spans (liftoff through landing / hazard contact) and
locomotion spans (consecutive frames of running, waiting, or descending
between primitives). One parent skill-level span per episode carries the
scenario family as the requested goal. Termination reasons distinguish
success, failure, interruption, timeout, environment termination, and
evaluator truncation so a timeout can never be studied as a success.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from retroagi.core.temporal import HierarchicalTransition, TemporalGoal

_JUMP_ACTIONS = {2, 4, 5}
_LOCOMOTION_NAMES = {0: "wait", 1: "run_right", 3: "run_left"}
_PROGRESS_STALL_FRAMES = 8


def _locomotion_name(action: int) -> str:
    if action in _JUMP_ACTIONS:
        return "airborne_carry"
    return _LOCOMOTION_NAMES.get(int(action), f"action_{int(action)}")


def _episode_end_reason(record: Mapping[str, Any]) -> str:
    if record.get("goal"):
        return "success"
    if record.get("death") or record.get("attempt_failed"):
        return "failure"
    if record.get("terminated"):
        return "environment_termination"
    if record.get("truncated"):
        return "timeout"
    return "evaluator_truncation"


def build_block_smb_temporal_spans(
    records: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
    scenario_id: str,
    stage: str = "block_smb",
    seed: int | None = None,
    source: str = "real",
    policy_version: str = "",
    family_goal: str = "",
) -> list[HierarchicalTransition]:
    if not records:
        return []

    def common(kind_index: int, start: int, end: int, **kwargs: Any) -> HierarchicalTransition:
        return HierarchicalTransition(
            episode_id=episode_id,
            scenario_id=scenario_id,
            stage=stage,
            seed=seed,
            source=source,
            policy_version=policy_version,
            transition_id=f"{episode_id}:p{kind_index}",
            start_frame=start,
            end_frame=end,
            parent_id=f"{episode_id}:skill0",
            **kwargs,
        )

    spans: list[HierarchicalTransition] = []
    index = 0
    span_counter = 0
    n = len(records)
    while index < n:
        record = records[index]
        if record.get("started"):
            is_jump = int(record.get("action", -1)) in _JUMP_ACTIONS
            # Jump primitive: runs while the executor reports the primitive
            # active; ends on landing, hazard contact, episode end, or the
            # rollout budget.
            start = index
            events: list[dict[str, Any]] = [{"event": "liftoff", "frame": start}] if is_jump else []
            end = index
            reason = "evaluator_truncation"
            while end < n:
                r = records[end]
                if r.get("released") and all(e["event"] != "release" for e in events):
                    events.append({"event": "release", "frame": end})
                    if not is_jump:
                        # Steady primitives (walk/wait) complete at their
                        # duration boundary — the release marker frame.
                        reason = "success"
                        break
                if r.get("attempt_failed"):
                    events.append({"event": "landing", "frame": end, "outcome": "miss"})
                    reason = "failure"
                    break
                if r.get("cancelled"):
                    events.append({"event": "hazard_contact", "frame": end})
                    reason = "interruption"
                    break
                if r.get("landed"):
                    events.append({"event": "landing", "frame": end})
                    reason = "success"
                    break
                if r.get("death"):
                    events.append({"event": "death", "frame": end})
                    reason = "failure"
                    break
                if r.get("goal"):
                    events.append({"event": "success", "frame": end})
                    reason = "success"
                    break
                if r.get("terminated"):
                    reason = "environment_termination"
                    break
                if r.get("truncated"):
                    events.append({"event": "timeout", "frame": end})
                    reason = "timeout"
                    break
                if (
                    end + 1 < n
                    and not records[end + 1].get("active")
                    and not records[end + 1].get("started")
                    and not records[end + 1].get("landed")
                    and not records[end + 1].get("cancelled")
                ):
                    # Executor no longer active next frame without a recorded
                    # physical ending (e.g. safety valve): close here. The
                    # landing frame itself reports active=False with
                    # landed=True, so it must NOT trigger this early close —
                    # that would end every landed jump one frame short as an
                    # interruption and zero out the landing metrics.
                    reason = "interruption"
                    break
                if end == n - 1:
                    events.append({"event": "truncation", "frame": end})
                    reason = "evaluator_truncation"
                    break
                end += 1
            displacement = float(records[end].get("x_after", 0.0)) - float(
                records[start].get("x_before", 0.0)
            )
            span_action = int(record.get("action", -1))
            held = sum(
                1
                for i in range(start, end + 1)
                if (
                    int(records[i].get("action", -1)) in _JUMP_ACTIONS
                    if is_jump
                    else int(records[i].get("action", -1)) == span_action
                )
            )
            spans.append(
                common(
                    span_counter,
                    start,
                    end,
                    level="motor_primitive",
                    termination_reason=reason,
                    success=reason == "success",
                    failure_category=(
                        str(records[end].get("stomp_outcome") or "death")
                        if reason == "failure"
                        else (
                            str(records[end].get("stomp_outcome") or "")
                            if reason != "success"
                            else ""
                        )
                    ),
                    interruption_source="hazard_contact" if reason == "interruption" else "",
                    command={
                        "primitive": (
                            "jump" if is_jump else _locomotion_name(int(record.get("action", -1)))
                        ),
                        "action": int(record.get("action", -1)),
                        "held_frames": held,
                    },
                    state_before={
                        "x": records[start].get("x_before"),
                        "y": records[start].get("y_before"),
                    },
                    state_after={
                        "x": records[end].get("x_after"),
                        "y": records[end].get("y_after"),
                    },
                    outcome={"displacement": displacement},
                    events=tuple(events),
                )
            )
            span_counter += 1
            index = end + 1
            continue

        # Locomotion span: consecutive frames of one action class with no
        # active primitive.
        start = index
        name = _locomotion_name(int(record.get("action", -1)))
        end = index
        stall_frames = 0
        events = []
        while end < n:
            r = records[end]
            if float(r.get("x_after", 0.0)) == float(r.get("x_before", 0.0)) and name in (
                "run_right",
                "run_left",
            ):
                stall_frames += 1
                if stall_frames == _PROGRESS_STALL_FRAMES:
                    events.append({"event": "progress_stall", "frame": end})
            else:
                stall_frames = 0
            if r.get("death"):
                events.append({"event": "death", "frame": end})
                break
            if r.get("goal"):
                events.append({"event": "success", "frame": end})
                break
            if r.get("terminated") or r.get("truncated"):
                break
            nxt = end + 1
            if nxt >= n:
                break
            nrec = records[nxt]
            if nrec.get("started") or _locomotion_name(int(nrec.get("action", -1))) != name:
                break
            end = nxt
        last = records[end]
        if last.get("goal"):
            reason = "success"
        elif last.get("death") or last.get("attempt_failed"):
            reason = "failure"
        elif last.get("terminated"):
            reason = "environment_termination"
        elif last.get("truncated"):
            reason = "timeout"
            events.append({"event": "timeout", "frame": end})
        elif end == n - 1:
            reason = "evaluator_truncation"
            events.append({"event": "truncation", "frame": end})
        else:
            # Ended because the next frame starts a different behavior: the
            # span delivered control to the next decision.
            reason = "success"
        spans.append(
            common(
                span_counter,
                start,
                end,
                level="motor_primitive",
                termination_reason=reason,
                success=reason == "success",
                failure_category=(
                    str(records[end].get("stomp_outcome") or "death")
                    if reason == "failure"
                    else str(records[end].get("stomp_outcome") or "") if reason != "success" else ""
                ),
                command={"primitive": name},
                state_before={
                    "x": records[start].get("x_before"),
                    "y": records[start].get("y_before"),
                },
                state_after={
                    "x": records[end].get("x_after"),
                    "y": records[end].get("y_after"),
                },
                outcome={
                    "displacement": float(last.get("x_after", 0.0))
                    - float(records[start].get("x_before", 0.0))
                },
                events=tuple(events),
            )
        )
        span_counter += 1
        index = end + 1

    parent = HierarchicalTransition(
        episode_id=episode_id,
        scenario_id=scenario_id,
        stage=stage,
        seed=seed,
        source=source,
        policy_version=policy_version,
        transition_id=f"{episode_id}:skill0",
        level="skill",
        start_frame=0,
        end_frame=n - 1,
        termination_reason=_episode_end_reason(records[-1]),
        failure_category=(
            str(records[-1].get("stomp_outcome") or "") if not records[-1].get("goal") else ""
        ),
        success=bool(records[-1].get("goal")),
        goal=TemporalGoal(
            level="skill",
            goal_type=family_goal or scenario_id or "unspecified",
        ).to_json_dict(),
        child_ids=tuple(s.transition_id for s in spans),
        events=(),
    )
    return [parent, *spans]
