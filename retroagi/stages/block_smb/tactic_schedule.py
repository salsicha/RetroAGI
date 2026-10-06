"""Each layout's tactics, in order: which tactic is right where. Training only.

Every Block SMB layout carries ``scenario["tactics"]``, a list of segments
played in order. The simulator follows it every frame (``track``):

- the goal counts only once Mario is in the last segment;
- standing on a platform the current segment forbids ends the episode as a
  loss (Mario took the wrong path);
- leaving a hold segment's area before the segment ends is also a loss.

A segment is a dictionary:

- ``stance``: advance, alternate_route, hold_area or retreat (SCHEDULE_STANCES:
  the teacher's own plan, not the tactic tokens; teacher_tokens.teacher_tactic
  turns where Mario is going under it into one of tokens.TACTICS);
- ``direction``: 1 (right) or -1 (left). For advance, alternate_route and
  retreat the way Mario travels; for hold_area the way he faces;
- ``kind``: "plain" (the stance holds throughout), or "bridge", "plant" or
  "monster", whose teachers change the stance inside the segment
  (teacher_tokens.teacher_tactic);
- ``route``: platform indices to stand on, in order (optional);
- ``forbidden``: platform indices that must not be stood on (optional);
- ``avoid``: platform indices the teacher's route keeps off, without
  standing on them being a loss (optional);
- ``area``: for hold_area, how far (pixels) Mario may move from where the
  segment began, relative to the platform supporting him then;
- ``keep_behind``: for a monster segment, a line Mario should stay behind
  until he jumps over it (monster.py);
- ``stomp``: an enemy the teacher goes to stomp while the segment lasts
  (smb_coaching.training_target), optional;
- ``keep_alive``: enemies that must remain alive; killing one fails the segment,
  used to distinguish bypassing an enemy from stomping it;
- ``end``: when the segment is over, one of
  ``{"on": i}`` standing on platform i,
  ``{"reach_x": x}`` Mario's left edge at or past x in the segment's direction,
  ``{"frames": n}`` n frames after it began,
  ``{"bridge_crossed": True}`` the moving platform has been crossed,
  ``{"past_enemy": i}`` standing beyond enemy i, the way the segment goes
  (or the enemy is defeated),
  ``{"goal": True}`` never (the last segment).

None of this ever reaches a policy: it supplies teacher tokens, route labels
and whether an episode was won.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

SCHEDULE_STANCES = ("advance", "alternate_route", "hold_area", "retreat")
SEGMENT_KINDS = ("plain", "bridge", "plant", "monster")
END_KEYS = ("on", "reach_x", "frames", "bridge_crossed", "past_enemy", "goal")


def segment(
    stance: str,
    direction: int = 1,
    *,
    kind: str = "plain",
    route: Iterable[int] = (),
    forbidden: Iterable[int] = (),
    avoid: Iterable[int] = (),
    area: Optional[float] = None,
    keep_behind: Optional[float] = None,
    stomp: Optional[int] = None,
    keep_alive: Iterable[int] = (),
    **end: Any,
) -> dict:
    """One segment (see the module notes); ``end`` is one keyword, e.g. on=3."""
    built = {
        "stance": stance,
        "direction": int(direction),
        "kind": kind,
        "route": [int(i) for i in route],
        "forbidden": [int(i) for i in forbidden],
        "avoid": [int(i) for i in avoid],
        "keep_alive": [int(i) for i in keep_alive],
        "end": dict(end) or {"goal": True},
    }
    if area is not None:
        built["area"] = float(area)
    if keep_behind is not None:
        built["keep_behind"] = float(keep_behind)
    if stomp is not None:
        built["stomp"] = int(stomp)
    check_segment(built)
    return built


def check_segment(seg: dict) -> None:
    if seg.get("stance") not in SCHEDULE_STANCES:
        raise ValueError(f"unknown stance {seg.get('stance')!r}")
    if seg.get("direction") not in (-1, 1):
        raise ValueError("a segment's direction is -1 or 1")
    if seg.get("kind", "plain") not in SEGMENT_KINDS:
        raise ValueError(f"unknown segment kind {seg.get('kind')!r}")
    end = seg.get("end") or {}
    if len(end) != 1 or next(iter(end)) not in END_KEYS:
        raise ValueError(f"a segment ends one way, one of {END_KEYS}: {end!r}")


def check_schedule(schedule: list, platform_count: int, enemy_count: int) -> None:
    """Raise unless ``schedule`` is a valid list of segments for this layout."""
    if not schedule:
        raise ValueError("a layout needs at least one tactic segment")
    for i, seg in enumerate(schedule):
        check_segment(seg)
        last = i == len(schedule) - 1
        if ("goal" in seg["end"]) != last:
            raise ValueError("only the last segment, and always it, ends at the goal")
        for index in (*seg.get("route", ()), *seg.get("forbidden", ()), *seg.get("avoid", ())):
            if not 0 <= index < platform_count:
                raise ValueError(f"platform {index} is not in the layout")
        for key in ("on",):
            if key in seg["end"] and not 0 <= seg["end"][key] < platform_count:
                raise ValueError(f"platform {seg['end'][key]} is not in the layout")
        for enemy in (seg["end"].get("past_enemy"), seg.get("stomp")):
            if enemy is not None and not 0 <= enemy < enemy_count:
                raise ValueError(f"enemy {enemy} is not in the layout")
        if any(not 0 <= i < enemy_count for i in seg.get("keep_alive", ())):
            raise ValueError("keep_alive names an enemy outside the layout")


def advance_only(direction: int = 1) -> list:
    """The schedule of a layout that is advance all the way."""
    return [segment("advance", direction)]


# ── The simulator's side (env calls these) ────────────────────────────────────


def start(env, schedule: Optional[list]) -> None:
    """Begin following ``schedule`` (at reset)."""
    env._tactics = [dict(seg) for seg in (schedule or advance_only())]
    check_schedule(env._tactics, len(env.platforms), len(env.enemies))
    env._tactic_index = 0
    env._route_done = 0
    env._off_route = False
    _begin(env)


def _begin(env) -> None:
    env._tactic_started = env.steps
    env._tactic_anchor = env.mario["x"]
    env._tactic_anchor_platform = _support_index(env)
    env._tactic_anchor_platform_x = (
        env.platforms[env._tactic_anchor_platform]["rect"].x
        if env._tactic_anchor_platform is not None
        else 0
    )
    env._route_done = 0


_ADVANCE = None


def current(env) -> dict:
    """The current segment (advance all the way for a simulator without tactics)."""
    global _ADVANCE
    tactics = getattr(env, "_tactics", None)
    if tactics is None:
        _ADVANCE = _ADVANCE or segment("advance", 1)
        return _ADVANCE
    return tactics[env._tactic_index]


def _support_index(env) -> Optional[int]:
    support = env.mario.get("_platform") if env.mario["on_ground"] else None
    return next((i for i, p in enumerate(env.platforms) if p is support), None)


def _over(env, seg: dict) -> bool:
    end = seg["end"]
    m = env.mario
    if "goal" in end:
        return False
    if "on" in end:
        return _support_index(env) == end["on"]
    if "reach_x" in end:
        return (m["x"] - end["reach_x"]) * seg["direction"] >= 0
    if "frames" in end:
        return env.steps - env._tactic_started >= end["frames"]
    if "bridge_crossed" in end:
        return bool(env._bridge_crossed)
    if "past_enemy" in end:
        enemy = env.enemies[end["past_enemy"]]
        if enemy["dead"]:
            return True
        beyond = (
            m["x"] >= enemy["x"] + enemy["w"]
            if seg["direction"] > 0
            else m["x"] + m["w"] <= enemy["x"]
        )
        return bool(m["on_ground"] and beyond)
    raise ValueError(f"unknown segment end {end!r}")


def track(env) -> bool:
    """Follow the schedule after a frame. Returns True when Mario broke it
    (stood on a forbidden platform, or left a hold area early)."""
    seg = current(env)
    on = _support_index(env)
    route = seg.get("route", ())
    if on is not None and on in route:
        # Standing on a route platform, the rest of the route starts after it
        # (also after slipping back onto an earlier one).
        env._route_done = route.index(on) + 1
    if on is not None and on in seg.get("forbidden", ()):
        env._off_route = True
    if any(env.enemies[i]["dead"] for i in seg.get("keep_alive", ())):
        env._off_route = True
    area = seg.get("area")
    if area is not None:
        anchor = env._tactic_anchor
        platform = env._tactic_anchor_platform
        if platform is not None:
            anchor += env.platforms[platform]["rect"].x - env._tactic_anchor_platform_x
        if abs(env.mario["x"] - anchor) > area or (
            seg["stance"] == "hold_area" and platform is not None and on != platform
        ):
            env._off_route = True
    while env._tactic_index < len(env._tactics) - 1 and _over(env, current(env)):
        env._tactic_index += 1
        _begin(env)
        # The new segment's route may already be under Mario's feet.
        seg = current(env)
        if on is not None and on in seg.get("route", ())[:1]:
            env._route_done = 1
    # The camera is a physical left boundary. Once an overshoot has scrolled
    # a required retreat line offscreen, repeatedly requesting that retreat
    # can never complete it. End the failed route instead of collecting
    # hundreds of unlabelled, unrecoverable recovery decisions.
    seg = current(env)
    if (
        seg["direction"] < 0
        and "reach_x" in seg["end"]
        and seg["end"]["reach_x"] < env.camera_x
    ):
        env._off_route = True
    return env._off_route


def in_kind(env, kind: str) -> bool:
    """Whether the rule of a segment kind ("bridge", "plant", "monster") applies now:
    in a segment of that kind, or anywhere in a layout that names no such segment
    (a layout made without tactics, e.g. while its own route is being found)."""
    if current(env)["kind"] == kind:
        return True
    tactics = getattr(env, "_tactics", None) or ()
    return not any(seg["kind"] == kind for seg in tactics)


def goal_allowed(env) -> bool:
    """The goal counts only in the last segment."""
    return env._tactic_index == len(env._tactics) - 1


def next_route_platform(env) -> Optional[int]:
    """The current segment's next platform to stand on, if it has a route left."""
    route = current(env).get("route", ())
    done = getattr(env, "_route_done", 0)
    return route[done] if done < len(route) else None


STATE_FIELDS = (
    "_tactic_index",
    "_tactic_started",
    "_tactic_anchor",
    "_tactic_anchor_platform",
    "_tactic_anchor_platform_x",
    "_route_done",
    "_off_route",
)
