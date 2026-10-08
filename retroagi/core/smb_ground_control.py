"""Receding-horizon control of a fixed ground destination from visual feedback."""

from copy import copy
from dataclasses import dataclass

from .smb_trajectory import overlap, predict

MPC_FRAMES = 8
BOARD_MARGIN = 4


def stopping_distance(motion):
    """Terminal braking cost, not a rollout of distant terrain/platform positions."""
    motion = copy(motion)
    distance = 0
    for _ in range(64):
        if motion.x_speed == 0:
            break
        dx, _, _ = motion.advance(direction=0, jump=False, grounded=True, y=0, run=False)
        distance += dx
    return distance


def stable_on(box, left, right, margin=BOARD_MARGIN):
    return box[0] >= left + margin and box[2] <= right - margin


@dataclass
class GroundMove:
    feedback: object
    distance: float
    start: float
    feet: float
    height: float = 0
    elapsed: int = 0
    done: bool = False
    status: str = "running"
    started_moving: bool | None = None
    target: object = None
    target_offset: float = 0
    boarded: bool = False
    stalled: int = 0

    def press(self, scene):
        self.elapsed += 1
        if scene is None or scene.mario.box is None:
            self.done, self.status = True, "lost_observation"
            return 0
        from .smb_spatial_feedback import moving_support

        support_box = moving_support(scene)
        moving = support_box is not None
        platforms = [t for t in self.feedback.tracks.tracks if t.kind == "platform"]
        center = (scene.mario.box[0] + scene.mario.box[2]) / 2
        remaining = self.distance - (self.feedback.displacement - self.start)
        if self.started_moving is None:
            self.started_moving = moving
            goal = center + remaining
            candidates = [
                t
                for t in platforms
                if t.box[0] <= goal <= t.box[2] and abs(t.box[1] - self.feet - self.height) <= 4
            ]
            static_goal = any(
                not s.moving and s.x0 <= goal <= s.x1 and abs(s.top - self.feet - self.height) <= 4
                for s in scene.surfaces
            )
            if len(candidates) == 1 and not static_goal:
                self.target = candidates[0]
                self.target_offset = goal - self.target.box[0]
        # An approach may initially point past the visible platform. Bind it
        # on first contact, but do not report completion on an edge overlap.
        if moving and not self.started_moving and self.target is None:
            matches = [t for t in platforms if t.box == support_box]
            if len(matches) == 1:
                self.target = matches[0]
                half = (scene.mario.box[2] - scene.mario.box[0]) / 2
                inset = half + BOARD_MARGIN
                width = self.target.box[2] - self.target.box[0]
                self.target_offset = inset if remaining >= 0 else width - inset
        if moving:
            self.boarded = True
        if self.target is not None:
            if not any(t is self.target for t in platforms):
                self.done, self.status = True, "lost_support_target"
                return 0
            remaining = self.target.box[0] + self.target_offset - center
        speed = self.feedback.motion.x_speed / 16
        grounded = scene.mario.on_something and scene.mario.support != "air"
        target_supported = self.target is None or (
            support_box == self.target.box
            and stable_on(scene.mario.box, self.target.box[0], self.target.box[2])
        )
        if self.started_moving and self.target is None:
            # A shore destination finishes on that shore, fully inside its edge.
            target_supported = not moving and any(
                not surface.moving
                and surface.x0 <= center + remaining <= surface.x1
                and abs(surface.top - scene.mario.box[3]) <= 4
                and stable_on(scene.mario.box, surface.x0, surface.x1)
                for surface in scene.surfaces
            )
        if (
            target_supported
            and grounded
            and len(self.feedback.velocities) >= 2
            and abs(scene.mario.box[3] - self.feet - self.height) <= 4
            and abs(remaining) <= 0.5
            and abs(speed) <= 0.25
        ):
            self.done, self.status = True, "arrived"
            return 0
        if self.elapsed >= 192:
            self.done, self.status = True, "travel_timeout"
            return 0
        if not grounded:
            box = scene.mario.box
            goal = ((box[0] + box[2]) / 2 + remaining, self.feet + self.height)
            options = []
            for direction in (-1, 0, 1):
                landing = predict(
                    scene,
                    self.feedback.tracks.tracks,
                    self.feedback.motion,
                    box,
                    goal,
                    direction,
                    0,
                )
                if landing.safe:
                    options.append((landing.error, abs(direction), direction))
            direction = min(options)[-1] if options else (remaining > 0) - (remaining < 0)
            self.status = "descending" if options else "no_safe_continuation"
            return {-1: 3, 0: 0, 1: 1}[direction]
        # Each candidate accelerates briefly, then brakes. Replan after one
        # actual frame, preserving the original destination throughout.
        candidates = []
        stopped = stopping_distance(self.feedback.motion)
        shore_goal = None
        if moving and self.target is None:
            goal = center + remaining
            shore_goal = next(
                (
                    s
                    for s in scene.surfaces
                    if not s.moving and s.x0 <= goal <= s.x1 and abs(s.top - self.feet) <= 4
                ),
                None,
            )
        distant_closed = False
        if shore_goal is not None:
            support_track = next((t for t in platforms if t.box == support_box), None)
            forecast = (
                support_track.distant_forecast(max(MPC_FRAMES + 1, min(64, abs(remaining) / 2.5)))
                if support_track
                else None
            )
            if forecast is not None:
                _, future_box, uncertainty = forecast
                gap_now = max(shore_goal.x0 - support_box[2], support_box[0] - shore_goal.x1, 0)
                gap_later = max(shore_goal.x0 - future_box[2], future_box[0] - shore_goal.x1, 0)
                # A confident distant forecast can close a transfer request;
                # it never certifies immediate support or replaces observations.
                distant_closed = gap_later > gap_now + uncertainty * 2 and gap_now > 0
        for direction in (-1, 0, 1):
            for duration in range(1, MPC_FRAMES + 1):
                motion = copy(self.feedback.motion)
                x, y, right, feet = scene.mario.box
                width = right - x
                moved = 0
                safe = True
                for frame in range(MPC_FRAMES):
                    # Release to brake without reversing into an oscillation.
                    control = direction if frame < duration else 0
                    dx, _, _ = motion.advance(
                        direction=control, jump=False, grounded=True, y=y, run=control > 0
                    )
                    x += dx
                    moved += dx
                    supports = [(s.x0, s.x1, s.top, None) for s in scene.surfaces if not s.moving]
                    supports += [
                        (
                            t.box[0] + (t.velocity or (0, 0))[0] * (frame + 1),
                            t.box[2] + (t.velocity or (0, 0))[0] * (frame + 1),
                            t.box[1] + (t.velocity or (0, 0))[1] * (frame + 1),
                            t,
                        )
                        for t in platforms
                    ]
                    contact = [
                        s
                        for s in supports
                        if s[0] < x + width
                        and s[1] > x
                        and (abs(s[2] - feet) <= 4 or 0 < s[2] - feet <= self.height + 4)
                    ]
                    if not contact:
                        safe = False
                        break
                    # Moving supports carry position, not player momentum.
                    support = min(
                        contact, key=lambda s: (abs(s[2] - feet), -s[0] if remaining >= 0 else s[1])
                    )
                    if support[3] is not None:
                        carry = (support[3].velocity or (0, 0))[0]
                        x += carry
                        moved += carry
                    body = (x, y, x + width, feet)
                    # Carry can remove the overlap used to select a support.
                    if not (support[0] < x + width and support[1] > x):
                        safe = False
                        break
                    if any(overlap(body, b.box) for b in scene.blocks) or any(
                        overlap(body, p) for p in scene.pipes
                    ):
                        safe = False
                        break
                    for hazard in self.feedback.tracks.tracks:
                        if hazard.kind == "platform":
                            continue
                        if overlap(body, hazard.forecast(frame + 1)):
                            safe = False
                            break
                    if not safe:
                        break
                if safe:
                    brake = stopping_distance(motion)
                    stop_left, stop_right = x + brake, x + width + brake
                    # The short horizon must end in a state that can stop on
                    # support. A coast off an edge is not a viable terminal state.
                    terminal_safe = any(
                        left < stop_right and stop_left < right
                        for left, right, top, _ in supports
                        if abs(top - feet) <= 4
                    )
                    if not terminal_safe:
                        continue
                    if self.target is not None and (moving or self.boarded):
                        target_box = self.target.forecast(MPC_FRAMES)
                        if support[3] is not self.target:
                            continue
                        # Keep enough interior room to brake; do not take a
                        # shortcut back to the shore to minimize world distance.
                        was_stable = stable_on(
                            scene.mario.box, self.target.box[0], self.target.box[2], margin=0
                        )
                        if was_stable:
                            if not (target_box[0] <= stop_left and stop_right <= target_box[2]):
                                continue
                        else:
                            old_overlap = min(scene.mario.box[2], self.target.box[2]) - max(
                                scene.mario.box[0], self.target.box[0]
                            )
                            new_overlap = min(stop_right, target_box[2]) - max(
                                stop_left, target_box[0]
                            )
                            if new_overlap <= old_overlap:
                                continue
                    goal_motion = (
                        (self.target.velocity or (0, 0))[0] * MPC_FRAMES if self.target else 0
                    )
                    terminal_error = remaining + goal_motion - moved
                    cost = abs(terminal_error - brake) + abs(motion.x_speed / 16) * 0.3
                    progress = direction * remaining > 0 and abs(speed) < 0.25
                    candidates.append((cost, not progress, abs(direction), duration, direction))
        if not candidates:
            # Return control to skill instead of retaining an impossible target.
            self.done, self.status = True, "no_safe_path"
            direction = -(speed > 0) + (speed < 0)
        else:
            direction = min(candidates)[-1]
            self.status = "braking" if direction * speed < 0 else "running"
            self.stalled = self.stalled + 1 if direction == 0 and abs(speed) <= 0.25 else 0
            near_edge = moving and not stable_on(
                (
                    scene.mario.box[0] + stopped,
                    scene.mario.box[1],
                    scene.mario.box[2] + stopped,
                    scene.mario.box[3],
                ),
                support_box[0],
                support_box[2],
                margin=6,
            )
            if shore_goal is not None and ((distant_closed and near_edge) or self.stalled >= 8):
                self.done, self.status = True, "transfer_recheck"
                direction = -(speed > 0) + (speed < 0)
        return {-1: 3, 0: 0, 1: 1}[direction]
