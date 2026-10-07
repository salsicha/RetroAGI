"""Receding-horizon control of a fixed ground destination from visual feedback."""

from copy import copy
from dataclasses import dataclass

from .smb_trajectory import overlap, predict, translated


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

    def press(self, scene):
        self.elapsed += 1
        if scene is None or scene.mario.box is None:
            self.done, self.status = True, "lost_observation"
            return 0
        from .smb_spatial_feedback import moving_support

        moving = moving_support(scene) is not None
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
        if (
            self.started_moving
            and not moving
            and scene.mario.on_something
            and scene.mario.support != "air"
        ):
            self.done, self.status = True, "dismounted"
            return 0
        if moving and not self.started_moving:
            # Boarding completes the ground approach. Let skill choose a ride
            # or dismount; never chase the old floor waypoint off the bridge.
            self.done, self.status = True, "boarded"
            return 0
        if self.target is not None:
            if not any(t is self.target for t in platforms):
                self.done, self.status = True, "lost_support_target"
                return 0
            remaining = self.target.box[0] + self.target_offset - center
        speed = self.feedback.motion.x_speed / 16
        grounded = scene.mario.on_something and scene.mario.support != "air"
        if (
            grounded
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
        for direction in (-1, 0, 1):
            for duration in (*range(1, 17), 24, 32):
                motion = copy(self.feedback.motion)
                x, y, right, feet = scene.mario.box
                width = right - x
                moved = 0
                safe = True
                for frame in range(48):
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
                    if any(overlap(body, b.box) for b in scene.blocks) or any(
                        overlap(body, p) for p in scene.pipes
                    ):
                        safe = False
                        break
                    for hazard in self.feedback.tracks.tracks:
                        if hazard.kind == "platform":
                            continue
                        vx, vy = hazard.velocity or (0, 0)
                        if overlap(
                            body, translated(hazard.box, vx * (frame + 1), vy * (frame + 1))
                        ):
                            safe = False
                            break
                    if not safe:
                        break
                if safe:
                    goal_motion = (self.target.velocity or (0, 0))[0] * 48 if self.target else 0
                    cost = abs(remaining + goal_motion - moved) + abs(motion.x_speed / 16) * 2
                    # A stationary tie must not permanently starve a tiny move.
                    progress = direction * remaining > 0 and abs(speed) < 0.25
                    candidates.append((cost, not progress, abs(direction), duration, direction))
        if not candidates:
            # Return control to skill instead of retaining an impossible target.
            self.done, self.status = True, "no_safe_path"
            direction = -(speed > 0) + (speed < 0)
        else:
            direction = min(candidates)[-1]
            self.status = "braking" if direction * speed < 0 else "running"
        return {-1: 3, 0: 0, 1: 1}[direction]
