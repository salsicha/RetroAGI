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

    def press(self, scene):
        self.elapsed += 1
        if scene is None or scene.mario.box is None:
            self.done, self.status = True, "lost_observation"
            return 0
        remaining = self.distance - (self.feedback.displacement - self.start)
        speed = self.feedback.motion.x_speed / 16
        grounded = scene.mario.on_something and scene.mario.support != "air"
        if (
            grounded
            and len(self.feedback.velocities) >= 2
            and abs(scene.mario.box[3] - self.feet - self.height) <= 4
            and abs(remaining) <= 1
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
            for duration in (1, 4, 8, 16):
                motion = copy(self.feedback.motion)
                x, y, right, feet = scene.mario.box
                width = right - x
                moved = 0
                safe = True
                for frame in range(32):
                    control = (
                        direction
                        if frame < duration
                        else -(motion.x_speed > 0) + (motion.x_speed < 0)
                    )
                    dx, _, _ = motion.advance(direction=control, jump=False, grounded=True, y=y)
                    x += dx
                    moved += dx
                    if not any(
                        s.x0 < x + width
                        and s.x1 > x
                        and (abs(s.top - feet) <= 4 or 0 < s.top - feet <= self.height + 4)
                        for s in scene.surfaces
                    ):
                        safe = False
                        break
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
                    cost = abs(remaining - moved) + abs(motion.x_speed / 16) * 2
                    candidates.append((cost, abs(direction), duration, direction))
        if not candidates:
            # Return control to skill instead of retaining an impossible target.
            self.done, self.status = True, "no_safe_path"
            direction = -(speed > 0) + (speed < 0)
        else:
            direction = min(candidates)[-1]
            self.status = "braking" if direction * speed < 0 else "running"
        return {-1: 3, 0: 0, 1: 1}[direction]
