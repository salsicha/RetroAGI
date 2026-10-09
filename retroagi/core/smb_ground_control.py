"""Receding-horizon control of a fixed ground destination from visual feedback."""

from copy import copy
from dataclasses import dataclass

MPC_FRAMES = 8


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
        if self.target is not None:
            if not any(t is self.target for t in platforms):
                self.target = None
            else:
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
        # Only Mario's response to buttons is predicted. No support, obstacle,
        # hazard or arrival-safety veto can override the requested destination.
        candidates = []
        for direction in (-1, 0, 1):
            for duration in range(1, MPC_FRAMES + 1):
                motion = copy(self.feedback.motion)
                moved = 0
                for frame in range(MPC_FRAMES):
                    control = direction if frame < duration else 0
                    dx, _, _ = motion.advance(
                        direction=control,
                        jump=False,
                        grounded=grounded,
                        y=scene.mario.box[1],
                        run=control > 0,
                    )
                    moved += dx
                brake = stopping_distance(motion)
                cost = abs(remaining - moved - brake) + abs(motion.x_speed / 16) * 0.3
                progress = direction * remaining > 0 and abs(speed) < 0.25
                candidates.append((cost, not progress, abs(direction), duration, direction))
        direction = min(candidates)[-1]
        self.status = (
            "descending" if not grounded else ("braking" if direction * speed < 0 else "running")
        )
        return {-1: 3, 0: 0, 1: 1}[direction]
