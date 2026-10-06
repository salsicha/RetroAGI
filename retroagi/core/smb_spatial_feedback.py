"""Local execution feedback from successive pictures, never simulator state.

The network still receives only a destination. Its timed plan is a proposal:
observable motion lets the executor stop a run at that destination and
check a grounded jump against terrain and moving hazards with the shared
NES motion model. No unchecked jump launches while motion is unknown.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

from .actions import SMBAction
from .smb_pixel_types import VISIBLE_COLUMNS
from .smb_trajectory import VisualTracks, plan_flight


def terrain_edges(scene):
    """Unclipped, static landmarks for separating scroll from player motion."""
    return [
        (s.top, x)
        for s in scene.surfaces
        if not s.moving
        for x in (s.x0, s.x1)
        if VISIBLE_COLUMNS[0] < x < VISIBLE_COLUMNS[1]
    ]


def camera_shift(before, after):
    old, new = terrain_edges(before), terrain_edges(after)
    shifts = [
        x0 - x1 for y0, x0 in old for y1, x1 in new if abs(y0 - y1) <= 1 and abs(x0 - x1) <= 8
    ]
    if not shifts:
        return None
    counts = Counter(shifts)
    return min(counts, key=lambda shift: (-counts[shift], abs(shift)))


def moving_support_offset(scene):
    if scene.mario.box is None or scene.mario.support != "moving_platform":
        return None
    x0, _, x1, feet = scene.mario.box
    supports = [
        b for b in scene.moving_platforms if b[0] < x1 and b[2] > x0 and abs(b[1] - feet) <= 4
    ]
    if len(supports) != 1:
        return None
    return (x0 + x1) / 2 - supports[0][0]


@dataclass
class SpatialFeedback:
    previous: object = None
    velocities: deque = field(default_factory=lambda: deque(maxlen=4))
    speed: float | None = None
    displacement: float = 0.0
    tracked: bool = False
    destination: object = None
    start_displacement: float = 0.0
    tracks: VisualTracks = field(default_factory=VisualTracks)
    flight: object = None
    status: str = "idle"

    def observe(self, scene):
        self.tracked = False
        shift = None
        if scene.mario.box is None:
            self.previous = None
            self.velocities.clear()
            self.speed = None
            self.tracks.observe(scene, None)
            return
        if self.previous is not None and self.previous.mario.box is not None:
            shift = camera_shift(self.previous, scene)
            if shift is None:
                # Outside the camera-follow column, displacement is visible
                # directly. Do not interpret a scrolling, centred Mario as rest.
                if all(abs(s.mario.box[0] - 85) > 3 for s in (self.previous, scene)):
                    shift = 0
            # With no landmark and a scrolling player, world speed is unknown.
            dx = None
            if shift is not None:
                old = self.previous.mario.box
                now = scene.mario.box
                dx = (now[0] + now[2] - old[0] - old[2]) / 2 + shift
                if abs(dx) <= 6:
                    self.displacement += dx
                    self.tracked = True
                else:
                    dx = None
            old_offset, offset = moving_support_offset(self.previous), moving_support_offset(scene)
            if old_offset is not None and offset is not None:
                # The bridge carries position, not NES takeoff momentum.
                # Measure Mario relative to it, even if no static edge is visible.
                dx = offset - old_offset
            if dx is not None and abs(dx) <= 6:
                self.velocities.append(dx)
                self.speed = sum(self.velocities) / len(self.velocities)
            else:
                self.velocities.clear()
                self.speed = None
        self.tracks.observe(scene, shift)
        if self.flight is not None and shift is not None:
            self.flight.shift(shift)
        self.previous = scene

    def begin(self, destination, scene, proposed):
        from .smb_executor import HOLD_GROUND, ActionPlan

        self.flight = None
        self.status = "unplanned"
        self.destination = destination
        self.start_displacement = self.displacement
        if destination is not None and destination.mode == "jump" and scene.mario.box is None:
            self.status = "lost_observation"
            return ActionPlan(HOLD_GROUND, 1)
        grounded_jump = bool(
            destination is not None
            and destination.mode == "jump"
            and scene.mario.box is not None
            and scene.mario.support != "air"
            and scene.mario.on_something
        )
        if (
            destination is not None
            and scene.mario.support == "air"
            and proposed.action in (SMBAction.RIGHT_JUMP, SMBAction.LEFT_JUMP, SMBAction.JUMP)
        ):
            # A second action cannot silently extend the preceding jump hold.
            # Airborne steering releases A; only a grounded command starts a jump.
            coast = {
                SMBAction.RIGHT_JUMP: SMBAction.RIGHT,
                SMBAction.LEFT_JUMP: SMBAction.LEFT,
                SMBAction.JUMP: SMBAction.NOOP,
            }[proposed.action]
            return ActionPlan(coast, proposed.frames)
        if grounded_jump:
            if self.speed is not None and len(self.velocities) >= 2 and self.tracks.ready:
                self.flight = plan_flight(scene, self.tracks, self.speed, destination, proposed)
                if self.flight is not None:
                    self.status = "checked"
                    action = {-1: 4, 0: 5, 1: 2}[self.flight.direction]
                    return ActionPlan(action, self.flight.hold)
                self.status = "no_safe_trajectory"
                return ActionPlan(HOLD_GROUND, 1)
            self.status = "observing_motion"
            return ActionPlan(HOLD_GROUND, 1)
        return proposed

    def arrived(self):
        target = self.destination
        if target is None or target.mode != "run" or not self.tracked or target.x == 0:
            return False
        distance = self.displacement - self.start_displacement
        return distance * (1 if target.x > 0 else -1) >= abs(target.x)
