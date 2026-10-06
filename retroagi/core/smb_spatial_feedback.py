"""Local execution feedback from successive pictures, never simulator state.

The network still receives only a destination. Its timed plan is a proposal:
observable motion lets the executor stop a run at that destination and
calibrate a grounded jump's hold against the shared NES motion model.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache

from .actions import SMBAction
from .smb_physics import NESPlayerMotion
from .smb_pixel_types import VISIBLE_COLUMNS


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


@lru_cache(maxsize=32768)
def landing_for_hold(speed, direction, hold, target_y):
    """Unobstructed first descending crossing of the requested foot height."""
    motion = NESPlayerMotion(
        x_speed=round(speed * 16),
        moving=1 if speed > 0 else -1 if speed < 0 else 0,
        facing=direction or 1,
    )
    x = y = 0
    for frame in range(160):
        dx, dy, _ = motion.advance(
            direction=direction,
            jump=frame < hold,
            grounded=frame == 0,
            y=y,
        )
        old_y = y
        x, y = x + dx, y + dy
        if frame > 0 and dy >= 0 and old_y <= target_y <= y:
            return x
    return None


def jump_plan(destination, speed, proposed):
    """Calibrate only when the local model can actually reach the destination.

    Unreachable endpoints retain the network's proposal. Terrain selection
    belongs to skill; this does not certify collisions, search the level or use
    a teacher. Live-enemy scenes are excluded by the caller.
    """
    from .smb_executor import ActionPlan

    direction = (destination.x > 0) - (destination.x < 0)
    action = {1: SMBAction.RIGHT_JUMP, -1: SMBAction.LEFT_JUMP, 0: SMBAction.JUMP}[direction]
    candidates = []
    for hold in range(1, 33):
        landed = landing_for_hold(speed, direction, hold, destination.y)
        if landed is not None:
            candidates.append((abs(landed - destination.x), abs(hold - proposed.frames), hold))
    if not candidates:
        return proposed
    error, _, hold = min(candidates)
    return ActionPlan(action, hold) if error <= 4 else proposed


@dataclass
class SpatialFeedback:
    previous: object = None
    velocities: deque = field(default_factory=lambda: deque(maxlen=4))
    speed: float | None = None
    displacement: float = 0.0
    tracked: bool = False
    destination: object = None
    start_displacement: float = 0.0
    start_feet: float = 0.0
    grounded_jump: bool = False
    calibrated: bool = False

    def observe(self, scene):
        self.tracked = False
        if scene.mario.box is None:
            self.previous = None
            self.velocities.clear()
            self.speed = None
            return
        if self.previous is not None and self.previous.mario.box is not None:
            shift = camera_shift(self.previous, scene)
            # With no landmark and a scrolling player, world speed is unknown.
            # Away from the scrolling region an unchanged full-width floor is
            # insufficient evidence either: retain the learned proposal.
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
        self.previous = scene

    def begin(self, destination, scene, proposed):
        from .smb_executor import ActionPlan

        self.destination = destination
        self.start_displacement = self.displacement
        self.grounded_jump = bool(
            destination is not None
            and destination.mode == "jump"
            and scene.mario.box is not None
            and scene.mario.support != "air"
            and scene.mario.on_something
            # A ballistic endpoint cannot certify contact with an enemy that
            # moves during the flight, or clearance over a live hazard.
            and not any(e.kind != "defeated" for e in scene.enemies)
        )
        self.calibrated = False
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
        if self.grounded_jump:
            self.start_feet = scene.mario.box[3]
            if self.speed is not None and len(self.velocities) >= 2:
                self.calibrated = True
                return jump_plan(destination, max(-2.5, min(2.5, self.speed)), proposed)
        return proposed

    def calibrate_launch(self, scene, plan, pressed):
        # An episode may start at a running takeoff, before two pictures exist.
        # Two frames provide measured horizontal speed. A low launch could be
        # standing OR walking, so do not guess its takeoff state from height.
        # Never recalibrate midway through the flight.
        if (
            self.grounded_jump
            and not self.calibrated
            and pressed == 2
            and scene.mario.box is not None
            and scene.mario.support == "air"
        ):
            self.calibrated = True
            dy = scene.mario.box[3] - self.start_feet
            if dy <= -9 and self.speed is not None and len(self.velocities) >= 2:
                return jump_plan(self.destination, max(-2.5, min(2.5, self.speed)), plan)
        return plan

    def arrived(self):
        target = self.destination
        if target is None or target.mode != "run" or not self.tracked or target.x == 0:
            return False
        distance = self.displacement - self.start_displacement
        return distance * (1 if target.x > 0 else -1) >= abs(target.x)
