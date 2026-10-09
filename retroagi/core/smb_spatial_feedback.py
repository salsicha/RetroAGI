"""Local execution feedback from successive pictures, never simulator state.

Skill supplies a destination directly. Predictive ground and flight controllers
select the buttons; no learned action network or duration proposal is needed.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

from .actions import SMBAction
from .smb_ground_control import GroundMove
from .smb_physics import NESPlayerMotion
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
        # A moving platform can occlude a floor edge; that apparent static
        # endpoint moves with the bridge and cannot measure camera scroll.
        and not any(abs(x - edge) <= 4 for b in scene.moving_platforms for edge in (b[0], b[2]))
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


def moving_support(scene):
    """Resolve support geometrically when the classifier calls a bridge ground."""
    if scene.mario.box is None or scene.mario.support == "air":
        return None
    x0, _, x1, feet = scene.mario.box
    supports = [
        b
        for b in scene.moving_platforms
        if b[0] < x1 + 2 and b[2] > x0 - 2 and abs(b[1] - feet) <= 4
    ]
    if len(supports) != 1:
        return None
    support = supports[0]
    # A shore to the right takes support at the dismount overlap.
    if any(
        not s.moving and s.x0 >= support[0] and s.x0 < x1 and s.x1 > x0 and abs(s.top - feet) <= 4
        for s in scene.surfaces
    ):
        return None
    return support


def moving_support_offset(scene):
    support = moving_support(scene)
    if support is None:
        return None
    x0, _, x1, _ = scene.mario.box
    return (x0 + x1) / 2 - support[0]


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
    travel: object = None
    motion: NESPlayerMotion = field(default_factory=NESPlayerMotion)
    predicted_dx: float | None = None
    motion_ready: bool = False
    camera_position: float = 0.0
    start_feet: float = 0.0
    observed_frames: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=24))

    @property
    def stalled(self):
        return (
            self.destination is not None
            and self.destination.mode != "hold"
            and len(self.positions) == self.positions.maxlen
            and all(
                max(p[k] for p in self.positions) - min(p[k] for p in self.positions) <= 2
                for k in (0, 1)
            )
        )

    def report(self, scene):
        """Measured command outcome for skill, never a judgement of goal safety."""
        from .tokens import EXECUTION_WIDTH

        if self.destination is None or scene.mario.box is None:
            return [0.0] * EXECUTION_WIDTH
        box = scene.mario.box
        dx = self.displacement - self.start_displacement
        dy = box[3] - self.start_feet
        rx, ry = self.destination.x - dx, self.destination.y - dy
        if self.destination.mode == "hold":
            rx = ry = 0.0
        elif self.flight is not None:
            rx = self.flight.goal[0] - (box[0] + box[2]) / 2
            ry = self.flight.goal[1] - box[3]
        elif self.travel is not None and self.travel.target is not None:
            rx = self.travel.target.box[0] + self.travel.target_offset - (box[0] + box[2]) / 2
        return [
            1.0,
            float(self.stalled),
            float(abs(rx) <= 4 and abs(ry) <= 4),
            min(self.observed_frames / 192, 1.0),
            dx / 256,
            dy / 240,
            rx / 256,
            ry / 240,
        ]

    def executed(self, button, scene):
        """Propagate the actual button history, including release and braking."""
        if scene.mario.box is None:
            self.predicted_dx = None
            self.motion_ready = False
            return
        direction = 1 if button in (1, 2) else -1 if button in (3, 4) else 0
        self.predicted_dx, _, _ = self.motion.advance(
            direction=direction,
            jump=button in (2, 4, 5),
            grounded=scene.mario.on_something and scene.mario.support != "air",
            y=scene.mario.box[1],
            run=direction > 0,
        )

    def observe(self, scene):
        self.tracked = False
        shift = None
        if scene.mario.box is None:
            self.previous = None
            self.positions.clear()
            self.velocities.clear()
            self.speed = None
            self.predicted_dx = None
            self.motion_ready = False
            self.tracks.observe(scene, None)
            return
        if self.previous is not None and self.previous.mario.box is not None:
            shift = camera_shift(self.previous, scene)
            if shift is None:
                # Outside the camera-follow column, displacement is visible
                # directly. Do not interpret a scrolling, centred Mario as rest.
                if all(abs(s.mario.box[0] - 85) > 3 for s in (self.previous, scene)):
                    shift = 0
                elif self.predicted_dx is not None:
                    # Featureless scrolling floor supplies no world landmark.
                    # Dead-reckon from actual buttons until visual edges return.
                    shift = self.predicted_dx - (scene.mario.box[0] - self.previous.mario.box[0])
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
            before_support, support = moving_support(self.previous), moving_support(scene)
            if before_support is not None and support is not None:
                # Use the visible opposite edge when the bridge is clipped;
                # a viewport boundary cannot measure support-relative velocity.
                edge = 2 if min(before_support[0], support[0]) <= VISIBLE_COLUMNS[0] else 0
                old, now = self.previous.mario.box, scene.mario.box
                dx = (now[0] + now[2] - old[0] - old[2]) / 2
                dx -= support[edge] - before_support[edge]
            elif (before_support is None) != (support is None):
                # Do not mix carried world displacement into player momentum.
                self.velocities.clear()
                dx = None
            old, now = self.previous.mario.box, scene.mario.box
            if abs((now[2] - now[0]) - (old[2] - old[0])) > 1:
                # A changing visual silhouette moves its center without player
                # motion. Do not turn that deformation into a velocity reversal.
                dx = None
            if dx is not None and abs(dx) <= 6:
                self.velocities.append(dx)
                self.speed = sum(self.velocities) / len(self.velocities)
                # Keep fractional physics through ordinary pixel quantization;
                # correct substantial disagreement (e.g. a wall or changed speed).
                # Bootstrap unknown spawn momentum before a jump may plan.
                # Once initialized, retain four-sample filtering so pixel
                # quantization does not overwrite fractional acceleration.
                samples_needed = 4 if self.motion_ready else 2
                takeoff_bucket_changed = (
                    scene.mario.on_something
                    and scene.mario.support != "air"
                    and self.previous.mario.support == "air"
                    and len(self.velocities) == 4
                    and max(self.velocities) - min(self.velocities) <= 1
                    and sum(abs(self.speed * 16) >= n for n in (9, 16, 25, 28))
                    != sum(abs(self.motion.x_speed) >= n for n in (9, 16, 25, 28))
                )
                if len(self.velocities) >= samples_needed and (
                    abs(self.speed - self.motion.x_speed / 16) >= 0.75 or takeoff_bucket_changed
                ):
                    self.motion.x_speed = round(self.speed * 16)
                    self.motion.moving = (self.speed > 0) - (self.speed < 0)
                    if self.flight is not None:
                        # Flight owns a copy of the physical state; correcting
                        # only this tracker leaves its next prediction stale.
                        self.flight.motion.x_speed = self.motion.x_speed
                        self.flight.motion.moving = self.motion.moving
                if len(self.velocities) >= 2:
                    self.motion_ready = True
            else:
                self.velocities.clear()
                self.speed = None
        if shift is not None:
            self.camera_position += shift
        self.tracks.observe(scene, shift)
        if self.flight is not None and shift is not None:
            self.flight.shift(shift)
        if self.destination is not None:
            self.observed_frames += 1
            self.positions.append((self.displacement, scene.mario.box[3]))
        self.previous = scene

    def begin(self, destination, scene, proposed=None):
        from .smb_executor import HOLD_GROUND, ActionPlan

        self.flight = None
        self.travel = None
        self.status = "unplanned"
        previous_destination = self.destination
        self.destination = destination
        self.start_displacement = self.displacement
        self.start_feet = scene.mario.box[3] if scene.mario.box else 0.0
        self.observed_frames = 0
        if (
            destination is None
            or destination.mode == "hold"
            or previous_destination is None
            or previous_destination.mode == "hold"
        ):
            self.positions.clear()
        if destination is None:
            if proposed is None:
                raise ValueError("the executor needs a destination")
            return proposed  # Exact teacher playback only.
        if proposed is None and destination.mode == "hold":
            self.status = "holding"
            return ActionPlan(HOLD_GROUND, 1)
        if proposed is None and destination.mode == "run":
            if scene.mario.box is None:
                self.status = "lost_observation"
                return ActionPlan(HOLD_GROUND, 1)
            self.travel = GroundMove(
                self, destination.x, self.displacement, scene.mario.box[3], destination.y
            )
            return ActionPlan(0, 1)
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
            and (
                proposed is None
                or proposed.action in (SMBAction.RIGHT_JUMP, SMBAction.LEFT_JUMP, SMBAction.JUMP)
            )
        ):
            # A second action cannot silently extend the preceding jump hold.
            # Airborne steering releases A; only a grounded command starts a jump.
            coast = (
                (1 if destination.x > 0 else 3 if destination.x < 0 else 0)
                if proposed is None
                else {
                    SMBAction.RIGHT_JUMP: SMBAction.RIGHT,
                    SMBAction.LEFT_JUMP: SMBAction.LEFT,
                    SMBAction.JUMP: SMBAction.NOOP,
                }[proposed.action]
            )
            return ActionPlan(coast, proposed.frames if proposed is not None else 1)
        if grounded_jump:
            # Use the motion estimate already available; a jump request must
            # not insert waiting frames to collect additional observations.
            speed = (
                self.speed
                if proposed is not None and self.speed is not None
                else self.motion.x_speed / 16
            )
            self.flight = plan_flight(
                scene,
                self.tracks,
                speed,
                destination,
                proposed,
                motion=self.motion if proposed is None else None,
            )
            self.status = "attempting"
            action = {-1: 4, 0: 5, 1: 2}[self.flight.direction]
            return ActionPlan(action, max(1, self.flight.hold))
        if proposed is None:
            self.status = "observing_support"
            return ActionPlan(HOLD_GROUND, 1)
        return proposed

    def arrived(self):
        if self.travel is not None:
            return self.travel.done
        target = self.destination
        if target is None or target.mode != "run" or not self.tracked or target.x == 0:
            return False
        distance = self.displacement - self.start_displacement
        return distance * (1 if target.x > 0 else -1) >= abs(target.x)
