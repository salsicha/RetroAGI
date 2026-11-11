"""Mario's motion and the camera's, estimated from successive pictures and
the buttons pressed, never from simulator state.

MotionEstimator watches the scene vision gives each frame (observe) and is
told the button pressed (executed):
- the camera's scroll, from still landmarks (the ends of floors and steps)
  seen in both pictures; with none in view, dead-reckoned from what the
  motion model says the button did (from SCROLL_FROM on);
- Mario's speed: the motion model run on the buttons pressed, corrected
  when what is seen disagrees (a wall, a bounce off an enemy);
- every enemy and moving platform followed frame to frame with its velocity
  (tracks).
A controller built on the estimate adds what it needs through the hooks
(_lost, _speed_corrected, _observed).
"""

from collections import Counter, deque
from dataclasses import dataclass, field

from retroagi.domains.smb.physics import NESPlayerMotion

from .pixel_types import VISIBLE_COLUMNS
from .scene_labels import fixed_surfaces
from .tracks import VisualTracks

# Left of this screen column (Mario's drawn left side) the camera never
# moves: the real game starts scrolling as he passes about 80 and follows him
# only partly until 112 (5-1: a pixel a frame on screen, a pixel and a half
# of scroll), Block SMB's old camera at 85. From it on, with no landmark to
# measure the scroll by (5-1's long bare floor), the scroll is dead-reckoned
# from the buttons pressed (taken for none, the camera estimate stood still,
# Mario seemed to crawl and a Koopa to run at 1.6 pixels a frame).
SCROLL_FROM = 77


def terrain_edges(scene):
    """Unclipped, static landmarks for separating scroll from player motion."""
    return [
        (s.top, x)
        # Not the floors cut where enemies stand: those ends walk with them.
        for s in fixed_surfaces(scene)
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
        for s in fixed_surfaces(scene)
    ):
        return None
    return support


@dataclass
class MotionEstimator:
    previous: object = None
    velocities: deque = field(default_factory=lambda: deque(maxlen=4))
    # The motion model's own moves for the same frames (whole pixels, as
    # seen ones are): their sums differing by 2 or more is a real error.
    predicted_moves: deque = field(default_factory=lambda: deque(maxlen=4))
    speed: float | None = None
    displacement: float = 0.0
    tracked: bool = False
    tracks: VisualTracks = field(default_factory=VisualTracks)
    motion: NESPlayerMotion = field(default_factory=NESPlayerMotion)
    predicted_dx: float | None = None
    # The vertical move the motion model predicted for this frame (pixels,
    # down positive; None when unknown).
    predicted_dy: float | None = None
    motion_ready: bool = False
    camera_position: float = 0.0
    # Mario's feet's movement since the last frame (pixels, down positive;
    # 0 when either frame lost him): the vertical part of his estimate.
    feet_velocity: float = 0.0

    def _lost(self) -> None:
        """Mario is not in the picture."""

    def _speed_corrected(self, scene) -> None:
        """The motion model's speed was just set to the seen one."""

    def _observed(self, scene, shift) -> None:
        """A picture with Mario in it was taken in (``shift``: the camera's
        scroll since the last, None if unknown)."""

    def executed(self, button, scene):
        """Propagate the actual button history, including release and braking."""
        if scene.mario.box is None:
            self.predicted_dx = self.predicted_dy = None
            self.motion_ready = False
            return
        direction = 1 if button in (1, 2) else -1 if button in (3, 4) else 0
        self.predicted_dx, self.predicted_dy, _ = self.motion.advance(
            direction=direction,
            jump=button in (2, 4, 5),
            grounded=scene.mario.on_something and scene.mario.support != "air",
            y=scene.mario.box[1],
            run=direction > 0,
        )

    def observe(self, scene):
        self.tracked = False
        shift = None
        reckoned = False
        if scene.mario.box is None:
            self.previous = None
            self.feet_velocity = 0.0
            self._lost()
            self.velocities.clear()
            self.predicted_moves.clear()
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
                if all(s.mario.box[0] < SCROLL_FROM for s in (self.previous, scene)):
                    shift = 0
                elif self.predicted_dx is not None:
                    # Featureless scrolling floor supplies no world landmark.
                    # Dead-reckon from actual buttons until visual edges return.
                    old, now = self.previous.mario.box, scene.mario.box
                    shift = self.predicted_dx - (now[0] + now[2] - old[0] - old[2]) / 2
                    reckoned = True
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
                self.predicted_moves.clear()
                dx = None
            old, now = self.previous.mario.box, scene.mario.box
            if (now[2] - now[0]) != (old[2] - old[0]):
                # A changing visual silhouette moves its center without player
                # motion. Do not turn that deformation into a velocity reversal
                # (a 1-pixel wobble of the seen box's left edge read as drift
                # to the right while Mario moved left, and the flight then
                # held its jump 7 frames too long).
                dx = None
            if dx is not None and abs(dx) <= 6:
                self.velocities.append(dx)
                self.predicted_moves.append(self.predicted_dx)
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
                # The model's moves over the last four frames off the seen ones
                # by 2 pixels or more (pixel rounding makes at most 1): a wall
                # the model did not know stopped him (a jump against a step's
                # side left it 0.37 pixels a frame fast, under 0.75, and a
                # flight check then braked him into 1-1's last pit).
                drifted = (
                    len(self.predicted_moves) == 4
                    and None not in self.predicted_moves
                    and abs(sum(self.velocities) - sum(self.predicted_moves)) >= 2
                )
                if len(self.velocities) >= samples_needed and (
                    abs(self.speed - self.motion.x_speed / 16) >= 0.75 or takeoff_bucket_changed or drifted
                ):
                    self.motion.x_speed = round(self.speed * 16)
                    self.motion.moving = (self.speed > 0) - (self.speed < 0)
                    self._speed_corrected(scene)
                if len(self.velocities) >= 2:
                    self.motion_ready = True
            else:
                self.velocities.clear()
                self.predicted_moves.clear()
                self.speed = None
        if shift is not None:
            self.camera_position += shift
        self.tracks.observe(scene, shift, reckoned)
        self._observed(scene, shift)
        self.feet_velocity = (
            float(scene.mario.box[3] - self.previous.mario.box[3])
            if self.previous is not None and self.previous.mario.box is not None
            else 0.0
        )
        airborne = not (scene.mario.on_something and scene.mario.support != "air")
        if airborne and self.predicted_dy is not None and self.predicted_dy >= 1 and self.feet_velocity <= -2:
            # Seen going up where the model had him falling: bounced off an
            # enemy (a stomp, as the game's $FC). Without this the model fell
            # on through every bounce and each plan made in the air was wrong;
            # a pixel's wobble at the top of a jump is no bounce (taken for
            # one, it threw Mario's jump over 1-1's second pit short).
            self.motion.bounce()
            self.motion.y_force = 0
        self.previous = scene
