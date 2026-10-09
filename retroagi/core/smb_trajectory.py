"""Destination following with Mario motion prediction, never world collision checks.

Vision supplies poses and target identities. The controller predicts only Mario's
response to buttons; terrain, enemies and edge safety are the skill's responsibility.
"""

from copy import copy
from dataclasses import dataclass, field

from .smb_physics import NESPlayerMotion

HORIZON = 96
POSITION_TOLERANCE = 8
APPROACH_FRAMES = (0, *range(4, 65, 4))


def translated(box, dx=0, dy=0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


@dataclass
class Track:
    box: tuple
    kind: str
    samples: list = field(default_factory=list)
    identity: int = 0
    distant: list = field(default_factory=list)
    forecast_age: int = 0

    def distant_forecast(self, frames):
        """A timed LSTM endpoint, rebased by actual observations; never extrapolated."""
        usable = [
            p
            for p in self.distant
            if p[0] - self.forecast_age >= frames
            and p[0] > self.forecast_age
            and p[3] <= 4
            and p[4] >= 0.8
        ]
        if not usable:
            return None
        horizon, dx, dy, sigma, visible = min(usable)
        return horizon - self.forecast_age, translated(self.box, dx, dy), sigma

    @property
    def velocity(self):
        if not self.samples:
            return None
        vx, vy = tuple(sum(s[k] for s in self.samples) / len(self.samples) for k in (0, 1))
        return vx, vy


@dataclass
class VisualTracks:
    tracks: list = field(default_factory=list)
    next_identity: int = 1

    def observe(self, scene, scroll):
        objects = [(e.box, e.kind) for e in scene.enemies if e.kind != "defeated"]
        objects += [(b, "platform") for b in scene.moving_platforms]
        updated, used = [], set()
        for box, kind in objects:
            matches = []
            if scroll is not None:
                for i, old in enumerate(self.tracks):
                    if i in used or old.kind != kind:
                        continue
                    dx, dy = box[0] - old.box[0] + scroll, box[1] - old.box[1]
                    if abs(dx) <= 8 and abs(dy) <= 8:
                        matches.append((abs(dx) + abs(dy), i, dx, dy))
            matches.sort()
            track = Track(box, kind, identity=self.next_identity)
            self.next_identity += 1
            if matches and (len(matches) == 1 or matches[1][0] - matches[0][0] > 1):
                _, i, dx, dy = matches[0]
                old = self.tracks[i]
                # Do not let observation order transfer a target identity to
                # a neighbour when two visible objects both match this track.
                competitors = sorted(
                    abs(other[0] - old.box[0] + scroll) + abs(other[1] - old.box[1])
                    for other, other_kind in objects
                    if other_kind == kind
                    and abs(other[0] - old.box[0] + scroll) <= 8
                    and abs(other[1] - old.box[1]) <= 8
                )
                if len(competitors) > 1 and competitors[1] - competitors[0] <= 1:
                    updated.append(track)
                    continue
                if abs(dx) + abs(dy) != competitors[0]:
                    updated.append(track)
                    continue
                used.add(i)
                velocity = old.velocity
                reversed_course = velocity is not None and any(
                    abs(v) > 0.1 and delta * v < 0 for delta, v in zip((dx, dy), velocity)
                )
                # Preserve this visual identity across motion and reversals.
                # Flights may hold a reference to this particular walker.
                track = old
                track.distant = [
                    (h, px - dx, py - dy, sigma, visible)
                    for h, px, py, sigma, visible in track.distant
                ]
                track.forecast_age += 1
                track.box = box
                track.samples = ([] if reversed_course else old.samples) + [(dx, dy)]
                track.samples = track.samples[-32:]
            updated.append(track)
        self.tracks = updated

    @property
    def ready(self):
        return all(len(t.samples) >= 2 for t in self.tracks)


@dataclass
class Prediction:
    reached: bool
    error: float
    steps: list
    reason: str


def predict(
    scene,
    tracks,
    motion,
    box,
    goal,
    direction,
    hold,
    *,
    grounded=False,
    target=None,
    approach=0,
    approach_direction=None,
):
    """Predict self-motion to the requested height, without testing world objects.

    Crossing the goal's height on descent is a scoring plane, not a predicted
    collision or proof of support. Even an unreachable destination gets a
    finite closest-attempt score and an executable sequence.
    """
    motion = copy(motion)
    x, y, right, feet = box
    width, height = right - x, feet - y
    steps, best = [], float("inf")
    previous_feet = feet
    for frame in range(HORIZON + approach):
        preparing = frame < approach
        jump = approach <= frame < approach + hold
        control = approach_direction if preparing and approach_direction is not None else direction
        dx, dy, _ = motion.advance(
            direction=control,
            jump=jump,
            grounded=grounded and (preparing or frame == approach),
            y=y,
            run=control > 0,
        )
        x += dx
        if not preparing:
            y += dy
        body = (x, y, x + width, y + height)
        button = {
            (-1, True): 4,
            (0, True): 5,
            (1, True): 2,
            (-1, False): 3,
            (0, False): 0,
            (1, False): 1,
        }[control, jump]
        steps.append((body, copy(motion), button))
        if preparing:
            continue
        error = abs(x + width / 2 - goal[0]) + 2 * abs(y + height - goal[1])
        best = min(best, error)
        if dy >= 0 and previous_feet <= goal[1] <= y + height:
            return Prediction(error <= POSITION_TOLERANCE, error, steps, "target_height")
        previous_feet = y + height
    return Prediction(False, best, steps, "closest_attempt")


@dataclass
class Flight:
    """One attempted jump; actual observations, not collision predictions, end it."""

    goal: tuple
    motion: NESPlayerMotion
    direction: int
    hold: int
    prediction: Prediction
    tracks: VisualTracks
    elapsed: int = 0
    done: bool = False
    status: str = "attempting"
    released: bool = False
    target: Track | None = None
    target_offset: tuple = (0, 0)
    approach: int = 0
    approach_direction: int | None = None

    def shift(self, scroll):
        self.goal = (self.goal[0] - scroll, self.goal[1])

    def press(self, scene):
        if scene is None or scene.mario.box is None:
            self.released = True
            self.done, self.status = True, "lost_observation"
            return 0
        if self.target is not None:
            if any(t is self.target for t in self.tracks.tracks):
                b = self.target.box
                self.goal = (
                    (b[0] + b[2]) / 2 + self.target_offset[0],
                    b[1] + self.target_offset[1],
                )
            else:
                # Continue toward the last requested point; never acquire a neighbour.
                self.target = None
                self.status = "lost_target"
        observed_ground = scene.mario.on_something and scene.mario.support != "air"
        if self.elapsed < self.approach and not observed_ground:
            self.approach = 0
            self.released = True
        delay = max(0, self.approach - self.elapsed)
        airborne = max(0, self.elapsed - self.approach)
        remaining = 0 if self.released else max(0, self.hold - airborne)
        grounded = observed_ground and self.elapsed <= self.approach
        check = predict(
            scene,
            [],
            self.motion,
            scene.mario.box,
            self.goal,
            self.direction,
            remaining,
            grounded=grounded,
            approach=delay,
            approach_direction=self.approach_direction,
        )
        if not delay and not check.reached:
            options = []
            holds = (0,) if self.released else ((remaining,) if grounded else (remaining, 0))
            for direction in (-1, 0, 1):
                for hold in set(holds):
                    candidate = predict(
                        scene,
                        [],
                        self.motion,
                        scene.mario.box,
                        self.goal,
                        direction,
                        hold,
                        grounded=grounded,
                    )
                    options.append(
                        (
                            candidate.error,
                            direction != self.direction,
                            hold != remaining,
                            direction,
                            hold,
                            candidate,
                        )
                    )
            _, _, _, self.direction, remaining, check = min(options, key=lambda x: x[:5])
            self.hold = airborne + remaining
        if not remaining and not delay:
            self.released = True
        self.prediction = check
        _, self.motion, button = check.steps[0]
        self.elapsed += 1
        if self.elapsed >= HORIZON + self.approach:
            self.done, self.status = True, "flight_timeout"
        return button


def plan_flight(scene, tracks, speed, destination, proposed=None, *, motion=None):
    """Always attempt the goal. Approach durations stay on the four-frame grid."""
    box = scene.mario.box
    goal = ((box[0] + box[2]) / 2 + destination.x, box[3] + destination.y)
    motion = copy(motion) if motion is not None else NESPlayerMotion(x_speed=round(speed * 16))
    direction = (destination.x > 0) - (destination.x < 0)
    candidates = []
    chosen = None
    delays = (0,) if proposed is not None else APPROACH_FRAMES
    maneuvers = [(direction, direction)]
    if proposed is None:
        maneuvers += [(d, -d) for d in (direction, -direction) if d]
    for flight_direction, approach_direction in maneuvers:
        for delay in delays:
            group = []
            if flight_direction != approach_direction and not delay:
                continue
            for hold in range(1, 33):
                prediction = predict(
                    scene,
                    [],
                    motion,
                    box,
                    goal,
                    flight_direction,
                    hold,
                    grounded=True,
                    approach=delay,
                    approach_direction=approach_direction,
                )
                group.append(
                    (
                        not prediction.reached,
                        round(prediction.error / 4),
                        len(prediction.steps) + delay * 0.1,
                        hold,
                        flight_direction,
                        approach_direction,
                        delay,
                        prediction,
                    )
                )
            candidates.extend(group)
            reached = [c for c in group if c[0] is False]
            if reached:
                chosen = min(reached, key=lambda c: c[:7])
                break
        if chosen is not None:
            break
    _, _, _, hold, direction, approach_direction, delay, prediction = chosen or min(
        candidates, key=lambda c: c[:7]
    )
    targets = [
        t
        for t in tracks.tracks
        if t.kind in ("walker", "platform")
        and t.box[0] - 4 <= goal[0] <= t.box[2] + 4
        and abs(goal[1] - t.box[1]) <= 4
    ]
    target = targets[0] if len(targets) == 1 else None
    offset = (
        (goal[0] - (target.box[0] + target.box[2]) / 2, goal[1] - target.box[1])
        if target
        else (0, 0)
    )
    return Flight(
        goal,
        motion,
        direction,
        hold,
        prediction,
        tracks,
        target=target,
        target_offset=offset,
        approach=delay,
        approach_direction=approach_direction,
    )
