"""Bounded visual jump planning. No simulator, teacher, object IDs or RAM.

Predictions use observed rectangles and constant object velocity over one jump.
They are rechecked on every picture; they are not a guarantee about unseen
terrain, future enemy turns, or errors in the vision model.
"""

from copy import copy
from dataclasses import dataclass, field

from .actions import SMBAction
from .smb_physics import NESPlayerMotion

HORIZON = 96
POSITION_TOLERANCE = 8


def overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def translated(box, dx=0, dy=0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


@dataclass
class Track:
    box: tuple
    kind: str
    samples: list = field(default_factory=list)

    @property
    def velocity(self):
        if not self.samples:
            return None
        return tuple(sum(s[k] for s in self.samples) / len(self.samples) for k in (0, 1))


@dataclass
class VisualTracks:
    tracks: list = field(default_factory=list)

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
            track = Track(box, kind)
            if matches and (len(matches) == 1 or matches[1][0] - matches[0][0] > 1):
                _, i, dx, dy = matches[0]
                used.add(i)
                old = self.tracks[i]
                velocity = old.velocity
                reversed_course = velocity is not None and any(
                    abs(v) > 0.1 and delta * v < 0 for delta, v in zip((dx, dy), velocity)
                )
                track.samples = ([] if reversed_course else old.samples) + [(dx, dy)]
                track.samples = track.samples[-32:]
            updated.append(track)
        self.tracks = updated

    @property
    def ready(self):
        return all(len(t.samples) >= 2 for t in self.tracks)


def geometry(scene, tracks):
    solids = [(b.box, (0, 0)) for b in scene.blocks]
    solids += [(b, (0, 0)) for b in scene.pipes]
    solids += [(t.box, t.velocity or (0, 0)) for t in tracks if t.kind == "platform"]
    # Raised ground can be reported only as adjacent surface spans, without
    # block boxes. Their shared edge exposes the step's vertical face.
    surfaces = [s for s in scene.surfaces if not s.moving]
    for surface in surfaces:
        lower = [
            s.top
            for s in surfaces
            if s.top > surface.top and (abs(s.x1 - surface.x0) <= 1 or abs(s.x0 - surface.x1) <= 1)
        ]
        if lower:
            solids.append(((surface.x0, surface.top, surface.x1, max(lower)), (0, 0)))
    tops = [((s.x0, s.top, s.x1, s.top), (0, 0)) for s in scene.surfaces if not s.moving]
    tops += [((b[0], b[1], b[2], b[1]), v) for b, v in solids]
    return solids, tops


def takeoff_box(scene):
    """Ground contact disambiguates a few platform pixels merged into Mario."""
    box = scene.mario.box
    supports = [
        s.top
        for s in scene.surfaces
        if s.x0 < box[2] and s.x1 > box[0] and abs(s.top - box[3]) <= 8
    ]
    if scene.mario.on_something and supports:
        feet = min(supports, key=lambda top: abs(top - box[3]))
        if feet - box[1] >= 8:
            return (box[0], box[1], box[2], feet)
    return box


@dataclass
class Prediction:
    safe: bool
    reached: bool
    error: float
    steps: list
    reason: str


def predict(scene, tracks, motion, box, goal, direction, hold, *, grounded=False):
    """Sweep Mario's body through terrain and forecast object positions.

    Resolve wall/ceiling contacts, rather than pretending every flight is an
    unobstructed parabola. Stop at the first support or intended walker stomp.
    """
    motion = copy(motion)
    solids, tops = geometry(scene, tracks)
    hazards = [t for t in tracks if t.kind != "platform"]
    steps = []
    w, h = box[2] - box[0], box[3] - box[1]
    x, y = box[:2]
    for frame in range(HORIZON):
        old = (x, y, x + w, y + h)
        jump = frame < hold
        dx, dy, _ = motion.advance(direction=direction, jump=jump, grounded=grounded, y=y)
        grounded = False
        x += dx
        for rect, velocity in solids:
            r = translated(rect, velocity[0] * (frame + 1), velocity[1] * (frame + 1))
            if overlap((x, y, x + w, y + h), r):
                if dx > 0:
                    x = r[0] - w
                elif dx < 0:
                    x = r[2]
                motion.wall_contact()
        previous_y = y
        y += dy
        for rect, velocity in solids:
            r = translated(rect, velocity[0] * (frame + 1), velocity[1] * (frame + 1))
            if x < r[2] and x + w > r[0] and dy < 0 and y <= r[3] <= previous_y:
                y = r[3]
                motion.vertical_contact()
        landed = False
        for rect, velocity in tops:
            r = translated(rect, velocity[0] * (frame + 1), velocity[1] * (frame + 1))
            if dy >= 0 and x < r[2] and x + w > r[0] and previous_y + h <= r[1] <= y + h:
                y = r[1] - h
                landed = True
                motion.vertical_contact()
                break
        body = (x, y, x + w, y + h)
        button = {
            (-1, True): 4,
            (1, True): 2,
            (0, True): 5,
            (-1, False): 3,
            (1, False): 1,
            (0, False): 0,
        }[direction, jump]
        steps.append((body, copy(motion), button))
        for hazard in hazards:
            vx, vy = hazard.velocity or (0, 0)
            now = translated(hazard.box, vx * (frame + 1), vy * (frame + 1))
            before = translated(hazard.box, vx * frame, vy * frame)
            # Sub-frame relative sweeps also catch a fast hazard crossing the
            # body between pictures. A pixel of clearance for lethal objects.
            margin = 0 if hazard.kind == "walker" else 1 + (frame + 1) / max(1, len(hazard.samples))
            collision = any(
                overlap(
                    tuple(old[k] + (body[k] - old[k]) * t for k in range(4)),
                    (
                        before[0] + vx * t - margin,
                        before[1] + vy * t - margin,
                        before[2] + vx * t + margin,
                        before[3] + vy * t + margin,
                    ),
                )
                for t in (0.25, 0.5, 0.75, 1.0)
            )
            if collision:
                intended_stomp = (
                    hazard.kind == "walker"
                    and dy > 0
                    and old[3] <= (now[1] + now[3]) / 2
                    and abs(goal[1] - now[1]) <= 8
                    and now[0] - 4 <= goal[0] <= now[2] + 4
                )
                if intended_stomp:
                    return Prediction(True, True, abs(x + w / 2 - goal[0]), steps, "stomp")
                return Prediction(False, False, float("inf"), steps, "hazard")
        if landed:
            # Reaching the point a frame before an enemy arrives is not a safe
            # arrival. Include residual momentum and a short reaction window;
            # the next decision cannot stop Mario instantaneously.
            for hazard in hazards:
                vx, vy = hazard.velocity or (0, 0)
                for later in range(1, 9):
                    time = frame + 1 + later
                    margin = (
                        0 if hazard.kind == "walker" else 1 + time / max(1, len(hazard.samples))
                    )
                    enemy = translated(hazard.box, vx * time, vy * time)
                    enemy = (
                        enemy[0] - margin,
                        enemy[1] - margin,
                        enemy[2] + margin,
                        enemy[3] + margin,
                    )
                    arrival = translated(body, motion.x_speed / 16 * later)
                    if overlap(arrival, enemy):
                        return Prediction(False, False, float("inf"), steps, "unsafe_arrival")
            error = abs(x + w / 2 - goal[0]) + 2 * abs(y + h - goal[1])
            return Prediction(True, error <= POSITION_TOLERANCE, error, steps, "landed")
        if y > 256:
            break
    return Prediction(False, False, float("inf"), steps, "no_visible_landing")


@dataclass
class Flight:
    """One physical jump: hold, release, coast; never press A again in flight."""

    goal: tuple
    motion: NESPlayerMotion
    direction: int
    hold: int
    prediction: Prediction
    tracks: VisualTracks
    elapsed: int = 0
    done: bool = False
    status: str = "checked"
    released: bool = False

    def shift(self, scroll):
        self.goal = (self.goal[0] - scroll, self.goal[1])

    def press(self, scene):
        if scene is None or scene.mario.box is None:
            self.released = True
            self.done, self.status = True, "lost_observation"
            return int(SMBAction.NOOP)
        remaining = max(0, self.hold - self.elapsed) if not self.released else 0
        box = takeoff_box(scene) if self.elapsed == 0 else scene.mario.box
        check = predict(
            scene,
            self.tracks.tracks,
            self.motion,
            box,
            self.goal,
            self.direction,
            remaining,
            grounded=self.elapsed == 0,
        )
        if not check.safe:
            alternatives = []
            # Steering can change within this same maneuver. A released jump
            # stays released, including after a missing/unsafe observation.
            for direction in (-1, 0, 1):
                for hold in {0, remaining, max(0, 32 - self.elapsed) if not self.released else 0}:
                    candidate = predict(
                        scene,
                        self.tracks.tracks,
                        self.motion,
                        box,
                        self.goal,
                        direction,
                        hold,
                        grounded=self.elapsed == 0,
                    )
                    if candidate.safe and candidate.reached:
                        alternatives.append(
                            (
                                candidate.error,
                                direction != self.direction,
                                direction,
                                hold,
                                candidate,
                            )
                        )
            if alternatives:
                _, _, self.direction, remaining, check = min(alternatives, key=lambda a: a[:4])
                self.hold = self.elapsed + remaining
                self.status = "steering_correction"
            else:
                # A short emergency drop can look collision-free only because
                # it lands before the approaching enemy arrives. Never choose
                # an unrelated early landing or cut a committed jump blindly.
                self.status = "no_safe_continuation"
        self.prediction = check
        if check.steps:
            _, self.motion, button = check.steps[0]
        else:
            button = 0
        if not remaining:
            self.released = True
        self.elapsed += 1
        if self.elapsed >= HORIZON:
            self.done, self.status = True, "flight_timeout"
        return int(button)


def plan_flight(scene, tracks, speed, destination, proposed):
    box = takeoff_box(scene)
    goal = ((box[0] + box[2]) / 2 + destination.x, box[3] + destination.y)
    direction = (destination.x > 0) - (destination.x < 0)
    motion = NESPlayerMotion(
        x_speed=round(speed * 16),
        moving=(speed > 0) - (speed < 0),
        facing=1 if scene.mario.facing_right else -1,
    )
    candidates = []
    for hold in range(1, 33):
        prediction = predict(
            scene, tracks.tracks, motion, box, goal, direction, hold, grounded=True
        )
        if prediction.safe and prediction.reached:
            candidates.append(
                (round(prediction.error / 4), abs(hold - proposed.frames), hold, prediction)
            )
    if not candidates:
        return None
    _, _, hold, prediction = min(candidates, key=lambda c: c[:3])
    return Flight(goal, motion, direction, hold, prediction, tracks)
