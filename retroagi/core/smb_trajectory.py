"""Bounded visual jump planning. No simulator state, teacher targets or game IDs.

Predictions use observed rectangles, velocity and witnessed patrol reversals.
They are rechecked on every picture; they are not a guarantee about unseen
terrain, future enemy turns, or errors in the vision model.
"""

from copy import copy
from dataclasses import dataclass, field

from .actions import SMBAction
from .smb_collision import stomp_contact, walker_body
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
    turns: list = field(default_factory=list)
    horizontal_speeds: list = field(default_factory=list)

    def forecast(self, frames):
        vx, vy = self.velocity or (0, 0)
        x = self.box[0] + vx * frames
        if len(self.turns) >= 2:
            low, high = min(self.turns), max(self.turns)
            width = high - low
            if width > 2:
                phase = (x - low) % (2 * width)
                x = low + (phase if phase <= width else 2 * width - phase)
        return translated(self.box, x - self.box[0], vy * frames)

    @property
    def velocity(self):
        if not self.samples:
            return None
        vx, vy = tuple(sum(s[k] for s in self.samples) / len(self.samples) for k in (0, 1))
        if vx and self.kind == "walker" and self.horizontal_speeds:
            vx = (1 if vx > 0 else -1) * sum(self.horizontal_speeds) / len(self.horizontal_speeds)
        return vx, vy


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
                track.turns = [x - scroll for x in track.turns]
                if reversed_course and kind == "walker" and velocity[0] * dx < 0:
                    track.turns = (track.turns + [old.box[0] - scroll])[-4:]
                track.box = box
                track.samples = ([] if reversed_course else old.samples) + [(dx, dy)]
                track.samples = track.samples[-32:]
                # A turn changes direction, not patrol speed. Keep magnitude
                # evidence instead of forecasting a whole jump from two rounded
                # pixel displacements immediately after a reversal.
                track.horizontal_speeds = (track.horizontal_speeds + [abs(dx)])[-32:]
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
        # Explicit block/pipe boxes already provide the underside. Extending
        # a floating ledge down to an adjacent step invents a low ceiling.
        if any(
            abs(rect[1] - surface.top) <= 1 and rect[0] < surface.x1 and rect[2] > surface.x0
            for rect, _ in solids
        ):
            continue
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
    target: Track | None = None


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
    for frame in range(HORIZON + approach):
        old = (x, y, x + w, y + h)
        if frame == approach and approach and not grounded:
            return Prediction(False, False, float("inf"), steps, "lost_takeoff_support")
        jump = approach <= frame < approach + hold
        control = (
            approach_direction if frame < approach and approach_direction is not None else direction
        )
        dx, dy, _ = motion.advance(
            direction=control, jump=jump, grounded=grounded, y=y, run=control > 0
        )
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
                if frame < approach:
                    x += velocity[0]
                break
        body = (x, y, x + w, y + h)
        button = {
            (-1, True): 4,
            (1, True): 2,
            (0, True): 5,
            (-1, False): 3,
            (1, False): 1,
            (0, False): 0,
        }[control, jump]
        steps.append((body, copy(motion), button))
        for hazard in hazards:
            vx, vy = hazard.velocity or (0, 0)
            now = hazard.forecast(frame + 1)
            before = hazard.forecast(frame)
            if hazard.kind == "walker":
                now, before = walker_body(now), walker_body(before)
            vx, vy = now[0] - before[0], now[1] - before[1]
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
            # Walkers use the engine's discrete contact test. A swept near miss
            # cannot certify a stomp that never overlaps at a physical frame.
            if hazard.kind == "walker":
                collision = overlap(body, now)
            if collision:
                intended_stomp = (
                    hazard.kind == "walker"
                    and stomp_contact(body, now, motion.y_speed + motion.y_force / 256)
                    and (
                        hazard is target
                        if target is not None
                        else abs(goal[1] - now[1]) <= 8 and now[0] - 4 <= goal[0] <= now[2] + 4
                    )
                )
                if intended_stomp:
                    aim = (now[0] + now[2]) / 2 if target is not None else goal[0]
                    error = abs(x + w / 2 - aim)
                    # A forecast that just grazes a sprite edge is brittle to
                    # rounded pixels and subpixel patrol phase. Aim for overlap
                    # through at least half the smaller physical body.
                    return Prediction(
                        True,
                        error <= min(w, now[2] - now[0]) / 2,
                        error,
                        steps,
                        "stomp",
                        hazard,
                    )
                return Prediction(False, False, float("inf"), steps, "hazard")
        if frame < approach:
            if not landed:
                return Prediction(False, False, float("inf"), steps, "unsafe_approach")
            grounded = True
            continue
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
                    enemy = hazard.forecast(time)
                    if hazard.kind == "walker":
                        enemy = walker_body(enemy)
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
            return Prediction(
                True, target is None and error <= POSITION_TOLERANCE, error, steps, "landed"
            )
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
    target: Track | None = None
    approach: int = 0
    approach_direction: int | None = None
    seeking: bool = False

    def shift(self, scroll):
        self.goal = (self.goal[0] - scroll, self.goal[1])

    def press(self, scene):
        if scene is None or scene.mario.box is None:
            self.released = True
            self.done, self.status = True, "lost_observation"
            return int(SMBAction.NOOP)
        if self.seeking:
            return self.approach_press(scene)
        if (
            self.approach
            and self.elapsed <= self.approach
            and (not scene.mario.on_something or scene.mario.support == "air")
        ):
            # A scheduled takeoff cannot create ground contact. Cancel the
            # run-up/jump press, but retain airborne steering to the destination.
            # Dropping the maneuver here would discard its landing target.
            self.released = True
            self.approach = 0
            self.status = "lost_takeoff_support"
        approach = max(0, self.approach - self.elapsed)
        grounded = (
            self.elapsed <= self.approach
            and scene.mario.on_something
            and scene.mario.support != "air"
        )
        airborne_frames = max(0, self.elapsed - self.approach)
        remaining = max(0, self.hold - airborne_frames) if not self.released else 0
        box = takeoff_box(scene) if self.elapsed <= self.approach else scene.mario.box
        check = predict(
            scene,
            self.tracks.tracks,
            self.motion,
            box,
            self.goal,
            self.direction,
            remaining,
            grounded=grounded,
            target=self.target,
            approach=approach,
            approach_direction=self.approach_direction,
        )
        target_visible = self.target is None or any(t is self.target for t in self.tracks.tracks)
        if not check.safe or not check.reached:
            alternatives = []
            # Steering can change within this same maneuver. A released jump
            # stays released, including after a missing/unsafe observation.
            for direction in (-1, 0, 1):
                holds = (
                    range(max(0, 32 - airborne_frames) + 1)
                    if not self.released and target_visible
                    else {0, remaining, max(0, 32 - airborne_frames) if not self.released else 0}
                )
                for hold in holds:
                    candidate = predict(
                        scene,
                        self.tracks.tracks,
                        self.motion,
                        box,
                        self.goal,
                        direction,
                        hold,
                        grounded=grounded,
                        target=self.target,
                        approach=approach,
                        approach_direction=self.approach_direction,
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
                self.hold = airborne_frames + remaining
                self.status = "steering_correction"
            else:
                # A short emergency drop can look collision-free only because
                # it lands before the approaching enemy arrives. Never choose
                # an unrelated early landing or cut a committed jump blindly.
                self.status = "no_safe_continuation"
        if self.target is not None:
            if not target_visible:
                # A lost/ambiguous track is not a stomp and cannot be silently
                # replaced by another enemy. Continue the physical maneuver.
                self.status = "lost_target"
            elif check.reason == "stomp":
                contact = self.target.forecast(len(check.steps))
                self.goal = ((contact[0] + contact[2]) / 2, contact[1])
        self.prediction = check
        if check.steps:
            _, self.motion, button = check.steps[0]
        else:
            button = 0
        if not remaining:
            self.released = True
        self.elapsed += 1
        if self.elapsed >= HORIZON + self.approach:
            self.done, self.status = True, "flight_timeout"
        return int(button)

    def approach_press(self, scene):
        """Retain the destination while making a short, braking-safe approach."""
        from .tokens import SkillToken

        box = takeoff_box(scene)
        if self.target is not None and not any(t is self.target for t in self.tracks.tracks):
            self.done, self.status = True, "lost_target"
            return 0
        if not scene.mario.on_something or scene.mario.support == "air":
            self.done, self.status = True, "lost_takeoff_support"
            return 0
        destination = SkillToken(
            "jump", self.goal[0] - (box[0] + box[2]) / 2, self.goal[1] - box[3]
        )
        flight = plan_flight(
            scene,
            self.tracks,
            self.motion.x_speed / 16,
            destination,
            motion=self.motion,
            allow_approach=False,
            target=self.target,
        )
        if flight is not None:
            self.__dict__.update(flight.__dict__)
            return self.press(scene)
        direction = (destination.x > 0) - (destination.x < 0)
        options = []
        for control in dict.fromkeys((direction, 0, -direction)):
            motion = copy(self.motion)
            x, y, right, feet = box
            width = right - x
            steps = []
            safe = True
            solids, tops = geometry(scene, self.tracks.tracks)
            # Accelerate for at most four frames, then prove a stop on visible
            # support. Reobserve after just the first physical frame.
            for frame in range(32):
                command = control if frame < 4 else 0
                dx, _, _ = motion.advance(
                    direction=command, jump=False, grounded=True, y=y, run=command > 0
                )
                x += dx
                body = (x, y, x + width, feet)
                support = any(
                    r[0] < x + width and r[2] > x and abs(r[1] - feet) <= 1
                    for r, velocity in tops
                    if velocity == (0, 0)
                )
                blocked = any(overlap(body, r) for r, _ in solids)
                danger = any(
                    overlap(body, (r[0] - 8, r[1], r[2] + 8, r[3]))
                    for t in self.tracks.tracks
                    if t.kind != "platform"
                    for r in [t.forecast(frame + 1)]
                )
                if not support or blocked or danger:
                    safe = False
                    break
                steps.append((body, copy(motion), {-1: 3, 0: 0, 1: 1}[command]))
            if safe and steps:
                options.append((abs(self.goal[0] - x - width / 2), control == 0, steps))
        self.elapsed += 1
        if self.elapsed >= 96:
            self.done, self.status = True, "approach_timeout"
            return 0
        if not options:
            self.status = "waiting_for_intercept"
            self.motion.advance(direction=0, jump=False, grounded=True, y=box[1], run=False)
            return 0
        _, _, steps = min(options, key=lambda v: v[:2])
        _, self.motion, button = steps[0]
        self.prediction = Prediction(True, False, abs(destination.x), steps, "approaching")
        self.status = "approaching_target" if button else "waiting_for_intercept"
        return button


def plan_flight(
    scene,
    tracks,
    speed,
    destination,
    proposed=None,
    *,
    motion=None,
    allow_approach=True,
    target=None,
):
    box = takeoff_box(scene)
    goal = ((box[0] + box[2]) / 2 + destination.x, box[3] + destination.y)
    direction = (destination.x > 0) - (destination.x < 0)
    motion = (
        copy(motion)
        if motion is not None
        else NESPlayerMotion(
            x_speed=round(speed * 16),
            moving=(speed > 0) - (speed < 0),
            facing=1 if scene.mario.facing_right else -1,
        )
    )
    # Try the direct approach first; only expand to reversing maneuvers when
    # it cannot reach the destination. The landing delta is not the run-up.
    if target is None:
        possible = [
            t
            for t in tracks.tracks
            if t.kind == "walker"
            and abs(goal[1] - t.box[1]) <= 8
            and min(t.box[0], t.forecast(HORIZON)[0]) - 8
            <= goal[0]
            <= max(t.box[2], t.forecast(HORIZON)[2]) + 8
        ]
        if len(possible) == 1:
            target = possible[0]
    moving_intercept = target is not None and abs((target.velocity or (0, 0))[0]) > 0.1
    # For moving targets, advance the approach one observed frame at a time;
    # do not repeatedly enumerate long run-ups against an uncertain forecast.
    delays = (0,) if proposed is not None or moving_intercept else (0, *range(4, 65, 4))
    maneuvers = [(direction, direction)]
    if moving_intercept:
        maneuvers += [(d, d) for d in (0, -direction) if d != direction]
    if proposed is None and not moving_intercept:
        maneuvers += [(d, -d) for d in (direction, -direction) if d]
    for flight_direction, approach_direction in maneuvers:
        for delay in delays:
            if flight_direction != approach_direction and not delay:
                continue
            candidates = []
            for hold in range(1, 33):
                prediction = predict(
                    scene,
                    tracks.tracks,
                    motion,
                    box,
                    goal,
                    flight_direction,
                    hold,
                    grounded=True,
                    approach=delay,
                    approach_direction=approach_direction,
                    target=target,
                )
                contact = target.forecast(len(prediction.steps)) if target is not None else None
                uncertain_intercept = (
                    moving_intercept
                    and len(target.turns) < 2
                    and abs((contact[0] + contact[2]) / 2 - goal[0]) > POSITION_TOLERANCE
                )
                if prediction.safe and prediction.reached and not uncertain_intercept:
                    candidates.append(
                        (
                            round(prediction.error / 4),
                            (
                                abs(hold - proposed.frames)
                                if proposed is not None
                                else len(prediction.steps)
                            ),
                            hold,
                            prediction,
                        )
                    )
            if candidates:
                _, _, hold, prediction = min(candidates, key=lambda c: c[:3])
                target = prediction.target
                if target is not None and any(c[3].target is not target for c in candidates):
                    target = None
                return Flight(
                    goal,
                    motion,
                    flight_direction,
                    hold,
                    prediction,
                    tracks,
                    target=target,
                    approach=delay,
                    approach_direction=approach_direction,
                )
    if (
        proposed is None
        and allow_approach
        and any(
            t.kind == "walker"
            and abs(goal[1] - t.box[1]) <= 8
            and t.box[0] - 64 <= goal[0] <= t.box[2] + 64
            for t in tracks.tracks
        )
    ):
        return Flight(
            goal,
            motion,
            direction,
            0,
            Prediction(False, False, float("inf"), [], "approaching"),
            tracks,
            target=target,
            seeking=True,
        )
    return None
