"""Destination following with Mario motion prediction, never world collision checks.

Vision supplies poses and target identities. The controller predicts only Mario's
response to buttons; terrain, enemies and edge safety are the skill's responsibility.
"""

from copy import copy
from dataclasses import dataclass, field

from .smb_physics import NESPlayerMotion

HORIZON = 96
# Whether a flight re-predicts its path every frame (searching the steering and
# the remaining hold that best reach the goal from what vision now shows). Off
# for Block SMB training (user decision 2026-10-10): a flight then plays the
# hold and steering chosen at takeoff (_best_hold), then keeps steering toward
# the goal. The per-frame search was half the cost of every teacher trial.
REPLAN_IN_FLIGHT = False
POSITION_TOLERANCE = 1


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

    def landing_box(self):
        """Where a walker will be when the coming action (this jump) ends: its
        memory forecast once trusted (rebased by what has since been seen),
        else where it is now."""
        if self.kind == "walker":
            forecast = self.distant_forecast(1)
            if forecast is not None:
                return forecast[1]
        return self.box

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
    for frame in range(HORIZON):
        jump = frame < hold
        control = direction
        dx, dy, _ = motion.advance(
            direction=control,
            jump=jump,
            grounded=grounded and frame == 0,
            y=y,
            run=control > 0,
        )
        x += dx
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
        # Only the first step's motion is read (Flight.press); copying every
        # step's dominated the cost of a prediction.
        steps.append((body, copy(motion) if frame == 0 else None, button))
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

    def shift(self, scroll):
        self.goal = (self.goal[0] - scroll, self.goal[1])

    def press(self, scene):
        if scene is None or scene.mario.box is None:
            self.released = True
            self.done, self.status = True, "lost_observation"
            return 0
        if not REPLAN_IN_FLIGHT:
            return self._planned_button()
        if self.target is not None:
            if any(t is self.target for t in self.tracks.tracks):
                b = self.target.landing_box()
                self.goal = (
                    (b[0] + b[2]) / 2 + self.target_offset[0],
                    b[1] + self.target_offset[1],
                )
            else:
                # Continue toward the last requested point; never acquire a neighbour.
                self.target = None
                self.status = "lost_target"
        observed_ground = scene.mario.on_something and scene.mario.support != "air"
        remaining = 0 if self.released else max(0, self.hold - self.elapsed)
        grounded = observed_ground and self.elapsed == 0
        check = predict(
            scene,
            [],
            self.motion,
            scene.mario.box,
            self.goal,
            self.direction,
            remaining,
            grounded=grounded,
        )
        if not check.reached:
            options = []
            holds = (
                (0,)
                if self.released
                else ((remaining,) if grounded else tuple(range(max(0, 32 - self.elapsed) + 1)))
            )
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
            self.hold = self.elapsed + remaining
        if not remaining:
            self.released = True
        self.prediction = check
        _, self.motion, button = check.steps[0]
        self.elapsed += 1
        if self.elapsed >= HORIZON:
            self.done, self.status = True, "flight_timeout"
        return button

    def _planned_button(self):
        """The takeoff plan's button for this frame (open loop): the chosen hold
        and steering, then steering toward the goal with the button released."""
        steps = self.prediction.steps
        if self.elapsed < len(steps):
            button = steps[self.elapsed][2]
        else:
            button = {-1: 3, 0: 0, 1: 1}[self.direction]
        self.elapsed += 1
        if self.elapsed >= self.hold:
            self.released = True
        if self.elapsed >= HORIZON:
            self.done, self.status = True, "flight_timeout"
        return button


def _best_hold(scene, motion, box, goal, direction):
    """The jump-button hold (1-32 frames) whose predicted flight best reaches ``goal``."""
    candidates = []
    for hold in range(1, 33):
        prediction = predict(scene, [], motion, box, goal, direction, hold, grounded=True)
        candidates.append(
            (
                not prediction.reached,
                round(prediction.error / 4),
                len(prediction.steps),
                hold,
                prediction,
            )
        )
    _, _, _, hold, prediction = min(candidates, key=lambda c: c[:4])
    return hold, prediction


def hold_paths(scene, motion, box, direction):
    """Mario's predicted body box (x0, y0, x1, y1) every frame of the jump
    horizon for each hold of 1 to 32 frames, steering ``direction``: a list
    of 32 arrays [frames, 4], the path of a jump replayed open loop."""
    import numpy as np

    far = (0.0, -1e9)  # never crossed: the whole horizon
    return [
        np.array(
            [
                b
                for b, _, _ in predict(
                    scene, [], motion, box, far, direction, hold, grounded=True
                ).steps
            ],
            dtype=float,
        )
        for hold in range(1, 33)
    ]


def best_holds(scene, motion, box, goals, direction, paths=None):
    """_best_hold for many goals at once: [(hold, reached, error)] in the order
    of ``goals`` ((x, y) points), exactly as _best_hold would choose each
    (``paths``: hold_paths for the same start, if already predicted).

    A hold's predicted path does not depend on the goal (predict reads the
    goal only to stop at its height and to score the attempt), so each of the
    32 paths is predicted once and scored for every goal.
    """
    import numpy as np

    gx = np.array([g[0] for g in goals], dtype=float)
    gy = np.array([g[1] for g in goals], dtype=float)
    keys, chosen = [], []
    paths = paths if paths is not None else hold_paths(scene, motion, box, direction)
    for hold, bodies in enumerate(paths, start=1):
        steps = bodies
        center = (bodies[:, 0] + bodies[:, 2]) / 2
        feet = bodies[:, 3]
        tops = np.concatenate(([box[1]], bodies[:, 1]))
        dy = np.diff(tops)
        before = np.concatenate(([box[3]], feet[:-1]))
        error = np.abs(center[:, None] - gx[None, :]) + 2 * np.abs(feet[:, None] - gy[None, :])
        crossing = (
            (dy[:, None] >= 0) & (before[:, None] <= gy[None, :]) & (gy[None, :] <= feet[:, None])
        )
        crossed = crossing.any(axis=0)
        first = np.where(crossed, crossing.argmax(axis=0), len(steps) - 1)
        at = error[first, np.arange(len(goals))]
        best_so_far = np.minimum.accumulate(error, axis=0)[first, np.arange(len(goals))]
        reached = crossed & (at <= POSITION_TOLERANCE)
        score = np.where(crossed, at, best_so_far)
        length = np.where(crossed, first + 1, len(steps))
        keys.append(np.stack([~reached, np.round(score / 4), length, np.full(len(goals), hold)]))
        chosen.append((reached, score))
    keys = np.stack(keys)  # [hold, 4, goal]
    out = []
    for g in range(len(goals)):
        k = min(range(32), key=lambda h: tuple(keys[h, :, g]))
        out.append((k + 1, bool(chosen[k][0][g]), float(chosen[k][1][g])))
    return out


def plan_flight(scene, tracks, speed, destination, proposed=None, *, motion=None):
    """Choose an immediate takeoff; skill must request any preparation as a run."""
    box = scene.mario.box
    goal = ((box[0] + box[2]) / 2 + destination.x, box[3] + destination.y)
    motion = copy(motion) if motion is not None else NESPlayerMotion(x_speed=round(speed * 16))
    direction = (destination.x > 0) - (destination.x < 0)
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
    if target is not None:
        # A walker's landing is planned where the memory forecasts it will be
        # when this jump ends (Track.landing_box), not where it stands now.
        b = target.landing_box()
        goal = ((b[0] + b[2]) / 2 + offset[0], b[1] + offset[1])
    hold, prediction = _best_hold(scene, motion, box, goal, direction)
    return Flight(
        goal,
        motion,
        direction,
        hold,
        prediction,
        tracks,
        target=target,
        target_offset=offset,
    )
