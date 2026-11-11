"""A dynamics model identified from what is seen (interfaces.Dynamics), for
worlds of boxes moving in a plane: x to the right, y downward, solids as
boxes with a top to stand on.

Nothing here knows what any thing is. Each thing's motion is fitted to its
own recent places (its ``track``): how fast it moves, whether it speeds up
downward (it falls) or not (it flies), whether it rests on a solid, comes
out of one, or bounces. What the fit of one thing shows is remembered for
its ``type`` (any label the state source gives: things of one type are
taken to move alike), so the second thing of a type is predicted from its
first sighting: how hard that type falls, how fast it leaves the ground when
it bounces, how far below its box its feet are, how far out of a solid it
comes. The memory starts empty with each new model (one a play), with the
domain's priors for what has not been seen.

A thing needs ``box`` (x0, y0, x1, y1 now), ``vx`` and ``vy`` (the state
source's own estimate, used until its track says more), ``track`` (its
places as seen, oldest first, ending now: (x, y) of one point of it each
frame; None or short: not known yet), ``type`` and ``ref`` (which thing it
is, from frame to frame). A solid needs ``x0``, ``top``, ``x1``, ``bottom``
and ``kind``. Each thing is given ``path``: its box each frame from now.

How a thing moves on, by what is seen of it now:
- inside a solid (its box overlaps one) and not of a type seen to fall, or
  seen to come out of one: it passes through solids, straight on at its
  type's pace; rising, until it is out of the solid by as much as its type
  has been seen to come out (its own height, if never seen);
- resting (still vertically, a solid's top just under it): it moves along
  the top at its speed, turns back at a wall it walks into, and falls off
  the end under its type's gravity;
- in the air and moving up or down, or of a type seen to fall: it falls on
  until it lands on a top (rising, it passes through what is above it), and
  leaves the top again at its type's bounce speed if that type has been
  seen to bounce;
- in the air otherwise, or of a type seen to move on steadily inside
  solids: it flies straight on through everything.
A type seen to turn back often with no wall in its way (it paces to and
fro) is taken to move sideways at its track's average speed, not its
latest.
A track shorter than ``measured`` frames says nothing yet: the state
source's own estimate of its speed stands.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numba import njit

# A fit explains a stretch of a track if it is nowhere further off than
# this (pixels: places are whole pixels).
FIT = 1.0
STILL = 0.75
# The simplest motion (still, steady, speeding up) that explains at least
# this share of the longest stretch any of them explains is the one taken.
SHARE = 0.75
# A box this far into a solid on both axes (pixels), with this share of
# its area in it, is inside it (a thing landing on a corner is not).
INSIDE = 2.0
INSIDE_SHARE = 0.3
# Speeding up downward at least this much is falling (pixels a frame, a frame).
FALLS = 0.04
# A fall seen over at least this many frames tells its type's gravity; a
# thing still that long on a top, how far below its box its feet are (at the
# top of a slow hop it is still a few frames).
TOLD = 8
# A move down of at least this much, then RISES moves up of as much each,
# is a bounce (pixels a frame; a thing set back a pixel or two as it lands
# has not bounced).
BOUNCED = 1.0
RISES = 3
# A thing still for this many frames after coming out of a solid has come
# out as far as it does.
OUT_FRAMES = 4
# A steady move seen over at least this many frames tells its type's pace
# in and out of a solid; a thing moving on inside a solid that long tells
# that solids do not stop its type.
PACED = 6
# A steady fall seen over at least this many frames is the fastest its type
# falls (a slow fall looks steady over fewer).
FASTEST_FRAMES = 12
# A type whose sideways runs between turns with no wall in the way last
# fewer frames than this (the middle one of its last TURNS) paces to and fro.
PACES = 32
TURNS = 5
# A wall this near ahead of a thing that turned explains the turn (pixels).
WALL_NEAR = 6.0
# A falling thing lands on a top it is already this far below (pixels: the
# game lets a landing sink in a pixel or two before it sets it back).
SUNK = 3.0


@dataclass
class Seen:
    """What has been seen of the things of one type (None: not yet)."""

    gravity: Optional[float] = None  # speeding up downward, in the air
    fastest: Optional[float] = None  # the steady speed its fall ends at
    fell: float = 0.0  # the fastest it has been seen to fall
    bounce: Optional[float] = None  # speed upward as it leaves the ground it landed on
    feet: Optional[float] = None  # a top it rests on, below its box's bottom
    out: Optional[float] = None  # its box's bottom above a solid's top it came out of, once still
    through: bool = False  # seen moving on inside a solid: solids do not stop it
    runs: tuple = ()  # frames its last sideways runs lasted before a turn no wall explains

    @property
    def paces(self) -> bool:
        """Whether it turns back often with no wall in its way."""
        return len(self.runs) >= 2 and sorted(self.runs)[len(self.runs) // 2] < PACES
    pace: Optional[float] = None  # its steady speed up or down in and out of a solid (the most seen)


@njit(cache=True)
def _stretch(series, degree, tolerance):
    """The longest recent stretch of ``series`` (oldest first) that a
    polynomial of ``degree`` in time fits within ``tolerance``: (its
    frames, the speed now, the speeding up); 0 frames if none does."""
    n = series.shape[0]
    for w in range(n, degree + 1, -1):
        s = series[n - w:]
        t = np.arange(-(w - 1), 1).astype(np.float64)
        if degree == 0:
            speed, more = 0.0, 0.0
            fitted = np.full(w, s.mean())
        elif degree == 1:
            tm, sm = t.mean(), s.mean()
            speed = ((t - tm) * (s - sm)).sum() / ((t - tm) ** 2).sum()
            more = 0.0
            fitted = sm + speed * (t - tm)
        else:
            a = np.empty((w, 3))
            a[:, 0] = 1.0
            a[:, 1] = t
            a[:, 2] = t * t
            c = np.linalg.solve(a.T @ a, a.T @ s)
            speed, more = c[1], 2.0 * c[2]
            fitted = a @ c
        if np.abs(s - fitted).max() <= tolerance:
            return w, speed, more
    return 0, 0.0, 0.0


@njit(cache=True)
def _motion(series, curved):
    """(speed now, speeding up, frames it has been still, frames a steady
    speed was read over) of one axis of a track: the simplest of still,
    steady and (``curved``) speeding up that explains SHARE of the longest
    stretch any explains; (nan, 0, 0, 0) when the track is too short to
    say."""
    still, _, _ = _stretch(series, 0, STILL)
    steady, speed, _ = _stretch(series, 1, FIT)
    bent, bent_speed, more = (0, 0.0, 0.0)
    if curved:
        bent, bent_speed, more = _stretch(series, 2, FIT)
        if bent < 5:
            bent = 0
    longest = max(still, steady, bent)
    if bent and more < FALLS:
        # (a curve that is not a fall: a thing slowing to a stop, or noise)
        bent, longest = 0, max(still, steady)
    if still >= 2 and still >= SHARE * longest:
        return 0.0, 0.0, still, 0
    if steady >= 3 and steady >= SHARE * longest:
        return speed, 0.0, 0, steady
    if bent:
        return bent_speed, more, 0, 0
    return np.nan, 0.0, 0, 0


@njit(cache=True)
def _roll(things, solids, frames, wall_depth, sunk):
    """Each thing's box for ``frames`` frames. A row of ``things``: box (4),
    vx, vy, gravity, fastest fall, through (1: solids do not stop it),
    resting (1), feet (a top it stands on, below its box's bottom), bounce
    (speed upward off a landing; 0: it stays down), the highest and the
    lowest its top may go (through things: nan, no end)."""
    out = np.empty((things.shape[0], frames + 1, 4))
    for e in range(things.shape[0]):
        x0, y0, x1, y1 = things[e, 0], things[e, 1], things[e, 2], things[e, 3]
        vx, vy, gravity, fastest = things[e, 4], things[e, 5], things[e, 6], things[e, 7]
        through, resting, feet, bounce = things[e, 8] > 0, things[e, 9] > 0, things[e, 10], things[e, 11]
        highest, lowest = things[e, 12], things[e, 13]
        out[e, 0] = (x0, y0, x1, y1)
        for k in range(1, frames + 1):
            if through:
                dy = vy
                if not np.isnan(highest) and y0 + dy < highest:
                    dy, vy = min(0.0, highest - y0), 0.0
                if not np.isnan(lowest) and y0 + dy > lowest:
                    dy, vy = max(0.0, lowest - y0), 0.0
                x0, x1, y0, y1 = x0 + vx, x1 + vx, y0 + dy, y1 + dy
                out[e, k] = (x0, y0, x1, y1)
                continue
            nx0, nx1 = x0 + vx, x1 + vx
            if vx != 0.0:  # a wall (a solid beside it, rising above its feet) turns it back
                for i in range(solids.shape[0]):
                    if (nx0 < solids[i, 2] and solids[i, 0] < nx1 and solids[i, 1] < y1 - wall_depth
                            and y0 < solids[i, 3] and not (x0 < solids[i, 2] and solids[i, 0] < x1)):
                        vx = -vx
                        nx0, nx1 = x0 + vx, x1 + vx
                        break
            dy = 0.0
            if resting:
                under = False
                for i in range(solids.shape[0]):
                    if nx0 < solids[i, 2] and solids[i, 0] < nx1 and abs(solids[i, 1] - (y1 + feet)) <= 1.0:
                        under = True
                        break
                if not under:
                    resting, vy = False, 0.0  # off its end: it falls
            if not resting:
                vy = min(vy + gravity, fastest)
                dy = vy
                if vy > 0:
                    low = y1 + feet
                    for i in range(solids.shape[0]):
                        if nx0 < solids[i, 2] and solids[i, 0] < nx1 and low - sunk <= solids[i, 1] <= low + vy:
                            dy = solids[i, 1] - low
                            if bounce > 0:
                                vy = -bounce
                            else:
                                resting, vy = True, 0.0
                            break
            x0, x1, y0, y1 = nx0, nx1, y0 + dy, y1 + dy
            out[e, k] = (x0, y0, x1, y1)
    return out


class ObservedDynamics:
    """The model (module notes). The priors, for what has not been seen:
    ``gravity`` and ``fastest`` (how a thing falls: pixels a frame gained
    each frame, and at most), ``reach`` (how far from a box's bottom the top it
    rests on may be), ``wall_depth`` (a side this far below a solid's top
    is a wall), ``measured`` (frames of a track before it is believed over
    the state source's estimate), ``ignored`` (kinds of solid that stop
    none of these things)."""

    def __init__(self, gravity: float = 0.5, fastest: float = 4.5, reach: float = 6.0, wall_depth: float = 2.0,
                 measured: int = 8, ignored: Sequence[str] = ()):
        self.gravity, self.fastest, self.reach, self.wall_depth = gravity, fastest, reach, wall_depth
        self.measured, self.ignored = measured, tuple(ignored)
        self.seen: dict = {}  # type -> Seen
        self.came_out: dict = {}  # ref -> the top of the solid it was seen coming out of (it goes back in there)
        self.inside: dict = {}  # ref -> frames it has been inside a solid
        self.heading: dict = {}  # ref -> [the way it moves sideways (1, -1), frames it has]

    def _solids(self, solids: Sequence) -> np.ndarray:
        kept = [(s.x0, s.top, s.x1, s.bottom) for s in solids if s.kind not in self.ignored]
        return np.array(kept, np.float64).reshape(-1, 4)

    def _track(self, thing) -> Optional[np.ndarray]:
        """``thing``'s track if it is long enough to believe."""
        if thing.track is None or len(thing.track) < self.measured:
            return None
        return np.asarray(thing.track, np.float64).reshape(-1, 2)

    def _read(self, thing, solids: np.ndarray) -> tuple:
        """What ``thing``'s track and the solids say of it now: (vx, vy,
        speeding up downward, frames still vertically, frames its steady
        vertical speed was read over, the solid it is inside or -1, the top
        it rests on or None)."""
        vx, vy, more, still, steady = thing.vx, thing.vy, 0.0, 0, 0
        track = self._track(thing)
        if track is not None:
            speed, _, _, _ = _motion(track[:, 0].copy(), False)
            if not np.isnan(speed):
                vx = speed
            speed, more, still, steady = _motion(track[:, 1].copy(), True)
            if not np.isnan(speed):
                vy = speed
            if (self.seen.get(thing.type) or Seen()).paces:
                vx = (track[-1, 0] - track[0, 0]) / (len(track) - 1)
        x0, y0, x1, y1 = thing.box
        inside, deepest = -1, 0.0
        for i, (sx0, top, sx1, bottom) in enumerate(solids):
            across, down = min(x1, sx1) - max(x0, sx0), min(y1, bottom) - max(y0, top)
            if min(across, down) >= INSIDE and across * down >= INSIDE_SHARE * (x1 - x0) * (y1 - y0) and (
                    across * down > deepest):
                inside, deepest = i, across * down
        rests = None
        if inside < 0 and vy == 0.0:
            # (the top nearest its box's bottom: some boxes reach below their feet)
            tops = [top for sx0, top, sx1, _ in solids if x0 < sx1 and sx0 < x1 and -self.reach <= top - y1 <= self.reach]
            rests = min(tops, key=lambda top: abs(top - y1)) if tops else None
        return vx, vy, more, still, steady, inside, rests

    def observe(self, things: Sequence, solids: Sequence) -> None:
        """Remember for each thing's type what its track shows now."""
        packed = self._solids(solids)
        present = set()
        for thing in things:
            present.add(thing.ref)
            seen = self.seen.setdefault(thing.type, Seen())
            vx, vy, more, still, steady, inside, rests = self._read(thing, packed)
            x0, y0, x1, y1 = thing.box
            # A thing in a solid that neither falls nor is of a type seen to
            # (a hop passes up through the blocks above it) is passing
            # through it.
            passing = inside >= 0 and more < FALLS and seen.gravity is None
            if passing:
                self.inside[thing.ref] = self.inside.get(thing.ref, 0) + 1
                if self.inside[thing.ref] >= PACED and (vx != 0.0 or vy != 0.0):
                    seen.through = True
                if vy < 0 or self._track(thing) is None:  # (first seen in a solid: it is coming out of it)
                    self.came_out[thing.ref] = packed[inside, 1]
            else:
                self.inside.pop(thing.ref, None)
                if inside < 0 and thing.ref in self.came_out and still >= OUT_FRAMES:
                    seen.out = self.came_out[thing.ref] - y1
            self._turns(thing, seen, packed)
            passing = passing or thing.ref in self.came_out
            if passing and vy != 0.0 and steady >= PACED:
                # (its full speed: slowing to a stop it shows less)
                seen.pace, seen.through = max(seen.pace or 0.0, abs(vy)), True
            if rests is not None and still >= TOLD and not passing:
                seen.feet = rests - y1
            track = self._track(thing)
            if track is None or passing or seen.through:
                continue
            if rests is None and more >= FALLS:
                frames, _, _ = _stretch(track[:, 1].copy(), 2, FIT)
                if frames >= TOLD:
                    seen.gravity = more
            if rests is None and seen.gravity is not None:
                seen.fell = max(seen.fell, vy)
                if steady >= FASTEST_FRAMES and vy >= 1.0:
                    seen.fastest = vy
            # A bounce: down, then up at once and on. Where it turned tells
            # how far below its box its feet are; the rise since (from the
            # frame after: the turn itself can set it back a pixel or two),
            # how fast it left.
            moves = np.diff(track[:, 1])
            for k in range(len(moves) - RISES, 0, -1):
                if moves[k - 1] >= BOUNCED and (moves[k:k + RISES] <= -BOUNCED).all():
                    low = track[k, 1] + (y1 - y0)  # its box's bottom where it turned
                    at = track[k, 0] - track[-1, 0]
                    tops = [top for sx0, top, sx1, _ in packed
                            if x0 + at < sx1 and sx0 < x1 + at and -self.reach <= top - low <= self.reach]
                    if tops:
                        seen.feet = min(tops, key=lambda top: abs(top - low)) - low
                    rise = track[k + 1:, 1].copy()
                    frames, speed, gained = _stretch(rise, 2, FIT)
                    if frames == len(rise) >= TOLD:
                        seen.bounce = -(speed - gained * frames)
                    break
        for kept in (self.came_out, self.inside, self.heading):
            for ref in [ref for ref in kept if ref not in present]:
                del kept[ref]

    def _turns(self, thing, seen: Seen, solids: np.ndarray) -> None:
        """Note the run a thing ended if it has turned back with no wall
        ahead of it (its latest sideways speed, whatever its type)."""
        track = self._track(thing)
        if track is None:
            return
        speed, _, _, _ = _motion(track[:, 0].copy(), False)
        if np.isnan(speed) or speed == 0.0:
            return
        way = 1 if speed > 0 else -1
        heading = self.heading.setdefault(thing.ref, [way, 0])
        if heading[0] == way:
            heading[1] += 1
            return
        x0, y0, x1, y1 = thing.box
        walled = any(
            top < y1 - self.wall_depth and y0 < bottom
            and (0.0 <= sx0 - x1 <= WALL_NEAR if heading[0] > 0 else 0.0 <= x0 - sx1 <= WALL_NEAR)
            for sx0, top, sx1, bottom in solids
        )
        if not walled:
            seen.runs = (seen.runs + (heading[1],))[-TURNS:]
        self.heading[thing.ref] = [way, 1]

    def _prior_gravity(self) -> float:
        """How a type not yet seen to fall is taken to: as the types seen
        to do (their middle one), or the prior."""
        known = sorted(seen.gravity for seen in self.seen.values() if seen.gravity is not None)
        return known[len(known) // 2] if known else self.gravity

    def paths(self, things: Sequence, solids: Sequence, frames: int) -> None:
        if not things:
            return
        packed = self._solids(solids)
        rows = np.empty((len(things), 14))
        for row, thing in zip(rows, things):
            seen = self.seen.get(thing.type) or Seen()
            vx, vy, more, _, steady, inside, rests = self._read(thing, packed)
            x0, y0, x1, y1 = thing.box
            feet = seen.feet if seen.feet is not None else 0.0
            gravity, through, resting, highest = 0.0, True, False, np.nan
            falls = more >= FALLS or seen.gravity is not None
            came_out = packed[inside, 1] if inside >= 0 and not falls else self.came_out.get(thing.ref)
            passing = came_out is not None or (seen.through and rests is None)
            if passing and seen.pace is not None and (vy != 0.0 or (inside >= 0 and self._track(thing) is None)):
                # just begun to move, or just seen: at its type's pace (out, if not seen to go in)
                vy = seen.pace if vy > 0 else -seen.pace
            if came_out is not None:
                if vy < 0:  # coming out of a solid: as far as its type does
                    out = seen.out if seen.out is not None else y1 - y0
                    highest = came_out - out - (y1 - y0)
            elif rests is not None:
                through, resting, feet = False, True, rests - y1
                gravity = seen.gravity if seen.gravity is not None else self._prior_gravity()
            elif not seen.through and (vy != 0.0 or falls):
                through = False
                gravity = more if more >= FALLS else seen.gravity if seen.gravity is not None else self._prior_gravity()
                if more < FALLS and steady:  # a steady speed read over frames it was gaining speed: their middle's
                    vy += gravity * (steady - 1) / 2
            fastest = max(seen.fastest if seen.fastest is not None else self.fastest, seen.fell)
            row[:] = (x0, y0, x1, y1, vx, vy, gravity, fastest, through, resting, feet,
                      (seen.bounce or 0.0) if not through else 0.0, highest, np.nan)
        for thing, path in zip(things, _roll(rows, packed, frames, self.wall_depth, SUNK)):
            thing.path = path
