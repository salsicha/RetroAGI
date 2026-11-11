"""Enemies and moving platforms followed from picture to picture.

Vision reports boxes only. VisualTracks keeps one Track per thing seen, with
its identity and its velocity (the mean of its last moves in world pixels:
the screen's move plus the camera's scroll), through frames in which vision
misses it.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


def translated(box, dx=0, dy=0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


# The executor's first-order state estimate of each tracked sprite (user,
# 2026-10-16: vision reports boxes only; the executor estimates position and
# velocity, assuming constant velocity, and the world model and the collision
# trackers read its estimates). Velocity is the mean world movement per frame
# over the last ESTIMATE_SAMPLES sightings: boxes move by whole pixels, so a
# walker at 0.3 pixels a frame moves 0 or 1 between two frames (its last
# movement continued for 64 frames was 21 pixels off on average, the mean of
# 16 sightings 3). A reversal starts the mean again.
ESTIMATE_SAMPLES = 16
# A track not seen in a frame (vision missed it, or it crossed another) is
# kept this many frames, moved by its velocity, and found again near there
# with its identity and velocity.
LOST_FRAMES = 8


@dataclass
class Track:
    box: tuple
    kind: str
    samples: list = field(default_factory=list)
    identity: int = 0
    # Frames since it was last seen (0: seen this frame).
    missed: int = 0
    # Its object type as vision last told it (EnemyView.type; -1 not told).
    type: int = -1

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
    # Training only: exact(track, frames) -> where the simulator will have
    # moved the walker the track follows, ``frames`` from now (a box in the
    # scene's pixels), or None. A jump aimed at a walker then lands where it
    # will be (aimed_goal): the ground truth a prediction will stand for
    # (the target tracker's collision point; user, 2026-10-14/15). None when
    # the agent plays without the teacher.
    exact: Optional[Callable] = None
    # Tracks not seen this frame, for up to LOST_FRAMES (Track.missed).
    lost: list = field(default_factory=list)

    def observe(self, scene, scroll, reckoned=False):
        """The tracks after seeing ``scene``, the camera having moved
        ``scroll`` pixels (None: unknown); ``reckoned``: that scroll was
        dead-reckoned, a fraction off the game's whole-pixel step, so one
        frame's move against a track's course is no turn (it reset every
        track's speed on 5-1's bare floor)."""
        objects = [(e.box, e.kind, e.type) for e in scene.enemies if e.kind != "defeated"]
        objects += [(b, "platform", -1) for b in scene.moving_platforms]
        # Tracks unseen since an earlier frame move on by their velocity (and
        # the scroll) to where they should be now.
        lost = []
        if scroll is not None:
            for old in self.lost:
                vx, vy = old.velocity or (0.0, 0.0)
                old.box = translated(old.box, vx - scroll, vy)
                lost.append(old)
        found = set()
        updated, used = [], set()
        for box, kind, seen_type in objects:
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
                    for other, other_kind, _ in objects
                    if other_kind == kind
                    and abs(other[0] - old.box[0] + scroll) <= 8
                    and abs(other[1] - old.box[1]) <= 8
                )
                if len(competitors) > 1 and competitors[1] - competitors[0] <= 1:
                    track.type = seen_type
                    updated.append(track)
                    continue
                if abs(dx) + abs(dy) != competitors[0]:
                    track.type = seen_type
                    updated.append(track)
                    continue
                used.add(i)
                velocity = old.velocity
                reversed_course = not reckoned and velocity is not None and any(
                    abs(v) > 0.1 and delta * v < 0 for delta, v in zip((dx, dy), velocity)
                )
                # Preserve this visual identity across motion and reversals.
                # Flights may hold a reference to this particular walker.
                track = old
                track.box = box
                track.samples = ([] if reversed_course else old.samples) + [(dx, dy)]
                track.samples = track.samples[-ESTIMATE_SAMPLES:]
            elif not matches:
                track = self._found_again(box, kind, lost, found) or track
            track.type = seen_type
            updated.append(track)
        if scroll is None:
            self.lost = []
        else:
            # Seen last frame but not now: kept where they should be next.
            missing = [t for i, t in enumerate(self.tracks) if i not in used]
            for old in missing:
                vx, vy = old.velocity or (0.0, 0.0)
                old.box = translated(old.box, vx - scroll, vy)
            kept = []
            for old in missing + [t for t in lost if t.identity not in found]:
                old.missed += 1
                if old.missed <= LOST_FRAMES:
                    kept.append(old)
            self.lost = kept
        self.tracks = updated

    def _found_again(self, box, kind, lost, found):
        """The lost track this unmatched sighting continues (one whose
        predicted box is within 8 pixels, unambiguously), with its identity
        and velocity, the movement over the gap added as one sample; or
        None."""
        near = sorted(
            (abs(box[0] - old.box[0]) + abs(box[1] - old.box[1]), j)
            for j, old in enumerate(lost)
            if old.kind == kind
            and old.identity not in found
            and abs(box[0] - old.box[0]) <= 8
            and abs(box[1] - old.box[1]) <= 8
        )
        if not near or (len(near) > 1 and near[1][0] - near[0][0] <= 1):
            return None
        old = lost[near[0][1]]
        vx, vy = old.velocity or (0.0, 0.0)
        gap = old.missed + 1
        sample = (vx + (box[0] - old.box[0]) / gap, vy + (box[1] - old.box[1]) / gap)
        old.samples = (old.samples + [sample])[-ESTIMATE_SAMPLES:]
        old.box, old.missed = box, 0
        found.add(old.identity)
        return old

    @property
    def ready(self):
        return all(len(t.samples) >= 2 for t in self.tracks)
