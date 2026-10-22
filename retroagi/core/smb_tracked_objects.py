"""The list of moving things the system keeps (user, 2026-10-21): every enemy
and moving platform vision has reported, up to CAPACITY of them, those out of
sight kept where they vanished.

- An entry is updated every frame its thing is seen (the executor's track
  identity says which one it is); out of sight it stays where it was last
  seen, in the world (it moves on screen only with the camera), with the
  velocity it was last estimated to have and how long it has been unseen.
  There is no time limit.
- A sighting under a new identity of the same kind within REFIND pixels of
  where an entry out of sight vanished is that entry found again (a plant up
  out of its pipe, an enemy back on screen); the nearest, unambiguously.
- Vision showing an enemy defeated where an entry out of sight vanished ends
  that entry.
- A full list makes room for a new thing by dropping the entry seen longest
  ago; one seen this frame is never dropped.
- Each entry keeps its row from when it joins until it leaves, so whatever
  reads the list sees the same thing in the same row from frame to frame.

The world model (core/world_model) reads the list in place of vision's enemy
and platform tokens, and the collision trackers follow every enemy on it.
The list is built from the numbers each frame records (the executor's
kinematics, the sprite slots' identities, what vision shows of the enemies,
the scene row, where Mario is), rounded as the recordings keep them, so the
list made at play and the one made again from a recording are the same
(sightings, record_objects).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .smb_kinematics import CONTEXT, ENEMY_SLOTS, SPRITE_SLOTS, SPRITE_STATE, TRACKED
from .smb_observer import row_objects

CAPACITY = 20
# A sighting of a thing's kind this near (pixels, its middle and top) to where
# an entry out of sight vanished is that entry found again.
REFIND = 8
# How long an entry has been out of sight, as read: log(1 + frames) /
# log(1 + UNSEEN_SCALE): 0 seen now, 1 after UNSEEN_SCALE frames, on past it
# (no limit, and no scale to change if the memory changes).
UNSEEN_SCALE = 256
# Each row: the trackers' TRACKED numbers, and whether it is a moving
# platform (the trackers follow the enemies only).
FIELDS = TRACKED + ("platform",)
COLUMN = {name: k for k, name in enumerate(FIELDS)}
ENEMY_KINDS = CONTEXT[: CONTEXT.index("other") + 1]


def unseen_value(frames: int) -> float:
    return math.log1p(frames) / math.log1p(UNSEEN_SCALE)


@dataclass
class Sighting:
    """One thing as vision reports it this frame (sightings)."""

    identity: int  # the executor's track identity
    kind: str  # an enemy kind (ENEMY_KINDS) or "platform"
    place: tuple  # its middle and top, in the world (pixels)
    velocity: tuple  # the executor's estimate (pixels a frame)
    seen: float  # how many sightings the estimate rests on (SPRITE_STATE "seen")
    context: np.ndarray  # [len(CONTEXT)]: what vision shows of it (kind, size, surface, type)
    slot: int  # its sprite slot this frame
    truth: int = -1  # recordings: the true thing it shows (-1 unknown)


@dataclass
class Entry:
    key: int  # the entry's own identity, kept when it is found again
    sighting: Sighting  # as last seen
    last_seen: int  # the frame it was last seen


@dataclass
class TrackedObjects:
    capacity: int = CAPACITY
    rows: list = field(default_factory=list)
    frame: int = 0
    next_key: int = 1

    def __post_init__(self):
        if not self.rows:
            self.rows = [None] * self.capacity

    def entries(self):
        return [e for e in self.rows if e is not None]

    def update(self, sightings, defeated=()) -> None:
        """One frame: ``sightings`` (Sighting), and where vision shows an
        enemy defeated (``defeated``: its middle and bottom in the world)."""
        self.frame += 1
        by_identity = {e.sighting.identity: e for e in self.entries()}
        new = []
        for s in sightings:
            entry = by_identity.get(s.identity)
            if entry is not None and entry.last_seen != self.frame:
                entry.sighting, entry.last_seen = s, self.frame
            else:
                new.append(s)
        for s in new:
            entry = self._found_again(s)
            if entry is not None:
                entry.sighting, entry.last_seen = s, self.frame
            else:
                self._add(s)
        for x, bottom in defeated:
            for i, entry in enumerate(self.rows):
                if (
                    entry is not None
                    and entry.last_seen < self.frame
                    and entry.sighting.kind != "platform"
                    and abs(entry.sighting.place[0] - x) <= REFIND
                    and abs(entry.sighting.place[1] + self._height(entry) - bottom) <= REFIND
                ):
                    self.rows[i] = None

    @staticmethod
    def _height(entry) -> float:
        return float(entry.sighting.context[CONTEXT.index("height")]) * 16

    def _found_again(self, s: Sighting) -> Optional[Entry]:
        near = sorted(
            (abs(e.sighting.place[0] - s.place[0]) + abs(e.sighting.place[1] - s.place[1]), e.key, e)
            for e in self.entries()
            if e.last_seen < self.frame
            and e.sighting.kind == s.kind
            and abs(e.sighting.place[0] - s.place[0]) <= REFIND
            and abs(e.sighting.place[1] - s.place[1]) <= REFIND
        )
        if not near or (len(near) > 1 and near[1][0] - near[0][0] <= 1):
            return None
        return near[0][2]

    def _add(self, s: Sighting) -> None:
        free = [i for i, e in enumerate(self.rows) if e is None]
        if free:
            row = free[0]
        else:
            # Full: the entry seen longest ago makes room (never one seen now).
            out_of_sight = [i for i, e in enumerate(self.rows) if e.last_seen < self.frame]
            if not out_of_sight:
                return
            row = min(out_of_sight, key=lambda i: (self.rows[i].last_seen, self.rows[i].key))
        self.rows[row] = Entry(self.next_key, s, self.frame)
        self.next_key += 1

    def key_of(self, identity: int) -> int:
        """The key of the entry the executor's track ``identity`` is seen as
        now (0: none)."""
        if not identity:
            return 0
        for e in self.entries():
            if e.sighting.identity == identity and e.last_seen == self.frame:
                return e.key
        return 0

    def read(self, position) -> tuple:
        """The list as read now, Mario at ``position`` (his world displacement
        and feet row): ([capacity, len(FIELDS)] rows, [capacity] keys (0
        empty), [capacity] true things (recordings; -1 unknown))."""
        rows = np.zeros((self.capacity, len(FIELDS)), np.float32)
        keys = np.zeros(self.capacity, np.int32)
        truths = np.full(self.capacity, -1, np.int32)
        for i, entry in enumerate(self.rows):
            if entry is None:
                continue
            s = entry.sighting
            row = rows[i]
            row[COLUMN["present"]] = 1.0
            row[COLUMN["x"]] = s.place[0] - position[0]
            row[COLUMN["y"]] = s.place[1] - position[1]
            row[COLUMN["vx"]], row[COLUMN["vy"]] = s.velocity
            row[COLUMN["seen"]] = s.seen
            row[COLUMN["unseen"]] = unseen_value(self.frame - entry.last_seen)
            row[COLUMN["walker"] : COLUMN["walker"] + len(CONTEXT)] = s.context
            row[COLUMN["platform"]] = float(s.kind == "platform")
            keys[i] = entry.key
            truths[i] = s.truth
        return rows, keys, truths


def _kept(values):
    """``values`` as the recordings keep them (float16)."""
    return np.asarray(values, np.float32).astype(np.float16).astype(np.float32)


def sightings(sprites, identities, context, c_row, position, truth=None) -> tuple:
    """This frame's sightings and defeated enemies (TrackedObjects.update)
    from the numbers a frame records: the executor's kinematics per sprite
    slot ([SPRITE_SLOTS, len(SPRITE_STATE)]), the slots' track identities,
    what vision shows of each enemy slot ([ENEMY_SLOTS, len(CONTEXT)]; eight
    numbers in recordings from before object types), the scene row
    (smb_observer.pack_c), Mario's position (world displacement, feet row)
    and, in recordings, each slot's true thing."""
    sprites, c_row = _kept(sprites), _kept(c_row)
    position = _kept(position)
    context = _kept(context)
    if context.shape[-1] < len(CONTEXT):
        context = np.concatenate(
            (context, np.zeros(context.shape[:-1] + (len(CONTEXT) - context.shape[-1],), np.float32)), -1
        )
    scene = row_objects(c_row)
    x, y = SPRITE_STATE.index("x"), SPRITE_STATE.index("y")
    vx, vy, seen = (SPRITE_STATE.index(n) for n in ("vx", "vy", "seen"))
    out = []
    enemies = {slot: (box, kind, kind_type) for slot, box, kind, kind_type in scene["enemies"]}
    platforms = dict(scene["moving_platforms"])
    for slot in range(SPRITE_SLOTS):
        identity = int(identities[slot])
        if not identity or sprites[slot, 0] <= 0:
            continue
        if slot < ENEMY_SLOTS:
            if slot not in enemies:
                continue
            box, kind, kind_type = enemies[slot]
            row = context[slot].copy()
            if kind_type >= 0:
                row[CONTEXT.index(f"type_{kind_type}")] = 1.0
        else:
            box = platforms.get(slot - ENEMY_SLOTS)
            if box is None:
                continue
            kind = "platform"
            row = np.zeros(len(CONTEXT), np.float32)
            row[CONTEXT.index("width")] = (box[2] - box[0]) / 16
            row[CONTEXT.index("height")] = (box[3] - box[1]) / 16
        out.append(
            Sighting(
                identity=identity,
                kind=kind,
                place=(float(sprites[slot, x] + position[0]), float(sprites[slot, y] + position[1])),
                velocity=(float(sprites[slot, vx]), float(sprites[slot, vy])),
                seen=float(sprites[slot, seen]),
                context=row,
                slot=slot,
                truth=-1 if truth is None else int(truth[slot]),
            )
        )
    defeated = []
    if scene["mario"] is not None:
        middle = (scene["mario"][0] + scene["mario"][2]) / 2
        for _, box, kind, _ in scene["enemies"]:
            if kind == "defeated":
                defeated.append(((box[0] + box[2]) / 2 - middle + position[0], box[3]))
    return out, defeated


def slot_keys(objects: TrackedObjects, identities) -> np.ndarray:
    """[SPRITE_SLOTS] the key of the entry each sprite slot's thing is (0: none)."""
    return np.asarray([objects.key_of(int(i)) for i in identities[:SPRITE_SLOTS]], np.int32)


def record_objects(record) -> dict:
    """The list frame by frame for a recorded episode (EpisodeTask.
    record_truth), made again from what it recorded: {"rows" [T, CAPACITY,
    len(FIELDS)], "keys" [T, CAPACITY], "truths" [T, CAPACITY], "slot_keys"
    [T, SPRITE_SLOTS]}."""
    frames = len(record.sprite_kinematics)
    objects = TrackedObjects()
    out = {
        "rows": np.zeros((frames, CAPACITY, len(FIELDS)), np.float32),
        "keys": np.zeros((frames, CAPACITY), np.int32),
        "truths": np.full((frames, CAPACITY), -1, np.int32),
        "slot_keys": np.zeros((frames, SPRITE_SLOTS), np.int32),
    }
    match = getattr(record, "slot_match", None)
    for t in range(frames):
        seen, defeated = sightings(
            record.sprite_kinematics[t],
            record.slot_identity[t],
            record.enemy_context[t],
            record.src_c[t],
            record.mario_position[t],
            None if match is None else match[t],
        )
        objects.update(seen, defeated)
        rows, keys, truths = objects.read(_kept(record.mario_position[t]))
        out["rows"][t], out["keys"][t], out["truths"][t] = rows, keys, truths
        out["slot_keys"][t] = slot_keys(objects, record.slot_identity[t])
    return out
