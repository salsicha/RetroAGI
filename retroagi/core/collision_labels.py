"""Collision labels made from a recorded episode's truth: what the collision
trackers learn (user, 2026-10-16: labels from recorded real paths).

For each frame t and true enemy, whether, when, where and how Mario and it
first touch within HORIZON frames if he carries on with the action he is
doing until it ends and then stands where it left him, against the enemy's
real path:

- frames: how many frames after t they first touch (0: they do not, as far
  as is known);
- kind: KINDS, a stomp (his feet coming down onto its top, by the game's
  rule smb_collision.stomp_contact; a recorded kill there is one) or a hit;
- place: the middle of his feet then, from the middle of his feet at t
  (pixels, x right, y down);
- censor: the first offset nothing is known from (the episode ended, or
  the enemy died another way, before any touch); 0 when known to HORIZON.

Truth arrays come from a recording: Block SMB's simulator
(stages/block_smb/truth_recording) and, later, Full SMB's memory, the same
way: per frame and one more after the last button, Mario's and each enemy's
collision box (world pixels), each enemy's state (ENEMY_STATES) and whether
it can be stomped.
"""

from dataclasses import dataclass

import numpy as np

HORIZON = 96
KINDS = ("none", "stomp", "hit")
ABSENT, ALIVE, DEAD = 0, 1, 2


@dataclass
class CollisionLabels:
    frames: np.ndarray  # [T, E] int16
    kind: np.ndarray  # [T, E] int8, KINDS index
    place: np.ndarray  # [T, E, 2] float32
    censor: np.ndarray  # [T, E] int16


def action_ends(decision_frames, frames: int) -> np.ndarray:
    """[frames] int: for each frame, the frame its action in progress ends at
    (the next decision; the last action at ``frames``, the state after the
    last button). Before the first decision Mario does nothing (he is
    watched standing): the frame itself."""
    ends = np.arange(frames, dtype=np.int64)
    starts = list(int(f) for f in decision_frames) + [frames]
    for start, end in zip(starts, starts[1:]):
        ends[start:end] = end
    return ends


def _overlap(a, b):
    """Whether boxes overlap (strictly), elementwise over leading axes."""
    return (
        (a[..., 0] < b[..., 2])
        & (b[..., 0] < a[..., 2])
        & (a[..., 1] < b[..., 3])
        & (b[..., 1] < a[..., 3])
    )


def collision_labels(truth: dict, ends: np.ndarray, horizon: int = HORIZON) -> CollisionLabels:
    """The labels of every frame (``ends``: action_ends) and true enemy."""
    mario = truth["mario"].astype(np.float32)
    enemies = truth["enemies"].astype(np.float32)
    state = truth["enemy_state"]
    stompable = truth["stompable"]
    last = len(mario) - 1  # the state after the last button
    frames = len(ends)
    count = enemies.shape[1] if enemies.ndim == 3 else 0
    out = CollisionLabels(
        frames=np.zeros((frames, count), np.int16),
        kind=np.zeros((frames, count), np.int8),
        place=np.zeros((frames, count, 2), np.float32),
        censor=np.zeros((frames, count), np.int16),
    )
    if not frames or not count:
        return out
    offsets = np.arange(1, horizon + 1)
    rows = np.arange(frames)[:, None] + offsets[None, :]  # [T, H]
    inside = rows <= last
    rows = np.minimum(rows, last)
    end = np.asarray(ends)[:, None]
    # Mario's path: his own until his action ends, then where it left him.
    path = np.minimum(rows, end)
    body = mario[path]  # [T, H, 4]
    before = mario[np.minimum(rows - 1, end)]
    vy = body[..., 3] - before[..., 3]
    start = mario[:frames]
    middle = (start[:, 0] + start[:, 2]) / 2
    for e in range(count):
        box = enemies[rows, e]  # [T, H, 4]
        now, previous = state[rows, e], state[rows - 1, e]
        killed = (now == DEAD) & (previous == ALIVE)
        touch = _overlap(body, box) & inside & ((now == ALIVE) | killed)
        # Nothing is known past the episode's end, or once it died of
        # something else (or before).
        unknown = ~inside | ((now == DEAD) & ~(killed & touch))
        first_touch = np.where(touch.any(1), touch.argmax(1), horizon)
        first_unknown = np.where(unknown.any(1), unknown.argmax(1), horizon)
        alive_now = state[:frames, e] != DEAD
        seen = alive_now & (first_touch < first_unknown)
        out.censor[:, e] = np.where(
            alive_now & ~seen & (first_unknown < horizon), first_unknown + 1, 0
        )
        t = np.flatnonzero(seen)
        s = first_touch[t]
        b, x = body[t, s], box[t, s]
        stomp = stompable[e] & (
            killed[t, s]
            | ((vy[t, s] > 0) & (b[:, 3] - vy[t, s] <= x[:, 1] + (x[:, 3] - x[:, 1]) // 2))
        )
        out.frames[t, e] = s + 1
        out.kind[t, e] = np.where(stomp, KINDS.index("stomp"), KINDS.index("hit"))
        out.place[t, e, 0] = (b[:, 0] + b[:, 2]) / 2 - middle[t]
        out.place[t, e, 1] = b[:, 3] - start[t, 3]
    return out


def slot_labels(labels: CollisionLabels, slot_match: np.ndarray, enemy_slots: int):
    """The labels each enemy scene slot's sprite carries at each frame (the
    true enemy it shows then: truth_recording.match_slots): frames, kind,
    place and censor per [T, enemy_slots], and "labelled" [T, enemy_slots]
    (False where the slot shows no matched enemy)."""
    match = slot_match[:, :enemy_slots].astype(np.int64)
    labelled = match >= 0
    index = np.where(labelled, match, 0)
    rows = np.arange(len(match))[:, None]

    def take(values):
        if values.shape[1] == 0:  # no enemy in the recording
            return np.zeros(match.shape + values.shape[2:], values.dtype)
        picked = values[rows, index]
        mask = labelled.reshape(labelled.shape + (1,) * (picked.ndim - 2))
        return np.where(mask, picked, 0)

    return {
        "frames": take(labels.frames),
        "kind": take(labels.kind),
        "place": take(labels.place),
        "censor": take(labels.censor),
        "labelled": labelled,
    }
