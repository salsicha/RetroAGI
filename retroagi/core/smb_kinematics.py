"""The executor's first-order kinematics, as the prediction networks read them.

Vision reports boxes only and estimates nothing; the executor keeps, for each
sprite it tracks, an estimate of where it is and how fast it moves (constant
velocity; smb_trajectory.Track), and one of Mario (SpatialFeedback). The world
model, at each action's start, and the collision trackers, every frame, read
these same numbers (user, 2026-10-16).

Per sprite slot, the scene's enemies then its moving platforms in their slot
order (smb_observer.packed_lists, the order of their scene tokens):

- present: the executor tracks it (else the row is zeros);
- x, y: its middle and top, pixels from the middle of Mario's feet (right,
  down);
- vx, vy: its movement per frame in the world (the camera's scroll removed);
- seen: how many sightings its velocity rests on, of ESTIMATE_SAMPLES.

Mario: vx (his measured world speed, else the executor's model of it), vy
(his feet's movement since the last frame), grounded.

What the executor remembers out of sight (smb_trajectory.MEMORY_FRAMES: a
plant in its pipe, an enemy past the screen's edge) the collision trackers
read too (remembered): up to MEMORY_SLOTS rows of TRACKED, the most recently
seen first.
"""

import numpy as np

from .smb_observer import SCENE_SLOTS, packed_lists
from .smb_trajectory import ESTIMATE_SAMPLES, LOST_FRAMES, MEMORY_FRAMES

SPRITE_STATE = ("present", "x", "y", "vx", "vy", "seen")
MARIO_STATE = ("vx", "vy", "grounded")
SPRITE_SLOTS = SCENE_SLOTS["enemies"] + SCENE_SLOTS["moving_platforms"]
ENEMY_SLOTS = SCENE_SLOTS["enemies"]
# What vision shows of an enemy besides where it is (enemy_context): its kind,
# its box's size (/16), and the surface it walks on (the far ends of the
# pieces on either side of it, from its middle, /64; supported 0 when none).
CONTEXT = ("walker", "plant", "other", "width", "height", "supported", "left", "right")
# Each enemy the collision trackers follow: its kinematics, how long it has
# been out of sight (frames / MEMORY_FRAMES; 0 seen now) and what vision
# shows (or last showed) of it.
TRACKED = SPRITE_STATE + ("unseen",) + CONTEXT
MEMORY_SLOTS = 4


def kinematics(scene, spatial) -> tuple[np.ndarray, np.ndarray]:
    """([SPRITE_SLOTS, len(SPRITE_STATE)], [len(MARIO_STATE)]) for ``scene``
    from the executor's estimates in ``spatial`` (SpatialFeedback), after it
    has observed the scene."""
    sprites = np.zeros((SPRITE_SLOTS, len(SPRITE_STATE)), np.float32)
    mario = np.zeros(len(MARIO_STATE), np.float32)
    if scene.mario.box is None:
        return sprites, mario
    box = scene.mario.box
    middle, feet = (box[0] + box[2]) / 2, box[3]
    lists = packed_lists(scene)
    objects = [
        (slot, e.box, e.kind) for slot, e in enumerate(lists["enemies"]) if e.kind != "defeated"
    ]
    objects += [
        (SCENE_SLOTS["enemies"] + slot, b, "platform")
        for slot, b in enumerate(lists["moving_platforms"])
    ]
    tracks = spatial.tracks.tracks
    for slot, b, kind in objects:
        matches = [t for t in tracks if t.kind == kind and t.box == b]
        if len(matches) != 1:
            continue
        track = matches[0]
        vx, vy = track.velocity or (0.0, 0.0)
        sprites[slot] = (
            1.0,
            (b[0] + b[2]) / 2 - middle,
            b[1] - feet,
            vx,
            vy,
            min(len(track.samples), ESTIMATE_SAMPLES) / ESTIMATE_SAMPLES,
        )
    speed = spatial.speed if spatial.speed is not None else spatial.motion.x_speed / 16
    grounded = scene.mario.on_something and scene.mario.support != "air"
    mario[:] = (speed, spatial.feet_velocity, float(grounded))
    return sprites, mario


def slot_identities(scene, spatial) -> np.ndarray:
    """[SPRITE_SLOTS] int32: the executor's track identity of each sprite
    slot's sprite (0: none): a slot's sprite changes as sprites come nearer
    or farther, its identity does not."""
    out = np.zeros(SPRITE_SLOTS, np.int32)
    if scene.mario.box is None:
        return out
    lists = packed_lists(scene)
    objects = [
        (slot, e.box, e.kind) for slot, e in enumerate(lists["enemies"]) if e.kind != "defeated"
    ]
    objects += [
        (SCENE_SLOTS["enemies"] + slot, b, "platform")
        for slot, b in enumerate(lists["moving_platforms"])
    ]
    for slot, b, kind in objects:
        matches = [t for t in spatial.tracks.tracks if t.kind == kind and t.box == b]
        if len(matches) == 1:
            out[slot] = matches[0].identity
    return out


def enemy_context(scene) -> np.ndarray:
    """[ENEMY_SLOTS, len(CONTEXT)]: each enemy slot's kind, size and the
    surface it walks on, as vision shows them (no estimation)."""
    out = np.zeros((ENEMY_SLOTS, len(CONTEXT)), np.float32)
    for slot, enemy in enumerate(packed_lists(scene)["enemies"]):
        if enemy.kind == "defeated":
            continue
        x0, y0, x1, y1 = enemy.box
        middle = (x0 + x1) / 2
        row = out[slot]
        row[CONTEXT.index(enemy.kind)] = 1.0
        row[CONTEXT.index("width")] = (x1 - x0) / 16
        row[CONTEXT.index("height")] = (y1 - y0) / 16
        # The surface it walks on is cut at it (smb_scene_labels.
        # split_at_enemies): its ends are the far ends of the pieces on
        # either side.
        level = [s for s in scene.surfaces if abs(s.top - y1) <= 3]
        left = [s for s in level if abs(s.x1 - x0) <= 3]
        right = [s for s in level if abs(s.x0 - x1) <= 3]
        if enemy.kind != "plant" and (left or right):
            row[CONTEXT.index("supported")] = 1.0
            row[CONTEXT.index("left")] = ((left[0].x0 if left else x0) - middle) / 64
            row[CONTEXT.index("right")] = ((right[0].x1 if right else x1) - middle) / 64
    return out


def remembered(scene, spatial) -> tuple[np.ndarray, np.ndarray]:
    """([MEMORY_SLOTS, len(TRACKED)], [MEMORY_SLOTS] int32 identities, 0
    none): the enemies the executor remembers that vision does not show now
    (VisualTracks.lost, the most recently seen first): where it expects each,
    from Mario's feet, the velocity that expectation moves by (none once it
    is held: smb_trajectory.Track.coast), the sightings its velocity rested
    on, how long it has been out of sight, and its kind and size when last
    seen (the surface it walked on is not known out of sight)."""
    rows = np.zeros((MEMORY_SLOTS, len(TRACKED)), np.float32)
    identities = np.zeros(MEMORY_SLOTS, np.int32)
    if scene.mario.box is None:
        return rows, identities
    box = scene.mario.box
    middle, feet = (box[0] + box[2]) / 2, box[3]
    kinds = CONTEXT[: CONTEXT.index("other") + 1]
    lost = sorted(
        (t for t in spatial.tracks.lost if t.kind in kinds), key=lambda t: (t.missed, t.identity)
    )
    column = {name: k for k, name in enumerate(TRACKED)}
    for row, track in zip(rows, lost[:MEMORY_SLOTS]):
        b = track.box
        vx, vy = (track.velocity or (0.0, 0.0)) if track.missed <= LOST_FRAMES else (0.0, 0.0)
        row[column["present"]] = 1.0
        row[column["x"]] = (b[0] + b[2]) / 2 - middle
        row[column["y"]] = b[1] - feet
        row[column["vx"]], row[column["vy"]] = vx, vy
        row[column["seen"]] = min(len(track.samples), ESTIMATE_SAMPLES) / ESTIMATE_SAMPLES
        row[column["unseen"]] = min(track.missed, MEMORY_FRAMES) / MEMORY_FRAMES
        row[column[track.kind]] = 1.0
        last = track.seen_box or b
        row[column["width"]] = (last[2] - last[0]) / 16
        row[column["height"]] = (last[3] - last[1]) / 16
    identities[: min(len(lost), MEMORY_SLOTS)] = [t.identity for t in lost[:MEMORY_SLOTS]]
    return rows, identities
