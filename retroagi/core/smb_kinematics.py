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
"""

import numpy as np

from .smb_observer import SCENE_SLOTS, packed_lists
from .smb_trajectory import ESTIMATE_SAMPLES

SPRITE_STATE = ("present", "x", "y", "vx", "vy", "seen")
MARIO_STATE = ("vx", "vy", "grounded")
SPRITE_SLOTS = SCENE_SLOTS["enemies"] + SCENE_SLOTS["moving_platforms"]


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
