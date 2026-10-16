"""The simulator's truth kept with a recorded episode (training only), for
labels computed afterwards from what really happened (core/collision_labels):
the collision trackers' and the world model's (user, 2026-10-16: labels from
recorded real paths, the same way for Full SMB with the emulator's memory).

Per frame, one row of MarioScenarioEnv.true_frame, and one more after the
last button (the state the final step left: the stomp or the death it made).
Which true enemy each scene slot shows (match_slots) ties the executor's
estimates and the vision rows to them.
"""

import numpy as np

from retroagi.core.smb_kinematics import SPRITE_SLOTS
from retroagi.core.smb_observer import SCENE_SLOTS, packed_lists

ENEMY_STATES = ("absent", "alive", "dead")
# A scene slot is matched to a true enemy (or moving platform) within this
# many pixels (box middles and tops, screen pixels), when no other is within
# MATCH_MARGIN more.
MATCH = 12
MATCH_MARGIN = 2


class TruthRecorder:
    """Collects true_frame rows; arrays() packs them."""

    def __init__(self):
        self.rows = []

    def add(self, frame: dict) -> None:
        self.rows.append(frame)

    def arrays(self) -> dict:
        """{"mario" [N,4], "enemies" [N,E,4], "enemies_drawn" [N,E,4],
        "enemy_state" [N,E] (ENEMY_STATES), "platforms" [N,P,4] (int16 world
        pixels, rounded), "camera" [N] float32}; N = frames + 1."""
        rows = self.rows
        count = len(rows)
        enemies = len(rows[0]["enemies"]) if rows else 0
        platforms = len(rows[0]["platforms"]) if rows else 0

        def boxes(key, width):
            out = np.zeros((count, width, 4), np.int16)
            for t, row in enumerate(rows):
                if row[key]:
                    out[t] = np.rint(np.asarray(row[key], np.float32))
            return out

        return {
            "mario": np.rint(np.asarray([r["mario"] for r in rows], np.float32)).astype(np.int16)
            if rows
            else np.zeros((0, 4), np.int16),
            "enemies": boxes("enemies", enemies),
            "enemies_drawn": boxes("enemies_drawn", enemies),
            "enemy_state": np.asarray([r["enemy_state"] for r in rows], np.int8).reshape(
                count, enemies
            ),
            "platforms": boxes("platforms", platforms),
            "camera": np.asarray([r["camera"] for r in rows], np.float32),
        }


def _nearest(point, candidates):
    """The index of the candidate (middle x, top) nearest ``point`` within
    MATCH, unambiguous by MATCH_MARGIN; else -1."""
    distances = sorted(
        (abs(point[0] - c[0]) + abs(point[1] - c[1]), i) for i, c in enumerate(candidates)
    )
    if not distances or distances[0][0] > MATCH:
        return -1
    if len(distances) > 1 and distances[1][0] - distances[0][0] <= MATCH_MARGIN:
        return -1
    return distances[0][1]


def match_slots(scene, frame: dict) -> np.ndarray:
    """[SPRITE_SLOTS] int16: for each enemy slot of the scene, the true enemy
    it shows (its drawn box, live ones), and for each moving-platform slot the
    true moving platform; -1 for an empty slot or no unambiguous match. Slots
    as smb_kinematics orders them."""
    out = np.full(SPRITE_SLOTS, -1, np.int16)
    camera = int(frame["camera"])
    lists = packed_lists(scene)
    live = [
        ((b[0] + b[2]) / 2 - camera, b[1]) if state == 1 else (1e9, 1e9)
        for b, state in zip(frame["enemies_drawn"], frame["enemy_state"])
    ]
    for slot, enemy in enumerate(lists["enemies"]):
        if enemy.kind == "defeated":
            continue
        point = ((enemy.box[0] + enemy.box[2]) / 2, enemy.box[1])
        out[slot] = _nearest(point, live)
    lifts = [((b[0] + b[2]) / 2 - camera, b[1]) for b in frame["platforms"]]
    for slot, box in enumerate(lists["moving_platforms"]):
        point = ((box[0] + box[2]) / 2, box[1])
        out[SCENE_SLOTS["enemies"] + slot] = _nearest(point, lifts)
    return out
