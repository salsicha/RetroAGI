"""The game's truth kept with a Full SMB recording (training only), in the
arrays Block SMB's recordings keep (legacy/block_smb/truth_recording), so the
collision trackers' and the world model's labels are made the same way
(core/collision_labels, core/action_outcomes; user, 2026-10-16).

The game reuses its six enemy slots as the level scrolls: a slot filled
again (set after it was empty, or holding another kind) is a new enemy.
FullTruthRecorder keeps one column per enemy, absent outside its life.
"""

import numpy as np

from retroagi.domains.smb.nes.truth import SLOTS
from retroagi.domains.smb.perception.scene_lists import SCENE_SLOTS, packed_lists
from retroagi.legacy.core.smb_kinematics import SPRITE_SLOTS


class FullTruthRecorder:
    """Collects true_frame rows, gives each enemy (a slot's fill) its own
    column, and packs them as truth_recording.TruthRecorder does."""

    def __init__(self):
        self.rows = []
        self.ids = []  # per frame, each slot's enemy column (-1: none)
        self.kinds = []  # per enemy column: (slot, kind, platform, stompable)
        self._current = [None] * SLOTS  # (kind, column) filling each slot now

    def add(self, frame: dict) -> list:
        """Keep a frame; returns each slot's enemy column at it (-1: empty)."""
        ids = []
        for slot, info in enumerate(frame["slots"]):
            current = self._current[slot]
            if not info["active"] or not (info["enemy"] or info["platform"]):
                if not info["active"]:
                    self._current[slot] = None
                ids.append(-1 if current is None or not info["active"] else current[1])
                continue
            if current is None or current[0] != info["kind"]:
                column = len(self.kinds)
                self.kinds.append((slot, info["kind"], info["platform"], info["stompable"]))
                current = self._current[slot] = (info["kind"], column)
            ids.append(current[1])
        self.rows.append(frame)
        self.ids.append(ids)
        return ids

    def arrays(self) -> dict:
        """As truth_recording.TruthRecorder.arrays, the columns being enemies
        (and lifts) over the whole recording, plus "level_x", "score",
        "coins", "time" [N] and "dying" [N]."""
        count = len(self.rows)
        enemies = [c for c, k in enumerate(self.kinds) if not k[2]]
        lifts = [c for c, k in enumerate(self.kinds) if k[2]]
        column = {c: i for i, c in enumerate(enemies)}
        lift_column = {c: i for i, c in enumerate(lifts)}
        boxes = np.zeros((count, len(enemies), 4), np.int16)
        drawn = np.zeros((count, len(enemies), 4), np.int16)
        state = np.zeros((count, len(enemies)), np.int8)
        platforms = np.zeros((count, len(lifts), 4), np.int16)
        alive_once = set()
        for t, (row, ids) in enumerate(zip(self.rows, self.ids)):
            for slot, c in enumerate(ids):
                info = row["slots"][slot]
                if c < 0:
                    continue
                if c in lift_column and info["drawn"] is not None:
                    platforms[t, lift_column[c]] = np.floor(info["drawn"])
                if c not in column:
                    continue
                i = column[c]
                if info["box"] is not None:
                    boxes[t, i] = np.floor(info["box"])
                if info["drawn"] is not None:
                    drawn[t, i] = np.floor(info["drawn"])
                if info["defeated"]:
                    state[t, i] = 2 if i in alive_once else 0
                elif info["box"] is not None:
                    state[t, i] = 1
                    alive_once.add(i)
        return {
            "mario": np.floor(np.asarray([r["mario"] for r in self.rows], np.float32)).astype(np.int16)
            if self.rows
            else np.zeros((0, 4), np.int16),
            "enemies": boxes,
            "enemies_drawn": drawn,
            "enemy_state": state,
            "platforms": platforms,
            "camera": np.asarray([r["camera"] for r in self.rows], np.float32),
            "stompable": np.asarray([self.kinds[c][3] for c in enemies], bool),
            "level_x": np.asarray([r["level_x"] for r in self.rows], np.int32),
            "score": np.asarray([r["score"] for r in self.rows], np.int32),
            "coins": np.asarray([r["coins"] for r in self.rows], np.int16),
            "time": np.asarray([r["time"] for r in self.rows], np.int16),
            "dying": np.asarray([r["dying"] for r in self.rows], bool),
        }

    def enemy_columns(self):
        """For each recorder column, its index among the enemies (-1: a lift)."""
        out, i = [], 0
        for _, _, platform, _ in self.kinds:
            out.append(-1 if platform else i)
            i += 0 if platform else 1
        return out


def match_slots(scene, frame: dict, ids: list) -> np.ndarray:
    """[SPRITE_SLOTS]: for each scene slot (smb_kinematics' order), the
    recorder column of the true enemy (or lift) it shows: its drawn box
    nearest the scene slot's box (middle and top, screen pixels) within 12
    pixels, unambiguously; -1 otherwise. Turned into enemy (or lift) indices
    by FullTruthRecorder afterwards."""
    from retroagi.legacy.block_smb.truth_recording import MATCH, MATCH_MARGIN

    out = np.full(SPRITE_SLOTS, -1, np.int32)
    camera = int(frame["camera"])

    def nearest(point, candidates):
        distances = sorted(
            (abs(point[0] - c[0]) + abs(point[1] - c[1]), column) for c, column in candidates
        )
        if not distances or distances[0][0] > MATCH:
            return -1
        if len(distances) > 1 and distances[1][0] - distances[0][0] <= MATCH_MARGIN:
            return -1
        return distances[0][1]

    enemies, lifts = [], []
    for slot, (info, column) in enumerate(zip(frame["slots"], ids)):
        if column < 0 or info["drawn"] is None:
            continue
        d = info["drawn"]
        point = ((d[0] + d[2]) / 2 - camera, d[1])
        if info["platform"]:
            lifts.append((point, column))
        elif info["enemy"] and not info["defeated"]:
            enemies.append((point, column))
    lists = packed_lists(scene)
    for slot, enemy in enumerate(lists["enemies"]):
        if enemy.kind != "defeated":
            out[slot] = nearest(((enemy.box[0] + enemy.box[2]) / 2, enemy.box[1]), enemies)
    for slot, box in enumerate(lists["moving_platforms"]):
        out[SCENE_SLOTS["enemies"] + slot] = nearest(((box[0] + box[2]) / 2, box[1]), lifts)
    return out
