"""The game's truth kept with a Full SMB recording (training only), read from
the emulator's memory every frame, in the arrays Block SMB's recordings keep
(stages/block_smb/truth_recording), so the collision trackers' and the world
model's labels are made the same way (core/collision_labels,
core/action_outcomes; user, 2026-10-16).

Per frame:

- Mario's collision box and every enemy slot's: the boxes the game itself
  tests contacts with ([BoundingBox_UL_Corner] 0x4AC, [EnemyBoundingBoxCoord]
  0x4B0 + 4 x slot; screen pixels, 0xFF off screen), in world pixels (the
  screen's left edge added), their rows unwrapped (_rows): the box bytes
  wrap past row 255, so a Mario falling into a pit read as standing at the
  top of the screen, and an action that ended in a pit as a 252-pixel rise
  (user, 2026-10-28: why Full SMB learning made the agent worse);
- each slot's drawn box (the union of its sprite pieces), to tie vision's
  boxes to the slots;
- whether each slot's enemy is absent (an empty slot, off screen, or not an
  enemy), alive or defeated (pixel_labels.object_of's rule), and whether it
  can be stomped;
- the lifts' drawn boxes; the screen's left edge;
- the level's progress: Mario's distance into it, the score, coins and time
  left, whether he is dying or has left the level.

The game reuses its six enemy slots as the level scrolls: a slot filled
again (set after it was empty, or holding another kind) is a new enemy.
FullTruthRecorder keeps one column per enemy, absent outside its life.
"""

import numpy as np

from retroagi.core.smb_kinematics import SPRITE_SLOTS
from retroagi.core.smb_observer import SCENE_SLOTS, packed_lists

from .pixel_labels import (
    ENEMY_ACTIVE,
    ENEMY_KIND,
    ENEMY_KIND_TYPES,
    ENEMY_POINTERS,
    ENEMY_STATE,
    HIDDEN_ROW,
    OWNER_DRAWINGS,
    SCROLL_PAGE,
    SCROLL_X,
    SPRITE_LIST,
)
from .play import LEVEL, PAGE, WORLD, X_ON_PAGE
from .vision_frames import _dying

SLOTS = 6
PLAYER_BOX = 0x4AC  # [BoundingBox_UL_Corner]: x0, y0, x1, y1 (screen pixels)
ENEMY_BOXES = 0x4B0  # [EnemyBoundingBoxCoord]: four bytes a slot
# [Player_Y_HighPos], [Player_Y_Position]; an enemy slot's at + slot: the
# screen page (1 on screen, 2 below it) and the row on it.
PLAYER_Y_PAGE, PLAYER_Y = 0xB5, 0xCE
ENEMY_Y_PAGE, ENEMY_Y = 0xB6, 0xCF
OFF_SCREEN = 0xFF
SCORE, COINS, TIME = 0x7DD, 0x75E, 0x7F8  # score: 6 digits; time: 3 digits
# Enemies Mario cannot stomp: Piranha Plant, Spiny, fire bars.
UNSTOMPABLE = frozenset((0x0D, 0x12, 0x1B, 0x1D))


def _digits(ram, address, count) -> int:
    value = 0
    for byte in ram[address : address + count]:
        value = value * 10 + int(byte) % 10
    return value


def _rows(box, page: int, y: int) -> tuple:
    """A box's top and bottom rows from the top of the screen, unwrapped:
    its bytes are rows modulo 256, a fixed offset from the object's own row
    ``y`` on screen ``page`` (1 on screen), which the game keeps apart."""
    top = (page - 1) * 256 + y + (box[1] - y) % 256
    return top, top + (box[3] - box[1]) % 256


def _drawn(ram, slot, kind, camera):
    """The world box of the sprite pieces the slot's kind draws (None: none)."""
    drawings = OWNER_DRAWINGS.get(f"enemy {kind:02X}", ())
    first = int(ram[ENEMY_POINTERS + slot]) // 4
    boxes = []
    for piece in range(first, min(first + 6, 64)):
        row, graphic, _, column = (int(v) for v in ram[SPRITE_LIST + 4 * piece : SPRITE_LIST + 4 * piece + 4])
        if piece and row < HIDDEN_ROW and graphic in drawings:
            boxes.append((column, row + 1, column + 8, row + 9))
    if not boxes:
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return (x0 + camera, y0, x1 + camera, y1)


def true_frame(ram, start=None) -> dict:
    """The game's state at this frame (module notes): {"mario", "slots": [per
    slot {"active", "kind", "box", "drawn", "defeated", "enemy",
    "platform", "stompable"}], "camera", "level_x", "score", "coins",
    "time", "dying", "finished"}. ``start``: the (world, level) the
    playthrough began in (``finished`` once it differs)."""
    camera = int(ram[SCROLL_PAGE]) * 256 + int(ram[SCROLL_X])
    box = [int(v) for v in ram[PLAYER_BOX : PLAYER_BOX + 4]]
    box[1], box[3] = _rows(box, int(ram[PLAYER_Y_PAGE]), int(ram[PLAYER_Y]))
    slots = []
    for slot in range(SLOTS):
        kind = int(ram[ENEMY_KIND + slot])
        active = bool(ram[ENEMY_ACTIVE + slot])
        state = int(ram[ENEMY_STATE + slot])
        kind_type = ENEMY_KIND_TYPES.get(kind)
        raw = [int(v) for v in ram[ENEMY_BOXES + 4 * slot : ENEMY_BOXES + 4 * slot + 4]]
        on_screen = active and raw[0] != OFF_SCREEN and raw[2] != OFF_SCREEN
        if on_screen:
            raw[1], raw[3] = _rows(raw, int(ram[ENEMY_Y_PAGE + slot]), int(ram[ENEMY_Y + slot]))
        slots.append(
            {
                "active": active,
                "kind": kind,
                "box": (raw[0] + camera, raw[1], raw[2] + camera, raw[3]) if on_screen else None,
                "drawn": _drawn(ram, slot, kind, camera) if active else None,
                "defeated": (not active) or bool(state & 0x20) or (kind == 0x06 and state == 4),
                "enemy": kind_type == "enemy",
                "platform": kind_type == "moving_platform",
                "stompable": kind not in UNSTOMPABLE,
            }
        )
    world_level = (int(ram[WORLD]), int(ram[LEVEL]))
    return {
        "mario": (box[0] + camera, box[1], box[2] + camera, box[3]),
        "slots": slots,
        "camera": float(camera),
        "level_x": int(ram[PAGE]) * 256 + int(ram[X_ON_PAGE]),
        "score": _digits(ram, SCORE, 6) * 10,
        "coins": int(ram[COINS]),
        "time": _digits(ram, TIME, 3),
        "dying": _dying(ram),
        "finished": start is not None and world_level != tuple(start),
    }


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
    from retroagi.stages.block_smb.truth_recording import MATCH, MATCH_MARGIN

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
