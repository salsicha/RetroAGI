"""The game's truth read from the emulator's memory each frame (true_frame):
for scoring, for the game-memory state source and for training labels,
never for a policy that plays from vision.

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
"""



from .game import PAGE, X_ON_PAGE, level_finished
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
# The game's object codes by name (the disassembly's), for reports.
KIND_NAMES = {
    0x00: "green Koopa", 0x02: "Buzzy Beetle", 0x03: "red Koopa", 0x05: "Hammer Bro", 0x06: "Goomba",
    0x07: "Blooper", 0x08: "Bullet Bill (frenzy)", 0x0A: "grey Cheep Cheep", 0x0B: "red Cheep Cheep",
    0x0C: "Podoboo", 0x0D: "Piranha Plant", 0x0E: "green Paratroopa (hopping)", 0x0F: "red Paratroopa",
    0x10: "green Paratroopa (flying)", 0x11: "Lakitu", 0x12: "Spiny", 0x14: "flying Cheep Cheep",
    0x15: "Bowser's flame", 0x1B: "fire bar", 0x1D: "fire bar", 0x2D: "Bowser", 0x2E: "power-up",
    0x32: "springboard", 0x33: "Bullet Bill",
}


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
    playthrough began in (``finished`` from the flagpole on, or once it
    differs: play.level_finished)."""
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
    return {
        "mario": (box[0] + camera, box[1], box[2] + camera, box[3]),
        "slots": slots,
        "camera": float(camera),
        "level_x": int(ram[PAGE]) * 256 + int(ram[X_ON_PAGE]),
        "score": _digits(ram, SCORE, 6) * 10,
        "coins": int(ram[COINS]),
        "time": _digits(ram, TIME, 3),
        "dying": _dying(ram),
        "finished": start is not None and level_finished(ram, start),
    }
