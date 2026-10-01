"""Exact per-pixel type labels for real Super Mario Bros frames, read from game memory.

Everything a frame shows is in the emulator's memory: the background tile map
and its colours (the picture chip's memory) and the game's own memory (its
block map, its objects, and its list of 64 small 8x8 sprite pieces). Each
frame is rebuilt from memory pixel by pixel, recording what drew each pixel:

* Background squares. Each 16x16 square of the play area shows one of the
  game's block drawings, read from the cartridge's own drawing table. The
  game's block map says which block code sits in the square; the drawing and
  the code together give the square's type (BLOCK_TYPES). Scenery (hills,
  bushes, clouds, the end castle) is drawn but kept out of the block map, so
  it is background.
* Sprite pieces. Each piece belongs to Mario, an enemy slot, a bouncing
  block, a loose coin or hammer, or a fireball, found from the game's
  per-object pointers into the piece list. Its drawn pixels take the owner's
  type, and its drawing must be one that owner is known to use.
* Pixels drawn in the backdrop colour, and the score bar, are background.

A frame is accepted only when the rebuilt picture equals the emulator's
picture at every visible pixel and every square and piece is explained;
otherwise label_frame raises Unexplained with the reason.

Timing: the game writes background tiles and colours at the very start of a
frame, so those are read from the saved state taken after the frame. The
scroll position, sprite pieces and objects were prepared during the previous
frame, so they are read from the saved state taken before it.

Labels describe the full 256x240 screen. The emulator shows only the middle
240x224 (8 pixels are cut from every edge); the image is padded back by
repeating its edge pixels (canonical_rgb), and the labels are padded the same
way, so every padded pixel has the label of the real pixel it copies.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retroagi.core.smb_pixel_types import TYPE_ID
from retroagi.core.smb_scene import canonical_rgb

SCREEN = (240, 256)
VISIBLE = (slice(8, 232), slice(8, 248))  # the part of the screen the emulator shows
SCORE_BAR_ROWS = 32  # drawn without scrolling; never part of the play area
_ROWS, _COLUMNS = np.mgrid[0:240, 0:256]

# Game memory addresses (SMB disassembly names in brackets).
SCROLL_PAGE, SCROLL_X = 0x71A, 0x71C  # [ScreenLeft_PageLoc], [ScreenLeft_X_Pos]
BLOCK_MAP = 0x500  # [Block_Buffer_1]; the second page follows 0xD0 bytes later
PLAYER_FLOAT_STATE = 0x1D  # [Player_State]; 0 means standing on something
SPRITE_LIST = 0x200  # 64 pieces of 4 bytes: row - 1, graphic, attributes, column
PLAYER_POINTER = 0x6E4  # [SprDataOffset]: byte offset of an object's first piece
ENEMY_POINTERS = 0x6E5  # six enemy slots
BLOCK_POINTERS = 0x6EC  # two bouncing-block slots
BUBBLE_POINTERS = 0x6EE  # three air-bubble slots
FIREBALL_POINTERS = 0x6F1  # Mario's two fireballs
MISC_POINTERS = 0x6F3  # nine loose items: coins out of blocks, Hammer Bro hammers
ENEMY_ACTIVE, ENEMY_KIND = 0x0F, 0x16  # [Enemy_Flag], [Enemy_ID] per slot
BLOCK_METATILE = 0x3E8  # [Block_Metatile]: the block code a bouncing block shows
MISC_STATE = 0x2A  # [Misc_State]: 0x80 set for a hammer, 1 for a coin
HIDDEN_ROW = 0xEF  # pieces at this row or lower are not drawn

# The cartridge's block drawing table: four pointers (low bytes, then high
# bytes) at this program offset, one per colour group, and each group's size.
_DRAWING_POINTERS = 0x0B08
_DRAWINGS_PER_GROUP = (0x27, 0x2E, 0x0A, 0x06)
# The block map keeps a code only if it is at least this for its colour group
# (code >> 6); smaller codes are scenery and stored as 0 [BlockBuffLowBounds].
_KEPT_FROM = (0x10, 0x51, 0x88, 0xC0)


def _codes(first, last, kind):
    return dict.fromkeys(range(first, last + 1), kind)


# The type of every block code the cartridge can draw, from the SMB
# disassembly's description of each drawing. Blank drawings (empty sky, hidden
# blocks, the space under a springboard) draw no pixels, so their type never
# reaches a label.
BLOCK_TYPES = {
    **_codes(0x00, 0x0F, "background"),  # sky, bushes, hills, bridge rail, chain, tree tops
    **_codes(0x10, 0x15, "pipe"),  # upright pipe ends and shafts
    **_codes(0x16, 0x1B, "ground"),  # tree-top and mushroom ledges
    **_codes(0x1C, 0x21, "pipe"),  # sideways pipe ends, shafts and joints
    0x22: "ground",  # sea plant
    0x23: "background",  # blank left while a hit block bounces
    **_codes(0x24, 0x26, "background"),  # flagpole ball and shaft; blank beside a vine
    **_codes(0x40, 0x50, "background"),  # ropes, pulleys, castle, fence, tree trunk, stems
    **_codes(0x51, 0x53, "brick"),
    0x54: "ground",  # the overworld floor
    **_codes(0x55, 0x5E, "brick"),  # bricks holding items, with and without the line
    **_codes(0x5F, 0x60, "background"),  # hidden blocks: nothing drawn
    0x61: "ground",  # stair block
    0x62: "ground",  # castle floor
    0x63: "ground",  # castle bridge
    **_codes(0x64, 0x66, "ground"),  # Bullet Bill cannon barrel, top and base
    0x67: "background",  # blank under a springboard
    0x68: "ground",  # half block under a springboard
    0x69: "ground",  # underwater rock
    0x6A: "ground",  # half block
    **_codes(0x6B, 0x6C, "pipe"),  # underwater pipe
    0x6D: "background",  # flagpole ball
    **_codes(0x80, 0x87, "background"),  # clouds, water and lava
    0x88: "ground",  # cloud block
    0x89: "ground",  # Bowser's bridge
    **_codes(0xC0, 0xC1, "question_block"),
    **_codes(0xC2, 0xC3, "coin"),
    0xC4: "ground",  # used block
    0xC5: "background",  # the axe at the end of a castle
}

# Enemy-slot kinds [Enemy_ID] whose pieces were inspected on real frames, and
# the type their pixels take. A frame showing any other kind is refused.
ENEMY_KIND_TYPES = {
    0x00: "enemy",  # green Koopa Troopa
    0x02: "enemy",  # Buzzy Beetle
    0x05: "enemy",  # Hammer Bro
    0x06: "enemy",  # Goomba
    0x0D: "enemy",  # Piranha Plant
    0x0E: "enemy",  # green Koopa Paratroopa
    0x11: "enemy",  # Lakitu
    0x12: "enemy",  # Spiny and its falling egg
    0x16: "background",  # fireworks after the flagpole
    0x1B: "enemy",  # fire bar turning clockwise
    0x1D: "enemy",  # fire bar turning anticlockwise
    0x2A: "moving_platform",  # lift
    0x2E: "background",  # power-up (mushroom, flower, star)
    0x2F: "background",  # vine
    0x30: "background",  # flag on the flagpole
    0x31: "background",  # flag on the castle
    0x32: "moving_platform",  # springboard: stood on, and it moves
    0x33: "enemy",  # Bullet Bill
}
# Sprite drawings that are never an object's body: the floating score numbers
# ("00", "10", "20", "40", "50", "80", "0", "1U", "P").
SCORE_DIGITS = frozenset((0x50, *range(0xF6, 0xFC), 0xFD, 0xFE))
# The vine also draws through other slots' pieces; these drawings are only vine.
VINE_DRAWINGS = frozenset((0xE0, 0xE1))
# The sprite drawings each owner draws, every pair inspected on real frames
# from all ten level starts. Any other pair makes the frame refused.
OWNER_DRAWINGS = {
    "mario": frozenset(
        (*range(0x00, 0x2C), *range(0x32, 0x44), *range(0x4A, 0x4E), 0x4F, *range(0x90, 0x94))
    ),
    "enemy 00": frozenset(
        (*range(0x69, 0x70), *range(0xA0, 0xAA), 0xEF)
    ),  # also wings just lost, a squashed Goomba's last frame
    "enemy 02": frozenset((*range(0xAA, 0xB2), 0xF4, 0xF5)),  # F4, F5: upside down
    "enemy 05": frozenset((0x7C, 0x7D, *range(0x88, 0x8D), *range(0xD1, 0xD6), 0xE2, 0xE3)),
    "enemy 06": frozenset((*range(0x70, 0x74), 0xEF)),  # EF: squashed
    "enemy 0D": frozenset((0xE5, 0xE6, *range(0xEB, 0xEF))),
    "enemy 0E": frozenset((*range(0x69, 0x6D), 0xA0, *range(0xA2, 0xA6), *range(0xA7, 0xAA))),
    "enemy 11": frozenset(range(0xB8, 0xBE)),
    "enemy 12": frozenset((0x8E, 0x8F, *range(0x94, 0x9E))),
    "enemy 16": frozenset(range(0x66, 0x69)),
    "enemy 1B": frozenset((0x64, 0x65)),
    "enemy 1D": frozenset((0x64, 0x65)),
    "enemy 2A": frozenset((0x75,)),
    "enemy 2E": frozenset((*range(0x76, 0x7A), 0x8D, 0xE4)),
    "enemy 30": frozenset((0x7E, 0x7F)),
    "enemy 31": frozenset(range(0x54, 0x58)),
    "enemy 32": frozenset(range(0xF0, 0xF4)),
    "enemy 33": frozenset(range(0xE7, 0xEB)),
    "block 51": frozenset((0x85, 0x86)),
    "block 58": frozenset((0x85, 0x86)),
    "block C4": frozenset((0x87,)),
    "coin": frozenset(range(0x60, 0x64)),
    "hammer": frozenset(range(0x80, 0x84)),
    "fireball": frozenset((0x64, 0x65)),
}


def _colour_table(known):
    table = np.full((64, 3), -1, dtype=np.int16)
    for number, rgb in known.items():
        table[number] = (rgb >> 16, (rgb >> 8) & 0xFF, rgb & 0xFF)
    return table


# The emulator's red, green and blue for each NES colour number seen on the ten
# level starts; -1 marks numbers never seen, so a frame using one is refused.
_EMULATOR_COLOURS = _colour_table(
    {
        0x00: 0x707470,
        0x07: 0x780800,
        0x0F: 0x000000,
        0x10: 0xB8BCB8,
        0x16: 0xD82800,
        0x17: 0xC84C08,
        0x18: 0x887000,
        0x1A: 0x00A800,
        0x1C: 0x008088,
        0x21: 0x38BCF8,
        0x22: 0x5894F8,
        0x27: 0xF89838,
        0x29: 0x80D010,
        0x30: 0xF8FCF8,
        0x36: 0xF8BCB0,
        0x37: 0xF8D8A8,
    }
)


class Unexplained(ValueError):
    """A frame that memory does not fully explain; it must not be used for training."""


@dataclass(frozen=True)
class LabelledFrame:
    image: np.ndarray  # [240, 256, 3] uint8, the emulator's picture padded to full size
    labels: np.ndarray  # [240, 256] uint8 smb_pixel_types.TYPE_ID
    on_ground: bool  # the game's own standing flag
    enemy_rects: tuple  # (x, y, w, h) screen box of each drawn enemy object
    family: str = ""  # the level the frame comes from


# ── The cartridge ─────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def cartridge():
    """The 512 8x8 graphics (colour index 0-3, 0 = not drawn) and each block code's tiles.

    Tiles are given as (top-left, top-right, bottom-left, bottom-right).
    """
    import retro

    path = retro.data.get_romfile_path("SuperMarioBros-Nes", inttype=retro.data.Integrations.STABLE)
    rom = Path(path).read_bytes()
    if rom[:4] != b"NES\x1a":
        raise ValueError(f"{path} is not an NES cartridge file")
    program = rom[16 : 16 + rom[4] * 0x4000]
    pictures = np.frombuffer(rom, np.uint8, rom[5] * 0x2000, 16 + len(program))
    pictures = pictures.reshape(-1, 16)
    low = np.unpackbits(pictures[:, :8], axis=1).reshape(-1, 8, 8)
    high = np.unpackbits(pictures[:, 8:], axis=1).reshape(-1, 8, 8)
    graphics = low | (high << 1)
    table = program[_DRAWING_POINTERS : _DRAWING_POINTERS + 8]
    starts = [(table[i] | table[i + 4] << 8) - 0x8000 for i in range(4)]
    for group in range(3):
        if starts[group] + 4 * _DRAWINGS_PER_GROUP[group] != starts[group + 1]:
            raise ValueError("unexpected cartridge: the block drawing table is not where expected")
    drawings = {}
    for group, (start, count) in enumerate(zip(starts, _DRAWINGS_PER_GROUP)):
        for index in range(count):
            # Stored column by column: top-left, bottom-left, top-right, bottom-right.
            top_left, bottom_left, top_right, bottom_right = program[
                start + 4 * index : start + 4 * index + 4
            ]
            drawings[group * 0x40 + index] = (top_left, top_right, bottom_left, bottom_right)
    if set(drawings) != set(BLOCK_TYPES):
        raise ValueError("BLOCK_TYPES does not cover exactly the cartridge's block codes")
    return graphics, drawings


@functools.lru_cache(maxsize=1)
def _codes_by_drawing():
    _, drawings = cartridge()
    found = {}
    for code, tiles in drawings.items():
        found.setdefault(tiles, []).append(code)
    return found


def _kept_in_block_map(code):
    return code >= _KEPT_FROM[code >> 6]


@functools.lru_cache(maxsize=None)
def square_type(code, tiles, table):
    """Type of a 16x16 square showing ``tiles`` where the block map holds ``code``.

    Returns None when the drawing has no drawn pixels. When the drawing is the
    code's own, the code's type is used. Otherwise the block map and the
    picture are one frame apart (a block just hit, broken or collected), and
    the drawing decides: among the codes that draw it, those the block map
    would hold here (scenery when the map holds 0, kept codes otherwise), or
    all of them if none are. They must agree on one type.
    """
    graphics, drawings = cartridge()
    if not any(graphics[table * 256 + tile].any() for tile in tiles):
        return None
    if code and drawings.get(code) == tiles:
        return BLOCK_TYPES[code]
    same = _codes_by_drawing().get(tiles, [])
    preferred = [c for c in same if _kept_in_block_map(c) == bool(code)]
    types = {BLOCK_TYPES[c] for c in (preferred or same)}
    if len(types) != 1:
        drawn = " ".join(f"{tile:02X}" for tile in tiles)
        raise Unexplained(f"square drawing [{drawn}] under block code {code:02X} has no one type")
    return types.pop()


# ── Saved states ──────────────────────────────────────────────────────────────


def state_blocks(state: bytes) -> dict:
    """The named memory blocks of an emulator saved state (FCEU format, uncompressed)."""
    if state[:3] != b"FCS":
        raise ValueError("not an FCEU saved state")
    blocks, position = {}, 16
    while position < len(state):
        size = int.from_bytes(state[position + 1 : position + 5], "little")
        end, position = position + 5 + size, position + 5
        while position < end:
            name = state[position : position + 4].rstrip(b"\0").decode("ascii")
            length = int.from_bytes(state[position + 4 : position + 8], "little")
            blocks[name] = np.frombuffer(state, np.uint8, length, position + 8)
            position += 8 + length
    for name in ("RAM", "NTAR", "PRAM", "PPUR"):
        if name not in blocks:
            raise ValueError(f"saved state has no {name} block")
    return blocks


# ── Rebuilding the picture ────────────────────────────────────────────────────


def _background(nametables, table, scroll):
    """Palette entry of each background pixel (0 = backdrop) and its world column."""
    graphics, _ = cartridge()
    world = np.where(_ROWS < SCORE_BAR_ROWS, _COLUMNS, _COLUMNS + scroll)
    base = ((world >> 8) & 1) * 0x400
    x, y = world & 0xFF, _ROWS
    tiles = nametables[base + (y >> 3) * 32 + (x >> 3)].astype(int)
    attributes = nametables[base + 0x3C0 + (y >> 5) * 8 + (x >> 5)]
    group = (attributes >> (((y >> 4) & 1) * 4 + ((x >> 4) & 1) * 2)) & 3
    index = graphics[table * 256 + tiles, y & 7, x & 7]
    return np.where(index > 0, group * 4 + index, 0), world


def _sprites(pieces, table):
    """Front-most drawn sprite piece of each pixel (-1 = none), its palette entry and
    whether it is set to show behind the background.

    The picture chip draws only the first eight pieces that cover a row, and a
    lower-numbered piece is drawn in front of a higher-numbered one.
    """
    graphics, _ = cartridge()
    piece = np.full(SCREEN, -1, dtype=int)
    entry = np.zeros(SCREEN, dtype=int)
    behind = np.zeros(SCREEN, dtype=bool)
    on_row = np.zeros(SCREEN[0], dtype=int)
    shown = np.zeros((64, SCREEN[0]), dtype=bool)
    for k in range(64):
        top = int(pieces[4 * k]) + 1
        rows = np.arange(top, min(top + 8, SCREEN[0]))
        on_row[rows] += 1
        shown[k, rows] = on_row[rows] <= 8
    for k in range(63, -1, -1):
        row, graphic, attributes, left = (int(v) for v in pieces[4 * k : 4 * k + 4])
        top = row + 1
        height, width = min(8, SCREEN[0] - top), min(8, SCREEN[1] - left)
        if row >= HIDDEN_ROW or height <= 0 or width <= 0:
            continue
        image = graphics[table * 256 + graphic]
        if attributes & 0x40:
            image = image[:, ::-1]
        if attributes & 0x80:
            image = image[::-1]
        image = image[:height, :width]
        drawn = (image > 0) & shown[k, top : top + height, None]
        area = (slice(top, top + height), slice(left, left + width))
        piece[area] = np.where(drawn, k, piece[area])
        entry[area] = np.where(drawn, 16 + (attributes & 3) * 4 + image, entry[area])
        behind[area] = np.where(drawn, bool(attributes & 0x20), behind[area])
    return piece, entry, behind


# ── Who drew each sprite piece ────────────────────────────────────────────────


def piece_claims(ram):
    """Every object that may have drawn each sprite piece, as {piece: [(rank, owner, slot)]}.

    Owners are "mario", "enemy XX" (XX = the slot's kind), "block XX" (XX = the
    bouncing block's code), "coin", "hammer", "misc", "fireball" and "bubble".
    Each object claims the pieces from its pointer onward, as many as that
    kind of object can use; it may use fewer, and pointers of objects that are
    no longer active stay in memory, so a piece can have several claimants.
    Rank orders them: Mario, then active enemies, active blocks, active loose
    items, active fireballs, and inactive objects last (a slot that has just
    become inactive was still drawn with its last kind).
    """
    claims = {}

    def claim(pointer, count, owner, slot, rank):
        first = int(ram[pointer]) // 4
        for k in range(first, min(first + count, 64)):
            if k:
                claims.setdefault(k, []).append((rank, owner, slot))

    for slot in range(6):
        owner = f"enemy {int(ram[ENEMY_KIND + slot]):02X}"
        claim(ENEMY_POINTERS + slot, 6, owner, slot, 8 if ram[ENEMY_ACTIVE + slot] else 2)
    for slot in range(2):
        owner = f"block {int(ram[BLOCK_METATILE + slot]):02X}"
        claim(BLOCK_POINTERS + slot, 4, owner, slot, 6 if ram[0x26 + slot] else 1)
    for slot in range(9):
        state = int(ram[MISC_STATE + slot])
        owner = "hammer" if state & 0x80 else "coin" if state == 1 else "misc"
        claim(MISC_POINTERS + slot, 2, owner, slot, 5 if state else 1)
    for slot in range(2):
        claim(FIREBALL_POINTERS + slot, 1, "fireball", slot, 4 if ram[0x24 + slot] else 1)
    for slot in range(3):
        claim(BUBBLE_POINTERS + slot, 1, "bubble", slot, 1)
    claim(PLAYER_POINTER, 8, "mario", 0, 9)
    return claims


def piece_owner(claims, graphic):
    """The one owner of a piece showing ``graphic``, from its claimants, or Unexplained.

    Only claimants known to draw that graphic (OWNER_DRAWINGS) qualify; the
    highest-ranked of them decides, and if several share that rank they must
    give the piece the same type.
    """
    able = [c for c in claims if graphic in OWNER_DRAWINGS.get(c[1], ())]
    if not able:
        names = ", ".join(sorted({owner for _, owner, _ in claims})) or "nobody"
        raise Unexplained(
            f"sprite drawing {graphic:02X} is claimed by {names}, none known to draw it"
        )
    best = max(rank for rank, _, _ in able)
    top = [(owner, slot) for rank, owner, slot in able if rank == best]
    if len({owner_type(owner) for owner, _ in top}) != 1:
        raise Unexplained(f"sprite drawing {graphic:02X} has claimants of different types: {top}")
    return top[0]


def owner_type(owner):
    """The pixel type of an owner named as in piece_claims, or None if it is not known."""
    if owner == "mario":
        return "mario"
    if owner.startswith("enemy "):
        return ENEMY_KIND_TYPES.get(int(owner[6:], 16))
    if owner.startswith("block "):
        return BLOCK_TYPES.get(int(owner[6:], 16))
    return {
        "coin": "coin",
        "hammer": "enemy",
        "fireball": "background",
        "bubble": "background",
    }.get(owner)


# ── Labels ────────────────────────────────────────────────────────────────────


def rebuild(before: bytes, after: bytes, *, check_sprites: bool = True) -> dict:
    """Rebuild a frame from memory: colours, the layers behind them and the pixel types.

    Returns colour numbers (the NES's 64 colours) for the full screen, the
    palette entry of every pixel's background and front sprite, which sprite
    piece shows at each pixel (-1 = background shows), the types, and the
    owners of the pieces. Raises Unexplained for anything memory cannot
    account for in the visible area. ``check_sprites=False`` skips the sprite
    owner checks (sprite pixels are then left as background); it exists only to
    survey which shapes each object draws when extending OWNER_DRAWINGS.
    """
    past, drawn = state_blocks(before), state_blocks(after)
    ram = past["RAM"]
    control, mask = int(drawn["PPUR"][0]), int(drawn["PPUR"][1])
    if control & 0x20:
        raise Unexplained("tall 8x16 sprite pieces are switched on")
    if mask & 0x18 != 0x18:
        raise Unexplained("background or sprite drawing is switched off")
    background_table, sprite_table = (control >> 4) & 1, (control >> 3) & 1
    scroll = (int(ram[SCROLL_PAGE]) & 1) * 256 + int(ram[SCROLL_X])
    nametables = drawn["NTAR"]
    background, world = _background(nametables, background_table, scroll)
    piece, sprite_entry, behind = _sprites(ram[SPRITE_LIST : SPRITE_LIST + 256], sprite_table)
    sprite_shows = (piece >= 0) & (~behind | (background == 0))
    entry = np.where(sprite_shows, sprite_entry, background)
    colours = drawn["PRAM"][entry] & 0x3F

    # Background: every visible square of the play area, typed from the block map.
    labels = np.full(SCREEN, TYPE_ID["background"], dtype=np.uint8)
    rows = slice(SCORE_BAR_ROWS, VISIBLE[0].stop)
    first, last = (scroll + VISIBLE[1].start) // 16, (scroll + VISIBLE[1].stop - 1) // 16
    squares = np.full((13, last - first + 1), TYPE_ID["background"], dtype=np.uint8)
    for column in range(first, last + 1):
        page, x = divmod(column, 16)
        tile_x, base = (column * 16 % 256) // 8, (page & 1) * 0x400
        for row in range((rows.stop - SCORE_BAR_ROWS + 15) // 16):
            code = int(ram[BLOCK_MAP + (page & 1) * 0xD0 + row * 16 + x])
            at = base + (SCORE_BAR_ROWS // 8 + 2 * row) * 32 + tile_x
            tiles = (
                int(nametables[at]),
                int(nametables[at + 1]),
                int(nametables[at + 32]),
                int(nametables[at + 33]),
            )
            kind = square_type(code, tiles, background_table)
            if kind is not None:
                squares[row, column - first] = TYPE_ID[kind]
    play = (slice(SCORE_BAR_ROWS, SCREEN[0]), slice(None))
    square_row = np.minimum((_ROWS[play] - SCORE_BAR_ROWS) // 16, 12)
    square_column = np.clip(world[play] // 16 - first, 0, squares.shape[1] - 1)
    labels[play] = np.where(
        background[play] > 0, squares[square_row, square_column], TYPE_ID["background"]
    )

    # Sprites: every piece showing in the visible area must have one known owner.
    claims = piece_claims(ram)
    pieces = ram[SPRITE_LIST : SPRITE_LIST + 256]
    showing = np.unique(piece[VISIBLE][sprite_shows[VISIBLE]])
    piece_types = np.full(64, TYPE_ID["background"], dtype=np.uint8)
    owners = {}
    for k in showing if check_sprites else ():
        graphic = int(pieces[4 * k + 1])
        if k == 0 or graphic in SCORE_DIGITS or graphic in VINE_DRAWINGS:
            continue  # the score bar's marker piece, score digits and vine are background
        owners[int(k)] = piece_owner(claims.get(int(k), []), graphic)
        piece_types[k] = TYPE_ID[owner_type(owners[int(k)][0])]
    labels = np.where(sprite_shows, piece_types[np.maximum(piece, 0)], labels).astype(np.uint8)
    return {
        "colours": colours,
        "background_entry": background,
        "sprite_entry": sprite_entry,
        "piece": np.where(sprite_shows, piece, -1),
        "labels": labels,
        "owners": owners,
        "claims": claims,
        "ram": ram,
    }


def label_frame(frame, before: bytes, after: bytes, *, family: str = "") -> LabelledFrame:
    """Exact labels for one emulator frame, or Unexplained.

    ``frame`` is the emulator's 224x240 picture of one step; ``before`` and
    ``after`` are the saved states taken just before and just after that step.
    """
    image = canonical_rgb(frame)
    built = rebuild(before, after)
    expected = _EMULATOR_COLOURS[built["colours"][VISIBLE]]
    wrong = np.any(expected != image[VISIBLE], axis=-1)
    if wrong.any():
        rows, columns = np.nonzero(wrong)
        raise Unexplained(
            f"rebuilt picture differs at {int(wrong.sum())} pixels "
            f"(rows {rows.min() + 8}-{rows.max() + 8}, columns {columns.min() + 8}-{columns.max() + 8})"
        )
    labels = np.pad(
        built["labels"][VISIBLE],
        (
            (VISIBLE[0].start, SCREEN[0] - VISIBLE[0].stop),
            (VISIBLE[1].start, SCREEN[1] - VISIBLE[1].stop),
        ),
        mode="edge",
    )
    # One box per enemy object: the visible pixels its pieces drew as enemy.
    piece, owners = built["piece"][VISIBLE], built["owners"]
    enemy = built["labels"][VISIBLE] == TYPE_ID["enemy"]
    objects = {}
    for k in np.unique(piece[enemy]):
        objects.setdefault(owners[int(k)], []).append(int(k))
    rects = []
    for members in objects.values():
        rows, columns = np.nonzero(enemy & np.isin(piece, members))
        rects.append(
            (
                int(columns.min()) + VISIBLE[1].start,
                int(rows.min()) + VISIBLE[0].start,
                int(columns.max() - columns.min()) + 1,
                int(rows.max() - rows.min()) + 1,
            )
        )
    return LabelledFrame(
        image=image,
        labels=labels,
        on_ground=int(built["ram"][PLAYER_FLOAT_STATE]) == 0,
        enemy_rects=tuple(sorted(rects)),
        family=family,
    )
