"""What a vision transformer reports to the policies: the objects drawn on screen.

Both SMB games describe a frame the same way (SceneObservation): Mario, every
enemy, coin, power-up, moving platform, pipe and block, every surface Mario
can stand on, and every gap in the floor. Everything is as drawn on the
256x240 screen: never the game's hidden collision bodies, and never anything
hidden (a plant inside its pipe is not reported). There is no speed or
motion: one picture, one scene.

Ground truth comes from one function for both games, scene_from_labels. It
reads exact per-pixel types and per-object identity (SceneLabels): from the
simulator's own drawing in Block SMB, and from the frame rebuilt out of
emulator memory in Full SMB. The vision transformer's scene is found from
the per-pixel types it predicts, by the same rules for both games: each
object (Mario, enemy, coin, power-up, moving platform) is a group of touching
pixels of its type (objects_from_types), and surfaces, gaps, blocks and pipes
come from structure_from_types. Only Mario's facing, his support, whether his
feet are on something (the landing signal) and each
enemy's kind are read from its other outputs. Exact visual support beneath
his feet establishes standing even when the contact head disagrees. A box
that sinks at most SINK_ROWS rows into a surface he mostly covers is support
read as Mario: its bottom is raised to the surface, where he stands. Other
contact predictions require nearby support or an enemy beneath his feet;
side contact cannot establish a landing. Type errors and touching objects can still distort the scene
(the truth keeps them apart; one group of pixels cannot).

Boxes are (x0, y0, x1, y1) in screen pixels, with x1 and y1 one past the last
drawn pixel. Every list runs left to right (then top to bottom). Only the
window the NES shows (smb_pixel_types.VISIBLE_ROWS / VISIBLE_COLUMNS) is
measured; outside it pictures and labels only repeat its edge.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy import ndimage

from .smb_pixel_types import SCREEN_SHAPE, TYPE_ID, VISIBLE_COLUMNS, VISIBLE_ROWS

Box = tuple[int, int, int, int]

# Compact objects, each reported with its own box.
OBJECT_CATEGORIES = ("mario", "enemy", "coin", "power_up", "moving_platform")
# What an enemy looks like it is: a walker (Goomba-like, stompable in Block
# SMB), a plant (out of a pipe), anything else, or one already defeated.
ENEMY_KINDS = ("walker", "plant", "other", "defeated")
SUPPORTS = ("air", "ground", "moving_platform")
# Pixel types Mario can stand on.
STANDABLE_TYPES = ("ground", "brick", "question_block", "pipe", "moving_platform")
_STANDABLE_IDS = tuple(TYPE_ID[name] for name in STANDABLE_TYPES)
# A surface is the top edge of a standable run at least this wide and deep;
# thinner fringes are drawing detail, not something to stand on.
MIN_SURFACE_WIDTH = 8
MIN_SURFACE_DEPTH = 3
# Pieces of one surface at most this many pixels apart, sideways or in
# height, are one surface (the rounded corners of question blocks, the bumpy
# tops of cloud blocks): Mario walks straight across. Real steps and gaps in
# both games are far larger.
SURFACE_JOIN = 2
# Holes in solid terrain at most this thick (mortar lines drawn in the
# backdrop colour, as in castle bricks) are filled when finding structure.
TERRAIN_HOLE = 2
# Gaps are columns with nothing standable in this band of rows near the bottom
# of the visible window, at least this wide.
FLOOR_BAND = (216, 232)
MIN_GAP_WIDTH = 8
# Rows under Mario's drawn feet checked for a moving platform beneath him.
SUPPORT_REACH = 2
# The vision transformer names the kind of enemy drawn in each 8x8 cell.
OBJECT_CELL = 8
CELL_GRID = (SCREEN_SHAPE[0] // OBJECT_CELL, SCREEN_SHAPE[1] // OBJECT_CELL)
# Pixels of one object type at most this far apart (sideways, up or down,
# diagonally) are one object: the pieces of one NES sprite can be a pixel or
# two apart.
OBJECT_JOIN = 1
# A group of fewer pixels is a speck, not an object, block or pipe.
MIN_OBJECT_PIXELS = 8
# Objects standing on a surface can cover part of its top row (NES sprites are
# drawn one row low, so feet overlap the ground's top row). For finding
# surfaces, a run of object pixels with standable terrain on both sides in the
# same row counts as that terrain.
_OBJECT_TYPE_IDS = tuple(TYPE_ID[name] for name in ("mario", "enemy", "coin", "power_up"))


@dataclass(frozen=True)
class MarioView:
    box: Optional[Box]  # None when Mario is not drawn
    facing_right: bool
    support: str  # one of SUPPORTS
    # His feet are on something in this picture: the ground, a moving platform,
    # or an enemy he is stomping. Turning on after he was in the air is a landing.
    on_something: bool


@dataclass(frozen=True)
class EnemyView:
    box: Box
    kind: str  # one of ENEMY_KINDS


@dataclass(frozen=True)
class BlockView:
    box: Box
    kind: str  # "brick" or "question_block"


@dataclass(frozen=True)
class Surface:
    x0: int
    x1: int
    top: int  # the row of the surface's top edge
    moving: bool


@dataclass(frozen=True)
class Gap:
    x0: int
    x1: int


@dataclass(frozen=True)
class SceneObservation:
    mario: MarioView
    enemies: tuple[EnemyView, ...] = ()
    coins: tuple[Box, ...] = ()
    power_ups: tuple[Box, ...] = ()
    moving_platforms: tuple[Box, ...] = ()
    pipes: tuple[Box, ...] = ()
    blocks: tuple[BlockView, ...] = ()
    surfaces: tuple[Surface, ...] = ()
    gaps: tuple[Gap, ...] = ()


@dataclass(frozen=True)
class SceneLabels:
    """The exact truth of one frame, from which its SceneObservation follows."""

    types: np.ndarray  # [240, 256] uint8 smb_pixel_types.TYPE_ID of every pixel
    instances: np.ndarray  # [240, 256] int32: which object drew each pixel, -1 for none
    categories: Mapping[int, str]  # object number -> one of OBJECT_CATEGORIES
    kinds: Mapping[int, str]  # enemy object number -> one of ENEMY_KINDS
    standing: bool  # the game's own flag: Mario stands on something
    facing_right: bool  # Mario as drawn
    stomping: bool  # the game's own event: this picture shows Mario landing on an enemy


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each run of True in a 1-D mask."""
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _component_boxes(mask: np.ndarray) -> list[Box]:
    """Box of each group of pixels that touch along an edge (specks left out)."""
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return []
    top, left = int(rows[0]), int(columns[0])
    groups, count = ndimage.label(mask[top : int(rows[-1]) + 1, left : int(columns[-1]) + 1])
    sizes = np.bincount(groups.ravel(), minlength=count + 1)
    boxes = []
    for number, (rows, columns) in enumerate(ndimage.find_objects(groups)[:count], start=1):
        if sizes[number] >= MIN_OBJECT_PIXELS:
            boxes.append(
                (columns.start + left, rows.start + top, columns.stop + left, rows.stop + top)
            )
    return sorted(boxes)


def _object_groups(mask: np.ndarray) -> list[tuple[Box, np.ndarray]]:
    """Pixels of one object type grouped into objects (pieces at most OBJECT_JOIN
    apart are one), specks left out: each group's box and its pixels inside
    the box, largest group first."""
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return []
    # Only the part of the picture holding this type's pixels (and room to join).
    top, left = max(int(rows[0]) - OBJECT_JOIN, 0), max(int(columns[0]) - OBJECT_JOIN, 0)
    part = mask[top : int(rows[-1]) + 1 + OBJECT_JOIN, left : int(columns[-1]) + 1 + OBJECT_JOIN]
    reach = 2 * OBJECT_JOIN + 1
    grown = ndimage.binary_dilation(part, np.ones((reach, reach), bool)) if OBJECT_JOIN else part
    groups, count = ndimage.label(grown, np.ones((3, 3), bool))
    groups = np.where(part, groups, 0)
    sizes = np.bincount(groups.ravel(), minlength=count + 1)
    found = []
    for number, where in enumerate(ndimage.find_objects(groups)[:count], start=1):
        if where is not None and sizes[number] >= MIN_OBJECT_PIXELS:
            box_rows, box_columns = where
            box = (
                box_columns.start + left,
                box_rows.start + top,
                box_columns.stop + left,
                box_rows.stop + top,
            )
            found.append((int(sizes[number]), box, groups[where] == number))
    return [(box, pixels) for _, box, pixels in sorted(found, key=lambda item: -item[0])]


def _left_to_right(boxes) -> tuple:
    return tuple(sorted(boxes, key=lambda box: (box[0], box[1])))


def _terrain_behind_objects(types: np.ndarray) -> np.ndarray:
    """Types with each object run that interrupts standable terrain taken as that terrain.

    Only for finding surfaces and gaps: Mario or an enemy standing on the
    ground covers part of its top row, which must not split the ground's top
    edge. A run of object pixels counts as the terrain it interrupts when the
    nearest pixels on both sides in its row are standable (a run touching the
    screen edge needs only its other side).
    """
    terrain = types.copy()
    objects = np.isin(types, _OBJECT_TYPE_IDS)
    standable = np.isin(types, _STANDABLE_IDS)
    width = types.shape[1]
    for row in np.flatnonzero(objects.any(axis=1)).tolist():
        for x0, x1 in _runs(objects[row]):
            left = x0 == 0 or standable[row, x0 - 1]
            right = x1 == width or standable[row, x1]
            if left and right and (x0 > 0 or x1 < width):
                terrain[row, x0:x1] = types[row, x0 - 1] if x0 > 0 else types[row, x1]
    return terrain


def _fill_thin_holes(mask: np.ndarray, size: int = TERRAIN_HOLE) -> np.ndarray:
    """The mask with runs of at most ``size`` False pixels between True ones set True.

    Done down each column, then along each row.
    """
    filled = mask.copy()
    for axis in (0, 1):
        current = np.moveaxis(filled, axis, 0)
        result = current.copy()
        for length in range(1, size + 1):
            hole = current[: -length - 1] & current[length + 1 :]
            for offset in range(length):
                hole = hole & ~current[1 + offset : current.shape[0] - length + offset]
            for offset in range(length):
                result[1 + offset : current.shape[0] - length + offset] |= hole
        filled = np.moveaxis(result, 0, axis)
    return filled


def _joined_surfaces(pieces: list[Surface]) -> list[Surface]:
    """Pieces that touch or nearly touch, sideways and in height, as one surface each.

    A joined surface spans all its pieces and lies at the highest one's top.
    """
    pieces = sorted(pieces, key=lambda p: (p.moving, p.x0, p.top))
    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if b.moving != a.moving or b.x0 - a.x1 > SURFACE_JOIN:
                break
            if abs(b.top - a.top) <= SURFACE_JOIN:
                parent[root(j)] = root(i)
    groups: dict[int, list[Surface]] = {}
    for i, piece in enumerate(pieces):
        groups.setdefault(root(i), []).append(piece)
    return [
        Surface(
            min(p.x0 for p in group),
            max(p.x1 for p in group),
            min(p.top for p in group),
            group[0].moving,
        )
        for group in groups.values()
    ]


def structure_from_types(types: np.ndarray) -> dict:
    """Surfaces, gaps, blocks and pipes of a frame from its per-pixel types.

    - surfaces: the top edge of every standable run (ground, brick, question
      block, pipe, moving platform) at least MIN_SURFACE_WIDTH wide whose
      standable pixels reach MIN_SURFACE_DEPTH rows down (below the screen
      counts as standable); moving surfaces are the tops of moving platforms.
      Mario or an enemy covering part of a surface's top row does not split
      it, holes in terrain at most TERRAIN_HOLE thick are filled, and pieces
      at most SURFACE_JOIN pixels apart (sideways and in height) are one
      surface at the highest piece's top;
    - gaps: runs of at least MIN_GAP_WIDTH columns with nothing standable in
      FLOOR_BAND;
    - blocks: each group of touching brick pixels, and of question block
      pixels, as one box with its kind;
    - pipes: each group of touching pipe pixels.
    """
    types = np.asarray(types)
    if types.shape != SCREEN_SHAPE:
        raise ValueError(f"types must be {SCREEN_SHAPE}, got {types.shape}")
    (top, bottom), (left, right) = VISIBLE_ROWS, VISIBLE_COLUMNS
    found = _window_structure(types[top:bottom, left:right])
    return {
        "surfaces": tuple(
            Surface(s.x0 + left, s.x1 + left, s.top + top, s.moving) for s in found["surfaces"]
        ),
        "gaps": tuple(Gap(g.x0 + left, g.x1 + left) for g in found["gaps"]),
        "blocks": tuple(BlockView(_shift(b.box, left, top), b.kind) for b in found["blocks"]),
        "pipes": tuple(_shift(box, left, top) for box in found["pipes"]),
    }


def _shift(box: Box, dx: int, dy: int) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _window_structure(types: np.ndarray) -> dict:
    """structure_from_types inside the visible window, in window coordinates."""
    terrain = _terrain_behind_objects(types)
    moving = terrain == TYPE_ID["moving_platform"]
    standable = _fill_thin_holes(np.isin(terrain, _STANDABLE_IDS))
    below = np.vstack((standable, np.ones((MIN_SURFACE_DEPTH - 1, standable.shape[1]), bool)))
    deep = standable.copy()
    for depth in range(1, MIN_SURFACE_DEPTH):
        deep &= below[depth : depth + standable.shape[0]]
    edge = deep.copy()
    edge[1:] &= ~standable[:-1]
    pieces = [
        Surface(x0, x1, row, is_moving)
        for row in np.flatnonzero(edge.any(axis=1)).tolist()
        for is_moving in (False, True)
        for x0, x1 in _runs(edge[row] & (moving[row] == is_moving))
    ]
    surfaces = [s for s in _joined_surfaces(pieces) if s.x1 - s.x0 >= MIN_SURFACE_WIDTH]
    floor = standable[FLOOR_BAND[0] - VISIBLE_ROWS[0] : FLOOR_BAND[1] - VISIBLE_ROWS[0]].any(axis=0)
    gaps = [Gap(x0, x1) for x0, x1 in _runs(~floor) if x1 - x0 >= MIN_GAP_WIDTH]
    blocks = [
        BlockView(box, kind)
        for kind in ("brick", "question_block")
        for box in _component_boxes(types == TYPE_ID[kind])
    ]
    return {
        "surfaces": tuple(sorted(surfaces, key=lambda s: (s.x0, s.top))),
        "gaps": tuple(gaps),
        "blocks": tuple(sorted(blocks, key=lambda b: (b.box[0], b.box[1]))),
        "pipes": _left_to_right(_component_boxes(types == TYPE_ID["pipe"])),
    }


def instance_boxes(instances: np.ndarray) -> dict[int, Box]:
    """The box of every object's drawn pixels inside the visible window."""
    (top, bottom), (left, right) = VISIBLE_ROWS, VISIBLE_COLUMNS
    window = np.asarray(instances)[top:bottom, left:right]
    boxes = {}
    # Labels shift by one so that "no object" (-1) becomes find_objects' ignored 0.
    for number, found in enumerate(ndimage.find_objects(window + 1)):
        if found is not None:
            rows, columns = found
            boxes[number] = (
                columns.start + left,
                rows.start + top,
                columns.stop + left,
                rows.stop + top,
            )
    return boxes


def mario_support(types: np.ndarray, box: Optional[Box], standing: bool) -> str:
    """Air unless standing; on a moving platform if one is drawn just under his feet."""
    if not standing:
        return "air"
    if box is not None:
        x0, _, x1, y1 = box
        under = np.asarray(types)[y1 : y1 + SUPPORT_REACH, x0:x1]
        if (under == TYPE_ID["moving_platform"]).any():
            return "moving_platform"
    return "ground"


def scene_from_labels(labels: SceneLabels) -> SceneObservation:
    """The true SceneObservation of a frame (the vision transformer's target)."""
    boxes = instance_boxes(labels.instances)
    by_category: dict[str, list[int]] = {name: [] for name in OBJECT_CATEGORIES}
    for number, category in labels.categories.items():
        if number in boxes:
            if category not in by_category:
                raise ValueError(f"unknown object category {category!r}")
            by_category[category].append(number)
    if len(by_category["mario"]) > 1:
        raise ValueError("a frame has more than one Mario")
    mario_box = boxes[by_category["mario"][0]] if by_category["mario"] else None
    enemies = []
    for number in by_category["enemy"]:
        kind = labels.kinds.get(number)
        if kind not in ENEMY_KINDS:
            raise ValueError(f"enemy object {number} has no known kind: {kind!r}")
        enemies.append(EnemyView(boxes[number], kind))
    return SceneObservation(
        mario=MarioView(
            box=mario_box,
            facing_right=bool(labels.facing_right),
            support=mario_support(labels.types, mario_box, labels.standing),
            on_something=bool(labels.standing or labels.stomping),
        ),
        enemies=tuple(sorted(enemies, key=lambda e: (e.box[0], e.box[1]))),
        coins=_left_to_right(boxes[n] for n in by_category["coin"]),
        power_ups=_left_to_right(boxes[n] for n in by_category["power_up"]),
        moving_platforms=_left_to_right(boxes[n] for n in by_category["moving_platform"]),
        **structure_from_types(labels.types),
    )


def scene_counts(scene: SceneObservation) -> dict[str, int]:
    """How many of each listed thing a scene holds."""
    return {
        name: len(getattr(scene, name))
        for name in (
            "enemies",
            "coins",
            "power_ups",
            "moving_platforms",
            "pipes",
            "blocks",
            "surfaces",
            "gaps",
        )
    }


# ── Training targets and decoding (shared by both games' vision transformers) ──


def box_overlap(a: Box, b: Box) -> float:
    """Area shared by two boxes over the area they cover together (0 to 1)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / area if area > 0 else 0.0


_GROUPED = {
    "mario": "mario",
    "enemies": "enemy",
    "coins": "coin",
    "power_ups": "power_up",
    "moving_platforms": "moving_platform",
}


def objects_from_types(types: np.ndarray, kind_probability: np.ndarray) -> dict:
    """Mario, enemies, coins, power-ups and moving platforms from per-pixel types.

    Each object is a group of touching pixels of its type (pieces at most
    OBJECT_JOIN apart are one group; groups under MIN_OBJECT_PIXELS are
    specks), its box the group's extent, inside the visible window. Mario is
    the largest group of Mario pixels. An enemy's kind is the most likely one
    over its pixels, from ``kind_probability`` [ENEMY_KINDS, 30, 32]: the kind
    the vision transformer sees in each 8x8 cell.
    """
    types = np.asarray(types)
    (top, bottom), (left, right) = VISIBLE_ROWS, VISIBLE_COLUMNS
    window = types[top:bottom, left:right]
    found: dict[str, list] = {}
    for name, type_name in _GROUPED.items():
        groups = _object_groups(window == TYPE_ID[type_name])
        if name == "mario":
            groups = groups[:1]
        if name == "enemies" and groups:
            cells = np.repeat(np.repeat(kind_probability, OBJECT_CELL, 1), OBJECT_CELL, 2)
            cells = cells[:, top:bottom, left:right]
            found[name] = [
                EnemyView(
                    _shift(box, left, top),
                    ENEMY_KINDS[
                        int(cells[:, box[1] : box[3], box[0] : box[2]][:, pixels].sum(1).argmax())
                    ],
                )
                for box, pixels in groups
            ]
        else:
            found[name] = [_shift(box, left, top) for box, _ in groups]
    return {
        "mario": found["mario"][0] if found["mario"] else None,
        "enemies": tuple(sorted(found["enemies"], key=lambda e: (e.box[0], e.box[1]))),
        "coins": _left_to_right(found["coins"]),
        "power_ups": _left_to_right(found["power_ups"]),
        "moving_platforms": _left_to_right(found["moving_platforms"]),
    }


def scene_targets(labels: SceneLabels) -> dict[str, np.ndarray]:
    """What the vision transformer is trained to output for one frame.

    types [240, 256]; kind [30, 32]: in each 8x8 cell holding enemy pixels,
    the ENEMY_KINDS index of the enemy drawing most of them (-1 elsewhere);
    facing (1 = right), support (SUPPORTS index), on_something (1 = his feet
    are on something), stomping (1 = on an enemy: weighted up, being rare) and
    mario_present are numbers.
    """
    scene = scene_from_labels(labels)
    rows, columns = CELL_GRID
    instances = np.asarray(labels.instances)
    kind = np.full((rows, columns), -1, np.int64)
    enemies = [n for n, category in labels.categories.items() if category == "enemy"]
    if enemies:
        # Pixels drawn by each enemy, counted per cell; each cell takes the enemy with most.
        counts = np.stack(
            [
                (instances == number).reshape(rows, OBJECT_CELL, columns, OBJECT_CELL).sum((1, 3))
                for number in enemies
            ]
        )
        most = counts.argmax(0)
        drawn = counts.max(0) > 0
        kinds = np.array([ENEMY_KINDS.index(labels.kinds[number]) for number in enemies])
        kind[drawn] = kinds[most[drawn]]
    return {
        "types": np.asarray(labels.types, np.uint8),
        "kind": kind,
        "facing": np.int64(scene.mario.facing_right),
        "support": np.int64(SUPPORTS.index(scene.mario.support)),
        "on_something": np.int64(scene.mario.on_something),
        "stomping": np.int64(bool(labels.stomping)),
        "mario_present": np.int64(scene.mario.box is not None),
    }


# Mario cannot be inside solid support. When his found box reaches at most
# SINK_ROWS rows below the top of a surface he mostly covers, with at least
# BODY_ROWS of him above it, those rows are support pixels read as Mario (seen
# on thin brick ledges): his feet are on that surface. A shorter blob there is
# not a standing Mario (a coin read as Mario was one).
SINK_ROWS = 6
BODY_ROWS = 8


def _feet_on_support(box, surfaces):
    """Mario's box with its bottom raised to the top of a surface it sank into."""
    if box is None:
        return box
    width = box[2] - box[0]
    for left, right, top in surfaces:
        cover = min(right, box[2]) - max(left, box[0])
        if 2 * cover >= width and box[1] + BODY_ROWS <= top < box[3] <= top + SINK_ROWS:
            return (box[0], box[1], box[2], top)
    return box


def decode_scene(heads: Mapping[str, "object"]) -> list:
    """SceneObservations from a vision transformer's raw heads (batched tensors).

    The per-pixel types are chosen where the tensors are (the graphics card,
    usually); only the choices come back to the processor.
    """
    import torch

    def chosen(name):
        return heads[name].argmax(1).to(torch.uint8).cpu().numpy()

    types = chosen("pixel_logits")
    kind_probability = torch.softmax(heads["kind_logits"].float(), dim=1).cpu().numpy()
    facing, support = chosen("facing_logits"), chosen("support_logits")
    on_something = chosen("on_something_logits")
    scenes = []
    for b in range(types.shape[0]):
        objects = objects_from_types(types[b], kind_probability[b])
        structure = structure_from_types(types[b])
        box = objects.pop("mario")
        # Resolve contradictory vision heads here, before any policy/controller
        # consumes contact. Side contact is not support beneath Mario's feet.
        surfaces = [(s.x0, s.x1, s.top) for s in structure["surfaces"]]
        surfaces += [(b[0], b[2], b[1]) for b in objects["moving_platforms"]]
        box = _feet_on_support(box, surfaces)
        standing = box is not None and any(
            left < box[2] and right > box[0] and abs(top - box[3]) <= 1
            for left, right, top in surfaces
        )
        stomping = box is not None and any(
            e.box[0] < box[2] and e.box[2] > box[0] and abs(e.box[1] - box[3]) <= 4
            for e in objects["enemies"]
        )
        exact_support = box is not None and any(
            left < box[2] and right > box[0] and top == box[3] for left, right, top in surfaces
        )
        contact = exact_support or bool(on_something[b]) and (standing or stomping)
        support_kind = SUPPORTS[int(support[b])] if contact and standing else "air"
        if exact_support and support_kind == "air":
            support_kind = mario_support(types[b], box, True)
        scenes.append(
            SceneObservation(
                mario=MarioView(
                    box=box,
                    facing_right=bool(facing[b]),
                    support=support_kind,
                    on_something=contact,
                ),
                **objects,
                **structure,
            )
        )
    return scenes
