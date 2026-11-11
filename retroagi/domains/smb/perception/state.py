"""The controller's state from what vision shows (perceived), in world
pixels: the game's collision boxes inside the drawn sprites, the solids from
the scene's pipes, blocks and surfaces (solids_of), each enemy's kind and
velocity from its track (walking as a walker until measured), items
followed from frame to frame (ItemTracker).
"""

from copy import copy
from typing import Optional

from retroagi.domains.smb.controller import State
from retroagi.domains.smb.planner import (
    BLOCK_KINDS,
    HOP_GRAVITY,
    SCREEN_BOTTOM,
    Enemy,
    Solid,
    floor_behind,
    screen_edge,
)

from .scene_lists import packed_lists

# Perceived enemies seen for fewer frames than this are taken to walk as
# walkers do, WALK pixels a frame toward Mario (a track found again gave a
# Goomba walking at him +1.2 pixels a frame from 5 whole-pixel moves, and
# Mario walked off 1-1's third pipe into it).
MEASURED = 8
WALK = 0.5
# Mario's body as the game collides it, inside the sprite vision draws: this
# wide about the drawn box's middle (pixels), its top this far below the
# drawn top. The drawn 16-pixel width stood him on a 16-pixel step past its
# edge, in the planner, while he slid off it.
BODY_WIDTH = 10
HEAD_ROOM = 4
# Big Mario (after a mushroom: drawn taller than BIG_DRAWN pixels) collides
# BIG_BODY_WIDTH wide, his top BIG_HEAD_ROOM below the drawn top.
BIG_DRAWN, BIG_BODY_WIDTH, BIG_HEAD_ROOM = 20, 12, 8
# A walker drawn taller than this is a Koopa, and wider than this more than
# one (_collision_box).
KOOPA_DRAWN, MERGED_DRAWN = 20, 20
# The planner's kind of enemy (smb_planner.ENEMY_KINDS) by the object type
# vision tells (pixel_labels.ENEMY_TYPE_NUMBERS: a Piranha Plant, a Koopa
# Paratroopa, a Bullet Bill); otherwise by vision's kind ("other": a flier).
# A Bullet Bill can be stomped.
TYPE_KINDS = {4: "plant", 5: "hopper", 9: "flier"}
BULLET_TYPE = 9
# A hopper's vertical speed is measured over its last this many frames
# (hopper_speed).
HOP_SAMPLES = 4
# Enemies seen starting in this many columns at the screen's left edge are
# left out: there vision read cut-off sprites as enemies walking toward
# Mario, and he jumped back over a pit at them (the game hides the first 8
# columns; a real enemy there is half gone and leaving).
LEFT_COLUMNS = 16
# Enemies seen narrower or shorter than this (pixels) are left out unless
# the screen's right edge cuts them (from RIGHT_EDGE on): fragments 2 to 8
# pixels wide in 1-1's second pit were read as enemies walking toward Mario
# (a Goomba is 16 by 16, 8 tall when stomped flat).
MIN_ENEMY = 6
RIGHT_EDGE = 240


def solids_of(scene, camera: float) -> list:
    """Every still thing vision reports as a solid, in world pixels: pipes
    and blocks by their boxes, the floor and steps from their tops down to
    the screen's bottom (but those the pipes and blocks of their height
    cover at least half of: their own tops), moving surfaces thin. A 2-pixel pipe vision
    made up at a step's end once hid the whole step Mario stood on."""
    lists = packed_lists(scene)
    out = []
    for slot, p in enumerate(lists.get("pipes", ())):
        out.append(Solid(p[0] + camera, p[1], p[2] + camera, SCREEN_BOTTOM, "pipe", ("pipes", slot)))
    for slot, b in enumerate(lists.get("blocks", ())):
        x0, y0, x1, y1 = b.box
        kind = "question" if getattr(b, "kind", "") == "question_block" else "block"
        out.append(Solid(x0 + camera, y0, x1 + camera, y1, kind, ("blocks", slot)))
    for slot, s in enumerate(lists.get("surfaces", ())):
        x0, x1 = s.x0 + camera, s.x1 + camera
        covered = 2 * sum(
            max(0.0, min(x1, o.x1) - max(x0, o.x0))
            for o in out if o.kind in ("pipe", *BLOCK_KINDS) and abs(o.top - s.top) <= 2
        ) >= x1 - x0
        if covered:
            continue
        bottom = s.top + 8 if s.moving else SCREEN_BOTTOM
        out.append(Solid(x0, s.top, x1, bottom, "moving" if s.moving else "ground", ("surfaces", slot)))
    return out


class ItemTracker:
    """Items vision shows (power-ups), followed frame to frame in world
    pixels, each with its velocity over its last SAMPLES moves (a new one
    taken to move as a mushroom comes out, MUSHROOM_SPEED to the right)."""

    SAMPLES = 8
    MUSHROOM_SPEED = 1.0

    def __init__(self):
        self.items, self.next_identity = [], 1  # [box, [(dx, dy)], identity]

    def observe(self, boxes) -> None:
        kept = []
        for box in boxes:
            old = min(self.items, key=lambda i: abs(i[0][0] - box[0]) + abs(i[0][1] - box[1]), default=None)
            if old is not None and abs(old[0][0] - box[0]) <= 8 and abs(old[0][1] - box[1]) <= 8:
                self.items.remove(old)
                kept.append([box, (old[1] + [(box[0] - old[0][0], box[1] - old[0][1])])[-self.SAMPLES:], old[2]])
            else:
                kept.append([box, [], self.next_identity])
                self.next_identity += 1
        self.items = kept

    def seen(self) -> list:
        out = []
        for box, samples, identity in self.items:
            if samples:
                vx = sum(d[0] for d in samples) / len(samples)
                vy = sum(d[1] for d in samples) / len(samples)
            else:
                vx, vy = self.MUSHROOM_SPEED, 0.0
            out.append(Enemy(tuple(box), vx, vy, ("items", identity), stompable=False))
        return out


def perceived(feedback, scene, items: Optional[ItemTracker] = None) -> Optional[State]:
    """The controller's state from what vision shows (``scene``) and the
    executor's estimates (``feedback``, smb_spatial_feedback.SpatialFeedback,
    already observing the scene), in world pixels (screen pixels plus its
    camera position):
    - Mario's body is his collision box inside the drawn one (BODY_WIDTH,
      HEAD_ROOM; big Mario's BIG_BODY_WIDTH, BIG_HEAD_ROOM); his motion is the estimate, ready once it has measured his
      speed; he stands when vision says so or when his feet rest still on a
      solid's top (vision's flag alone left him "in the air" in a third of
      the frames beside 1-1's fourth pipe);
    - the solids are vision's, none inside Mario's drawn box, floor gaps
      hidden by enemies joined, and the screen's left edge, a wall;
    - the enemies are its tracks seen this frame, each of the planner's
      kinds (TYPE_KINDS), a walker's and a plant's collision box inside its
      drawn one (_collision_box, _plant_box), none starting in the
      LEFT_COLUMNS, none smaller than MIN_ENEMY away from the right edge and
      none mostly inside Mario's drawn box (vision read big Mario's top half
      as an enemy on his head), with their averaged velocities (seen for fewer than
      MEASURED frames: a walker's WALK pixels a frame toward Mario, anything
      else still: 5-1's piranha plants, never measured as they rise and
      sink, seemed to walk through the air at Mario);
    - the items are ``items``' (an ItemTracker the caller feeds vision's
      power-ups), none if not given.
    None while vision does not show Mario."""
    if scene.mario.box is None:
        return None
    camera = feedback.camera_position
    box = scene.mario.box
    middle = (box[0] + box[2]) / 2 + camera
    drawn_on_screen = box
    big = box[3] - box[1] > BIG_DRAWN
    width, head = (BIG_BODY_WIDTH, BIG_HEAD_ROOM) if big else (BODY_WIDTH, HEAD_ROOM)
    body = (middle - width / 2, box[1] + head, middle + width / 2, box[3])
    enemies = []
    for t in feedback.tracks.tracks:
        if t.kind == "platform" or t.missed or t.box[0] < LEFT_COLUMNS:
            continue
        small = min(t.box[2] - t.box[0], t.box[3] - t.box[1]) < MIN_ENEMY
        if small and t.box[2] < RIGHT_EDGE:
            continue
        if _inside(t.box, drawn_on_screen):
            continue  # part of Mario's own sprite (big Mario's top half read as an enemy)
        world = (t.box[0] + camera, t.box[1], t.box[2] + camera, t.box[3])
        kind = TYPE_KINDS.get(t.type, "walker" if t.kind == "walker" else "plant" if t.kind == "plant" else "flier")
        if kind in ("walker", "hopper"):
            world = _collision_box(world)
        elif kind == "plant":
            world = _plant_box(world)
        if len(t.samples) >= MEASURED:
            vx, vy = t.velocity
            if kind == "hopper":
                vy = hopper_speed([dy for _, dy in t.samples])
        elif kind in ("walker", "hopper"):
            vx, vy = (-WALK if world[0] > body[0] else WALK), 0.0
        else:
            vx, vy = 0.0, 0.0  # a plant or anything else: still until measured
        stompable = kind in ("walker", "hopper") or t.type == BULLET_TYPE
        enemies.append(Enemy(world, vx, vy, ("enemies", t.identity), stompable=stompable, kind=kind))
    # Nothing solid lies inside Mario's own drawn box: at the screen's left
    # edge, parts of his cut-off sprite read as a 3-pixel pipe.
    drawn = (box[0] + camera, box[1], box[2] + camera, box[3])
    solids = [
        s for s in floor_behind(solids_of(scene, camera), enemies)
        if not (drawn[0] - 1 <= s.x0 and s.x1 <= drawn[2] + 1 and s.top < drawn[3] and drawn[1] < s.bottom)
    ] + [screen_edge(camera)]
    resting = abs(feedback.feet_velocity) < 1 and any(
        s.x0 < body[2] and body[0] < s.x1 and abs(s.top - body[3]) <= 2 for s in solids
    )
    motion = copy(feedback.motion)
    motion.origin_y += head  # the jump's origin, measured at the drawn top
    return State(
        body=body,
        motion=motion,
        grounded=(scene.mario.on_something and scene.mario.support != "air") or resting,
        solids=solids,
        enemies=enemies,
        ready=feedback.motion_ready,
        items=items.seen() if items is not None else [],
    )


def _collision_box(drawn) -> tuple:
    """A walker's collision box inside the sprite vision draws, as the game
    collides it (game memory, 1-1): from the drawn left side (the game
    measures it from the sprite's place, whichever way it faces), a Goomba
    (drawn 16 tall) 1 to 11 across and 6 tall ending 5 above the drawn
    bottom, a Koopa (drawn taller than KOOPA_DRAWN) 0 to 12 across and 12
    tall ending 4 above it. With the drawn boxes, a pass over a Goomba pair
    looked like a hit in the air, and the flight check braked Mario down in
    front of them. Drawn wider than one sprite (MERGED_DRAWN: vision draws
    two side by side as one), the box spans them all, its right side as
    far in from the drawn right (5-1: two Koopas read as one, the second
    left out, and Mario jumped back onto it)."""
    left, right, bottom = drawn[0], drawn[2], drawn[3]
    if drawn[3] - drawn[1] > KOOPA_DRAWN:
        x0, x1, top, low = left, left + 12, bottom - 16, bottom - 4
    else:
        x0, x1, top, low = left + 1, left + 11, bottom - 11, bottom - 5
    if right - left > MERGED_DRAWN:
        x1 = right - (left + 16 - x1)
    return (x0, top, x1, low)


def hopper_speed(moves) -> float:
    """A hopper's vertical speed now from its last vertical moves (pixels a
    frame, oldest first): the mean of the last HOP_SAMPLES, which is its
    speed in the middle of them, plus the gravity since (a 16-frame mean put
    a Paratroopa's landing 6 frames late, under Mario's)."""
    recent = moves[-HOP_SAMPLES:]
    return sum(recent) / len(recent) + HOP_GRAVITY * (len(recent) + 1) / 2


def _plant_box(drawn) -> tuple:
    """A piranha plant's collision box in the part of it vision draws (out of
    its pipe, from the plant's top): the game's 10 by 6, 3 in from its left
    side and 14 below its top (game memory, 5-1)."""
    return (drawn[0] + 3, drawn[1] + 14, drawn[0] + 13, drawn[1] + 20)


def _inside(box, outer) -> bool:
    """Whether at least half of ``box`` lies inside ``outer``."""
    w = max(0.0, min(box[2], outer[2]) - max(box[0], outer[0]))
    h = max(0.0, min(box[3], outer[3]) - max(box[1], outer[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area > 0 and w * h >= area / 2
