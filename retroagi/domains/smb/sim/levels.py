"""Random levels made of the real game's pieces (docs/choose-and-control-plan.md
phase 2; user, 2026-11-07: "The levels could be randomly generated and I
could still pass them").

A level is a run of World 1-1's pieces, in pixels as the game draws them
(16-pixel tiles, the floor's top at 208): pits 2 to 4 tiles wide, pipes 2 to
4 tiles tall, staircases up and down with a valley or a pit between (each
top one or two columns wide), rows of bricks and question blocks at the
game's two heights (4 and 8 tiles above the floor, Goombas sometimes walking
on the upper one; a question block holds a coin or a mushroom, big Mario
breaks bricks), question blocks over the floor, single Goombas and pairs,
Koopa Troopas (a stomp leaves a shell that a touch kicks), and from World
5-1 piranha plants in pipes, hopping Koopa Paratroopas and Bullet Bill
cannons, and a staircase before the flag, each separated by a few tiles of floor
(random_level). The floor is cut where a pipe or a step stands, as the game
draws it. Goombas follow the game's rules (env.py "game_rules": they wake as
they come on screen, walk left at half a pixel a frame, turn at walls, fall
off ledges), and the camera keeps Mario where the real game does
(SCROLL_X).

A skill scene (skill_scene) is a short stretch of the same pieces that
calls for one skill: one piece of a kind (SCENE_KINDS: a pipe, a pit,
enemies, bricks, stairs, question blocks, a cannon) or two pieces close
together ("pair"), Mario standing a few tiles before it, flat floor after
it and no flag. Scenes give outcome sampling its share of every piece, and
a score for each.
"""

import random
from typing import Optional, Sequence

from .piranha import EXPOSED_FRAMES, HIDDEN_FRAMES, RISE_FRAMES

PLANT_CYCLE = 2 * RISE_FRAMES + EXPOSED_FRAMES + HIDDEN_FRAMES

TILE = 16
FLOOR = 208
BOTTOM = 240
SCREEN = 256
# The real game keeps Mario's left side about here on screen as it scrolls
# (Full SMB, 1-1: 115 in nine scrolling frames of ten).
SCROLL_X = 115
# Mario starts standing on the floor this far in.
START_X = 40
# Floor before the first piece, between pieces, and after the last (tiles).
START_FLAT = 10
FLAT = (3, 8)
END_FLAT = 10
# The pieces and how often each comes.
PIECES = {"pipe": 3, "pit": 2, "goombas": 3, "bricks": 2, "stairs": 1, "questions": 1, "cannon": 1}
PIPE_TILES = (2, 3, 4)
PIT_TILES = (2, 2, 3, 3, 4)
STAIR_TILES = (3, 4)
STAIR_TOPS = (1, 2)
# Brick rows: their tops (pixels) and lengths (tiles).
LOW_ROW, HIGH_ROW = FLOOR - 4 * TILE, FLOOR - 8 * TILE
LOW_TILES, HIGH_TILES = (1, 5), (3, 8)
# In a brick row this share of tiles are question blocks; a question block
# holds a mushroom MUSHROOM_SHARE of the time, else a coin. The "questions"
# piece is 1 to 3 question blocks over the floor (1-1's first).
QUESTION_SHARE = 0.3
MUSHROOM_SHARE = 0.35
# The share of levels Mario starts big (as after 1-1's first mushroom). Both
# shares are above the game's so the outcome network sees mushrooms and big
# Mario often enough (at the game's, sampled states almost never had one).
BIG_START = 0.3
# Two Goombas walk this far apart (pixels), as in 1-1; an enemy piece is a
# Koopa Troopa this share of the time.
PAIR_GAP = 24
KOOPA_SHARE = 0.25
# World 5-1's things: a pipe holds a piranha plant PLANT_SHARE of the time
# (at a random point of its cycle); an enemy piece is a Koopa Paratroopa
# PARATROOPA_SHARE of the time; a cannon is CANNON_TILES tall.
PLANT_SHARE = 0.4
PARATROOPA_SHARE = 0.2
CANNON_TILES = (1, 2)
# A Goomba's body as the simulator places it: its feet this far below the
# y it is given.
GOOMBA_FEET = 14
# Frames a level may take per pixel of its length.
FRAMES_PER_PIXEL = 3
# The kinds of skill scene: one piece of a kind, or two pieces close together.
SCENE_KINDS = (*PIECES, "pair")
# A scene: its first piece starts this many tiles in (Mario stands at
# START_X: 5.5 to 11.5 tiles before it; in the game an enemy wakes as the
# screen reaches it, some 9 tiles ahead of him, and one put 3 tiles in
# started on top of him); a pair's pieces are this many tiles apart (the
# game's are often side by side: worlds 2-1, 3-1 and 5-1 have pieces 0 to 2
# tiles apart 28 times); it is passed once Mario stands SCENE_PASSED pixels
# beyond its last piece, and the floor goes on SCENE_AFTER tiles from there
# (further than a choice tried near its end and the probes after it can
# take him).
SCENE_START = (8, 14)
PAIR_FLAT = (0, 2)
SCENE_PASSED = 2 * TILE
SCENE_AFTER = 24


def _goomba(x: float, top: float, kind: Optional[str] = None) -> dict:
    out = {"x": float(x), "y": float(top - GOOMBA_FEET), "game_rules": True}
    if kind:
        out["kind"] = kind
    return out


def random_level(seed: int, pieces: int = 12, only: Optional[str] = None, start_flat: int = START_FLAT,
                 flat: Sequence[int] = FLAT, flag: bool = True) -> dict:
    """A level of ``pieces`` random pieces (each of the kind ``only``, if
    given) then the closing staircase and flag, the same for the same
    arguments (a Block SMB scenario). ``start_flat``: tiles of floor before
    the first piece; ``flat``: the fewest and most tiles between pieces.
    Without ``flag`` (a skill scene) flat floor follows the last piece and
    nothing is to be reached: metadata "scene" has where it is passed."""
    rng = random.Random(seed)
    x = start_flat * TILE
    pits, solids, enemies = [], [], []  # pits (x0, x1); solids [x, y, w, h, kind]
    contents = {}  # index in solids -> "coin" or "mushroom"

    def column(x0, tiles):
        top = FLOOR - tiles * TILE
        solids.append([x0, top, TILE, BOTTOM - top, "ground"])

    def block(x0, top, question):
        if question:
            contents[len(solids)] = "mushroom" if rng.random() < MUSHROOM_SHARE else "coin"
        solids.append([x0, top, TILE, TILE, "question_block" if question else "brick"])

    def row(x0, top, tiles):
        for k in range(tiles):
            block(x0 + k * TILE, top, rng.random() < QUESTION_SHARE)

    after = 0  # (tiles of floor after the last piece)
    for _ in range(pieces):
        kind = only or rng.choices(list(PIECES), weights=list(PIECES.values()))[0]
        if kind == "pipe":
            top = FLOOR - rng.choice(PIPE_TILES) * TILE
            solids.append([x, top, 2 * TILE, BOTTOM - top, "pipe"])
            if rng.random() < PLANT_SHARE:
                enemies.append({"kind": "piranha_plant", "x": float(x + TILE // 2), "pipe_top": float(top),
                                "phase": rng.randrange(PLANT_CYCLE)})
            x += 2 * TILE
        elif kind == "cannon":
            top = FLOOR - rng.choice(CANNON_TILES) * TILE
            solids.append([x, top, TILE, BOTTOM - top, "cannon"])
            x += TILE
        elif kind == "pit":
            width = rng.choice(PIT_TILES) * TILE
            pits.append((x, x + width))
            x += width
        elif kind == "goombas":
            roll = rng.random()
            if roll < PARATROOPA_SHARE:
                enemies.append(_goomba(x, FLOOR, "paratroopa"))
            elif roll < PARATROOPA_SHARE + KOOPA_SHARE:
                enemies.append(_goomba(x, FLOOR, "koopa"))
            else:
                enemies.append(_goomba(x, FLOOR))
                if rng.random() < 0.5:
                    enemies.append(_goomba(x + PAIR_GAP, FLOOR))
            x += 3 * TILE
        elif kind == "questions":
            for k in range(rng.randint(1, 3)):
                block(x + 2 * k * TILE, LOW_ROW, True)
            x += 5 * TILE
        elif kind == "bricks":
            low = rng.randint(*LOW_TILES)
            row(x, LOW_ROW, low)
            end = x + low * TILE
            if rng.random() < 0.5:
                start = x + rng.randint(1, 2) * TILE
                high = rng.randint(*HIGH_TILES)
                row(start, HIGH_ROW, high)
                if rng.random() < 0.5:
                    enemies.append(_goomba(start + (high - 1) * TILE, HIGH_ROW))
                end = max(end, start + high * TILE)
            x = end
        else:  # stairs: up, a valley of floor or a pit, down
            height = rng.choice(STAIR_TILES)
            # Each side's top one or two columns wide (1-1's second pair:
            # two up, one down).
            up_top, down_top = rng.choice(STAIR_TOPS), rng.choice(STAIR_TOPS)
            for k in range(height + up_top - 1):
                column(x, min(k + 1, height))
                x += TILE
            gap = 2 * TILE
            if rng.random() < 0.5:
                pits.append((x, x + gap))
            x += gap
            for k in range(height + down_top - 1):
                column(x, min(height, height + down_top - 1 - k))
                x += TILE
        after = rng.randint(*flat)
        x += after * TILE
    if flag:
        # The closing staircase, then the flag on the floor beyond.
        height = rng.randint(4, 8)
        for k in range(height):
            column(x, k + 1)
            x += TILE
        column(x, height)
        x += TILE + 4 * TILE
        goal = x
        width = x + END_FLAT * TILE
    else:
        goal = None
        passed = x - after * TILE + SCENE_PASSED
        width = passed + SCENE_AFTER * TILE

    # The floor, cut at pits and where pipes and steps stand.
    cuts = sorted(pits + [(s[0], s[0] + s[2]) for s in solids if s[4] in ("pipe", "ground", "cannon")])
    floor, start = [], 0
    for x0, x1 in cuts:
        if x0 > start:
            floor.append([start, FLOOR, x0 - start, BOTTOM - FLOOR, "ground"])
        start = max(start, x1)
    floor.append([start, FLOOR, width - start, BOTTOM - FLOOR, "ground"])
    platforms = floor + solids
    big = rng.random() < BIG_START
    return {
        "mario_big": big,
        "block_contents": {len(floor) + k: c for k, c in contents.items()},
        "task": {"family": "game_level"},
        "world_width": width,
        "scroll_x": SCROLL_X,
        "frame_budget": width * FRAMES_PER_PIXEL,
        "mario": [START_X, FLOOR - 12],
        "platforms": [p[:4] for p in platforms],
        "platform_kinds": [p[4] for p in platforms],
        "enemies": enemies,
        **({"goal": [goal, 0, TILE, BOTTOM]} if flag else {}),
        "metadata": {"game_level": {"seed": seed, "pieces": pieces}, **({} if flag else {"scene": {"passed": passed}})},
    }


def skill_scene(seed: int, kind: str) -> dict:
    """The skill scene of ``kind`` (SCENE_KINDS) for ``seed`` (module
    notes): passed once Mario stands beyond metadata "scene"'s "passed"."""
    if kind not in SCENE_KINDS:
        raise ValueError(f"no skill scene {kind!r}: one of {SCENE_KINDS}")
    start = random.Random(seed).randint(*SCENE_START)
    if kind == "pair":
        scene = random_level(seed, 2, start_flat=start, flat=PAIR_FLAT, flag=False)
    else:
        scene = random_level(seed, 1, only=kind, start_flat=start, flag=False)
    scene["metadata"]["scene"]["kind"] = kind
    return scene
