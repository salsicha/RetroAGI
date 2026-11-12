"""Random levels made of the real game's pieces (docs/choose-and-control-plan.md
phase 2; user, 2026-11-07: "The levels could be randomly generated and I
could still pass them").

A level is a run of the game's pieces in a random order, in pixels as the
game draws them (16-pixel tiles, the floor's top at 208). No level is a
real level's layout (docs/requirements.md): the kinds of piece, their sizes
and how close they stand were measured in the game (World 1-1 at first;
since 2026-11-11 Worlds 2-1, 3-1 and 5-1, scripts/dev/piece_catalogue.py),
and each level draws its own from those ranges (random_level):

- pits 2 to 4 tiles wide, some with a row of bricks over them;
- pipes 2 to 6 tiles tall, some with a piranha plant; one too tall to jump
  onto from the floor has a way up before it (a lower pipe, a step, or a
  ledge with a pit between);
- staircases: up and down with a valley or a pit between, up only (to 6
  tiles, Goombas sometimes on its steps, sometimes a pit at its foot), or a
  wall with steps down; lone steps one or two columns wide;
- a bridge: steps up to a walkway 4 tiles above a pit 5 to 8 tiles wide,
  Goombas sometimes on it;
- rows of bricks and question blocks 4 or 5 and 8 tiles above the floor,
  enemies sometimes walking on them (a question block holds a coin or a
  mushroom, big Mario breaks bricks); question blocks over the floor;
- Goombas, Koopa Troopas (a stomp leaves a shell that a touch kicks) and
  hopping Koopa Paratroopas, alone or in files of two or three;
- Bullet Bill cannons;
- Hammer Bros: one or two under and on two rows of bricks 4 and 8 tiles up
  (3-1's), jumping from row to row and throwing hammers;
- and a staircase before the flag.

Pieces stand 0 to 8 tiles of floor apart. The floor is cut where a pipe or
a step stands, as the game draws it. Walkers follow the game's rules
(env.py "game_rules": they wake as they come on screen, walk left at half a
pixel a frame, turn at walls, fall off ledges), and the camera keeps Mario
where the real game does (SCROLL_X).

A skill scene (skill_scene) is a short stretch of the same pieces that
calls for one skill: one piece of a kind (SCENE_KINDS: a pipe, a pit,
enemies, bricks, stairs, question blocks, a cannon, a step, a bridge,
Hammer Bros) or two pieces close together ("pair"), Mario standing a few tiles before it, flat floor after
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
# (In the game pieces often touch: in Worlds 2-1, 3-1 and 5-1 a third of
# them stand less than 2 tiles from the one before.)
START_FLAT = 10
FLAT = (0, 8)
END_FLAT = 10
# The pieces and how often each comes.
PIECES = {"pipe": 3, "pit": 2, "goombas": 3, "bricks": 2, "stairs": 2, "questions": 1, "cannon": 1, "step": 1,
          "bridge": 1, "hammer_bros": 1}
# Hammer Bros' rows of bricks: their length (tiles; 3-1's are 10 and 11),
# the second Bro there this share of the time; each Bro's first throw and
# jump after a random wait up to its usual gap (sim/env.HAMMER_FRAMES and
# HAMMER_BRO_JUMPS).
HAMMER_ROW_TILES = (6, 11)
SECOND_BRO = 0.5
# Mario jumps onto a thing up to REACH tiles above where he stands. Pipes
# (tiles tall; in those three worlds 2 (3 of them), 3 (5), 4 (6), 5 (1), 6
# (1)): one taller than REACH has a way up before it (random_level).
REACH = 4
PIPE_TILES = (2, 2, 3, 3, 3, 4, 4, 4, 5, 6)
PIT_TILES = (2, 2, 3, 3, 4)
# A pit has a row of bricks over it this share of the time, at one of these
# heights or both (2-1: 8 tiles up; 3-1: 5 and 8).
ROOFED_PIT = 0.25
ROOFS = ((5,), (8,), (5, 8))
# Straight after a pit (less than 2 tiles of floor) nothing stands taller
# than this (5-1: a 3-tile pipe at a pit's far edge), and the next pit is
# at least BETWEEN_PITS tiles on (3-1: two tiles between two pits).
AFTER_PIT = 3
BETWEEN_PITS = 2
# Staircases: up and down (STAIR_TILES tall), up only (STAIR_UP tall: 2-1's
# is 5, 3-1's 6; Goombas on its steps ON_STAIRS of the time, a pit at its
# foot PIT_AT_FOOT of the time), or a wall with steps down. Each top is one
# or two columns wide.
STAIR_SHAPES = ("both", "both", "up", "down")
STAIR_TILES = (3, 4)
STAIR_UP = (3, 4, 5, 6)
STAIR_TOPS = (1, 2)
ON_STAIRS = 0.3
PIT_AT_FOOT = 0.4
# A bridge's pit (tiles wide; 3-1's is 8) and the Goombas on its walkway.
BRIDGE_TILES = (5, 8)
ON_BRIDGE = (0, 0, 1, 2, 3)
# Brick rows: the lower one's height (tiles above the floor) and length,
# the upper one's top (pixels) and length; an enemy walks on the lower one
# ON_ROW of the time.
LOW_ROWS = (4, 4, 4, 5)
LOW_ROW, HIGH_ROW = FLOOR - 4 * TILE, FLOOR - 8 * TILE
LOW_TILES, HIGH_TILES = (1, 2, 3, 3, 4, 5, 5, 8, 11), (3, 8)
ON_ROW = 0.25
# In a brick row this share of tiles are question blocks; a question block
# holds a mushroom MUSHROOM_SHARE of the time, else a coin. The "questions"
# piece is 1 to 3 question blocks over the floor (1-1's first).
QUESTION_SHARE = 0.3
MUSHROOM_SHARE = 0.35
# The share of levels Mario starts big (as after 1-1's first mushroom). Both
# shares are above the game's so the outcome network sees mushrooms and big
# Mario often enough (at the game's, sampled states almost never had one).
BIG_START = 0.3
# Walkers in a file are this far apart (pixels), as 1-1's pairs, and a file
# is one of these long (the game's Koopas and Goombas come in twos and
# threes as often as alone); an enemy piece is of Koopa Troopas this share
# of the time.
PAIR_GAP = 24
GROUPS = (1, 1, 1, 2, 2, 3)
KOOPA_SHARE = 0.25
# World 5-1's things: a pipe holds a piranha plant PLANT_SHARE of the time
# (at a random point of its cycle); an enemy piece is of Koopa Paratroopas
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

    def pipe(x0, tiles):
        top = FLOOR - tiles * TILE
        solids.append([x0, top, 2 * TILE, BOTTOM - top, "pipe"])
        if rng.random() < PLANT_SHARE:
            enemies.append({"kind": "piranha_plant", "x": float(x0 + TILE // 2), "pipe_top": float(top),
                            "phase": rng.randrange(PLANT_CYCLE)})

    def pit(x0, tiles):
        pits.append((x0, x0 + tiles * TILE))

    def way_up(x0, tiles):
        """What stands before a pipe ``tiles`` tall, too tall to jump onto
        from the floor, and where the pipe then starts: a lower pipe (as in
        2-1), a step (3-1), or a ledge with a pit between (5-1)."""
        way = rng.choice(("pipe", "step", "ledge"))
        if way == "pipe":
            pipe(x0, rng.choice((3, 4)))
            return x0 + (2 + rng.randint(1, 2)) * TILE
        if way == "step":
            column(x0, rng.randint(tiles - 3, REACH))
            return x0 + (1 + rng.randint(0, 2)) * TILE
        for k in range(3):
            column(x0 + k * TILE, REACH)
        width = rng.randint(2, 3)
        pit(x0 + 3 * TILE, width)
        return x0 + (3 + width + rng.randint(0, 1)) * TILE

    after, behind_pit = 0, False  # (tiles of floor after the last piece; whether it ended in a pit)
    for _ in range(pieces):
        kind = only or rng.choices(list(PIECES), weights=list(PIECES.values()))[0]
        capped = behind_pit and after < 2  # (nothing tall at a pit's far edge: AFTER_PIT)
        ends_in_pit = False
        if kind == "pipe":
            tiles = min(rng.choice(PIPE_TILES), AFTER_PIT if capped else 9)
            if tiles > REACH:
                x = way_up(x, tiles)
            pipe(x, tiles)
            x += 2 * TILE
        elif kind == "cannon":
            top = FLOOR - rng.choice(CANNON_TILES) * TILE
            solids.append([x, top, TILE, BOTTOM - top, "cannon"])
            x += TILE
        elif kind == "pit":
            if behind_pit and after < BETWEEN_PITS:  # (two pits never run together)
                x += (BETWEEN_PITS - after) * TILE
            width = rng.choice(PIT_TILES)
            pit(x, width)
            if rng.random() < ROOFED_PIT:
                for tiles in rng.choice(ROOFS):
                    row(x, FLOOR - tiles * TILE, width)
            x += width * TILE
            ends_in_pit = True
        elif kind == "goombas":
            roll = rng.random()
            walker = ("paratroopa" if roll < PARATROOPA_SHARE
                      else "koopa" if roll < PARATROOPA_SHARE + KOOPA_SHARE else None)
            count = rng.choice(GROUPS)
            for k in range(count):
                enemies.append(_goomba(x + k * PAIR_GAP, FLOOR, walker))
            x += (2 + count) * TILE
        elif kind == "questions":
            for k in range(rng.randint(1, 3)):
                block(x + 2 * k * TILE, LOW_ROW, True)
            x += 5 * TILE
        elif kind == "bricks":
            low_top = FLOOR - rng.choice(LOW_ROWS) * TILE
            low = rng.choice(LOW_TILES)
            row(x, low_top, low)
            if low >= 2 and rng.random() < ON_ROW:
                enemies.append(_goomba(x + rng.randrange(1, low) * TILE, low_top, rng.choice((None, "koopa"))))
            end = x + low * TILE
            if rng.random() < 0.5:
                start = x + rng.randint(1, 2) * TILE
                high = rng.randint(*HIGH_TILES)
                row(start, HIGH_ROW, high)
                if rng.random() < 0.5:
                    enemies.append(_goomba(start + (high - 1) * TILE, HIGH_ROW))
                end = max(end, start + high * TILE)
            x = end
        elif kind == "stairs":
            shape = rng.choice(STAIR_SHAPES)
            if shape == "down" and capped:
                shape = "up"  # (a wall would stand at the pit's far edge)
            if shape == "both":  # up, a valley of floor or a pit, down
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
            elif shape == "up":  # up, then the drop from its top
                height, first = rng.choice(STAIR_UP), x
                for k in range(height + rng.choice(STAIR_TOPS) - 1):
                    column(x, min(k + 1, height))
                    x += TILE
                if rng.random() < ON_STAIRS:  # Goombas on its steps, coming down to meet him
                    for k in sorted(rng.sample(range(1, height), min(height - 1, rng.choice(GROUPS)))):
                        enemies.append(_goomba(first + k * TILE, FLOOR - (k + 1) * TILE))
                if rng.random() < PIT_AT_FOOT:
                    width = rng.choice((2, 2, 3))
                    pit(x, width)
                    x += width * TILE
                    ends_in_pit = True
            else:  # a wall he jumps onto, then steps down by one or two (2-1: 4 then 2)
                height = rng.choice(STAIR_TILES)
                while height > 0:
                    column(x, height)
                    x += TILE
                    height -= rng.choice((1, 2))
        elif kind == "step":  # one column, or two of unlike heights (3-1: 3 then 6)
            first = rng.randint(1, AFTER_PIT if capped else REACH)
            column(x, first)
            x += TILE
            if rng.random() < 0.5:
                column(x, rng.randint(max(1, first - 3), min(6, first + 3)))
                x += TILE
        elif kind == "hammer_bros":  # two rows of bricks, a Bro under them and perhaps one on the lower
            length = rng.randint(*HAMMER_ROW_TILES)
            row(x, LOW_ROW, length)
            row(x, HIGH_ROW, length - rng.randint(0, 1))
            spots = [(x + 2 * TILE, FLOOR)] + ([(x + (length - 3) * TILE, LOW_ROW)] if rng.random() < SECOND_BRO else [])
            for bx, top in spots:
                enemies.append({**_goomba(bx, top, "hammer_bro"), "throw_wait": rng.randint(1, 50),
                                "jump_wait": rng.randint(1, 241)})
            x += length * TILE
        else:  # bridge: steps up, a walkway over a long pit, a step or two down
            for k in range(REACH):
                column(x, k + 1)
                x += TILE
            width, top = rng.randint(*BRIDGE_TILES), FLOOR - REACH * TILE
            pit(x, width)
            solids.append([x, top, width * TILE, TILE, "ground"])
            for k in range(rng.choice(ON_BRIDGE)):
                enemies.append(_goomba(x + (width - 1) * TILE - k * PAIR_GAP, top))
            x += width * TILE
            column(x, REACH)
            x += TILE
            if rng.random() < 0.5:
                column(x, 2)
                x += TILE
        behind_pit = ends_in_pit
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

    # The floor, cut at pits and where pipes and steps stand (not under a bridge's walkway, which is in the air).
    cuts = sorted(pits + [(s[0], s[0] + s[2]) for s in solids
                          if s[4] in ("pipe", "ground", "cannon") and s[1] + s[3] >= FLOOR])
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
