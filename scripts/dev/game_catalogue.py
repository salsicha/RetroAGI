"""Every kind of thing the real game puts in its levels, and in which: read
from the game's memory while Mario is flown through each level, beside what
Block SMB has. A measuring device for building Block SMB
(docs/requirements.md, section 3): it says what Block SMB lacks; no layout
goes anywhere.

Each level is reached by writing its world, level and area numbers into the
game's memory and letting the timer run out (the game reloads the level
those numbers name). Mario is then held near the top of the screen, made
invincible and walked right, pushed on through walls; every frame the
game's enemy slots are read (which kind is in each). Things the game makes
as it goes (Cheep Cheeps, Bullet Bills, a Lakitu's Spinies) are caught as
well as those its level lists place. The three castles whose corridors loop
(4-4, 7-4, 8-4) are not flown to their ends.

The levels the finished agent is scored on (SCORED) are not read.

    python scripts/dev/game_catalogue.py [--output catalogue.json] [--workers 4]
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCORED = ("1-1", "4-1", "6-1", "7-1", "8-1")
# Game memory (the disassembly's names).
WORLD, LEVEL, AREA, HALFWAY, LIVES = 0x75F, 0x75C, 0x760, 0x75B, 0x75A  # WorldNumber, LevelNumber, AreaNumber, HalfwayPage, NumberofLives
TIMER = 0x7F8  # GameTimerDisplay: three digits
ENGINE, PLAYING, FLAGPOLE, LEVEL_END = 0x0E, 8, 4, 5  # GameEngineSubroutine and three of its values
PAGE, X = 0x6D, 0x86  # Player_PageLoc, Player_X_Position
Y, Y_PAGE, Y_SPEED, Y_FORCE, STAR = 0xCE, 0xB5, 0x9F, 0x433, 0x79F  # Player_Y_Position, ..._HighPos, ..._Y_Speed, ..._Y_MoveForce, StarInvincibleTimer
ENEMY_ON, ENEMY_ID, SLOTS = 0x0F, 0x16, 5  # Enemy_Flag, Enemy_ID
ENEMY_PAGE, ENEMY_X, ENEMY_Y_PAGE, ENEMY_Y = 0x6E, 0x87, 0xB6, 0xCF  # Enemy_PageLoc, _X_Position, _Y_HighPos, _Y_Position
CAMERA_PAGE, CAMERA_X = 0x71A, 0x71C  # ScreenLeft_PageLoc, ScreenLeft_X_Pos
# The block buffer (Block_Buffer_1 and _2): two screens of 16 columns by 13
# rows of the game's block codes, its first row 32 pixels from the top; each
# column is read while it is on the screen (pixel_labels.BLOCK_TYPES says
# what a code is).
BLOCKS, BLOCK_ROWS, BLOCK_TOP = 0x500, 13, 32
AREA_TYPE = 0x74E  # AreaType: 0 under water, 1 above ground, 2 underground, 3 castle
AREA_TYPES = ("water", "ground", "underground", "castle")
# Worlds with a short walk to a pipe before level 2: an area of its own,
# so level 2 on is one area further.
INTRO_BEFORE_2 = (1, 2, 4, 7)
FLY_HEIGHT = 20  # Mario's top, pixels from the screen's top: above pipes, steps and ceilings
STALLED = 900  # frames without getting further: the level's end, or a loop

# The game's codes (the disassembly's names).
NAMES = {
    0x00: "green Koopa", 0x02: "Buzzy Beetle", 0x03: "red Koopa", 0x05: "Hammer Bro", 0x06: "Goomba",
    0x07: "Blooper", 0x08: "Bullet Bill (frenzy)", 0x0A: "grey Cheep Cheep", 0x0B: "red Cheep Cheep",
    0x0C: "Podoboo (lava fireball)", 0x0D: "Piranha Plant", 0x0E: "green Paratroopa (hopping)",
    0x0F: "red Paratroopa (flying up and down)", 0x10: "green Paratroopa (flying across)", 0x11: "Lakitu",
    0x12: "Spiny", 0x14: "leaping Cheep Cheeps (frenzy)", 0x15: "Bowser's flames", 0x17: "Bullet Bill frenzy",
    0x1B: "fire bar (slow, clockwise)", 0x1C: "fire bar (fast, clockwise)", 0x1D: "fire bar (slow, anticlockwise)",
    0x1E: "fire bar (fast, anticlockwise)", 0x1F: "long fire bar",
    0x24: "balance lifts", 0x25: "lift going up and down", 0x26: "lift going up (looping)",
    0x27: "lift going down (looping)", 0x28: "lift going side to side", 0x29: "lift that falls when stood on",
    0x2A: "lift going right", 0x2B: "short lift going up (looping)", 0x2C: "short lift going down (looping)",
    0x2D: "Bowser", 0x2E: "power-up", 0x32: "springboard", 0x33: "Bullet Bill (from a cannon)",
}
# What Block SMB has, by the same codes (sim/levels.py, sim/env.py; a Buzzy
# Beetle walks and leaves a shell as a Koopa does, which without fireballs
# makes it one).
IN_BLOCK_SMB = {0x00, 0x02, 0x03, 0x05, 0x06, 0x08, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x14, 0x15, 0x17, 0x1B,
                0x1C, 0x1D, 0x1F, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2B, 0x2C, 0x2D, 0x2E, 0x32, 0x33}
# Not things to meet: the flag, the warp zone, Toad.
NOT_MET = {0x30, 0x31, 0x34, 0x35}


def area_of(world: int, level: int) -> int:
    return level - 1 + (world in INTRO_BEFORE_2 and level >= 2)


def fly(world: int, level: int, frames: int = 9000) -> dict:
    """Mario flown through one level: for each code met, the screens
    (256-pixel pages) it was met on and the places (x, y) where each was
    first seen; how far he got; the level's blocks, a column of BLOCK_ROWS
    block codes for each 16 pixels."""
    import retro

    env = retro.make("SuperMarioBros-Nes", state="Level2-1", inttype=retro.data.Integrations.STABLE, render_mode=None)
    env.reset()
    idle = np.zeros(len(env.buttons), dtype=np.int8)
    run = idle.copy()
    run[env.buttons.index("RIGHT")] = run[env.buttons.index("B")] = 1
    memory = env.data.memory

    def write(address, value):
        memory.assign(address, "|u1", value)

    for _ in range(5):
        env.step(idle)
    for address, value in ((WORLD, world - 1), (LEVEL, level - 1), (AREA, area_of(world, level)), (HALFWAY, 0),
                           (LIVES, 5), (TIMER, 0), (TIMER + 1, 0), (TIMER + 2, 1)):
        write(address, value)
    here = (world - 1, level - 1)
    started, seen, furthest, still, frame, columns, area = False, defaultdict(set), 0, 0, 0, {}, None
    places, present = defaultdict(set), [False] * SLOTS
    try:
        for frame in range(frames):
            ram = env.get_ram()
            started = started or (frame > 300 and ram[ENGINE] == PLAYING and (ram[WORLD], ram[LEVEL]) == here)
            if started:
                for address, value in ((Y, FLY_HEIGHT), (Y_PAGE, 1), (Y_SPEED, 0), (Y_FORCE, 0), (STAR, 30), (TIMER, 3)):
                    write(address, value)
            env.step(run if started else idle)
            if not started:
                continue
            ram = env.get_ram()
            area = AREA_TYPES[int(ram[AREA_TYPE]) & 3]
            x = int(ram[PAGE]) * 256 + int(ram[X])
            for slot in range(SLOTS):
                on = bool(ram[ENEMY_ON + slot])
                if on:
                    seen[int(ram[ENEMY_ID + slot])].add(x // 256)
                if on and not present[slot]:  # (where each thing is when it is first seen)
                    places[int(ram[ENEMY_ID + slot])].add((int(ram[ENEMY_PAGE + slot]) * 256 + int(ram[ENEMY_X + slot]),
                                                           int(ram[ENEMY_Y + slot])))
                present[slot] = on
            first = (int(ram[CAMERA_PAGE]) * 256 + int(ram[CAMERA_X])) // 16
            for column in range(first, first + 16):
                at = BLOCKS + ((column // 16) % 2) * 16 * BLOCK_ROWS + column % 16
                columns[column] = [int(ram[at + 16 * row]) for row in range(BLOCK_ROWS)]
            still = still + 1 if x <= furthest else 0
            furthest = max(furthest, x)
            if still and still % 6 == 0:  # against a wall: on through it
                x += 16 * (still // 6)
                write(PAGE, x // 256)
                write(X, x % 256)
            if ram[ENGINE] in (FLAGPOLE, LEVEL_END) or (ram[WORLD], ram[LEVEL]) != here or still > STALLED:
                break
    finally:
        env.close()
    return {"furthest": furthest, "frames": frame, "seen": {code: sorted(pages) for code, pages in sorted(seen.items())},
            "places": {code: sorted(where) for code, where in sorted(places.items())},
            "blocks": [columns.get(c, [0] * BLOCK_ROWS) for c in range(max(columns, default=-1) + 1)],
            "area": area}


def _fly(name: str):
    world, level = map(int, name.split("-"))
    return name, fly(world, level)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    names = [f"{w}-{l}" for w in range(1, 9) for l in range(1, 5) if f"{w}-{l}" not in SCORED]
    with Pool(args.workers) as pool:
        flown = dict(pool.map(_fly, names))
    where = defaultdict(list)
    for name, result in flown.items():
        for code in result["seen"]:
            if code not in NOT_MET:
                where[code].append(name)
    rows = [{"code": code, "name": NAMES.get(code, f"code 0x{code:02X}"), "in_block_smb": code in IN_BLOCK_SMB,
             "levels": levels} for code, levels in where.items()]
    rows.sort(key=lambda r: (r["in_block_smb"], -len(r["levels"]), r["code"]))
    for row in rows:
        print(f"{'has' if row['in_block_smb'] else 'LACKS':5}  {row['name']:38} {len(row['levels']):2} levels: "
              f"{', '.join(row['levels'])}")
    if args.output:
        args.output.write_text(json.dumps({"levels": flown, "kinds": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
