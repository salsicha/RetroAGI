"""Real levels copied into Block SMB (nes/clone.py, from the game's memory)
as practice worlds: the layouts of the levels set aside for building Block
SMB, with what the random levels do not have (a tall pipe behind a step, a
staircase over a pit, enemies on a row of bricks). The levels kept for
scoring the finished agent are not among them.

copied_level(seed) is one of the copies, Mario standing at its start or on
the floor somewhere along it (so its every part is met from many states),
its plants at a point of their cycle and its question blocks holding a coin
or a mushroom by the seed.
"""

import json
import random
from functools import lru_cache
from pathlib import Path

from .levels import BIG_START, FLOOR, MUSHROOM_SHARE, PLANT_CYCLE, START_X, TILE

COPIES = Path("data/block_smb_levels")
# Mario starts at the level's start this share of the time; else on a
# stretch of floor at least START_FLOOR tiles long, no enemy starting
# within CLEAR pixels ahead of him.
FROM_START = 0.25
START_FLOOR = 4
CLEAR = 96


@lru_cache(maxsize=None)
def copies() -> tuple:
    """The copied levels on disk (their scenarios), by name."""
    return tuple(json.loads(p.read_text()) for p in sorted(COPIES.glob("Level*-?.json")))


def copied_level(seed: int) -> dict:
    """The practice world of ``seed`` (module notes)."""
    levels = copies()
    if not levels:
        raise FileNotFoundError(f"no copied levels in {COPIES}: make them with nes/clone.py")
    rng = random.Random(seed)
    level = json.loads(json.dumps(levels[seed % len(levels)]))  # (its own copy to change)
    start = START_X
    if rng.random() >= FROM_START:
        floors = [p for p, kind in zip(level["platforms"], level["platform_kinds"])
                  if kind == "ground" and p[1] == FLOOR and p[2] >= START_FLOOR * TILE
                  and p[0] + p[2] < level["goal"][0] - 8 * TILE]
        spots = []
        for x0, _, width, _ in floors:
            for x in range(x0 + TILE, x0 + width - 2 * TILE, TILE):
                clear = not any(x - TILE <= e["x"] <= x + CLEAR for e in level["enemies"])
                under = not any(p[0] < x + 12 and x < p[0] + p[2] and p[1] < FLOOR and p[1] + p[3] > FLOOR - 2 * TILE
                                for p in level["platforms"])  # (no block or step where he stands)
                if clear and under:
                    spots.append(x)
        if spots:
            start = rng.choice(spots)
    level["mario"] = [start, FLOOR - 12]
    level["mario_big"] = rng.random() < BIG_START
    # What he has left behind is not met: the enemies that start before him are left out.
    level["enemies"] = [e for e in level["enemies"] if e["x"] > start + TILE]
    for enemy in level["enemies"]:
        if enemy.get("kind") == "piranha_plant":
            enemy["phase"] = rng.randrange(PLANT_CYCLE)
    level["block_contents"] = {k: ("mushroom" if rng.random() < MUSHROOM_SHARE else "coin")
                               for k in level["block_contents"]}
    level["metadata"]["game_level"]["start"] = start
    return level
