"""What the real game's levels are shaped like against what Block SMB makes:
from the block maps game_catalogue.py reads out of the game's memory (each
level not scored on, flown through), the pits, the floating ledges, the
ceilings and the stretches with no floor, beside the ranges Block SMB's
random levels draw from (sim/levels.py). A measuring device for building
Block SMB (docs/requirements.md, section 3): where the game goes beyond
Block SMB, Block SMB is what gets widened.

    python scripts/dev/level_shapes.py artifacts/catalogue/game_catalogue.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from retroagi.domains.smb.nes.pixel_labels import BLOCK_TYPES
from retroagi.domains.smb.sim import levels as made

SOLID = ("ground", "brick", "pipe", "question_block")
FLOOR_ROW = 11  # the block row whose top is the floor's (208 pixels down)
LOOPING = ("4-4", "7-4", "8-4")


def solid(code: int) -> bool:
    return BLOCK_TYPES.get(code) in SOLID


def runs(flags) -> list:
    """(start, length) of each run of True in ``flags``."""
    out, start = [], None
    for k, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            out.append((start, k - start))
            start = None
    return out


def shapes(blocks: list, area: str) -> dict:
    """A level's pits (tiles wide; a stretch with no floor), floating
    ledges (tiles wide, tiles above the floor; a run of solid blocks with
    nothing under it), and the share of it under a ceiling."""
    columns = len(blocks)
    floor = [solid(c[FLOOR_ROW]) or solid(c[FLOOR_ROW + 1]) for c in blocks]
    pits = [length for start, length in runs([not f for f in floor]) if start > 0 and start + length < columns]
    ledges = []
    for row in range(1, FLOOR_ROW):
        floating = [solid(c[row]) and not solid(c[row + 1]) and not solid(c[row - 1]) for c in blocks]
        ledges += [(length, FLOOR_ROW - row) for _, length in runs(floating)]
    ceiling = sum(solid(c[0]) for c in blocks) / max(1, columns)
    return {"pits": pits, "ledges": ledges, "ceiling": ceiling, "area": area, "columns": columns}


def spans(values) -> str:
    return ", ".join(f"{v} ({n})" for v, n in sorted(Counter(values).items())) or "none"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("catalogue", type=Path)
    args = parser.parse_args(argv)
    levels = json.loads(args.catalogue.read_text())["levels"]
    # (castles whose corridors loop are not flown to their ends: game_catalogue)
    every = {name: shapes(level["blocks"], level.get("area")) for name, level in sorted(levels.items())
             if name not in LOOPING}
    for name, s in every.items():
        print(f"{name}  {s['area'] or '?':11}  {s['columns']:3} columns  pits {spans(s['pits'])}")
    pits = [p for s in every.values() if s["area"] != "water" for p in s["pits"]]
    ledges = [l for s in every.values() if s["area"] != "water" for l in s["ledges"]]
    print()
    print(f"pit widths (tiles; a stretch with no floor, ledges or a bridge over it or not), levels not under water: "
          f"{spans(pits)}")
    print(f"   Block SMB: {sorted(set(made.PIT_TILES))}, bridged to {made.BRIDGE_TILES[1]}")
    print(f"floating ledge widths (tiles): {spans(w for w, _ in ledges)}")
    print(f"   Block SMB rows: {sorted(set(made.LOW_TILES))} and {made.HIGH_TILES}")
    print(f"floating ledge heights (tiles over the floor): {spans(h for _, h in ledges)}")
    print(f"   Block SMB: {sorted(set(made.LOW_ROWS))} and 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
