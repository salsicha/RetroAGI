"""What real levels are made of against what Block SMB makes: the pieces of
levels mapped from the game's memory (nes/obstacles.py: their pipes, pits,
steps and enemies) with their sizes and spacing, beside the ranges the
random-level generator draws from (sim/levels.py). Where the game goes
beyond Block SMB, Block SMB is what gets widened, never the policy.

Only levels set aside for building Block SMB are read (the default: worlds
2-1, 3-1 and 5-1); the others are kept for scoring the finished agent.

    python scripts/dev/piece_catalogue.py [--levels Level2-1 Level3-1 Level5-1]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from retroagi.domains.smb.nes.truth import KIND_NAMES
from retroagi.domains.smb.sim import levels as made

CALIBRATION = ("Level2-1", "Level3-1", "Level5-1")
# The enemies Block SMB has, by the game's code.
IN_BLOCK_SMB = {0x06: "Goomba", 0x00: "green Koopa", 0x0E: "hopping Paratroopa", 0x0D: "Piranha Plant",
                0x33: "Bullet Bill"}


def tiles(pixels: float) -> float:
    return round(pixels / made.TILE, 1)


def spans(values) -> str:
    counts = Counter(values)
    return ", ".join(f"{value:g} ({n})" for value, n in sorted(counts.items()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--levels", nargs="*", default=CALIBRATION)
    parser.add_argument("--benches", type=Path, default=Path("data/full_smb_obstacles"))
    args = parser.parse_args(argv)
    pipes, pits, steps, enemies, gaps = [], [], [], Counter(), []
    for level in args.levels:
        level_map = json.loads((args.benches / level / "index.json").read_text())["map"]
        pipes += [tiles(made.FLOOR - top) for _, _, top in level_map["pipes"]]
        pits += [tiles(x1 - x0) for x0, x1 in level_map["pits"]]
        steps += [tiles(made.FLOOR - top) for _, _, top in level_map["steps"]]
        enemies.update(kind for _, _, kind in level_map["enemies"])
        # Floor between one piece's end and the next one's start (pipes, pits and steps).
        pieces = sorted([(x0, x1) for x0, x1, _ in level_map["pipes"]] + [tuple(p) for p in level_map["pits"]]
                        + [(x0, x1) for x0, x1, _ in level_map["steps"]])
        gaps += [tiles(b[0] - a[1]) for a, b in zip(pieces, pieces[1:]) if b[0] - a[1] >= 0]
    print(f"levels read: {', '.join(args.levels)}")
    rows = [
        ("pipe heights (tiles)", spans(pipes), f"{made.PIPE_TILES}", max(pipes, default=0) <= max(made.PIPE_TILES)
         and min(pipes, default=9) >= min(made.PIPE_TILES)),
        ("pit widths (tiles)", spans(pits), f"{tuple(sorted(set(made.PIT_TILES)))}, to {made.BRIDGE_TILES[1]} bridged",
         max(pits, default=0) <= made.BRIDGE_TILES[1] and min(pits, default=9) >= min(made.PIT_TILES)),
        ("step heights (tiles)", spans(steps), f"1 to {max(made.STAIR_UP)}, 8 before the flag",
         max(steps, default=0) <= 8),
        ("floor between pieces (tiles)", spans(g for g in gaps if g <= 8), f"{made.FLAT[0]} to {made.FLAT[1]} "
         f"(pairs {made.PAIR_FLAT[0]} to {made.PAIR_FLAT[1]})", min(gaps, default=9) >= made.PAIR_FLAT[0]),
    ]
    print(f"{'piece':30} {'in the game (how many)':62} {'Block SMB draws':28} covered")
    for name, game, block, covered in rows:
        print(f"{name:30} {game:62} {block:28} {'yes' if covered else 'NO'}")
    print("enemies first seen, by kind:")
    for kind, count in enemies.most_common():
        name = KIND_NAMES.get(kind, f"object {kind:02X}")
        print(f"  {name:28} {count:4d}  {'in Block SMB' if kind in IN_BLOCK_SMB else 'NOT in Block SMB'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
