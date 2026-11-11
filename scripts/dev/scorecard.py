"""How far from a working agent: the three numbers of docs/generalization.md
for one or more measurement runs (folders of framework/bench.py and
framework/play.py results named LEVEL_{memory,vision}_network.json and
LEVEL_play_SOURCE.json), on the held-out levels:

- whole levels finished from vision, of the plays made (the agent itself);
- the share of tries at an obstacle that fail, from vision (what moves
  before levels start to finish), and how many obstacles that is between
  failures;
- the same with the state from game memory (perfect perception: what
  choosing and controlling lose without seeing).

A level has about 35 obstacles: half its plays finish at about 2% of tries
failing, 9 in 10 at about 0.3%.

    python scripts/dev/scorecard.py FOLDER [FOLDER ...] [--levels Level4-1 Level6-1 Level7-1 Level8-1]
"""

import argparse
import json
from pathlib import Path

HELD_OUT = ("Level4-1", "Level6-1", "Level7-1", "Level8-1")


def failing(folder: Path, levels, agent: str):
    """The share of tries at an obstacle that did not pass, over ``levels``
    (None: not measured)."""
    passed = tries = 0
    for level in levels:
        path = folder / f"{level}_{agent}.json"
        if not path.exists():
            return None
        for obstacle in json.loads(path.read_text())["obstacles"]:
            passed += obstacle["ends"].get("passed", 0)
            tries += sum(obstacle["ends"].values())
    return 1 - passed / max(1, tries)


def finished(folder: Path, levels, source: str):
    """(plays finished, plays) of whole levels (None: not played)."""
    done = plays = 0
    for level in levels:
        path = folder / f"{level}_play_{source}.json"
        if not path.exists():
            return None
        rows = json.loads(path.read_text())
        done += sum(row["end"] == "finished" for row in rows)
        plays += len(rows)
    return done, plays


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--levels", nargs="*", default=HELD_OUT, help="the levels scored (the held-out ones)")
    args = parser.parse_args(argv)
    print(f"{'run':24} {'levels finished':>16} {'tries failing':>14} {'obstacles a':>12} {'failing with':>13} "
          f"{'obstacles a':>12}")
    print(f"{'':24} {'from vision':>16} {'from vision':>14} {'failure':>12} {'game memory':>13} {'failure':>12}")
    for folder in args.folders:
        levels = finished(folder, args.levels, "vision")
        seen, true = failing(folder, args.levels, "vision_network"), failing(folder, args.levels, "memory_network")
        cells = [
            "-" if levels is None else f"{levels[0]} of {levels[1]}",
            "-" if seen is None else f"{seen:.0%}",
            "-" if not seen else f"{1 / seen:.1f}",
            "-" if true is None else f"{true:.0%}",
            "-" if not true else f"{1 / true:.1f}",
        ]
        print(f"{folder.name:24} {cells[0]:>16} {cells[1]:>14} {cells[2]:>12} {cells[3]:>13} {cells[4]:>12}")
    print(f"{'a working agent':24} {'most':>16} {'about 2%':>14} {'about 50':>12} {'about 2%':>13} {'about 50':>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
