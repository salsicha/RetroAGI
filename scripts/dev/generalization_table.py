"""One table of a measurement run over many levels (a folder of
framework/bench.py and framework/play.py results named
LEVEL_{memory,vision}_{network,scripted}.json and LEVEL_play_SOURCE.json):
per level and agent, the share of obstacles passed, the share of the plays
from distinct start states that got past their farthest obstacle, and
whole-level plays finished.

    python scripts/dev/generalization_table.py FOLDER [--tuned Level1-1 Level5-1]
"""

import argparse
import json
from pathlib import Path

from bench_table import by_start

AGENTS = ("memory_network", "memory_scripted", "vision_network")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--tuned", nargs="*", default=("Level1-1", "Level5-1"),
                        help="the levels the agent's rules were tuned on (listed first, pooled apart)")
    args = parser.parse_args(argv)
    levels = sorted({p.name.split("_")[0] for p in args.folder.glob("Level*_*.json")},
                    key=lambda level: (level not in args.tuned, level))
    agents = [a for a in AGENTS if any((args.folder / f"{level}_{a}.json").exists() for level in levels)]
    sources = [s for s in ("memory", "vision") if any((args.folder / f"{lv}_play_{s}.json").exists() for lv in levels)]
    print("| Level | " + " | ".join(a.replace("_", ", ") for a in agents)
          + "".join(f" | whole level, {s}" for s in sources) + " |")
    print("|---" * (1 + len(agents) + len(sources)) + "|")
    pooled = {}  # (tuned or not, agent) -> [obstacles passed, obstacles]
    for level in levels:
        cells = []
        for agent in agents:
            path = args.folder / f"{level}_{agent}.json"
            if not path.exists():
                cells.append("")
                continue
            results = json.loads(path.read_text())
            starts = by_start(results)
            cells.append(f"{results['passed']:.2f} ({sum(starts) / len(starts):.2f})")
            pool = pooled.setdefault((level in args.tuned, agent), [0.0, 0])
            pool[0] += results["passed"] * len(results["obstacles"])
            pool[1] += len(results["obstacles"])
        for source in sources:
            path = args.folder / f"{level}_play_{source}.json"
            if not path.exists():
                cells.append("")
                continue
            plays = json.loads(path.read_text())
            reached = sorted(p["progress"] for p in plays)
            cells.append(f"{sum(p['end'] == 'finished' for p in plays)}/{len(plays)}, x {reached[len(reached) // 2]}")
        print(f"| {level[5:]}{' (tuned)' if level in args.tuned else ''} | " + " | ".join(cells) + " |")
    for tuned in (True, False):
        cells = [f"**{pooled[(tuned, a)][0] / pooled[(tuned, a)][1]:.2f}**" if (tuned, a) in pooled else "" for a in agents]
        print(f"| **{'tuned' if tuned else 'untouched'}, all obstacles** | " + " | ".join(cells) + " |" * len(sources) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
