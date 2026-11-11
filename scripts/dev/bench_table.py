"""A table of obstacle bench results (framework/bench.py's JSON files).

For each file: the share of obstacles passed, and of the plays from
distinct start states that got past their farthest obstacle, with a 90%
interval from resampling the starts (obstacles sharing a start are not
independent: one death there counts against all of them).

    python scripts/dev/bench_table.py RESULT.json [RESULT.json ...]
"""

import json
import random
import sys
from pathlib import Path


def by_start(results: dict) -> list:
    """Per distinct start: the share of its tries (every mode and seed) that
    passed its farthest obstacle."""
    farthest = {}
    for obstacle in results["obstacles"]:
        farthest[obstacle["start_x"]] = obstacle  # (obstacles are in level order)
    shares = []
    for obstacle in farthest.values():
        tries = [t for mode_tries in obstacle["tries"].values() for t in mode_tries]
        shares.append(sum(t["end"] == "passed" for t in tries) / max(1, len(tries)))
    return shares


def interval(shares: list, draws: int = 2000, seed: int = 0) -> tuple:
    """The 5th and 95th percentiles of the mean of ``shares`` resampled."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(shares, k=len(shares))) / len(shares) for _ in range(draws))
    return means[int(0.05 * draws)], means[int(0.95 * draws)]


def main(argv=None) -> int:
    print(f"{'file':44} {'level':9} {'obstacles':>9} {'passed':>7} {'starts':>6} {'by start':>8}  90% interval")
    for name in argv if argv is not None else sys.argv[1:]:
        results = json.loads(Path(name).read_text())
        shares = by_start(results)
        low, high = interval(shares)
        print(f"{Path(name).name:44} {results['level']:9} {len(results['obstacles']):9d} {results['passed']:7.2f} "
              f"{len(shares):6d} {sum(shares) / max(1, len(shares)):8.2f}  {low:.2f}-{high:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
