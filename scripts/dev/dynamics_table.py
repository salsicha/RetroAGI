"""Compare dynamics models' scores (nes/dynamics_check.py's JSON files)
pooled over levels: for each kind of enemy and each model, the share of
predictions within reach of the truth at each number of frames ahead.

    python scripts/dev/dynamics_table.py RESULT.json [RESULT.json ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    pooled = defaultdict(lambda: [0, 0.0])  # (model, kind, ahead) -> predictions, near ones
    models, levels = [], []
    for name in argv if argv is not None else sys.argv[1:]:
        results = json.loads(Path(name).read_text())
        levels.append(results.pop("level"))
        for model, rows in results.items():
            if model not in models:
                models.append(model)
            for row in rows:
                cell = pooled[(model, row["kind"], row["ahead"])]
                cell[0] += row["predictions"]
                cell[1] += row["near"] * row["predictions"]
    aheads = sorted({ahead for _, _, ahead in pooled})
    kinds = sorted({kind for _, kind, _ in pooled}, key=lambda kind: -pooled[(models[0], kind, aheads[0])][0])
    print(" ".join(levels))
    print(f"{'kind':28} {'predictions':>11}  " + "   ".join(
        f"{model + ' at ' + '/'.join(str(a) for a in aheads):>24}" for model in models))
    for kind in kinds:
        cells = ["  ".join(f"{pooled[(model, kind, a)][1] / max(1, pooled[(model, kind, a)][0]):.2f}" for a in aheads)
                 for model in models]
        print(f"{kind:28} {pooled[(models[0], kind, aheads[0])][0]:11d}  " + "   ".join(f"{c:>24}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
