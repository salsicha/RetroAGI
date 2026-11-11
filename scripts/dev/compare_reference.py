"""Compare two folders of recorded behaviour (the refactoring check): random
levels played (levels_*.json), outcomes sampled (samples/part_*.npz) and
obstacle benches (mem*.json, vis*.json, scripted_*.json). Play is
deterministic, so a behaviour-preserving change gives the same files.

    python scripts/dev/compare_reference.py REFERENCE NEW
"""

import json
import sys
from pathlib import Path

import numpy as np


def _results(path: Path):
    """A bench's tries per obstacle, or the file's whole content."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "obstacles" in data:
        return [(o["name"], o["tries"]) for o in data["obstacles"]]
    return data


def compare(reference: Path, new: Path) -> list:
    """What differs, one line each (empty: the same)."""
    out = []
    for path in sorted(new.glob("*.json")):
        ref = reference / path.name
        if not ref.exists():
            out.append(f"{path.name}: no reference")
            continue
        if _results(ref) != _results(path):
            out.append(f"{path.name}: differs")
    parts = sorted((new / "samples").glob("part_*.npz")) if (new / "samples").exists() else []
    for part in parts:
        a, b = dict(np.load(reference / "samples" / part.name)), dict(np.load(part))
        for key in a:
            if key not in b or a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
                out.append(f"samples/{part.name}: {key} differs")
    return out


if __name__ == "__main__":
    differences = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print("\n".join(differences) or "identical")
    raise SystemExit(1 if differences else 0)
