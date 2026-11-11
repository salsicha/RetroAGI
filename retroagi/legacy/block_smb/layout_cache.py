"""Invalidate certified layouts and teacher routes when their code changes."""

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """One code version per process; both disk caches share the same contract."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    unused = {"layered_train.py", "cli.py", "vision.py", "vision_frames.py"}
    # The curriculum's own code, and the game's they are certified against:
    # its motion and controller, the simulator, perception.
    paths = [*root.glob("legacy/block_smb/*.py"), *root.glob("legacy/core/smb_*.py"), root / "legacy/core/tokens.py",
             *root.glob("domains/smb/*.py"), *root.glob("domains/smb/sim/*.py"),
             *root.glob("domains/smb/perception/*.py")]
    for path in sorted(paths):
        if path.name not in unused:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
