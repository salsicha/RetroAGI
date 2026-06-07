"""Compatibility wrapper for the full-SMB emulator runner."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.stages.full_smb.run import main


if __name__ == "__main__":
    steps = 200
    if len(sys.argv) > 2 and sys.argv[1] == "--steps":
        steps = int(sys.argv[2])
    main(num_steps=steps)
