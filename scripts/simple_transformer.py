"""Compatibility wrapper for the synthetic 1D curriculum stage."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.stages.synthetic_1d.train import *  # noqa: F401,F403
from retroagi.stages.synthetic_1d.train import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()
