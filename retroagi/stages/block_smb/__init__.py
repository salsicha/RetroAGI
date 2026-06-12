"""Stage 2: scriptable low-resolution Super Mario Bros environment."""

from .adapter import BLOCK_SMB_SPEC, SCENARIOS_DIR, BlockSMBStage
from .env import MarioScenarioEnv
from .vision import BLOCK_SEMANTIC_CLASSES, BlockVisionTransformer

__all__ = [
    "BLOCK_SEMANTIC_CLASSES",
    "BLOCK_SMB_SPEC",
    "BlockSMBStage",
    "BlockVisionTransformer",
    "MarioScenarioEnv",
    "SCENARIOS_DIR",
]
