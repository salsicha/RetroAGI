"""Stage 2: scriptable low-resolution Super Mario Bros environment."""

from .adapter import (
    BLOCK_SMB_SPEC,
    SCENARIOS_DIR,
    BlockSMBObservationConfig,
    BlockSMBStage,
)
from .env import MarioScenarioEnv
from .vision import (
    BLOCK_SEMANTIC_CLASSES,
    DEFAULT_BLOCK_VIT_CHECKPOINT,
    FALLBACK_BLOCK_VIT_CHECKPOINT,
    BlockVITLoadResult,
    BlockVisionTransformer,
    load_block_vit_checkpoint,
    set_block_vit_trainable,
)

__all__ = [
    "BLOCK_SEMANTIC_CLASSES",
    "BLOCK_SMB_SPEC",
    "BlockSMBObservationConfig",
    "BlockSMBStage",
    "BlockVITLoadResult",
    "BlockVisionTransformer",
    "DEFAULT_BLOCK_VIT_CHECKPOINT",
    "FALLBACK_BLOCK_VIT_CHECKPOINT",
    "MarioScenarioEnv",
    "SCENARIOS_DIR",
    "load_block_vit_checkpoint",
    "set_block_vit_trainable",
]
