"""Stage 2: scriptable low-resolution Super Mario Bros environment."""

from .adapter import (
    BLOCK_SMB_SPEC,
    SCENARIOS_DIR,
    BlockSMBObservationConfig,
    BlockSMBStage,
)
from .env import BlockSMBRewardConfig, MarioScenarioEnv
from .train import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
    BlockSMBReplayBuffer,
    BlockSMBTrainingConfig,
    BlockSMBTrajectory,
    BlockSMBTransition,
    SequentialBlockSMBVectorEnv,
    build_curriculum,
    evaluate_block_smb,
    restore_block_smb_checkpoint,
    save_block_smb_checkpoint,
    train_and_evaluate_block_smb,
    train_block_smb_epoch,
)
from .success import (
    FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS,
    BlockSMBSuccessThreshold,
    evaluate_fixed_success_thresholds,
    evaluate_success_threshold,
    fixed_scenario_success_threshold,
)
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
    "BLOCK_SMB_CHECKPOINT_KIND",
    "BLOCK_SMB_MODEL_NAME",
    "BlockSMBReplayBuffer",
    "BlockSMBTrainingConfig",
    "BlockSMBTrajectory",
    "BlockSMBTransition",
    "SequentialBlockSMBVectorEnv",
    "build_curriculum",
    "evaluate_block_smb",
    "restore_block_smb_checkpoint",
    "save_block_smb_checkpoint",
    "train_and_evaluate_block_smb",
    "train_block_smb_epoch",
    "BLOCK_SEMANTIC_CLASSES",
    "BLOCK_SMB_SPEC",
    "BlockSMBObservationConfig",
    "BlockSMBRewardConfig",
    "BlockSMBSuccessThreshold",
    "BlockSMBStage",
    "BlockVITLoadResult",
    "BlockVisionTransformer",
    "DEFAULT_BLOCK_VIT_CHECKPOINT",
    "FALLBACK_BLOCK_VIT_CHECKPOINT",
    "FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS",
    "MarioScenarioEnv",
    "SCENARIOS_DIR",
    "evaluate_fixed_success_thresholds",
    "evaluate_success_threshold",
    "fixed_scenario_success_threshold",
    "load_block_vit_checkpoint",
    "set_block_vit_trainable",
]
