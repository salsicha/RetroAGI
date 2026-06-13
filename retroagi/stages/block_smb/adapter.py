"""Adapter from the block-SMB pygame environment to the shared training contract."""

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from retroagi.core import (
    SMBAction,
    StageBatch,
    StageSpec,
    VisionEncoder,
    VisionHierarchyProjector,
    block_smb_action,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.vision import BlockVisionTransformer

BLOCK_SMB_SPEC = StageSpec(
    name="block_smb",
    observation_kind="low-resolution pygame RGB plus symbolic state_vec",
    action_kind="shared SMBAction vocabulary",
    seq_len_a=8,
    ratio_ab=2,
    ratio_bc=4,
    vocab_size=20,
)

SCENARIOS_DIR = Path(__file__).with_name("scenarios")


class BlockSMBStage:
    """Stage adapter for scriptable pygame scenarios."""

    spec = BLOCK_SMB_SPEC

    def __init__(
        self,
        env: Optional[MarioScenarioEnv] = None,
        scenario: Optional[dict] = None,
        vision: Optional[VisionEncoder] = None,
    ):
        self.env = env or MarioScenarioEnv()
        self.scenario = scenario
        self.vision = vision or BlockVisionTransformer()
        if isinstance(self.vision, torch.nn.Module):
            self.vision.eval()
        self.vision_projector = VisionHierarchyProjector(self.spec)
        self.last_info: Mapping[str, Any] = {}

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(scenario=self.scenario, seed=seed)
        self.last_info = info
        return obs

    def step(self, action: SMBAction | int):
        obs, reward, terminated, truncated, info = self.env.step(block_smb_action(action))
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def encode_observation(self, observation: np.ndarray, info: Optional[Mapping[str, Any]] = None) -> StageBatch:
        """Convert block-SMB vision and symbolic state into the hierarchy."""
        info = info or self.last_info
        state_vec = np.asarray(info["state_vec"], dtype=np.float32)
        with torch.no_grad():
            vision = self.vision.encode(observation)

        return self.vision_projector.project(
            vision,
            state=torch.as_tensor(state_vec, device=vision.position.device),
            metadata={"raw_observation_shape": observation.shape, "info": info},
        )

