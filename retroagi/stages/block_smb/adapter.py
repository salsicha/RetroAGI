"""Adapter from the block-SMB pygame environment to the shared training contract."""

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from retroagi.core import StageBatch, StageSpec
from retroagi.stages.block_smb.env import MarioScenarioEnv


BLOCK_SMB_SPEC = StageSpec(
    name="block_smb",
    observation_kind="low-resolution pygame RGB plus symbolic state_vec",
    action_kind="discrete SMB movement buttons",
    seq_len_a=8,
    ratio_ab=2,
    ratio_bc=4,
    vocab_size=20,
)

SCENARIOS_DIR = Path(__file__).with_name("scenarios")


class BlockSMBStage:
    """Stage adapter for scriptable pygame scenarios."""

    spec = BLOCK_SMB_SPEC

    def __init__(self, env: Optional[MarioScenarioEnv] = None, scenario: Optional[dict] = None):
        self.env = env or MarioScenarioEnv()
        self.scenario = scenario
        self.last_info: Mapping[str, Any] = {}

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(scenario=self.scenario, seed=seed)
        self.last_info = info
        return obs

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def encode_observation(self, observation: np.ndarray, info: Optional[Mapping[str, Any]] = None) -> StageBatch:
        """Convert symbolic block-SMB state into the common A/B/C tensor layout.

        This is intentionally simple for now: C receives a resized state vector,
        while A/B are reserved token streams that later curriculum code can fill
        with explicit goals and mid-level plans.
        """
        info = info or self.last_info
        state_vec = np.asarray(info["state_vec"], dtype=np.float32)
        src_c = np.resize(state_vec, self.spec.seq_len_c).reshape(1, self.spec.seq_len_c)

        return StageBatch(
            src_a=torch.zeros((1, self.spec.seq_len_a), dtype=torch.long),
            target_a=None,
            src_b=torch.zeros((1, self.spec.seq_len_b), dtype=torch.long),
            target_b=None,
            src_c=torch.tensor(src_c, dtype=torch.float),
            target_c=None,
            metadata={"raw_observation_shape": observation.shape, "info": info},
        )

