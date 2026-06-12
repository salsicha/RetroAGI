"""Adapter from the block-SMB pygame environment to the shared training contract."""

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from retroagi.core import StageBatch, StageSpec, VisionEncoder
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.vision import BlockVisionTransformer


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
        """Convert block-SMB vision and symbolic state into the hierarchy."""
        info = info or self.last_info
        state_vec = np.asarray(info["state_vec"], dtype=np.float32)
        with torch.no_grad():
            vision = self.vision.encode(observation)

        semantic_ids = vision.semantic_ids.reshape(vision.semantic_ids.shape[0], -1)
        src_a = _resize_tokens(semantic_ids, self.spec.seq_len_a, self.spec.vocab_size)
        src_b = _resize_tokens(semantic_ids, self.spec.seq_len_b, self.spec.vocab_size)

        class_probs = vision.semantic_logits.softmax(dim=1).mean(dim=(-2, -1))
        visual_state = torch.cat(
            (
                vision.position.float(),
                class_probs.float(),
                torch.as_tensor(state_vec, device=vision.position.device).reshape(1, -1),
            ),
            dim=1,
        )
        src_c = _resize_features(visual_state, self.spec.seq_len_c)

        return StageBatch(
            src_a=src_a,
            target_a=None,
            src_b=src_b,
            target_b=None,
            src_c=src_c,
            target_c=None,
            metadata={"raw_observation_shape": observation.shape, "info": info, "vision": vision},
        )


def _resize_tokens(tokens: torch.Tensor, length: int, vocab_size: int) -> torch.Tensor:
    indices = torch.linspace(0, tokens.shape[1] - 1, length, device=tokens.device).long()
    return tokens.index_select(1, indices).long().remainder(vocab_size)


def _resize_features(features: torch.Tensor, length: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        features.unsqueeze(1), size=length, mode="linear", align_corners=False
    ).squeeze(1)

