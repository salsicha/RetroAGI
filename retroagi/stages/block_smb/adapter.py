"""Adapter from the block-SMB pygame environment to the shared training contract."""

from collections import deque
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BlockSMBObservationConfig:
    """Preprocessing contract for Block SMB policy observations."""

    frame_stack: int = 4
    state_min: float = -1.0
    state_max: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        if self.state_min >= self.state_max:
            raise ValueError("state_min must be smaller than state_max")


class BlockSMBStage:
    """Stage adapter for scriptable pygame scenarios."""

    spec = BLOCK_SMB_SPEC

    def __init__(
        self,
        env: Optional[MarioScenarioEnv] = None,
        scenario: Optional[dict] = None,
        vision: Optional[VisionEncoder] = None,
        observation_config: BlockSMBObservationConfig = BlockSMBObservationConfig(),
    ):
        self.env = env or MarioScenarioEnv()
        self.scenario = scenario
        self.vision = vision or BlockVisionTransformer()
        self.observation_config = observation_config
        if isinstance(self.vision, torch.nn.Module):
            self.vision.eval()
        self.vision_projector = VisionHierarchyProjector(self.spec)
        self.last_info: Mapping[str, Any] = {}
        self._frame_stack: deque[torch.Tensor] = deque(
            maxlen=self.observation_config.frame_stack
        )
        self._frame_mask: deque[bool] = deque(
            maxlen=self.observation_config.frame_stack
        )
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(scenario=self.scenario, seed=seed)
        self.last_info = info
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False
        self._reset_frame_stack(obs)
        return obs

    def step(self, action: SMBAction | int):
        obs, reward, terminated, truncated, info = self.env.step(
            block_smb_action(action)
        )
        self.last_info = info
        self._last_episode_mask = 0.0 if terminated or truncated else 1.0
        self._last_terminal = terminated
        self._last_truncated = truncated
        self._append_frame(obs, valid=True)
        return obs, reward, terminated, truncated, info

    def encode_observation(
        self, observation: np.ndarray, info: Optional[Mapping[str, Any]] = None
    ) -> StageBatch:
        """Convert block-SMB vision and symbolic state into the hierarchy."""
        info = info or self.last_info
        if not self._frame_stack:
            self._reset_frame_stack(observation)
        normalized_observation = self._normalize_observation(observation)
        if not torch.equal(self._frame_stack[-1], normalized_observation):
            self._append_frame(observation, valid=True)
        state_vec = self._normalize_state_vec(info["state_vec"])
        with torch.no_grad():
            vision = self.vision.encode(normalized_observation)

        return self.vision_projector.project(
            vision,
            state=torch.as_tensor(state_vec, device=vision.position.device),
            metadata={
                "raw_observation_shape": observation.shape,
                "observation": self._observation_metadata(vision.position.device),
                "episode": {
                    "mask": torch.tensor(
                        [self._last_episode_mask],
                        dtype=torch.float32,
                        device=vision.position.device,
                    ),
                    "terminated": self._last_terminal,
                    "truncated": self._last_truncated,
                },
                "info": info,
            },
        )

    def _reset_frame_stack(self, observation: np.ndarray) -> None:
        self._frame_stack.clear()
        self._frame_mask.clear()
        normalized = self._normalize_observation(observation)
        padding = self.observation_config.frame_stack - 1
        for _ in range(padding):
            self._frame_stack.append(normalized.clone())
            self._frame_mask.append(False)
        self._frame_stack.append(normalized)
        self._frame_mask.append(True)

    def _append_frame(self, observation: np.ndarray, *, valid: bool) -> None:
        self._frame_stack.append(self._normalize_observation(observation))
        self._frame_mask.append(valid)

    @staticmethod
    def _normalize_observation(observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[-1] not in (3, 4):
            raise ValueError(
                "Block SMB observations must have shape [H, W, C] with RGB or RGBA channels"
            )
        tensor = tensor[..., :3]
        if bool(tensor.numel()) and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def _normalize_state_vec(self, state_vec: Any) -> np.ndarray:
        state = np.asarray(state_vec, dtype=np.float32)
        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=self.observation_config.state_max,
            neginf=self.observation_config.state_min,
        )
        return np.clip(
            state,
            self.observation_config.state_min,
            self.observation_config.state_max,
        )

    def _observation_metadata(self, device: torch.device) -> dict[str, Any]:
        frame_stack = torch.stack(tuple(self._frame_stack), dim=0).permute(
            0, 3, 1, 2
        )
        return {
            "frame_stack": frame_stack.unsqueeze(0).to(device),
            "frame_mask": torch.tensor(
                tuple(self._frame_mask), dtype=torch.bool, device=device
            ).unsqueeze(0),
            "frame_stack_size": self.observation_config.frame_stack,
            "normalized_range": (0.0, 1.0),
            "state_range": (
                self.observation_config.state_min,
                self.observation_config.state_max,
            ),
        }

