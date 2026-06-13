"""Common data contracts shared by every curriculum stage."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

import torch


@dataclass(frozen=True)
class StageSpec:
    """Resolution and timing metadata defined in docs/tensor-contracts.md."""

    name: str
    observation_kind: str
    action_kind: str
    seq_len_a: int
    ratio_ab: int
    ratio_bc: int
    vocab_size: int

    @property
    def seq_len_b(self) -> int:
        return self.seq_len_a * self.ratio_ab

    @property
    def seq_len_c(self) -> int:
        return self.seq_len_b * self.ratio_bc


@dataclass
class StageBatch:
    """Canonical hierarchy tensors defined in docs/tensor-contracts.md."""

    src_a: torch.Tensor
    target_a: Optional[torch.Tensor]
    src_b: torch.Tensor
    target_b: Optional[torch.Tensor]
    src_c: torch.Tensor
    target_c: Optional[torch.Tensor]
    metadata: Optional[Mapping[str, Any]] = None


@dataclass
class AgentStep:
    """Outputs from one actor/world-model/critic refinement step."""

    actions_first_pass: torch.Tensor
    next_state_prediction: torch.Tensor
    criticism: torch.Tensor
    actions_second_pass: torch.Tensor
    logits_a: torch.Tensor
    w_b: torch.Tensor
    b_b: torch.Tensor


@dataclass(frozen=True)
class VisionSpec:
    """Describes a stage vision encoder's spatial and semantic contract."""

    name: str
    semantic_classes: tuple[str, ...]
    token_dim: int
    position_dim: int = 2

    @property
    def num_classes(self) -> int:
        return len(self.semantic_classes)


@dataclass
class VisionOutput:
    """Stage-independent vision tensors defined in docs/tensor-contracts.md."""

    position: torch.Tensor
    semantic_logits: torch.Tensor
    semantic_ids: torch.Tensor
    tokens: torch.Tensor
    metadata: Optional[Mapping[str, Any]] = None


class VisionEncoder(Protocol):
    """Common interface for synthetic, block-SMB, and full-SMB perception."""

    spec: VisionSpec

    def encode(self, observation: Any) -> VisionOutput:
        """Extract normalized position, semantics, and latent tokens."""


class StageAdapter(Protocol):
    """Shared environment lifecycle described in docs/stage-semantics.md."""

    spec: StageSpec

    def reset(self, seed: Optional[int] = None) -> Any:
        """Start an episode, retain reset metadata, and return its observation."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
        """Return observation, reward, terminated, truncated, and info."""

    def encode_observation(self, observation: Any, info: Mapping[str, Any]) -> StageBatch:
        """Convert a stage-native observation into the shared hierarchical batch."""

