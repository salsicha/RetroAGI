"""What a vision transformer says it reads and what it reports."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

VISION_SUPPORT_CLASSES = ("air", "ground", "platform")


@dataclass(frozen=True)
class VisionSpec:
    """Describes a stage vision encoder's spatial and semantic contract."""

    name: str
    semantic_classes: tuple[str, ...]
    token_dim: int
    position_dim: int = 2
    support_classes: tuple[str, ...] = VISION_SUPPORT_CLASSES

    @property
    def num_classes(self) -> int:
        return len(self.semantic_classes)

    @property
    def num_support_classes(self) -> int:
        return len(self.support_classes)


@dataclass
class VisionOutput:
    """Stage-independent vision tensors defined in docs/tensor-contracts.md."""

    position: torch.Tensor
    semantic_logits: torch.Tensor
    semantic_ids: torch.Tensor
    tokens: torch.Tensor
    metadata: Optional[Mapping[str, Any]] = None
    support_logits: Optional[torch.Tensor] = None
    support_ids: Optional[torch.Tensor] = None
