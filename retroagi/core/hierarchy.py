"""Deterministic fusion of shared vision outputs into hierarchy streams."""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .interfaces import StageBatch, StageSpec, VisionOutput


class VisionHierarchyProjector:
    """Map spatial semantics to A/B and fused continuous vision state to C."""

    def __init__(self, stage_spec: StageSpec):
        self.stage_spec = stage_spec

    def project(
        self,
        vision: VisionOutput,
        state: Optional[torch.Tensor] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StageBatch:
        self._validate_vision(vision)
        probabilities = vision.semantic_logits.float().softmax(dim=1)
        src_a = self._semantic_stream(probabilities, self.stage_spec.seq_len_a)
        src_b = self._semantic_stream(probabilities, self.stage_spec.seq_len_b)

        batch_size = probabilities.shape[0]
        position = vision.position.float()
        global_semantics = probabilities.mean(dim=(-2, -1))
        support_state = self._support_tensor(vision, batch_size, position.device)
        state = self._state_tensor(state, batch_size, position.device)
        fixed = torch.cat((position, global_semantics, support_state, state), dim=1)

        token_slots = self.stage_spec.seq_len_c - fixed.shape[1]
        if token_slots <= 0:
            raise ValueError(
                "C stream is too short for position, semantics, and state: "
                f"{self.stage_spec.seq_len_c} <= {fixed.shape[1]}"
            )

        token_content = torch.tanh(vision.tokens.float()).flatten(1).unsqueeze(1)
        token_summary = F.adaptive_avg_pool1d(token_content, token_slots).squeeze(1)
        src_c = torch.cat((fixed, token_summary), dim=1)

        position_end = position.shape[1]
        semantics_end = position_end + global_semantics.shape[1]
        support_end = semantics_end + support_state.shape[1]
        state_end = support_end + state.shape[1]
        fusion = {
            "a_semantic_regions": (1, self.stage_spec.seq_len_a),
            "b_semantic_regions": (1, self.stage_spec.seq_len_b),
            "c_position": (0, position_end),
            "c_semantic_probabilities": (position_end, semantics_end),
            "c_support_state": (semantics_end, support_end),
            "c_state": (support_end, state_end),
            "c_patch_tokens": (state_end, self.stage_spec.seq_len_c),
        }
        batch_metadata = dict(metadata or {})
        batch_metadata.update({"vision": vision, "vision_fusion": fusion})
        return StageBatch(
            src_a=src_a,
            target_a=None,
            src_b=src_b,
            target_b=None,
            src_c=src_c,
            target_c=None,
            metadata=batch_metadata,
        )

    def _validate_vision(self, vision: VisionOutput) -> None:
        if vision.semantic_logits.ndim != 4:
            raise ValueError("semantic_logits must have shape [B, K, H, W]")
        batch_size, classes = vision.semantic_logits.shape[:2]
        if classes > self.stage_spec.vocab_size:
            raise ValueError(
                f"{classes} semantic classes exceed vocab_size={self.stage_spec.vocab_size}"
            )
        if vision.position.ndim != 2 or vision.position.shape[0] != batch_size:
            raise ValueError("position must have shape [B, P]")
        if vision.support_logits is not None:
            if vision.support_logits.ndim != 2 or vision.support_logits.shape[0] != batch_size:
                raise ValueError("support_logits must have shape [B, S]")
        if vision.support_ids is not None:
            if vision.support_ids.ndim != 1 or vision.support_ids.shape[0] != batch_size:
                raise ValueError("support_ids must have shape [B]")
        if vision.tokens.ndim != 3 or vision.tokens.shape[0] != batch_size:
            raise ValueError("tokens must have shape [B, N, D]")

    @staticmethod
    def _semantic_stream(probabilities: torch.Tensor, length: int) -> torch.Tensor:
        if probabilities.shape[1] == 1:
            return torch.zeros(
                (probabilities.shape[0], length),
                dtype=torch.long,
                device=probabilities.device,
            )

        semantic_ids = probabilities.argmax(dim=1, keepdim=True)
        foreground_mask = semantic_ids.ne(0)
        foreground_present = F.adaptive_max_pool2d(
            foreground_mask.float(), (1, length)
        ).squeeze((1, 2)).bool()

        foreground_probabilities = probabilities[:, 1:] * foreground_mask
        foreground_regions = F.adaptive_max_pool2d(
            foreground_probabilities, (1, length)
        ).squeeze(2)
        _, foreground_id = foreground_regions.max(dim=1)
        return torch.where(
            foreground_present,
            foreground_id + 1,
            torch.zeros_like(foreground_id),
        ).long()

    @staticmethod
    def _state_tensor(
        state: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if state is None:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[0] != batch_size:
            raise ValueError(f"state must have shape [B, S], got {tuple(state.shape)}")
        return state

    @staticmethod
    def _support_tensor(
        vision: VisionOutput, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if vision.support_logits is None:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        support_logits = torch.as_tensor(vision.support_logits, dtype=torch.float32, device=device)
        return support_logits.softmax(dim=1)
