"""Shared vision models used across curriculum stages."""

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import VISION_SUPPORT_CLASSES, VisionOutput, VisionSpec
from .smb_pixel_types import PIXEL_TYPES, SCREEN_SHAPE, vision_output

SUPPORT_HEAD_STATE_KEYS = ("support_head.weight", "support_head.bias")


def image_tensor(observation: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert HWC/BHWC or CHW/BCHW image data to normalized BCHW tensors."""
    image = torch.as_tensor(observation, device=device)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"expected a 3D or 4D image, got shape {tuple(image.shape)}")
    if image.shape[-1] in (1, 3, 4):
        image = image.permute(0, 3, 1, 2)
    if image.shape[1] == 4:
        image = image[:, :3]
    image = image.float()
    if image.numel() and image.max() > 1:
        image = image / 255.0
    return image.contiguous()


def infer_agent_support_logits(
    semantic_logits: torch.Tensor,
    *,
    semantic_classes: tuple[str, ...],
    agent_class: Optional[str],
    ground_classes: tuple[str, ...],
    platform_classes: tuple[str, ...],
    support_classes: tuple[str, ...] = VISION_SUPPORT_CLASSES,
    floor_y_threshold: float = 0.82,
    scan_depth: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Infer air/ground/platform contact from predicted semantic labels."""

    support_ids = infer_agent_support_ids_from_labels(
        semantic_logits.argmax(dim=1),
        semantic_classes=semantic_classes,
        agent_class=agent_class,
        ground_classes=ground_classes,
        platform_classes=platform_classes,
        support_classes=support_classes,
        floor_y_threshold=floor_y_threshold,
        scan_depth=scan_depth,
    )
    if support_ids is None:
        return None
    return support_logits_from_ids(
        support_ids,
        len(support_classes),
        dtype=semantic_logits.dtype,
    )


def infer_agent_support_ids_from_labels(
    labels: torch.Tensor,
    *,
    semantic_classes: tuple[str, ...],
    agent_class: Optional[str],
    ground_classes: tuple[str, ...],
    platform_classes: tuple[str, ...],
    support_classes: tuple[str, ...] = VISION_SUPPORT_CLASSES,
    floor_y_threshold: float = 0.82,
    scan_depth: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Infer support IDs from hard semantic labels.

    Support classes are ordered as ``air, ground, platform`` by default. The
    contact test uses the first row below the agent at patch resolution and a
    small pixel tolerance for dense masks.
    """

    if (
        not agent_class
        or agent_class not in semantic_classes
        or not support_classes
        or "air" not in support_classes
        or "ground" not in support_classes
        or "platform" not in support_classes
    ):
        return None
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B, H, W]")

    class_to_id = {name: index for index, name in enumerate(semantic_classes)}
    ground_ids = {class_to_id[name] for name in ground_classes if name in class_to_id}
    platform_ids = {class_to_id[name] for name in platform_classes if name in class_to_id}
    terrain_ids = ground_ids | platform_ids
    if not terrain_ids:
        return None

    support_to_id = {name: index for index, name in enumerate(support_classes)}
    air_id = support_to_id["air"]
    ground_id = support_to_id["ground"]
    platform_id = support_to_id["platform"]
    agent_id = class_to_id[agent_class]

    batch, grid_h, grid_w = labels.shape
    output = torch.full((batch,), air_id, dtype=torch.long, device=labels.device)
    contact_scan_depth = _contact_scan_depth(grid_h) if scan_depth is None else int(scan_depth)
    if contact_scan_depth <= 0:
        raise ValueError("scan_depth must be positive")
    column_margin = 1

    for batch_index in range(batch):
        support_id = air_id
        agent_locations = torch.nonzero(labels[batch_index] == agent_id, as_tuple=False)
        if agent_locations.numel() > 0:
            rows = agent_locations[:, 0]
            cols = agent_locations[:, 1]
            bottom = int(rows.max().item())
            col_start = max(int(cols.min().item()) - column_margin, 0)
            col_end = min(int(cols.max().item()) + column_margin + 1, grid_w)
            row_end = min(bottom + contact_scan_depth + 1, grid_h)
            for row in range(bottom + 1, row_end):
                window = labels[batch_index, row, col_start:col_end]
                matches = [
                    int(value) for value in window.flatten().tolist() if int(value) in terrain_ids
                ]
                if not matches:
                    continue
                matched_id = matches[0]
                if matched_id in ground_ids and matched_id in platform_ids:
                    normalized_row = row / max(grid_h - 1, 1)
                    support_id = ground_id if normalized_row >= floor_y_threshold else platform_id
                elif matched_id in platform_ids:
                    support_id = platform_id
                else:
                    support_id = ground_id
                break
        output[batch_index] = support_id
    return output


def support_logits_from_ids(
    support_ids: torch.Tensor,
    num_support_classes: int,
    *,
    positive: float = 4.0,
    negative: float = -4.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert support class IDs to high-margin logits."""

    ids = torch.as_tensor(support_ids, dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("support_ids must have shape [B]")
    if num_support_classes <= 0:
        raise ValueError("num_support_classes must be positive")
    logits = torch.full(
        (ids.shape[0], num_support_classes),
        float(negative),
        dtype=dtype,
        device=ids.device,
    )
    return logits.scatter_(1, ids.unsqueeze(1), float(positive))


def _contact_scan_depth(grid_h: int) -> int:
    return max(1, int(round(float(grid_h) / 120.0)))


class SquareTransformer(nn.Module):
    """Cuts a picture into squares, describes each, and lets squares inform each other.

    Pictures must already be full size: stretching distorts the pixels and the
    square grid, so callers pad a cropped screen back to full size instead.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        drop: float,
    ):
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if dim % heads:
            raise ValueError("token dimension must be divisible by attention heads")
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.grid_size = (height // patch_size, width // patch_size)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def screen_image(self, observation: Any) -> torch.Tensor:
        """The observation as a normalized [B, 3, H, W] tensor of the full picture size."""
        image = image_tensor(observation, device=self.pos_embed.device)
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Vision expects {self.image_size[0]}x{self.image_size[1]} pictures, got "
                f"{tuple(image.shape[-2:])}; pad the screen to full size, never stretch it"
            )
        return image

    def square_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Each square's final description, [B, squares, dim], row by row."""
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        return self.norm(self.encoder(self.dropout(tokens + self.pos_embed)))


class PatchVisionTransformer(SquareTransformer):
    """Patch-level semantic ViT with a checkpoint-compatible head."""

    def __init__(
        self,
        semantic_classes: tuple[str, ...],
        image_size: tuple[int, int] = (240, 256),
        patch_size: int = 16,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        position_class: Optional[str] = None,
        support_ground_classes: tuple[str, ...] = (),
        support_platform_classes: tuple[str, ...] = (),
        support_floor_y_threshold: float = 0.82,
        support_prior_scale: float = 0.25,
        support_scan_depth: Optional[int] = None,
        name: str = "patch_vit",
    ):
        super().__init__(image_size, patch_size, dim, depth, heads, mlp_ratio, drop)
        self.spec = VisionSpec(name=name, semantic_classes=semantic_classes, token_dim=dim)
        self.position_class = position_class
        self.support_ground_classes = tuple(support_ground_classes)
        self.support_platform_classes = tuple(support_platform_classes)
        self.support_floor_y_threshold = float(support_floor_y_threshold)
        self.support_prior_scale = float(support_prior_scale)
        self.support_scan_depth = support_scan_depth
        self.head = nn.Linear(dim, self.spec.num_classes)
        self.support_head = nn.Linear(dim, self.spec.num_support_classes)
        nn.init.zeros_(self.support_head.weight)
        nn.init.zeros_(self.support_head.bias)

    def _position_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        batch, _, grid_h, grid_w = logits.shape
        if self.position_class not in self.spec.semantic_classes:
            return torch.zeros((batch, self.spec.position_dim), device=logits.device)

        class_id = self.spec.semantic_classes.index(self.position_class)
        weights = logits.softmax(dim=1)[:, class_id]
        weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        y = torch.linspace(0, 1, grid_h, device=logits.device)
        x = torch.linspace(0, 1, grid_w, device=logits.device)
        pos_x = (weights * x.view(1, 1, grid_w)).sum(dim=(1, 2))
        pos_y = (weights * y.view(1, grid_h, 1)).sum(dim=(1, 2))
        return torch.stack((pos_x, pos_y), dim=-1)

    def _support_from_logits(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        return infer_agent_support_logits(
            logits,
            semantic_classes=self.spec.semantic_classes,
            agent_class=self.position_class,
            ground_classes=self.support_ground_classes,
            platform_classes=self.support_platform_classes,
            support_classes=self.spec.support_classes,
            floor_y_threshold=self.support_floor_y_threshold,
            scan_depth=self.support_scan_depth,
        )

    def _support_context(self, tokens: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.position_class in self.spec.semantic_classes:
            class_id = self.spec.semantic_classes.index(self.position_class)
            weights = logits.softmax(dim=1)[:, class_id].flatten(1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return tokens.mean(dim=1)

    def support_targets_from_labels(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        return infer_agent_support_ids_from_labels(
            labels,
            semantic_classes=self.spec.semantic_classes,
            agent_class=self.position_class,
            ground_classes=self.support_ground_classes,
            platform_classes=self.support_platform_classes,
            support_classes=self.spec.support_classes,
            floor_y_threshold=self.support_floor_y_threshold,
            scan_depth=self.support_scan_depth,
        )

    def support_logits_from_ids(self, support_ids: torch.Tensor) -> torch.Tensor:
        return support_logits_from_ids(
            support_ids,
            self.spec.num_support_classes,
            dtype=self.support_head.weight.dtype,
        )

    def load_compatible_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ):
        """Load ViT checkpoints while allowing legacy checkpoints without support head."""

        result = self.load_state_dict(state_dict, strict=False)
        if not strict:
            return result
        allowed_missing = set(SUPPORT_HEAD_STATE_KEYS)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        unsupported_missing = [key for key in missing if key not in allowed_missing]
        if unsupported_missing or unexpected:
            messages = []
            if unsupported_missing:
                messages.append(f"Missing key(s): {unsupported_missing}")
            if unexpected:
                messages.append(f"Unexpected key(s): {unexpected}")
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                + "; ".join(messages)
            )
        return result

    def forward(self, image: torch.Tensor) -> VisionOutput:
        tokens = self.square_tokens(self.screen_image(image))
        batch = tokens.shape[0]
        grid_h, grid_w = self.grid_size
        logits = (
            self.head(tokens).transpose(1, 2).reshape(batch, self.spec.num_classes, grid_h, grid_w)
        )
        support_logits = self.support_head(self._support_context(tokens, logits))
        support_source = "learned_head"
        support_prior = self._support_from_logits(logits)
        if support_prior is not None and self.support_prior_scale != 0.0:
            support_logits = support_logits + self.support_prior_scale * support_prior.to(
                device=support_logits.device,
                dtype=support_logits.dtype,
            )
            support_source = "learned_head+semantic_contact"
        metadata = {
            "grid_size": self.grid_size,
            "image_size": self.image_size,
            "semantic_classes": self.spec.semantic_classes,
            "support_classes": self.spec.support_classes,
        }
        metadata["support_source"] = support_source
        return VisionOutput(
            position=self._position_from_logits(logits),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=tokens,
            metadata=metadata,
            support_logits=support_logits,
            support_ids=support_logits.argmax(dim=1) if support_logits is not None else None,
        )

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)


class PixelVisionTransformer(SquareTransformer):
    """Vision transformer that gives every pixel of the SMB screen one of PIXEL_TYPES.

    Both SMB games use this one class, each with its own trained weights. It
    cuts the 256x240 picture into 16x16 squares, describes each square and
    lets the squares inform each other over ``depth`` rounds. Each square's
    final description gives scores for its own 16x16 pixels and every type; a
    small per-pixel refinement then reads those scores with the pixel's colour.
    The common vision output follows from the scores by the shared rules
    (smb_pixel_types.vision_output).
    """

    def __init__(
        self,
        *,
        name: str = "smb_pixel_vit",
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 16,
        drop: float = 0.0,
        refine_dim: int = 32,
    ):
        super().__init__(SCREEN_SHAPE, patch_size, dim, depth, heads, 4.0, drop)
        # Output tokens are each square's mean type probabilities (vision_output).
        types = len(PIXEL_TYPES)
        self.spec = VisionSpec(name=name, semantic_classes=PIXEL_TYPES, token_dim=types)
        self.hidden_dim = dim
        self.refine_dim = refine_dim
        # Each square's description -> scores for its 16x16 pixels x every type.
        self.pixel_head = nn.Linear(dim, patch_size * patch_size * types)
        # Per pixel: that pixel's scores and its colour -> a correction.
        self.refine = nn.Sequential(
            nn.Conv2d(types + 3, refine_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, types, kernel_size=1),
        )

    def pixel_logits(self, observation: Any) -> torch.Tensor:
        """Per-pixel type scores [B, types, 240, 256]; what training compares to labels."""
        image = self.screen_image(observation)
        tokens = self.square_tokens(image)
        batch, size, types = image.shape[0], self.patch_size, self.spec.num_classes
        grid_h, grid_w = self.grid_size
        logits = self.pixel_head(tokens).view(batch, grid_h, grid_w, types, size, size)
        logits = logits.permute(0, 3, 1, 4, 2, 5).reshape(batch, types, *self.image_size)
        return logits + self.refine(torch.cat((logits, image), dim=1))

    def forward(self, observation: Any) -> VisionOutput:
        return vision_output(self.pixel_logits(observation))

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)

    def architecture(self) -> dict[str, Any]:
        """The settings that rebuild this model (stored in its checkpoints)."""
        return {
            "name": self.spec.name,
            "hidden_dim": self.hidden_dim,
            "depth": len(self.encoder.layers),
            "heads": int(self.encoder.layers[0].self_attn.num_heads),
            "patch_size": self.patch_size,
            "dropout": float(self.dropout.p),
            "metadata": {"refine_dim": self.refine_dim},
        }

    @classmethod
    def from_architecture(cls, settings: Mapping[str, Any]) -> "PixelVisionTransformer":
        return cls(
            name=str(settings["name"]),
            dim=int(settings["hidden_dim"]),
            depth=int(settings["depth"]),
            heads=int(settings["heads"]),
            patch_size=int(settings["patch_size"]),
            drop=float(settings["dropout"]),
            refine_dim=int(settings["metadata"]["refine_dim"]),
        )


class LinearVisionEncoder(nn.Module):
    """Map synthetic 1D observations into the common vision representation."""

    def __init__(self, vocab_size: int = 20, token_dim: int = 64):
        super().__init__()
        classes = tuple(f"value_{index}" for index in range(vocab_size))
        self.spec = VisionSpec("synthetic_1d", classes, token_dim, position_dim=1)
        self.embedding = nn.Embedding(vocab_size, token_dim)

    def forward(self, observation: torch.Tensor) -> VisionOutput:
        values = torch.as_tensor(observation, device=self.embedding.weight.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        values = values.long()
        tokens = self.embedding(values)
        logits = F.one_hot(values, num_classes=self.spec.num_classes).float().transpose(1, 2)
        logits = logits.unsqueeze(2)
        denom = max(values.shape[1] - 1, 1)
        position = values.argmax(dim=1).float().unsqueeze(-1) / denom
        return VisionOutput(
            position=position,
            semantic_logits=logits,
            semantic_ids=values.unsqueeze(1),
            tokens=tokens,
            metadata={
                "sequence_length": values.shape[1],
                "semantic_classes": self.spec.semantic_classes,
                "support_classes": self.spec.support_classes,
            },
        )

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(torch.as_tensor(observation))
