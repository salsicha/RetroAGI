"""Vision Transformer and semantic supervision for block SMB."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from retroagi.core import (
    build_checkpoint,
    ModelConfig,
    PatchVisionTransformer,
    VisionOutput,
    load_checkpoint,
    validate_checkpoint_compatibility,
)
from retroagi.core.vision import image_tensor


BLOCK_SEMANTIC_CLASSES = (
    "background",
    "mario",
    "platform",
    "coin",
    "goal",
    "enemy",
    "moving_platform",
)

BLOCK_CLASS_COLORS = {
    "background": ((107, 140, 255),),
    "mario": ((255, 0, 0), (255, 220, 0)),
    "platform": ((139, 69, 19),),
    "coin": ((255, 215, 0),),
    "goal": ((0, 255, 0),),
    "enemy": ((160, 32, 240), (100, 0, 160)),
    "moving_platform": ((80, 160, 40),),
}

DEFAULT_BLOCK_VIT_CHECKPOINT = Path("data/block_vit/block_vit.pth")
FALLBACK_BLOCK_VIT_CHECKPOINT = Path("data/block_smb_vit/block_vit.pth")


@dataclass(frozen=True)
class BlockVITLoadResult:
    model: "BlockVisionTransformer"
    checkpoint: dict[str, Any]
    path: Path
    frozen: bool


@dataclass(frozen=True)
class BlockVITPerceptionThresholds:
    """Minimum perception quality before policy failures blame the trainer."""

    min_accuracy: float = 0.95
    min_foreground_accuracy: float = 0.90
    min_mean_iou: float = 0.70
    max_position_rmse: float = 0.06
    min_position_within_tolerance: float = 0.90
    position_tolerance: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "min_accuracy",
            "min_foreground_accuracy",
            "min_mean_iou",
            "min_position_within_tolerance",
            "position_tolerance",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_position_rmse < 0:
            raise ValueError("max_position_rmse must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "min_accuracy": self.min_accuracy,
            "min_foreground_accuracy": self.min_foreground_accuracy,
            "min_mean_iou": self.min_mean_iou,
            "max_position_rmse": self.max_position_rmse,
            "min_position_within_tolerance": self.min_position_within_tolerance,
            "position_tolerance": self.position_tolerance,
        }


def _resolve_block_vit_checkpoint(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    if DEFAULT_BLOCK_VIT_CHECKPOINT.exists():
        return DEFAULT_BLOCK_VIT_CHECKPOINT
    return FALLBACK_BLOCK_VIT_CHECKPOINT


def set_block_vit_trainable(model: "BlockVisionTransformer", trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    model.train(trainable)


def _legacy_block_vit_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Normalize the original Block ViT trainer checkpoint to schema v1."""
    if "model_state" not in checkpoint:
        raise ValueError("legacy Block ViT checkpoint is missing model_state")
    legacy_config = checkpoint.get("config", {})
    if not isinstance(legacy_config, Mapping):
        legacy_config = {}
    vision_spec = checkpoint.get("vision_spec", {})
    if not isinstance(vision_spec, Mapping):
        vision_spec = {}

    semantic_classes = tuple(vision_spec.get("semantic_classes", BLOCK_SEMANTIC_CLASSES))
    model_config = {
        "name": "block_smb_vit",
        "hidden_dim": int(legacy_config.get("hidden_dim", legacy_config.get("dim", 64))),
        "depth": int(legacy_config.get("depth", 2)),
        "heads": int(legacy_config.get("heads", 4)),
        "patch_size": int(legacy_config.get("patch_size", 16)),
        "dropout": float(legacy_config.get("dropout", 0.1)),
    }
    return build_checkpoint(
        stage="block_smb",
        model_name="block_smb_vit",
        checkpoint_kind="vision_encoder",
        epoch=int(checkpoint.get("epoch", 0)),
        metrics=checkpoint.get("metrics", {}),
        config={
            "model": model_config,
            "legacy_training": dict(legacy_config),
        },
        specs={
            "vision": {
                "name": str(vision_spec.get("name", "block_smb_vit")),
                "semantic_classes": semantic_classes,
                "token_dim": int(vision_spec.get("token_dim", model_config["hidden_dim"])),
                "position_dim": int(vision_spec.get("position_dim", 2)),
            }
        },
        states={"model": checkpoint["model_state"]},
        metadata={
            "legacy_checkpoint": True,
            "source_path": str(checkpoint_path),
        },
    )


def load_block_vit_checkpoint(
    path: Optional[Path] = None,
    *,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> BlockVITLoadResult:
    """Load the supported Block SMB ViT checkpoint for policy training.

    Perception is frozen by default for policy training. Pass ``freeze=False``
    only for explicit fine-tuning experiments.
    """
    checkpoint_path = _resolve_block_vit_checkpoint(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Block ViT checkpoint not found at {checkpoint_path}; train it with "
            "scripts/vit/train_block_vit.py or pass an explicit checkpoint path"
        )

    from .adapter import BLOCK_SMB_SPEC

    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    except ValueError as exc:
        if "checkpoint_schema_version" not in str(exc):
            raise
        legacy_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if not isinstance(legacy_checkpoint, Mapping):
            raise ValueError("legacy Block ViT checkpoint must be a mapping") from exc
        checkpoint = _legacy_block_vit_checkpoint(legacy_checkpoint, checkpoint_path)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    model = BlockVisionTransformer(
        dim=int(model_config.get("hidden_dim", 64)),
        depth=int(model_config.get("depth", 2)),
        heads=int(model_config.get("heads", 4)),
        patch_size=int(model_config.get("patch_size", 16)),
        drop=float(model_config.get("dropout", 0.1)),
    ).to(device)
    validate_checkpoint_compatibility(
        checkpoint,
        stage=BLOCK_SMB_SPEC,
        model=ModelConfig(
            name="block_smb_vit",
            hidden_dim=model.spec.token_dim,
            depth=len(model.encoder.layers),
            heads=int(model.encoder.layers[0].self_attn.num_heads),
            patch_size=model.patch_size,
            dropout=float(model.dropout.p),
        ),
        vision=model.spec,
        checkpoint_kind="vision_encoder",
        required_states=("model",),
        context=f"Block ViT policy loader {checkpoint_path}",
    )
    model.load_state_dict(checkpoint["states"]["model"])
    set_block_vit_trainable(model, trainable=not freeze)
    if freeze:
        model.eval()
    return BlockVITLoadResult(
        model=model, checkpoint=checkpoint, path=checkpoint_path, frozen=freeze
    )


@torch.no_grad()
def evaluate_block_vit_perception(
    model: "BlockVisionTransformer",
    observations: Any,
    *,
    thresholds: BlockVITPerceptionThresholds = BlockVITPerceptionThresholds(),
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate Block ViT semantics and position against exact palette labels."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frames = torch.as_tensor(observations)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError("observations must have shape [N,H,W,C] or [N,C,H,W]")
    if frames.shape[0] <= 0:
        raise ValueError("observations must contain at least one frame")

    device = model.pos_embed.device
    was_training = model.training
    model.eval()
    samples = correct = foreground_correct = foreground_total = patches = 0
    position_squared_error = 0.0
    position_within_tolerance = 0
    intersections = torch.zeros(model.spec.num_classes, dtype=torch.float64)
    unions = torch.zeros(model.spec.num_classes, dtype=torch.float64)

    for start in range(0, frames.shape[0], batch_size):
        batch = frames[start : start + batch_size].to(device)
        labels = model.patch_targets(batch)
        positions = model.position_targets(batch)
        output = model.encode(batch)
        prediction = output.semantic_ids
        if prediction.shape != labels.shape:
            raise ValueError(
                "vision semantic_ids shape must match patch targets "
                f"{tuple(labels.shape)}, got {tuple(prediction.shape)}"
            )
        if output.position.shape != positions.shape:
            raise ValueError(
                "vision position shape must match position targets "
                f"{tuple(positions.shape)}, got {tuple(output.position.shape)}"
            )

        batch_size_actual = batch.shape[0]
        samples += batch_size_actual
        patches += labels.numel()
        correct += (prediction == labels).sum().item()
        foreground = labels != 0
        foreground_correct += (prediction[foreground] == labels[foreground]).sum().item()
        foreground_total += foreground.sum().item()
        position_delta = output.position - positions
        position_squared_error += position_delta.pow(2).sum().item()
        position_error = torch.linalg.vector_norm(position_delta, dim=1)
        position_within_tolerance += (
            position_error <= thresholds.position_tolerance
        ).sum().item()
        for class_id in range(model.spec.num_classes):
            predicted = prediction == class_id
            target = labels == class_id
            intersections[class_id] += (predicted & target).sum().cpu()
            unions[class_id] += (predicted | target).sum().cpu()

    if was_training:
        model.train()
    valid = unions > 0
    per_class_iou = {
        class_name: (
            float((intersections[index] / unions[index]).item())
            if bool(valid[index])
            else None
        )
        for index, class_name in enumerate(model.spec.semantic_classes)
    }
    mean_iou = (
        float((intersections[valid] / unions[valid]).mean().item())
        if bool(valid.any())
        else 0.0
    )
    accuracy = correct / max(patches, 1)
    foreground_accuracy = foreground_correct / max(foreground_total, 1)
    position_mse = position_squared_error / max(samples * model.spec.position_dim, 1)
    position_rmse = position_mse ** 0.5
    position_within_rate = position_within_tolerance / max(samples, 1)
    bottleneck_reasons = []
    if accuracy < thresholds.min_accuracy:
        bottleneck_reasons.append("semantic_accuracy")
    if foreground_accuracy < thresholds.min_foreground_accuracy:
        bottleneck_reasons.append("foreground_accuracy")
    if mean_iou < thresholds.min_mean_iou:
        bottleneck_reasons.append("mean_iou")
    if position_rmse > thresholds.max_position_rmse:
        bottleneck_reasons.append("position_rmse")
    if position_within_rate < thresholds.min_position_within_tolerance:
        bottleneck_reasons.append("position_within_tolerance")
    return {
        "samples": float(samples),
        "patches": float(patches),
        "foreground_patches": float(foreground_total),
        "accuracy": float(accuracy),
        "foreground_accuracy": float(foreground_accuracy),
        "mean_iou": float(mean_iou),
        "per_class_iou": per_class_iou,
        "position_mse": float(position_mse),
        "position_rmse": float(position_rmse),
        "position_within_tolerance": float(position_within_rate),
        "thresholds": thresholds.to_dict(),
        "bottleneck": bool(bottleneck_reasons),
        "bottleneck_reasons": bottleneck_reasons,
    }


class BlockVisionTransformer(PatchVisionTransformer):
    """Mid-level ViT that predicts block-SMB semantics and Mario position."""

    def __init__(
        self,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        patch_size: int = 16,
        drop: float = 0.1,
    ):
        super().__init__(
            semantic_classes=BLOCK_SEMANTIC_CLASSES,
            image_size=(240, 256),
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            drop=drop,
            position_class="mario",
            support_ground_classes=("platform",),
            support_platform_classes=("platform", "moving_platform"),
            name="block_smb_vit",
        )

    @torch.no_grad()
    def semantic_targets(self, observation: Any) -> torch.Tensor:
        """Build exact pixel labels from the simplified renderer's fixed palette."""
        image = (image_tensor(observation, device=self.pos_embed.device) * 255).round().to(torch.uint8)
        labels = torch.zeros(
            image.shape[0], image.shape[2], image.shape[3], dtype=torch.long, device=image.device
        )
        pixels = image.permute(0, 2, 3, 1)
        for class_id, class_name in enumerate(self.spec.semantic_classes):
            for color in BLOCK_CLASS_COLORS[class_name]:
                rgb = torch.tensor(color, dtype=torch.uint8, device=image.device)
                labels[(pixels == rgb).all(dim=-1)] = class_id
        return labels

    @torch.no_grad()
    def patch_targets(self, observation: Any) -> torch.Tensor:
        """Reduce exact pixel labels to the ViT patch grid with actor priority."""
        labels = self.semantic_targets(observation)
        one_hot = F.one_hot(labels, num_classes=self.spec.num_classes).permute(0, 3, 1, 2).float()
        present = F.adaptive_max_pool2d(one_hot, self.grid_size)
        priority = torch.tensor(
            (0, 7, 2, 5, 6, 5, 3), dtype=present.dtype, device=present.device
        ).view(1, -1, 1, 1)
        return (present * priority).argmax(dim=1)

    @torch.no_grad()
    def position_targets(self, observation: Any) -> torch.Tensor:
        """Return the normalized center of Mario from exact renderer labels."""
        mario_pixels = self.semantic_targets(observation) == self.spec.semantic_classes.index("mario")
        mass = mario_pixels.sum(dim=(1, 2)).clamp_min(1)
        height, width = mario_pixels.shape[-2:]
        y = torch.linspace(0, 1, height, device=mario_pixels.device)
        x = torch.linspace(0, 1, width, device=mario_pixels.device)
        return torch.stack(
            (
                (mario_pixels * x.view(1, 1, width)).sum(dim=(1, 2)) / mass,
                (mario_pixels * y.view(1, height, 1)).sum(dim=(1, 2)) / mass,
            ),
            dim=-1,
        )

    def training_loss(
        self,
        observation: Any,
        semantic_weight: float = 1.0,
        position_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute joint patch-segmentation and normalized Mario-position loss."""
        output = self.forward(observation)
        targets = self.patch_targets(observation)
        semantic_loss = F.cross_entropy(output.semantic_logits, targets)

        position_target = self.position_targets(observation)
        position_loss = F.mse_loss(output.position, position_target)
        total = semantic_weight * semantic_loss + position_weight * position_loss
        return total, {"semantic": semantic_loss, "position": position_loss}

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)
