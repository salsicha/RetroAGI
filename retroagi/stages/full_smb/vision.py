"""Full-SMB semantic segmentation vision encoders."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from retroagi.core import (
    PatchVisionTransformer,
    VisionOutput,
    VisionSpec,
    build_checkpoint,
    is_versioned_checkpoint,
    load_checkpoint,
)
from retroagi.core.vision import image_tensor

FULL_SMB_VIT_MODEL_NAME = "full_smb_vit"
FULL_SMB_VIT_CHECKPOINT_KIND = "vision_encoder"
FULL_SMB_VIT_CLASSES = (
    "sky",
    "ground",
    "brick",
    "question_block",
    "pipe",
    "coin",
    "goomba",
    "koopa",
    "mario",
    "mushroom",
    "hill",
    "cloud",
    "bush",
)
FULL_SMB_DEEPLAB_CLASSES = ("background", "floor", "box", "enemy", "brick", "mario")
FULL_SMB_SEMANTIC_CLASSES = FULL_SMB_VIT_CLASSES
DEFAULT_FULL_SMB_VIT_CHECKPOINT = Path("data/vit/full_smb_vit.pth")
FALLBACK_FULL_SMB_VIT_CHECKPOINT = Path("data/vit/vit_smb.pth")
LEGACY_DEEPLAB_CHECKPOINT = (
    Path(__file__).parents[3] / "scripts" / "segmentation" / "MarioSegmentationModel.pth"
)
DEFAULT_CHECKPOINT = DEFAULT_FULL_SMB_VIT_CHECKPOINT


@dataclass(frozen=True)
class FullSMBVITLoadResult:
    model: "FullSMBVisionTransformer"
    checkpoint: dict[str, Any]
    path: Path
    frozen: bool


class FullSMBVisionTransformer(PatchVisionTransformer):
    """Patch-level ViT semantic segmenter for full SMB frames."""

    def __init__(
        self,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
    ):
        super().__init__(
            semantic_classes=FULL_SMB_VIT_CLASSES,
            image_size=(240, 256),
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            position_class="mario",
            name=FULL_SMB_VIT_MODEL_NAME,
        )

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)


class FullSMBSegmentationVision(nn.Module):
    """Default Full SMB semantic segmenter backed by a ViT checkpoint."""

    def __init__(
        self,
        checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
        device: Optional[torch.device | str] = None,
        freeze: bool = True,
        *,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
    ):
        super().__init__()
        if checkpoint is None:
            self.model = FullSMBVisionTransformer(
                dim=dim,
                depth=depth,
                heads=heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
            )
            if device is not None:
                self.model.to(device)
            _set_trainable(self.model, trainable=not freeze)
            if freeze:
                self.model.eval()
            self.checkpoint = None
            self.checkpoint_path = None
            self.frozen = freeze
        else:
            load_result = load_full_smb_vit_checkpoint(
                checkpoint,
                device=device or "cpu",
                freeze=freeze,
            )
            self.model = load_result.model
            self.checkpoint = load_result.checkpoint
            self.checkpoint_path = load_result.path
            self.frozen = load_result.frozen
        self.spec = self.model.spec

    def forward(self, observation: Any) -> VisionOutput:
        return self.model.encode(observation)

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)


class FullSMBDeepLabSegmentationVision(nn.Module):
    """Legacy wrapper for the previous six-class DeepLab checkpoint."""

    def __init__(
        self,
        checkpoint: Optional[Path] = LEGACY_DEEPLAB_CHECKPOINT,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead

        self.spec = VisionSpec(
            "full_smb_deeplab",
            FULL_SMB_DEEPLAB_CLASSES,
            token_dim=len(FULL_SMB_DEEPLAB_CLASSES),
        )
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
        self.model.classifier = DeepLabHead(2048, self.spec.num_classes)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        if device is not None:
            self.to(device)
        self.model.eval()

    def forward(self, observation: Any) -> VisionOutput:
        device = next(self.model.parameters()).device
        image = image_tensor(observation, device=device)
        logits = self.model(image)["out"]
        semantic_ids = logits.argmax(dim=1)
        probabilities = logits.softmax(dim=1)

        mario = probabilities[:, self.spec.semantic_classes.index("mario")]
        weights = mario / mario.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        height, width = mario.shape[-2:]
        y = torch.linspace(0, 1, height, device=device)
        x = torch.linspace(0, 1, width, device=device)
        position = torch.stack(
            (
                (weights * x.view(1, 1, width)).sum(dim=(1, 2)),
                (weights * y.view(1, height, 1)).sum(dim=(1, 2)),
            ),
            dim=-1,
        )

        token_logits = F.adaptive_avg_pool2d(logits, (15, 16))
        tokens = token_logits.flatten(2).transpose(1, 2)
        return VisionOutput(
            position=position,
            semantic_logits=logits,
            semantic_ids=semantic_ids,
            tokens=tokens,
            metadata={
                "checkpoint_classes": self.spec.semantic_classes,
                "legacy_encoder": "deeplabv3_resnet50",
            },
        )

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)


def load_full_smb_vit_checkpoint(
    path: Optional[Path] = None,
    *,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> FullSMBVITLoadResult:
    checkpoint_path = _resolve_full_smb_vit_checkpoint(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Full SMB ViT checkpoint not found at {checkpoint_path}; train it with "
            "scripts/vit/train_vit.py or pass checkpoint=None for an untrained model"
        )

    checkpoint, state, model_config = _load_full_smb_vit_checkpoint_payload(
        checkpoint_path, device
    )
    model = FullSMBVisionTransformer(
        dim=int(model_config.get("hidden_dim", model_config.get("dim", 192))),
        depth=int(model_config.get("depth", 6)),
        heads=int(model_config.get("heads", 6)),
        patch_size=int(model_config.get("patch_size", 16)),
        mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
        drop=float(model_config.get("dropout", model_config.get("drop", 0.1))),
    ).to(device)
    model.load_state_dict(state)
    _set_trainable(model, trainable=not freeze)
    if freeze:
        model.eval()
    return FullSMBVITLoadResult(
        model=model,
        checkpoint=checkpoint,
        path=checkpoint_path,
        frozen=freeze,
    )


def build_full_smb_vit_checkpoint(
    model: FullSMBVisionTransformer,
    *,
    epoch: int = 0,
    metrics: Optional[Mapping[str, float]] = None,
    config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    model_config = {
        "name": FULL_SMB_VIT_MODEL_NAME,
        "hidden_dim": model.spec.token_dim,
        "depth": len(model.encoder.layers),
        "heads": int(model.encoder.layers[0].self_attn.num_heads),
        "patch_size": model.patch_size,
        "mlp_ratio": float(model.encoder.layers[0].linear1.out_features / model.spec.token_dim),
        "dropout": float(model.dropout.p),
    }
    checkpoint_config = {"model": model_config}
    if config:
        extra_config = dict(config)
        extra_model_config = extra_config.pop("model", None)
        if isinstance(extra_model_config, Mapping):
            model_config.update(dict(extra_model_config))
        checkpoint_config.update(extra_config)
    return build_checkpoint(
        stage="full_smb",
        model_name=FULL_SMB_VIT_MODEL_NAME,
        checkpoint_kind=FULL_SMB_VIT_CHECKPOINT_KIND,
        epoch=epoch,
        states={"model": model.state_dict()},
        metrics=metrics or {},
        config=checkpoint_config,
        specs={"vision": asdict(model.spec)},
        metadata=metadata or {},
    )


def _resolve_full_smb_vit_checkpoint(path: Optional[Path]) -> Path:
    if path is not None:
        checkpoint_path = Path(path)
        if (
            checkpoint_path == DEFAULT_FULL_SMB_VIT_CHECKPOINT
            and not checkpoint_path.exists()
            and FALLBACK_FULL_SMB_VIT_CHECKPOINT.exists()
        ):
            return FALLBACK_FULL_SMB_VIT_CHECKPOINT
        return checkpoint_path
    if DEFAULT_FULL_SMB_VIT_CHECKPOINT.exists():
        return DEFAULT_FULL_SMB_VIT_CHECKPOINT
    return FALLBACK_FULL_SMB_VIT_CHECKPOINT


def _load_full_smb_vit_checkpoint_payload(
    path: Path, device: str | torch.device
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], Mapping[str, Any]]:
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, Mapping):
        raise ValueError("Full SMB ViT checkpoint must be a mapping")

    if is_versioned_checkpoint(loaded):
        checkpoint = load_checkpoint(path, map_location=device)
        model_config = _model_config(checkpoint)
        state = checkpoint["states"]["model"]
        return checkpoint, state, model_config

    if "model_state" in loaded:
        state = loaded["model_state"]
        model_config = dict(loaded.get("config", {}))
    elif _looks_like_state_dict(loaded):
        state = loaded
        model_config = _infer_vit_config_from_state(state)
    else:
        raise ValueError("Full SMB ViT checkpoint is missing model state")

    model_config = _normalized_vit_model_config(model_config, state)
    checkpoint = build_checkpoint(
        stage="full_smb",
        model_name=FULL_SMB_VIT_MODEL_NAME,
        checkpoint_kind=FULL_SMB_VIT_CHECKPOINT_KIND,
        states={"model": state},
        config={"model": model_config},
        specs={
            "vision": {
                "name": FULL_SMB_VIT_MODEL_NAME,
                "semantic_classes": FULL_SMB_VIT_CLASSES,
                "token_dim": int(model_config["hidden_dim"]),
                "position_dim": 2,
            }
        },
        metadata={"legacy_checkpoint": True, "source_path": str(path)},
    )
    return checkpoint, state, model_config


def _model_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        return {}
    model_config = config.get("model", {})
    return model_config if isinstance(model_config, Mapping) else {}


def _looks_like_state_dict(checkpoint: Mapping[str, Any]) -> bool:
    return "pos_embed" in checkpoint and "patch_embed.weight" in checkpoint


def _infer_vit_config_from_state(
    state: Mapping[str, torch.Tensor]
) -> dict[str, int | float]:
    hidden_dim = int(state["pos_embed"].shape[-1])
    patch_size = int(state["patch_embed.weight"].shape[-1])
    depth = 0
    for key in state:
        if key.startswith("encoder.layers."):
            try:
                depth = max(depth, int(key.split(".")[2]) + 1)
            except (IndexError, ValueError):
                continue
    return {
        "hidden_dim": hidden_dim,
        "depth": depth or 6,
        "heads": 6,
        "patch_size": patch_size,
        "mlp_ratio": float(
            state.get("encoder.layers.0.linear1.weight").shape[0] / hidden_dim
            if "encoder.layers.0.linear1.weight" in state
            else 4.0
        ),
        "dropout": 0.1,
    }


def _normalized_vit_model_config(
    config: Mapping[str, Any], state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    inferred = _infer_vit_config_from_state(state)
    class_count = int(state["head.weight"].shape[0])
    if class_count != len(FULL_SMB_VIT_CLASSES):
        raise ValueError(
            "Full SMB ViT checkpoint class count does not match "
            f"{len(FULL_SMB_VIT_CLASSES)} classes: got {class_count}"
        )
    return {
        "name": FULL_SMB_VIT_MODEL_NAME,
        "hidden_dim": int(config.get("hidden_dim", config.get("dim", inferred["hidden_dim"]))),
        "depth": int(config.get("depth", inferred["depth"])),
        "heads": int(config.get("heads", inferred["heads"])),
        "patch_size": int(config.get("patch_size", inferred["patch_size"])),
        "mlp_ratio": float(config.get("mlp_ratio", inferred["mlp_ratio"])),
        "dropout": float(config.get("dropout", config.get("drop", inferred["dropout"]))),
    }


def _set_trainable(model: nn.Module, *, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    model.train(trainable)
