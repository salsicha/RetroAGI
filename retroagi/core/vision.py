"""Shared vision models used across curriculum stages."""

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import VisionOutput, VisionSpec
from .smb_pixel_types import PIXEL_TYPES, SCREEN_SHAPE, vision_output
from .smb_scene_labels import ENEMY_KINDS, OBJECT_CELL
from .smb_scene_labels import SUPPORTS as MARIO_SUPPORTS


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


class SceneVisionTransformer(SquareTransformer):
    """Vision transformer that reports the objects drawn on an SMB screen.

    Both SMB games use this one class, each with its own trained weights. It
    cuts the 256x240 picture into 16x16 squares, describes each square and
    lets the squares inform each other over ``depth`` rounds. From each
    square's final description it reads:

    - every pixel's type (smb_pixel_types.PIXEL_TYPES), then corrects each
      pixel's scores from the scores and colours of its 3x3 neighbourhood.
      The types are never sent to a policy: the scene is found from them
      (smb_scene_labels: objects are groups of touching pixels of a type;
      surfaces, gaps, blocks and pipes follow from structure_from_types);
    - for each of its four 8x8 cells, the kind of enemy drawn there (walker,
      plant, other, defeated);
    - Mario's facing, his support (air, ground or moving platform), and
      whether his feet are on something - the ground, a moving platform or an
      enemy he is stomping: the land detector, whose turning on after he was
      in the air is a landing. These are read from the squares weighted by
      how much of Mario each holds.

    scene() turns these into one smb_scene_labels.SceneObservation per
    picture, which is all a policy may see.
    """

    def __init__(
        self,
        *,
        name: str = "smb_scene_vit",
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 16,
        drop: float = 0.0,
        refine_dim: int = 32,
    ):
        super().__init__(SCREEN_SHAPE, patch_size, dim, depth, heads, 4.0, drop)
        if patch_size % OBJECT_CELL:
            raise ValueError("patch_size must be a multiple of the 8-pixel cell")
        types = len(PIXEL_TYPES)
        self.spec = VisionSpec(name=name, semantic_classes=PIXEL_TYPES, token_dim=types)
        self.hidden_dim = dim
        self.refine_dim = refine_dim
        self.cells = patch_size // OBJECT_CELL
        # Each square's description -> scores for its 16x16 pixels x every type.
        self.pixel_head = nn.Linear(dim, patch_size * patch_size * types)
        # Per pixel: the scores and colours of its 3x3 neighbourhood -> a correction.
        self.refine = nn.Sequential(
            nn.Conv2d(types + 3, refine_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, types, kernel_size=1),
        )
        # Per 8x8 cell: the enemy kind scores.
        self.kind_head = nn.Linear(dim, self.cells * self.cells * len(ENEMY_KINDS))
        # Mario's facing (left, right) and support.
        # Mario's facing (left, right), support, and feet on something (no, yes).
        self.mario_head = nn.Linear(dim, 2 + len(MARIO_SUPPORTS) + 2)

    def heads(self, observation: Any) -> dict[str, torch.Tensor]:
        """Every head's raw output for a batch of pictures.

        pixel_logits [B, types, 240, 256]; kind_logits [B, kinds, 30, 32];
        facing_logits [B, 2] (left, right); support_logits [B, 3];
        on_something_logits [B, 2] (no, yes: his feet are on something).
        """
        image = self.screen_image(observation)
        tokens = self.square_tokens(image)
        batch, size, types = image.shape[0], self.patch_size, self.spec.num_classes
        grid_h, grid_w = self.grid_size
        logits = self.pixel_head(tokens).view(batch, grid_h, grid_w, types, size, size)
        logits = logits.permute(0, 3, 1, 4, 2, 5).reshape(batch, types, *self.image_size)
        pixel_logits = logits + self.refine(torch.cat((logits, image), dim=1))

        cells, kinds = self.cells, len(ENEMY_KINDS)
        kind_logits = self.kind_head(tokens).view(batch, grid_h, grid_w, cells, cells, kinds)
        kind_logits = kind_logits.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, kinds, grid_h * cells, grid_w * cells
        )

        # Mario's state from the squares, weighted by how strongly each shows Mario.
        mario = pixel_logits[:, PIXEL_TYPES.index("mario")].detach()
        per_square = F.max_pool2d(mario.unsqueeze(1), size).flatten(1)
        weights = per_square.softmax(dim=-1).unsqueeze(-1)
        mario_state = self.mario_head((weights * tokens).sum(dim=1))
        return {
            "pixel_logits": pixel_logits,
            "kind_logits": kind_logits,
            "facing_logits": mario_state[:, :2],
            "support_logits": mario_state[:, 2 : 2 + len(MARIO_SUPPORTS)],
            "on_something_logits": mario_state[:, 2 + len(MARIO_SUPPORTS) :],
        }

    def pixel_logits(self, observation: Any) -> torch.Tensor:
        """Per-pixel type scores [B, types, 240, 256] (internal)."""
        return self.heads(observation)["pixel_logits"]

    @torch.no_grad()
    def scene(self, observation: Any) -> list:
        """One smb_scene_labels.SceneObservation per picture: all a policy may see."""
        from .smb_scene_labels import decode_scene

        return decode_scene(self.heads(observation))

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
            "metadata": {
                "refine_dim": self.refine_dim,
                "refine_reach": 3,
                "pixel_types": list(PIXEL_TYPES),
                "enemy_kinds": list(ENEMY_KINDS),
                "mario_supports": list(MARIO_SUPPORTS),
                "mario_on_something": True,
                "object_cell": OBJECT_CELL,
            },
        }

    @classmethod
    def from_architecture(cls, settings: Mapping[str, Any]) -> "SceneVisionTransformer":
        metadata = settings["metadata"]
        expected = {
            "refine_reach": 3,
            "pixel_types": list(PIXEL_TYPES),
            "enemy_kinds": list(ENEMY_KINDS),
            "mario_supports": list(MARIO_SUPPORTS),
            "mario_on_something": True,
            "object_cell": OBJECT_CELL,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"checkpoint {key} {metadata.get(key)!r} is not {value!r}")
        return cls(
            name=str(settings["name"]),
            dim=int(settings["hidden_dim"]),
            depth=int(settings["depth"]),
            heads=int(settings["heads"]),
            patch_size=int(settings["patch_size"]),
            drop=float(settings["dropout"]),
            refine_dim=int(metadata["refine_dim"]),
        )
