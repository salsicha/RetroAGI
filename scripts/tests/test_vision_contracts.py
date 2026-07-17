"""Contract tests pinning vision class semantics against their sources of truth.

The Full SMB DeepLab class permutation (bricks/boxes/enemies swapped between
the shipped checkpoint and the package tuple) survived because nothing tied
class order to an artifact. These tests pin:

- the dataset generator, ViT trainer, and package class lists to each other,
- the Block SMB renderer's palette to the package color table, and
- the shipped Full SMB ViT checkpoint's per-class behavior on labeled frames.
"""

import unittest
from pathlib import Path

import numpy as np
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.vision import BLOCK_CLASS_COLORS, BLOCK_SEMANTIC_CLASSES
from retroagi.stages.full_smb.vision import (
    FULL_SMB_VIT_CLASSES,
    load_full_smb_vit_checkpoint,
)

FULL_SMB_VIT_CHECKPOINT = Path("data/vit/full_smb_vit.pth")
FULL_SMB_VAL_DATASET = Path("data/vit/val.npz")
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _artifact_available(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return not handle.read(len(GIT_LFS_POINTER_PREFIX)).startswith(GIT_LFS_POINTER_PREFIX)


class TestFullSMBVITClassContracts(unittest.TestCase):
    def test_dataset_generator_trainer_and_package_class_lists_are_identical(self):
        from scripts.vit.generate_dataset import CLASSES as generator_classes
        from scripts.vit.train_vit import CLASSES as trainer_classes

        self.assertEqual(tuple(generator_classes), FULL_SMB_VIT_CLASSES)
        self.assertEqual(tuple(trainer_classes), FULL_SMB_VIT_CLASSES)

    def test_shipped_checkpoint_predicts_each_class_at_its_package_index(self):
        if not (
            _artifact_available(FULL_SMB_VIT_CHECKPOINT)
            and _artifact_available(FULL_SMB_VAL_DATASET)
        ):
            self.skipTest("shipped Full SMB ViT checkpoint or val dataset not available")

        data = np.load(FULL_SMB_VAL_DATASET)
        images = torch.tensor(data["images"][:8], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        labels = torch.tensor(data["labels"][:8], dtype=torch.long)
        loaded = load_full_smb_vit_checkpoint(FULL_SMB_VIT_CHECKPOINT)
        with torch.no_grad():
            predictions = loaded.model.encode(images).semantic_logits.argmax(1)

        overall = (predictions == labels).float().mean().item()
        self.assertGreaterEqual(overall, 0.9)
        # Per-class recall at the package index catches class-order
        # permutations: a shuffled class list scores ~0 for swapped classes.
        for class_id, class_name in enumerate(FULL_SMB_VIT_CLASSES):
            mask = labels == class_id
            if int(mask.sum()) < 5:
                continue
            recall = (predictions[mask] == class_id).float().mean().item()
            self.assertGreaterEqual(
                recall,
                0.7,
                f"class {class_name!r} recall {recall:.3f} at package index "
                f"{class_id}; the checkpoint and package class order disagree",
            )


class TestBlockSMBPaletteContract(unittest.TestCase):
    def test_renderer_emits_only_package_palette_colors(self):
        env = MarioScenarioEnv()
        try:
            env.reset(seed=7)
            frame = env.render()
            for _ in range(30):
                _observation, _reward, terminated, truncated, _info = env.step(1)
                frame = env.render()
                if terminated or truncated:
                    break
        finally:
            env.close()

        palette = {color for colors in BLOCK_CLASS_COLORS.values() for color in colors}
        rendered = {tuple(color) for color in frame.reshape(-1, 3).tolist()}
        unknown = rendered - palette
        self.assertFalse(
            unknown,
            "renderer produced colors missing from BLOCK_CLASS_COLORS: "
            f"{sorted(unknown)[:8]} — the palette and renderer have drifted",
        )

    def test_package_palette_names_match_semantic_classes(self):
        self.assertEqual(tuple(BLOCK_CLASS_COLORS), BLOCK_SEMANTIC_CLASSES)


if __name__ == "__main__":
    unittest.main()
