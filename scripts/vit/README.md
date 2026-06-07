# Super Mario Bros Vision Transformer

A self-contained pipeline that trains a Vision Transformer to perform
**patch-level semantic segmentation** of Super Mario Bros scenes, using
procedurally generated images built from **accurate, ripped SMB sprites**.

## Pipeline

```
extract_sprites.py  ->  generate_dataset.py  ->  train_vit.py
   (assets)               (data/vit/*.npz)         (model + metrics)
```

### 1. `extract_sprites.py`
Downloads the well-known `justinmeister/Mario-Level-1` sprite sheets (kept in
`assets/spritesheets/`) and slices 12 individual transparent sprites into
`assets/sprites/`. Crop coordinates are taken verbatim from that project's own
component code, so each sprite is pixel-accurate to the original NES art:

`brick, question_block, coin, mushroom, goomba, koopa, mario, ground, pipe,
hill, cloud, bush`

```bash
python scripts/vit/extract_sprites.py
```

### 2. `generate_dataset.py`
Composes random but plausible SMB scenes (256x240) by stamping sprites on a
16px tile grid: sky, ground (with gaps), pipes, floating brick/?-block rows,
coins, enemies, scenery, and Mario. As every sprite is stamped, its class id is
written to a per-pixel label canvas, then reduced to a **16x15 patch-class grid**
(one label per ViT patch) via a priority-aware majority vote so small actors
(coins, Mario) survive.

```bash
python scripts/vit/generate_dataset.py --train 5000 --val 1000
# -> data/vit/train.npz, val.npz, preview_*.png
```

13 classes: `sky, ground, brick, question_block, pipe, coin, goomba, koopa,
mario, mushroom, hill, cloud, bush`.

### 3. `train_vit.py`
A 2.87M-parameter ViT:
`Conv2d patch-embed (16x16) -> 240 tokens + learned positions ->
6 pre-norm Transformer blocks -> per-token linear head -> class per patch`.
Trained with class-balanced cross-entropy (sky is ~75% of patches).

```bash
python scripts/vit/train_vit.py --epochs 30 --batch 64 --dim 192 --depth 6
# -> data/vit/vit_smb.pth, predictions.png
```

## Results (1000 held-out scenes, 30 epochs, ~17 min on Apple MPS)

| Metric | Value |
|---|---|
| Overall patch accuracy | **99.94%** |
| Foreground accuracy (non-sky) | **99.89%** |
| Mean IoU | **99.14%** |

Per-class IoU ranges from 96.0% (mushroom, the rarest class) to 100% (sky,
ground, pipe). `predictions.png` shows `scene | ground-truth | prediction`
triplets — the predicted patch grid matches ground truth almost exactly.
