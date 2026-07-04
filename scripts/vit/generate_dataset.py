"""
generate_dataset.py
===================
Procedurally compose Super Mario Bros scenes from the extracted sprites and
emit pixel-accurate per-patch semantic labels for training a Vision Transformer.

Task: PATCH-LEVEL SEMANTIC SEGMENTATION plus Mario support-state supervision.
  - Each 256x240 RGB scene is built by stamping real SMB sprites on a tile grid.
  - As every sprite is stamped, its class id is also written to a per-pixel
    label canvas using the sprite's alpha mask -> ground truth is exact.
  - The pixel labels are reduced to a (H/P x W/P) grid (one class per ViT patch)
    via a priority-aware majority vote, so small actors (coins, Mario) survive.
  - Mario is placed on ground, on pipes/blocks, or in air so the trainer can
    derive air/ground/platform support labels from the semantic grid.

Run:
    python scripts/vit/generate_dataset.py --train 4000 --val 800
Outputs:
    data/vit/train.npz , data/vit/val.npz   (images uint8, labels int64)
    data/vit/preview_*.png                  (a few visual sanity checks)
"""
import os
import argparse
import numpy as np
from PIL import Image

HERE      = os.path.dirname(os.path.abspath(__file__))
PROJECT   = os.path.dirname(os.path.dirname(HERE))
SPRITE_DIR = os.path.join(PROJECT, "assets", "sprites")
OUT_DIR    = os.path.join(PROJECT, "data", "vit")

# ── Canvas / patch geometry ──────────────────────────────────────────────────
W, H      = 256, 240          # scene size (NES viewport)
PATCH     = 16                # ViT patch size
GW, GH    = W // PATCH, H // PATCH   # 16 x 15 label grid
TILE      = 16                # placement grid
SKY_RGB   = (107, 140, 255)

# ── Semantic classes ─────────────────────────────────────────────────────────
CLASSES = [
    "sky", "ground", "brick", "question_block", "pipe", "coin",
    "goomba", "koopa", "mario", "mushroom", "hill", "cloud", "bush",
]
CID = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Higher number = wins a patch even with fewer pixels (actors beat scenery).
PRIORITY = {
    "mario": 9, "mushroom": 8, "coin": 8, "goomba": 8, "koopa": 8,
    "question_block": 6, "brick": 6, "pipe": 6,
    "ground": 4, "hill": 2, "bush": 2, "cloud": 2, "sky": 0,
}
MIN_FRAC = 0.12   # a class must cover >=12% of a patch to claim it by priority


def load_sprites():
    sprites = {}
    for name in CLASSES:
        if name == "sky":
            continue
        p = os.path.join(SPRITE_DIR, f"{name}.png")
        sprites[name] = np.array(Image.open(p).convert("RGBA"))
    return sprites


class Scene:
    """Holds the RGB image and the per-pixel class-id canvas."""
    def __init__(self, rng):
        self.rng = rng
        self.img = np.zeros((H, W, 4), np.uint8)
        self.img[..., :3] = SKY_RGB
        self.img[..., 3] = 255
        self.lab = np.full((H, W), CID["sky"], np.int64)

    def stamp(self, sprite, x, y, cls):
        """Alpha-composite `sprite` at (x,y); write `cls` where sprite is opaque."""
        sh, sw = sprite.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + sw), min(H, y + sh)
        if x0 >= x1 or y0 >= y1:
            return
        sx0, sy0 = x0 - x, y0 - y
        sub = sprite[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        alpha = sub[..., 3:4].astype(np.float32) / 255.0
        dst = self.img[y0:y1, x0:x1, :3].astype(np.float32)
        self.img[y0:y1, x0:x1, :3] = (alpha * sub[..., :3] + (1 - alpha) * dst).astype(np.uint8)
        mask = sub[..., 3] > 40
        self.lab[y0:y1, x0:x1][mask] = CID[cls]

    def patch_labels(self):
        """Reduce per-pixel labels to a GH x GW patch-class grid."""
        grid = np.zeros((GH, GW), np.int64)
        for gy in range(GH):
            for gx in range(GW):
                patch = self.lab[gy*PATCH:(gy+1)*PATCH, gx*PATCH:(gx+1)*PATCH]
                counts = np.bincount(patch.ravel(), minlength=NUM_CLASSES)
                frac = counts / counts.sum()
                # priority-aware pick: highest-priority class above MIN_FRAC
                best, best_pri = None, -1
                for c in range(NUM_CLASSES):
                    if frac[c] >= MIN_FRAC and PRIORITY[CLASSES[c]] > best_pri:
                        best, best_pri = c, PRIORITY[CLASSES[c]]
                grid[gy, gx] = best if best is not None else int(counts.argmax())
        return grid


def build_scene(sprites, rng):
    s = Scene(rng)
    support_surfaces = []

    # ── Background scenery ────────────────────────────────────────────────────
    for _ in range(rng.randint(0, 3)):                       # clouds
        s.stamp(sprites["cloud"], rng.randint(0, W-26), rng.randint(8, 80), "cloud")
    ground_top = H - 2 * TILE                                # two ground rows
    for _ in range(rng.randint(0, 2)):                       # hills
        hh = sprites["hill"].shape[0]
        s.stamp(sprites["hill"], rng.randint(-20, W-40), ground_top - hh, "hill")
    for _ in range(rng.randint(0, 3)):                       # bushes
        bh = sprites["bush"].shape[0]
        s.stamp(sprites["bush"], rng.randint(0, W-16), ground_top - bh, "bush")

    # ── Ground with occasional gaps ───────────────────────────────────────────
    gap_cols = set()
    if rng.random() < 0.5:
        g0 = rng.randint(2, GW-4); gap_cols = {g0, g0+1}
    for gx in range(GW):
        if gx in gap_cols:
            continue
        for row in range(2):
            s.stamp(sprites["ground"], gx*TILE, ground_top + row*TILE, "ground")
        support_surfaces.append((gx * TILE, (gx + 1) * TILE, ground_top, "ground"))

    # ── Pipes (sit on the ground) ─────────────────────────────────────────────
    for _ in range(rng.randint(0, 2)):
        ph, pw = sprites["pipe"].shape[:2]
        px = rng.randint(0, GW-3) * TILE
        s.stamp(sprites["pipe"], px, ground_top - ph, "pipe")
        support_surfaces.append((px, px + pw, ground_top - ph, "pipe"))

    # ── Floating block rows (bricks + question blocks) ────────────────────────
    for _ in range(rng.randint(1, 3)):
        row_y = rng.randint(2, GH-4) * TILE
        x = rng.randint(1, GW-5) * TILE
        for _ in range(rng.randint(2, 5)):
            if x > W - TILE:
                break
            kind = "question_block" if rng.random() < 0.35 else "brick"
            s.stamp(sprites[kind], x, row_y, kind)
            support_surfaces.append((x, x + TILE, row_y, kind))
            x += TILE

    # ── Coins (floating) ──────────────────────────────────────────────────────
    for _ in range(rng.randint(0, 5)):
        s.stamp(sprites["coin"], rng.randint(0, W-8), rng.randint(2*TILE, ground_top-TILE), "coin")

    # ── Enemies on the ground ─────────────────────────────────────────────────
    for _ in range(rng.randint(0, 3)):
        kind = "goomba" if rng.random() < 0.6 else "koopa"
        eh = sprites[kind].shape[0]
        s.stamp(sprites[kind], rng.randint(0, W-16), ground_top - eh, kind)

    # ── A loose mushroom now and then ─────────────────────────────────────────
    if rng.random() < 0.25:
        s.stamp(sprites["mushroom"], rng.randint(0, W-16), ground_top - 16, "mushroom")

    # ── Mario on ground, on platforms, or in air ──────────────────────────────
    mh = sprites["mario"].shape[0]
    mw = sprites["mario"].shape[1]
    x, y = choose_mario_pose(support_surfaces, rng, mw=mw, mh=mh, ground_top=ground_top)
    s.stamp(sprites["mario"], x, y, "mario")

    return s


def choose_mario_pose(support_surfaces, rng, *, mw, mh, ground_top):
    """Choose a Mario location that diversifies support-state labels."""

    roll = rng.random()
    platform_surfaces = [
        surface for surface in support_surfaces if surface[3] in {"pipe", "brick", "question_block"}
    ]
    ground_surfaces = [surface for surface in support_surfaces if surface[3] == "ground"]

    if roll < 0.20:
        max_y = max(2 * TILE, ground_top - mh - TILE)
        return rng.randint(0, max(W - mw, 0)), rng.randint(2 * TILE, max_y)

    if roll < 0.45 and platform_surfaces:
        return _pose_on_surface(rng.choice(platform_surfaces), rng, mw=mw, mh=mh)

    if ground_surfaces:
        return _pose_on_surface(rng.choice(ground_surfaces), rng, mw=mw, mh=mh)

    return rng.randint(0, max(W - mw, 0)), ground_top - mh


def _pose_on_surface(surface, rng, *, mw, mh):
    x0, x1, top, _kind = surface
    min_x = max(0, int(x0))
    max_x = min(W - mw, int(x1) - mw)
    if max_x < min_x:
        x = max(0, min(W - mw, int(round((x0 + x1 - mw) / 2))))
    else:
        x = rng.randint(min_x, max_x)
    return x, int(top) - mh


def make_split(n, sprites, seed):
    rng = np.random.RandomState(seed)
    import random as pyr
    r = pyr.Random(seed)
    imgs = np.zeros((n, H, W, 3), np.uint8)
    labs = np.zeros((n, GH, GW), np.int64)
    for i in range(n):
        s = build_scene(sprites, r)
        imgs[i] = s.img[..., :3]
        labs[i] = s.patch_labels()
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{n}")
    return imgs, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val",   type=int, default=800)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sprites = load_sprites()

    print(f"Generating {args.train} training scenes...")
    tr_x, tr_y = make_split(args.train, sprites, seed=0)
    np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), images=tr_x, labels=tr_y)

    print(f"Generating {args.val} validation scenes...")
    va_x, va_y = make_split(args.val, sprites, seed=12345)
    np.savez_compressed(os.path.join(OUT_DIR, "val.npz"), images=va_x, labels=va_y)

    # class distribution
    counts = np.bincount(tr_y.ravel(), minlength=NUM_CLASSES)
    print("\nPatch-class distribution (train):")
    for c, name in enumerate(CLASSES):
        print(f"  {name:14s} {counts[c]:8d}  ({100*counts[c]/counts.sum():5.2f}%)")

    # previews
    for i in range(4):
        Image.fromarray(tr_x[i]).save(os.path.join(OUT_DIR, f"preview_{i}.png"))
    print(f"\nSaved dataset + previews to {OUT_DIR}")
    print(f"Classes ({NUM_CLASSES}): {CLASSES}")


if __name__ == "__main__":
    main()
