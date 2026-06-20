"""
train_vit.py
============
A Vision Transformer that performs PATCH-LEVEL SEMANTIC SEGMENTATION of
procedurally generated Super Mario Bros scenes.

Pipeline:
  Conv patch-embed (16x16) -> [240 tokens] + learned positions
  -> N Transformer encoder blocks -> per-token linear head -> class per patch.

Trains on data/vit/{train,val}.npz (see generate_dataset.py) and reports
overall accuracy, foreground accuracy, per-class recall and mean IoU.

Run:
    python scripts/vit/train_vit.py --epochs 20 --batch 64
Outputs:
    data/vit/vit_smb.pth          trained weights
    data/vit/predictions.png      side-by-side scene / GT / prediction
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from retroagi.core import PatchVisionTransformer, select_device


HERE    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
DATA    = os.path.join(PROJECT, "data", "vit")

CLASSES = ["sky", "ground", "brick", "question_block", "pipe", "coin",
           "goomba", "koopa", "mario", "mushroom", "hill", "cloud", "bush"]
NUM_CLASSES = len(CLASSES)
PATCH = 16
W, H  = 256, 240
GW, GH = W // PATCH, H // PATCH      # 16 x 15


# ── Model ─────────────────────────────────────────────────────────────────────
class ViTSegmenter(PatchVisionTransformer):
    """Compatibility wrapper around the shared patch vision encoder."""

    def __init__(self, dim=192, depth=6, heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__(
            semantic_classes=tuple(CLASSES),
            image_size=(H, W),
            patch_size=PATCH,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            position_class="mario",
            name="smb_sprite_vit",
        )

    def forward(self, x):
        return super().forward(x).semantic_logits

    def encode(self, observation):
        return PatchVisionTransformer.forward(self, observation)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_split(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    imgs = torch.tensor(d["images"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labs = torch.tensor(d["labels"], dtype=torch.long)
    return TensorDataset(imgs, labs)


# ── Metrics ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    correct_c = np.zeros(NUM_CLASSES); total_c = np.zeros(NUM_CLASSES)
    n_correct = n_total = fg_correct = fg_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        n_correct += (pred == y).sum().item(); n_total += y.numel()
        fg = y != 0
        fg_correct += (pred[fg] == y[fg]).sum().item(); fg_total += fg.sum().item()
        p, t = pred.cpu().numpy().ravel(), y.cpu().numpy().ravel()
        for c in range(NUM_CLASSES):
            pc, tc = p == c, t == c
            inter[c] += np.logical_and(pc, tc).sum()
            union[c] += np.logical_or(pc, tc).sum()
            correct_c[c] += np.logical_and(pc, tc).sum(); total_c[c] += tc.sum()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    recall = np.where(total_c > 0, correct_c / np.maximum(total_c, 1), np.nan)
    return {
        "acc": n_correct / n_total,
        "fg_acc": fg_correct / max(fg_total, 1),
        "miou": np.nanmean(iou),
        "iou": iou, "recall": recall,
    }


# ── Visualization ─────────────────────────────────────────────────────────────
def save_predictions(model, ds, device, path, n=4):
    import colorsys
    from PIL import Image, ImageDraw
    cols = [(135, 206, 255)] + [tuple(int(c*255) for c in colorsys.hsv_to_rgb(i/12, 0.9, 0.95)) for i in range(12)]
    model.eval()
    rows = []
    with torch.no_grad():
        for k in range(n):
            x, y = ds[k]
            pred = model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
            base = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            def overlay(grid):
                im = Image.fromarray(base).convert("RGB"); dr = ImageDraw.Draw(im)
                for gy in range(GH):
                    for gx in range(GW):
                        c = int(grid[gy, gx])
                        if c == 0:
                            continue
                        dr.rectangle([gx*PATCH, gy*PATCH, (gx+1)*PATCH-1, (gy+1)*PATCH-1],
                                     outline=cols[c], width=2)
                return np.array(im)
            rows.append(np.concatenate([base, overlay(y.numpy()), overlay(pred)], axis=1))
    out = np.concatenate(rows, axis=0)
    Image.fromarray(out).resize((out.shape[1]*2, out.shape[0]*2), Image.NEAREST).save(path)


# ── Train ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--dim",    type=int, default=192)
    ap.add_argument("--depth",  type=int, default=6)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")

    train_ds, val_ds = load_split("train"), load_split("val")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # class-balanced loss: sky dominates ~75% of patches
    counts = np.bincount(train_ds.tensors[1].numpy().ravel(), minlength=NUM_CLASSES).astype(np.float64)
    weights = (1.0 / np.sqrt(counts + 1)); weights = weights / weights.mean()
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model = ViTSegmenter(dim=args.dim, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        m = evaluate(model, val_ld, device)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss {running/len(train_ds):.4f} "
              f"| acc {m['acc']*100:5.2f}% | fg_acc {m['fg_acc']*100:5.2f}% | mIoU {m['miou']*100:5.2f}%")
        if m["miou"] > best_miou:
            best_miou = m["miou"]
            torch.save(model.state_dict(), os.path.join(DATA, "vit_smb.pth"))

    # final per-class report
    model.load_state_dict(torch.load(os.path.join(DATA, "vit_smb.pth"), map_location=device))
    m = evaluate(model, val_ld, device)
    print("\n── Final per-class metrics (val) ──")
    print(f"{'class':16s}{'recall':>9s}{'IoU':>9s}")
    for c, name in enumerate(CLASSES):
        r = m["recall"][c]; i = m["iou"][c]
        print(f"{name:16s}{('%.1f%%'%(r*100)) if not np.isnan(r) else '   n/a':>9s}"
              f"{('%.1f%%'%(i*100)) if not np.isnan(i) else '   n/a':>9s}")
    print(f"\nOverall acc {m['acc']*100:.2f}% | foreground acc {m['fg_acc']*100:.2f}% | mIoU {m['miou']*100:.2f}%")

    save_predictions(model, val_ds, device, os.path.join(DATA, "predictions.png"))
    print(f"Saved weights -> {os.path.join(DATA,'vit_smb.pth')}")
    print(f"Saved predictions -> {os.path.join(DATA,'predictions.png')}")


if __name__ == "__main__":
    main()
