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
    data/vit/full_smb_vit.pth     versioned vision checkpoint
    data/vit/vit_smb.pth          legacy raw weights
    data/vit/predictions.png      side-by-side scene / GT / prediction
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.core import save_checkpoint as save_versioned_checkpoint, select_device
from retroagi.stages.full_smb.vision import (
    FullSMBVisionTransformer,
    build_full_smb_vit_checkpoint,
)


HERE    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
DATA    = os.path.join(PROJECT, "data", "vit")
VERSIONED_CHECKPOINT = os.path.join(DATA, "full_smb_vit.pth")
LEGACY_CHECKPOINT = os.path.join(DATA, "vit_smb.pth")

CLASSES = ["sky", "ground", "brick", "question_block", "pipe", "coin",
           "goomba", "koopa", "mario", "mushroom", "hill", "cloud", "bush"]
NUM_CLASSES = len(CLASSES)
PATCH = 16
W, H  = 256, 240
GW, GH = W // PATCH, H // PATCH      # 16 x 15


# ── Model ─────────────────────────────────────────────────────────────────────
class ViTSegmenter(FullSMBVisionTransformer):
    """Compatibility wrapper around the shared patch vision encoder."""

    def __init__(self, dim=192, depth=6, heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            patch_size=PATCH,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

    def forward(self, x):
        return super().forward(x).semantic_logits

    def encode(self, observation):
        return FullSMBVisionTransformer.forward(self, observation)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_split(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    imgs = torch.tensor(d["images"], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labs = torch.tensor(d["labels"], dtype=torch.long)
    return TensorDataset(imgs, labs)


# ── Metrics ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, support_weight=1.0):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    correct_c = np.zeros(NUM_CLASSES); total_c = np.zeros(NUM_CLASSES)
    n_correct = n_total = fg_correct = fg_total = 0
    support_correct = support_total = 0
    support_loss_total = semantic_loss_total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model.encode(x)
        semantic_loss = F.cross_entropy(output.semantic_logits, y)
        support_y = support_targets_from_labels(model, y)
        support_loss = F.cross_entropy(output.support_logits, support_y)
        pred = output.semantic_logits.argmax(1)
        support_pred = output.support_ids
        batch_size = x.shape[0]
        semantic_loss_total += semantic_loss.item() * batch_size
        support_loss_total += support_loss.item() * batch_size
        n_correct += (pred == y).sum().item(); n_total += y.numel()
        support_correct += (support_pred == support_y).sum().item()
        support_total += support_y.numel()
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
        "loss": (semantic_loss_total + support_weight * support_loss_total) / len(loader.dataset),
        "semantic_loss": semantic_loss_total / len(loader.dataset),
        "support_loss": support_loss_total / len(loader.dataset),
        "support_accuracy": support_correct / max(support_total, 1),
        "iou": iou, "recall": recall,
    }


def scalar_metrics(metrics):
    values = {
        "accuracy": float(metrics["acc"]),
        "foreground_accuracy": float(metrics["fg_acc"]),
        "mean_iou": float(metrics["miou"]),
        "loss": float(metrics["loss"]),
        "semantic_loss": float(metrics["semantic_loss"]),
        "support_loss": float(metrics["support_loss"]),
        "support_accuracy": float(metrics["support_accuracy"]),
    }
    for index, name in enumerate(CLASSES):
        iou = metrics["iou"][index]
        recall = metrics["recall"][index]
        if not np.isnan(iou):
            values[f"iou/{name}"] = float(iou)
        if not np.isnan(recall):
            values[f"recall/{name}"] = float(recall)
    return values


def save_training_checkpoint(model, args, *, epoch, metrics):
    checkpoint_path = Path(args.checkpoint)
    legacy_checkpoint_path = Path(args.legacy_checkpoint)
    checkpoint = build_full_smb_vit_checkpoint(
        model,
        epoch=epoch,
        metrics=scalar_metrics(metrics),
        config={
            "model": {
                "hidden_dim": args.dim,
                "depth": args.depth,
                "heads": args.heads,
                "patch_size": PATCH,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.drop,
            },
            "training": {
                "batch_size": args.batch,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "support_weight": args.support_weight,
            },
            "data": {
                "train": os.path.relpath(os.path.join(DATA, "train.npz"), PROJECT),
                "val": os.path.relpath(os.path.join(DATA, "val.npz"), PROJECT),
            },
        },
        metadata={
            "training_script": "scripts/vit/train_vit.py",
            "legacy_raw_checkpoint": os.path.relpath(
                legacy_checkpoint_path.resolve(), PROJECT
            ),
        },
    )
    save_versioned_checkpoint(checkpoint_path, checkpoint)
    torch.save(model.state_dict(), legacy_checkpoint_path)


def support_targets_from_labels(model, labels):
    support = model.support_targets_from_labels(labels)
    if support is None:
        raise ValueError("could not infer Full SMB ViT support targets")
    return support


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
    ap.add_argument("--heads",  type=int, default=6)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--checkpoint", default=VERSIONED_CHECKPOINT)
    ap.add_argument("--legacy-checkpoint", default=LEGACY_CHECKPOINT)
    ap.add_argument("--support-weight", type=float, default=1.0)
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

    model = ViTSegmenter(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_miou = None
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            output = model.encode(x)
            semantic_loss = criterion(output.semantic_logits, y)
            support_y = support_targets_from_labels(model, y)
            support_loss = F.cross_entropy(output.support_logits, support_y)
            loss = semantic_loss + args.support_weight * support_loss
            loss.backward(); opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        m = evaluate(model, val_ld, device, support_weight=args.support_weight)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss {running/len(train_ds):.4f} "
              f"| acc {m['acc']*100:5.2f}% | fg_acc {m['fg_acc']*100:5.2f}% "
              f"| mIoU {m['miou']*100:5.2f}% | support_acc {m['support_accuracy']*100:5.2f}%")
        # Always checkpoint the first epoch (and skip NaN mIoU comparisons) so a
        # checkpoint exists for the final report step below.
        if best_miou is None or (not np.isnan(m["miou"]) and m["miou"] > best_miou):
            best_miou = -float("inf") if np.isnan(m["miou"]) else m["miou"]
            save_training_checkpoint(model, args, epoch=epoch + 1, metrics=m)

    # final per-class report
    model.load_compatible_state_dict(torch.load(args.legacy_checkpoint, map_location=device))
    m = evaluate(model, val_ld, device, support_weight=args.support_weight)
    print("\n── Final per-class metrics (val) ──")
    print(f"{'class':16s}{'recall':>9s}{'IoU':>9s}")
    for c, name in enumerate(CLASSES):
        r = m["recall"][c]; i = m["iou"][c]
        print(f"{name:16s}{('%.1f%%'%(r*100)) if not np.isnan(r) else '   n/a':>9s}"
              f"{('%.1f%%'%(i*100)) if not np.isnan(i) else '   n/a':>9s}")
    print(f"\nOverall acc {m['acc']*100:.2f}% | foreground acc {m['fg_acc']*100:.2f}% "
          f"| mIoU {m['miou']*100:.2f}% | support acc {m['support_accuracy']*100:.2f}%")

    save_predictions(model, val_ds, device, os.path.join(DATA, "predictions.png"))
    print(f"Saved checkpoint -> {args.checkpoint}")
    print(f"Saved legacy weights -> {args.legacy_checkpoint}")
    print(f"Saved predictions -> {os.path.join(DATA,'predictions.png')}")


if __name__ == "__main__":
    main()
