"""Training, checkpoints and measurements for the shared per-pixel vision transformer.

Everything here is game-independent: it takes screens (uint8 [N, 240, 256, 3])
and their true pixel types (uint8 [N, 240, 256], smb_pixel_types.TYPE_ID), so
Block SMB (labels from the simulator's drawing) and Full SMB (labels from game
memory) train and are measured by the same code.
"""

import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from .compatibility import validate_checkpoint_compatibility
from .config import ModelConfig, to_plain_data
from .smb_pixel_types import PIXEL_TYPES, TYPE_ID, compare, mario_position, mario_standing
from .vision import PixelVisionTransformer

# ── Training ──────────────────────────────────────────────────────────────────


def pixel_type_weights(labels, *, power: float = 0.5, cap: float = 50.0) -> torch.Tensor:
    """Cross-entropy weights that lift rare types (Mario, coins, enemies, ? blocks).

    Each type present is weighted by (share of pixels)^-power, capped at ``cap``
    times the most common type's weight, and scaled so an average pixel weighs one.
    """
    labels = torch.as_tensor(np.asarray(labels))
    counts = torch.bincount(labels.flatten().long(), minlength=len(PIXEL_TYPES)).double()
    shares = counts / counts.sum().clamp_min(1)
    weights = shares.clamp_min(1e-9).pow(-power)
    weights = torch.minimum(weights, weights[counts > 0].min() * cap)
    weights = weights / (weights * shares).sum().clamp_min(1e-12)
    return weights.float()


def array_batches(images, labels, batch_size: int, seed: int) -> Iterator[tuple]:
    """Endless shuffled minibatches from in-memory screens and labels."""
    images, labels = np.asarray(images), np.asarray(labels)
    if images.shape[:3] != labels.shape or images.shape[1:] != (240, 256, 3):
        raise ValueError("expected images [N, 240, 256, 3] and labels [N, 240, 256]")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(images))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            chosen = np.sort(order[start : start + batch_size])
            yield torch.from_numpy(images[chosen]), torch.from_numpy(labels[chosen])


def screens_to_device(images, device: torch.device) -> torch.Tensor:
    images = torch.as_tensor(images).to(device, non_blocking=True)
    return images.permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def evaluate_pixel_types(model, images, labels, weights, device, batch_size: int = 64) -> dict:
    """Weighted loss, share of pixels correct and each type's IoU on held-out screens."""
    was_training = model.training
    model.eval()
    types = len(PIXEL_TYPES)
    confusion = torch.zeros(types, types, dtype=torch.long, device=device)
    loss_sum = 0.0
    for start in range(0, len(images), batch_size):
        image = screens_to_device(images[start : start + batch_size], device)
        target = torch.as_tensor(labels[start : start + batch_size]).to(device).long()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model.pixel_logits(image)
        loss_sum += F.cross_entropy(logits.float(), target, weight=weights).item() * len(image)
        guess = logits.argmax(1)
        confusion += torch.bincount(
            (target * types + guess).flatten(), minlength=types * types
        ).view(types, types)
    model.train(was_training)
    confusion = confusion.double()
    hits = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - hits
    present = confusion.sum(1) > 0
    iou = hits / union.clamp_min(1)
    return {
        "loss": loss_sum / len(images),
        "pixels_correct": float(hits.sum() / confusion.sum()),
        "mean_type_iou": float(iou[present].mean()),
        "type_iou": {
            name: float(iou[i]) if present[i] else None for i, name in enumerate(PIXEL_TYPES)
        },
    }


def train_pixel_vision(
    model: PixelVisionTransformer,
    batches: Iterable,
    *,
    epochs: int,
    steps_per_epoch: int,
    held_out_images,
    held_out_labels,
    weights: torch.Tensor,
    device: torch.device,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_steps: int = 500,
    gradient_clip_norm: float = 1.0,
    uniform_weights_from: float = 0.5,
    on_epoch: Optional[Callable[[int, dict, bool], None]] = None,
) -> dict:
    """Per-pixel cross-entropy training with rare types weighted up.

    ``batches`` yields (screens uint8 [B, 240, 256, 3], labels uint8
    [B, 240, 256]). Rare types are weighted up so they are learned early;
    from ``uniform_weights_from`` (a share of training) the weights blend
    linearly to uniform, so training ends on plain per-pixel accuracy instead
    of a bias toward rare types at uncertain edges. After each epoch the
    held-out screens are measured (plain cross-entropy) and
    ``on_epoch(epoch, metrics, improved)`` is called; ``improved`` marks a new
    best mean type IoU, the moment to save a checkpoint. Returns the best metrics.
    """
    model.to(device).train()
    weights = weights.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = epochs * steps_per_epoch

    def learning_rate_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    batches = iter(batches)
    best, started, step = None, time.time(), 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        for _ in range(steps_per_epoch):
            images, labels = next(batches)
            image = screens_to_device(images, device)
            target = torch.as_tensor(labels).to(device, non_blocking=True).long()
            progress = step / max(total_steps, 1)
            blend = max(0.0, progress - uniform_weights_from) / max(1 - uniform_weights_from, 1e-9)
            step_weights = weights + (1.0 - weights) * min(blend, 1.0)
            step += 1
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model.pixel_logits(image)
            loss = F.cross_entropy(logits.float(), target, weight=step_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += loss.item()
        metrics = evaluate_pixel_types(model, held_out_images, held_out_labels, None, device)
        metrics.update(
            epoch=epoch,
            train_loss=running / steps_per_epoch,
            minutes=(time.time() - started) / 60,
        )
        improved = best is None or metrics["mean_type_iou"] >= best["mean_type_iou"]
        if improved:
            best = metrics
        if on_epoch is not None:
            on_epoch(epoch, metrics, improved)
    return best


def train_pixel_vision_on_arrays(
    model: PixelVisionTransformer,
    images,
    labels,
    *,
    held_out_images,
    held_out_labels,
    epochs: int,
    device: torch.device,
    batch_size: int = 48,
    seed: int = 0,
    **options,
) -> dict:
    """train_pixel_vision on in-memory screens; an epoch is one pass over them."""
    return train_pixel_vision(
        model,
        array_batches(images, labels, batch_size, seed),
        epochs=epochs,
        steps_per_epoch=max(len(images) // batch_size, 1),
        held_out_images=held_out_images,
        held_out_labels=held_out_labels,
        weights=pixel_type_weights(labels),
        device=device,
        **options,
    )


def epoch_line(metrics: dict) -> str:
    ious = " ".join(
        f"{name}={value * 100:.2f}"
        for name, value in metrics["type_iou"].items()
        if value is not None
    )
    return (
        f"Epoch {metrics['epoch']:03d} train={metrics['train_loss']:.5f} "
        f"held-out={metrics['loss']:.5f} pixels={metrics['pixels_correct'] * 100:.3f}% "
        f"mean IoU={metrics['mean_type_iou'] * 100:.2f}% [{ious}] "
        f"({metrics['minutes']:.1f} min)"
    )


# ── Checkpoints ───────────────────────────────────────────────────────────────


def save_pixel_vision_checkpoint(
    path: Path,
    model: PixelVisionTransformer,
    *,
    stage: str,
    metrics: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    epoch: int = 0,
    trainer: str = "",
) -> None:
    config = dict(config or {})
    config["model"] = model.architecture()
    save_checkpoint(
        path,
        build_checkpoint(
            stage=stage,
            model_name=model.spec.name,
            checkpoint_kind="vision_encoder",
            epoch=epoch,
            metrics=dict(metrics),
            config=config,
            specs={"vision": to_plain_data(model.spec)},
            states={"model": model.state_dict()},
            metadata={"trainer": trainer},
        ),
    )


def load_pixel_vision_checkpoint(
    path: Path,
    *,
    stage,
    model_class=PixelVisionTransformer,
    device: str | torch.device = "cpu",
) -> tuple[PixelVisionTransformer, dict]:
    """Rebuild the model a checkpoint describes and load its weights."""
    checkpoint = load_checkpoint(path, map_location=device)
    settings = checkpoint["config"]["model"]
    model = model_class.from_architecture(settings).to(device)
    validate_checkpoint_compatibility(
        checkpoint,
        stage=stage,
        model=ModelConfig(name=model.spec.name),
        vision=model.spec,
        checkpoint_kind="vision_encoder",
        required_states=("model",),
        context=f"pixel vision checkpoint {path}",
    )
    model.load_state_dict(checkpoint["states"]["model"])
    return model, checkpoint


# ── Measurements ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_pixel_vision(model, frames, *, batch_size: int = 32) -> dict[str, Any]:
    """The shared vision measurements, identical for both games.

    ``frames`` yields records with ``image`` (240x256x3 uint8), ``labels`` (the
    true type of each pixel), ``on_ground`` (the game's own standing flag),
    ``enemy_rects`` ((x, y, w, h) screen box of each enemy object) and
    optionally ``family`` (a grouping name for the per-family breakdown).

    - pixels_correct and each type's found/correct: compare() over all frames'
      pixels pooled together;
    - mario_found: of frames whose true labels show Mario, the share whose
      predicted labels show him; position error: pixels between the predicted
      and true label images' Mario centres, over frames where both show him;
    - standing_agreement: of frames whose true labels show Mario, the share
      where mario_standing of the predicted labels equals on_ground (the true
      labels' agreement is reported as the rule's own ceiling);
    - enemies_seen: of enemy objects with at least one true enemy pixel, the
      share with at least one of those pixels predicted enemy.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    predicted, truth, families = [], [], []
    errors, mario_frames, mario_found = [], 0, 0
    standing_agree = truth_standing_agree = 0
    enemies = enemies_seen = 0
    batch = []

    def flush():
        nonlocal mario_frames, mario_found, standing_agree, truth_standing_agree
        nonlocal enemies, enemies_seen
        output = model.encode(np.stack([frame.image for frame in batch]))
        labels = output.metadata["pixel_labels"].cpu().numpy().astype(np.uint8)
        for frame, guess in zip(batch, labels):
            true_labels = np.asarray(frame.labels)
            predicted.append(guess)
            truth.append(true_labels)
            families.append(getattr(frame, "family", "frames"))
            true_centre = mario_position(true_labels)
            if true_centre is not None:
                mario_frames += 1
                centre = mario_position(guess)
                if centre is not None:
                    mario_found += 1
                    dx, dy = centre[0] - true_centre[0], centre[1] - true_centre[1]
                    errors.append(float(np.hypot(dx, dy)))
                standing_agree += mario_standing(guess) == bool(frame.on_ground)
                truth_standing_agree += mario_standing(true_labels) == bool(frame.on_ground)
            for x, y, w, h in frame.enemy_rects:
                left, top = max(x, 0), max(y, 0)
                window = (slice(top, max(y + h, top)), slice(left, max(x + w, left)))
                drawn = true_labels[window] == TYPE_ID["enemy"]
                if drawn.any():
                    enemies += 1
                    enemies_seen += bool((guess[window][drawn] == TYPE_ID["enemy"]).any())
        batch.clear()

    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    model.train(was_training)
    if not predicted:
        raise ValueError("frames must contain at least one frame")
    pooled = compare(np.stack(predicted), np.stack(truth))
    by_family = {}
    for name in sorted(set(families)):
        members = [i for i, family in enumerate(families) if family == name]
        result = compare(
            np.stack([predicted[i] for i in members]), np.stack([truth[i] for i in members])
        )
        by_family[name] = {"frames": len(members), "pixels_correct": result["pixels_correct"]}
    errors = np.asarray(errors)
    return {
        "frames": len(predicted),
        "pixels_correct": pooled["pixels_correct"],
        "types": {name: pooled[name] for name in PIXEL_TYPES},
        "mario_frames": mario_frames,
        "mario_found": mario_found / mario_frames if mario_frames else None,
        "mario_position_error_px": (
            {
                "mean": float(errors.mean()),
                "median": float(np.median(errors)),
                "p95": float(np.percentile(errors, 95)),
                "max": float(errors.max()),
            }
            if errors.size
            else None
        ),
        "standing_agreement": standing_agree / mario_frames if mario_frames else None,
        "true_label_standing_agreement": (
            truth_standing_agree / mario_frames if mario_frames else None
        ),
        "enemies": enemies,
        "enemies_seen": enemies_seen / enemies if enemies else None,
        "families": by_family,
    }


def _percent(value) -> str:
    return "-" if value is None else f"{value * 100:.3f}%"


def vision_report(metrics: Mapping[str, Any]) -> str:
    """The plain table both games' evaluation scripts print."""
    lines = [
        f"Held-out frames: {metrics['frames']}",
        f"Pixels correct: {_percent(metrics['pixels_correct'])}",
        "",
        f"{'type':<16}{'true pixels':>14}{'found':>12}{'correct':>12}",
    ]
    for name in PIXEL_TYPES:
        row = metrics["types"][name]
        lines.append(
            f"{name:<16}{row['true_pixels']:>14}"
            f"{_percent(row['found']):>12}{_percent(row['correct']):>12}"
        )
    error = metrics["mario_position_error_px"] or {}
    lines += [
        "",
        f"Frames where Mario is found: {_percent(metrics['mario_found'])} "
        f"of {metrics['mario_frames']}",
        "Mario position error (px): "
        + (", ".join(f"{key} {value:.2f}" for key, value in error.items()) or "-"),
        "Standing/air agrees with the game: "
        f"{_percent(metrics['standing_agreement'])} "
        f"(true labels: {_percent(metrics['true_label_standing_agreement'])})",
        f"Enemies seen: {_percent(metrics['enemies_seen'])} of {metrics['enemies']}",
    ]
    families = sorted(metrics["families"].items(), key=lambda item: item[1]["pixels_correct"])
    if len(families) > 1:
        lines += ["", "Weakest groups (pixels correct):"]
        lines += [
            f"  {name:<28}{_percent(row['pixels_correct']):>12}  ({row['frames']} frames)"
            for name, row in families[:5]
        ]
    return "\n".join(lines)


def seeded(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
