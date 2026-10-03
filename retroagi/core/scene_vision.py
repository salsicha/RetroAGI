"""Training, checkpoints and measurements for the shared scene vision transformer.

Everything here is game-independent: it takes screens (uint8 [N, 240, 256, 3])
and their exact labels (smb_scene_labels.SceneLabels, turned into training
targets by scene_targets), so Block SMB (labels from the simulator's drawing)
and Full SMB (labels read from game memory) are trained and measured by the
same code and report the same numbers with the same meaning.
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
from .smb_pixel_types import PIXEL_TYPES, VISIBLE_COLUMNS, VISIBLE_ROWS
from .smb_scene_labels import (
    ENEMY_KINDS,
    SUPPORTS,
    SceneObservation,
    box_overlap,
    scene_from_labels,
    scene_targets,
)
from .vision import SceneVisionTransformer

# A found object matches a true one when their boxes overlap at least this much.
MATCH_OVERLAP = 0.5
# A found surface matches a true one at most this many rows away in height,
# overlapping at least MATCH_OVERLAP of their joint width.
SURFACE_HEIGHT_TOLERANCE = 2
# ── Targets and batches ───────────────────────────────────────────────────────

TARGET_KEYS = ("types", "kind", "facing", "support", "on_something", "stomping", "mario_present")
# A picture of Mario landing on an enemy counts this many times in the
# feet-on-something loss: such pictures are rare and look like being in the air.
# Pictures of Mario landing on an enemy he stomps count this many times in the
# land detector's training loss: rare, so weighted up, but not so far that they
# pull the shared model away from everything else (at 20 the enemy kinds,
# moving platforms and power-ups of the Full SMB model came out clearly worse).
STOMP_WEIGHT = 4.0


def frame_targets(labels) -> dict[str, torch.Tensor]:
    """scene_targets as tensors (what a data loader yields per frame)."""
    return {
        name: torch.from_numpy(np.asarray(value)) for name, value in scene_targets(labels).items()
    }


def stack_targets(targets: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    return {name: torch.stack([torch.as_tensor(t[name]) for t in targets]) for name in TARGET_KEYS}


def pixel_type_weights(types, *, power: float = 0.5, cap: float = 50.0) -> torch.Tensor:
    """Cross-entropy weights that lift rare pixel types (Mario, coins, enemies, ...).

    Each type present is weighted by (share of pixels)^-power, capped at ``cap``
    times the most common type's weight, and scaled so an average pixel weighs one.
    """
    types = torch.as_tensor(np.asarray(types))
    counts = torch.bincount(types.flatten().long(), minlength=len(PIXEL_TYPES)).double()
    shares = counts / counts.sum().clamp_min(1)
    weights = shares.clamp_min(1e-9).pow(-power)
    weights = torch.minimum(weights, weights[counts > 0].min() * cap)
    weights = weights / (weights * shares).sum().clamp_min(1e-12)
    return weights.float()


def screens_to_device(images, device: torch.device) -> torch.Tensor:
    images = torch.as_tensor(images).to(device, non_blocking=True)
    return images.permute(0, 3, 1, 2).float().div_(255.0)


# ── Losses ────────────────────────────────────────────────────────────────────


def scene_losses(
    out: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    pixel_weights: Optional[torch.Tensor] = None,
    stomp_weight: float = STOMP_WEIGHT,
) -> dict[str, torch.Tensor]:
    """Every head's loss and their sum ("total").

    - pixels: per-pixel cross-entropy on the types (rare types weighted up);
    - kind: enemy kind in every cell holding enemy pixels;
    - facing, support, feet on something: Mario's, over frames that show him
      (pictures of a stomp weighted up by ``stomp_weight``).
    """
    types = targets["types"].long()
    pixel_logits = out["pixel_logits"].float()
    losses = {"pixels": F.cross_entropy(pixel_logits, types, weight=pixel_weights)}
    nothing = pixel_logits.sum() * 0.0
    enemy_cells = targets["kind"] >= 0
    if enemy_cells.any():
        kind_logits = out["kind_logits"].float().permute(0, 2, 3, 1)[enemy_cells]
        losses["kind"] = F.cross_entropy(kind_logits, targets["kind"][enemy_cells])
    else:
        losses["kind"] = nothing
    mario = targets["mario_present"].bool()
    if mario.any():
        losses["facing"] = F.cross_entropy(
            out["facing_logits"].float()[mario], targets["facing"][mario]
        )
        losses["support"] = F.cross_entropy(
            out["support_logits"].float()[mario], targets["support"][mario]
        )
        weight = 1.0 + (stomp_weight - 1.0) * targets["stomping"][mario].float()
        each = F.cross_entropy(
            out["on_something_logits"].float()[mario],
            targets["on_something"][mario],
            reduction="none",
        )
        losses["on_something"] = (each * weight).sum() / weight.sum()
    else:
        losses["facing"] = losses["support"] = losses["on_something"] = nothing
    losses["total"] = sum(losses.values())
    return losses


def _to_device(targets: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value).to(device, non_blocking=True)
        for name, value in targets.items()
    }


@torch.no_grad()
def held_out_loss(model, images, targets, device, batch_size: int = 32) -> dict[str, float]:
    """Mean of each loss over held-out screens, every pixel and picture counted
    once (no weighting): the measure checkpoints are chosen by."""
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    for start in range(0, len(images), batch_size):
        image = screens_to_device(images[start : start + batch_size], device)
        batch = _to_device(
            {name: value[start : start + batch_size] for name, value in targets.items()}, device
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model.heads(image)
        for name, value in scene_losses(out, batch, stomp_weight=1.0).items():
            sums[name] = sums.get(name, 0.0) + float(value) * len(image)
    model.train(was_training)
    return {name: value / len(images) for name, value in sums.items()}


# ── Training ──────────────────────────────────────────────────────────────────


def train_scene_vision(
    model: SceneVisionTransformer,
    batches: Iterable,
    *,
    epochs: int,
    steps_per_epoch: int,
    held_out_images,
    held_out_targets: Mapping[str, Any],
    pixel_weights: torch.Tensor,
    device: torch.device,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_steps: int = 500,
    gradient_clip_norm: float = 1.0,
    uniform_weights_from: float = 0.5,
    on_epoch: Optional[Callable[[int, dict, bool], None]] = None,
) -> dict:
    """Train every head of the scene vision transformer together.

    ``batches`` yields (screens uint8 [B, 240, 256, 3], targets) with targets
    stacked from scene_targets (stack_targets). Rare pixel types are weighted
    up so they are learned early; from ``uniform_weights_from`` (a share of
    training) the weights blend linearly to uniform. After each epoch the
    held-out screens are measured and ``on_epoch(epoch, metrics, improved)``
    is called; ``improved`` marks a new lowest held-out total loss, the moment
    to save a checkpoint. Returns the best metrics.
    """
    model.to(device).train()
    pixel_weights = pixel_weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        running: dict[str, float] = {}
        for _ in range(steps_per_epoch):
            images, targets = next(batches)
            image = screens_to_device(images, device)
            targets = _to_device(targets, device)
            progress = step / max(total_steps, 1)
            blend = max(0.0, progress - uniform_weights_from) / max(1 - uniform_weights_from, 1e-9)
            weights = pixel_weights + (1.0 - pixel_weights) * min(blend, 1.0)
            step += 1
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model.heads(image)
            losses = scene_losses(out, targets, weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value) / steps_per_epoch
        metrics = {
            "epoch": epoch,
            "train": running,
            "held_out": held_out_loss(model, held_out_images, held_out_targets, device),
            "minutes": (time.time() - started) / 60,
        }
        improved = best is None or metrics["held_out"]["total"] <= best["held_out"]["total"]
        if improved:
            best = metrics
        if on_epoch is not None:
            on_epoch(epoch, metrics, improved)
    return best


def epoch_line(metrics: Mapping[str, Any]) -> str:
    held = metrics["held_out"]
    parts = " ".join(f"{name}={value:.4f}" for name, value in held.items() if name != "total")
    return (
        f"Epoch {metrics['epoch']:03d} train={metrics['train']['total']:.4f} "
        f"held-out={held['total']:.4f} [{parts}] ({metrics['minutes']:.1f} min)"
    )


# ── Checkpoints ───────────────────────────────────────────────────────────────


def save_scene_vision_checkpoint(
    path: Path,
    model: SceneVisionTransformer,
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
            metrics=to_plain_data(dict(metrics)),
            config=config,
            specs={"vision": to_plain_data(model.spec)},
            states={"model": model.state_dict()},
            metadata={"trainer": trainer},
        ),
    )


def load_scene_vision_checkpoint(
    path: Path,
    *,
    stage,
    model_class=SceneVisionTransformer,
    device: str | torch.device = "cpu",
) -> tuple[SceneVisionTransformer, dict]:
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
        context=f"scene vision checkpoint {path}",
    )
    model.load_state_dict(checkpoint["states"]["model"])
    return model, checkpoint


# ── Measurements ──────────────────────────────────────────────────────────────


def _match(found: list, true: list, same) -> list[tuple[int, int]]:
    """Greedy one-to-one pairs (found index, true index) for which same() holds, best first."""
    candidates = sorted(
        (
            (-score, i, j)
            for i, a in enumerate(found)
            for j, b in enumerate(true)
            if (score := same(a, b)) is not None
        )
    )
    used_found, used_true, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_found and j not in used_true:
            used_found.add(i)
            used_true.add(j)
            pairs.append((i, j))
    return pairs


def _box_match(a, b) -> Optional[float]:
    overlap = box_overlap(a, b)
    return overlap if overlap >= MATCH_OVERLAP else None


def _edge_error(a, b) -> float:
    return float(max(abs(x - y) for x, y in zip(a, b)))


def _surface_match(a, b) -> Optional[float]:
    if a.moving != b.moving or abs(a.top - b.top) > SURFACE_HEIGHT_TOLERANCE:
        return None
    shared = min(a.x1, b.x1) - max(a.x0, b.x0)
    joint = max(a.x1, b.x1) - min(a.x0, b.x0)
    score = shared / joint if joint > 0 else 0.0
    return score if score >= MATCH_OVERLAP else None


def _gap_match(a, b) -> Optional[float]:
    shared = min(a.x1, b.x1) - max(a.x0, b.x0)
    joint = max(a.x1, b.x1) - min(a.x0, b.x0)
    score = shared / joint if joint > 0 else 0.0
    return score if score >= MATCH_OVERLAP else None


class _Tally:
    """Found / true / matched counts and edge errors of one kind of thing."""

    def __init__(self):
        self.found = self.true = self.matched = 0
        self.errors: list[float] = []

    def add(self, found: list, true: list, same, edges) -> list[tuple[int, int]]:
        pairs = _match(found, true, same)
        self.found += len(found)
        self.true += len(true)
        self.matched += len(pairs)
        self.errors += [_edge_error(edges(found[i]), edges(true[j])) for i, j in pairs]
        return pairs

    def report(self) -> dict[str, Any]:
        errors = np.asarray(self.errors)
        return {
            "true": self.true,
            "found": self.found,
            "recall": self.matched / self.true if self.true else None,
            "precision": self.matched / self.found if self.found else None,
            "edge_error_px": (
                {
                    "mean": float(errors.mean()),
                    "p95": float(np.percentile(errors, 95)),
                    "max": float(errors.max()),
                }
                if errors.size
                else None
            ),
        }


def compare_scenes(found: SceneObservation, true: SceneObservation, tallies: dict) -> None:
    """Add one frame's comparison to the tallies (see evaluate_scene_vision)."""
    mario = tallies["mario"]
    if true.mario.box is not None:
        mario["frames"] += 1
        if (
            found.mario.box is not None
            and box_overlap(found.mario.box, true.mario.box) >= MATCH_OVERLAP
        ):
            mario["found"] += 1
            mario["edge_errors"].append(_edge_error(found.mario.box, true.mario.box))
        mario["facing"] += found.mario.facing_right == true.mario.facing_right
        mario["support"] += found.mario.support == true.mario.support
        mario["support_confusion"][true.mario.support][found.mario.support] += 1
        mario["on_something"] += found.mario.on_something == true.mario.on_something
        if true.mario.on_something and true.mario.support == "air":
            mario["stomps"] += 1
            mario["stomps_seen"] += found.mario.on_something
    elif found.mario.box is not None:
        mario["false"] += 1
    pairs = tallies["enemy"].add(
        [e.box for e in found.enemies], [e.box for e in true.enemies], _box_match, lambda b: b
    )
    tallies["enemy_kind"]["matched"] += len(pairs)
    tallies["enemy_kind"]["correct"] += sum(
        found.enemies[i].kind == true.enemies[j].kind for i, j in pairs
    )
    for name in ("coins", "power_ups", "moving_platforms", "pipes"):
        tallies[name].add(
            list(getattr(found, name)), list(getattr(true, name)), _box_match, lambda b: b
        )
    tallies["blocks"].add(
        [b for b in found.blocks],
        [b for b in true.blocks],
        lambda a, b: _box_match(a.box, b.box) if a.kind == b.kind else None,
        lambda b: b.box,
    )
    tallies["surfaces"].add(
        list(found.surfaces), list(true.surfaces), _surface_match, lambda s: (s.x0, s.x1, s.top)
    )
    tallies["gaps"].add(list(found.gaps), list(true.gaps), _gap_match, lambda g: (g.x0, g.x1))


def _new_tallies() -> dict:
    tallies = {
        "mario": {
            "frames": 0,
            "found": 0,
            "false": 0,
            "facing": 0,
            "support": 0,
            "edge_errors": [],
            "support_confusion": {a: {b: 0 for b in SUPPORTS} for a in SUPPORTS},
            "on_something": 0,
            "stomps": 0,
            "stomps_seen": 0,
        },
        "enemy_kind": {"matched": 0, "correct": 0},
    }
    for name in (
        "enemy",
        "coins",
        "power_ups",
        "moving_platforms",
        "pipes",
        "blocks",
        "surfaces",
        "gaps",
    ):
        tallies[name] = _Tally()
    return tallies


def _report(tallies: dict, frames: int, pixels_correct: Optional[float]) -> dict[str, Any]:
    mario = tallies["mario"]
    errors = np.asarray(mario["edge_errors"])
    shown = mario["frames"]
    report = {
        "frames": frames,
        "pixels_correct": pixels_correct,
        "mario": {
            "frames": shown,
            "found": mario["found"] / shown if shown else None,
            "false_in_frames_without_mario": mario["false"],
            "edge_error_px": (
                {
                    "mean": float(errors.mean()),
                    "p95": float(np.percentile(errors, 95)),
                    "max": float(errors.max()),
                }
                if errors.size
                else None
            ),
            "facing": mario["facing"] / shown if shown else None,
            "support": mario["support"] / shown if shown else None,
            "support_confusion": mario["support_confusion"],
            "on_something": mario["on_something"] / shown if shown else None,
            "stomp_pictures": mario["stomps"],
            "stomps_seen": mario["stomps_seen"] / mario["stomps"] if mario["stomps"] else None,
        },
        "enemy_kind_accuracy": (
            tallies["enemy_kind"]["correct"] / tallies["enemy_kind"]["matched"]
            if tallies["enemy_kind"]["matched"]
            else None
        ),
    }
    for name in (
        "enemy",
        "coins",
        "power_ups",
        "moving_platforms",
        "pipes",
        "blocks",
        "surfaces",
        "gaps",
    ):
        report["enemies" if name == "enemy" else name] = tallies[name].report()
    return report


@torch.no_grad()
def evaluate_scene_vision(model, frames, *, batch_size: int = 32) -> dict[str, Any]:
    """The shared vision measurements, identical for both games.

    ``frames`` yields records with ``image`` (240x256x3 uint8), ``scene``
    (the exact smb_scene_labels.SceneLabels) and optionally ``family`` (a
    grouping name). Each frame's found scene (model.scene) is compared with
    its true scene (scene_from_labels):

    - Mario: found when a found box overlaps his true box at least half (of
      their joint area); edge error is the largest of the four edges'
      distances; facing and support (air / ground / moving platform) agree
      with the truth, over frames that show him;
    - enemies, coins, power-ups, moving platforms, pipes and blocks: found
      and true objects paired one to one by overlap (at least half);
      recall = paired / true, precision = paired / found; enemy kind accuracy
      over paired enemies;
    - surfaces: paired when at most 2 rows apart in height and overlapping at
      least half of their joint width; gaps: paired by the same overlap;
    - pixels correct: of the per-pixel types (an internal check only).
    """
    was_training = model.training
    model.eval()
    tallies = _new_tallies()
    by_family: dict[str, dict] = {}
    correct = total = 0
    count = 0
    batch: list = []
    window = (slice(*VISIBLE_ROWS), slice(*VISIBLE_COLUMNS))

    def flush():
        nonlocal correct, total, count
        images = np.stack([frame.image for frame in batch])
        out = model.heads(images)
        from .smb_scene_labels import decode_scene

        found_scenes = decode_scene(out)
        predicted_types = out["pixel_logits"].argmax(1).cpu().numpy()
        for frame, found, types in zip(batch, found_scenes, predicted_types):
            true = scene_from_labels(frame.scene)
            compare_scenes(found, true, tallies)
            family = getattr(frame, "family", "frames")
            compare_scenes(found, true, by_family.setdefault(family, _new_tallies()))
            truth = np.asarray(frame.scene.types)
            correct += int((types[window] == truth[window]).sum())
            total += truth[window].size
            count += 1
        batch.clear()

    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    model.train(was_training)
    if not count:
        raise ValueError("frames must contain at least one frame")
    report = _report(tallies, count, correct / total)
    report["families"] = {
        name: {
            "mario_found": family["mario"]["found"] / family["mario"]["frames"]
            if family["mario"]["frames"]
            else None,
            "support": family["mario"]["support"] / family["mario"]["frames"]
            if family["mario"]["frames"]
            else None,
            "enemy_recall": family["enemy"].report()["recall"],
            "surface_recall": family["surfaces"].report()["recall"],
            "gap_recall": family["gaps"].report()["recall"],
        }
        for name, family in sorted(by_family.items())
    }
    return report


def _percent(value) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def scene_report(metrics: Mapping[str, Any]) -> str:
    """The plain table both games' evaluation scripts print."""
    mario = metrics["mario"]
    error = mario["edge_error_px"] or {}
    lines = [
        f"Held-out frames: {metrics['frames']}",
        "",
        f"Mario: found in {_percent(mario['found'])} of {mario['frames']} frames that show him; "
        f"wrongly found in {mario['false_in_frames_without_mario']} frames without him",
        "  box edge error (px): "
        + (", ".join(f"{key} {value:.2f}" for key, value in error.items()) or "-"),
        f"  facing right/left correct: {_percent(mario['facing'])}",
        f"  standing / in the air / on a moving platform correct: {_percent(mario['support'])}",
        f"  feet on something (the land detector) correct: {_percent(mario['on_something'])}; "
        f"on an enemy he is stomping: {_percent(mario['stomps_seen'])} "
        f"of {mario['stomp_pictures']} pictures",
        "",
        f"{'objects':<18}{'true':>8}{'found':>8}{'recall':>10}{'precision':>11}"
        f"{'edge err p95':>14}",
    ]
    for name in (
        "enemies",
        "coins",
        "power_ups",
        "moving_platforms",
        "pipes",
        "blocks",
        "surfaces",
        "gaps",
    ):
        row = metrics[name]
        p95 = row["edge_error_px"]["p95"] if row["edge_error_px"] else None
        lines.append(
            f"{name:<18}{row['true']:>8}{row['found']:>8}{_percent(row['recall']):>10}"
            f"{_percent(row['precision']):>11}{'-' if p95 is None else f'{p95:.1f} px':>14}"
        )
    lines += [
        "",
        f"Enemy kind (walker / plant / other / defeated) correct: "
        f"{_percent(metrics['enemy_kind_accuracy'])}",
        f"Pixels given the right type (internal check): {_percent(metrics['pixels_correct'])}",
    ]
    families = metrics.get("families", {})
    if len(families) > 1:
        weakest = sorted(
            families.items(),
            key=lambda item: (
                min(value for value in item[1].values() if value is not None)
                if any(value is not None for value in item[1].values())
                else 1.0
            ),
        )[:5]
        lines += ["", "Weakest groups:"]
        for name, row in weakest:
            lines.append(
                f"  {name:<26} Mario {_percent(row['mario_found'])}, standing {_percent(row['support'])}, "
                f"enemies {_percent(row['enemy_recall'])}, surfaces {_percent(row['surface_recall'])}, "
                f"gaps {_percent(row['gap_recall'])}"
            )
    return "\n".join(lines)


def seeded(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def array_batches(
    images, targets: Mapping[str, Any], batch_size: int, seed: int
) -> Iterator[tuple]:
    """Endless shuffled minibatches from in-memory screens and stacked targets."""
    images = np.asarray(images)
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(images))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            chosen = np.sort(order[start : start + batch_size])
            yield (
                torch.from_numpy(images[chosen]),
                {name: torch.as_tensor(value)[chosen] for name, value in targets.items()},
            )


__all__ = [
    "ENEMY_KINDS",
    "array_batches",
    "compare_scenes",
    "epoch_line",
    "evaluate_scene_vision",
    "frame_targets",
    "held_out_loss",
    "load_scene_vision_checkpoint",
    "pixel_type_weights",
    "save_scene_vision_checkpoint",
    "scene_losses",
    "scene_report",
    "seeded",
    "stack_targets",
    "train_scene_vision",
]
