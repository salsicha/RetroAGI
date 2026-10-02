"""Train the Full SMB vision transformer to report the objects drawn on screen.

Frames are real emulator frames: every level start (vision_frames.LEVELS)
played from its saved start by a random player that is
rewound after each death (retroagi/stages/full_smb/vision_frames.py). Worker
processes play fresh episodes throughout training, so frames rarely repeat.
Labels are read from game memory (pixel_labels.label_frame): every accepted
frame is rebuilt from memory and matches the emulator's picture at every
visible pixel, and every object is what drew its pixels; frames memory cannot
explain are skipped. The model and the training loop are the shared ones
(retroagi.core.vision.SceneVisionTransformer, retroagi.core.scene_vision),
with the same settings as the Block SMB model.
Progress is monitored on separate plays of the levels;
scripts/vision/evaluate_full_vision.py measures on further fresh plays.

Example:
    python scripts/vit/train_full_vit.py --epochs 40 --samples-per-epoch 40000
"""

import argparse
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.core import (
    CheckpointConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    select_device,
    validate_stage_spec,
)
from retroagi.core.scene_vision import (
    epoch_line,
    frame_targets,
    pixel_type_weights,
    save_scene_vision_checkpoint,
    seeded,
    stack_targets,
    train_scene_vision,
)
from retroagi.core.smb_pixel_types import PIXEL_TYPES
from retroagi.stages.full_smb.adapter import FULL_SMB_SPEC
from retroagi.stages.full_smb.vision import DEFAULT_FULL_VIT_CHECKPOINT, FullVisionTransformer
from retroagi.stages.full_smb.vision_frames import LEVELS, labelled_frames

DEFAULT_OUTPUT = PROJECT_ROOT / DEFAULT_FULL_VIT_CHECKPOINT
DEFAULT_SEED = 7
# Frames per play before the level is restarted; most plays finish sooner.
PLAY_FRAMES = 9_000


@dataclass(frozen=True)
class TrainConfig:
    environment: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            stage="full_smb", seed=DEFAULT_SEED, rollout_steps=PLAY_FRAMES
        )
    )
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="full_smb_vit",
            hidden_dim=128,
            depth=4,
            heads=4,
            patch_size=16,
            dropout=0.0,
            metadata={"refine_dim": 32},
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            epochs=40,
            samples_per_epoch=40_000,
            batch_size=48,
            learning_rate=1e-3,
            weight_decay=0.05,
            seed=DEFAULT_SEED,
            gradient_clip_norm=1.0,
        )
    )
    evaluation: EvaluationConfig = field(
        default_factory=lambda: EvaluationConfig(
            samples=3_000,
            seed=DEFAULT_SEED + 1_000,
            metrics=("loss", "pixels_correct", "mean_type_iou"),
        )
    )
    checkpoints: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(
            output_path=DEFAULT_OUTPUT, best_metric="held_out_total_loss", best_mode="min"
        )
    )
    # Keep every `every`-th frame of a play; monitor plays keep fewer.
    every: int = 3
    monitor_every: int = 16
    monitor_plays_per_level: int = 2
    workers: int = 12
    warmup_steps: int = 500
    weight_power: float = 0.5

    def __post_init__(self) -> None:
        if self.training.samples_per_epoch is None or self.evaluation.samples is None:
            raise ValueError("samples_per_epoch and evaluation samples must be set")
        if min(self.every, self.monitor_every, self.monitor_plays_per_level) <= 0:
            raise ValueError("frame intervals and monitor plays must be positive")

    def to_dict(self) -> dict:
        return ExperimentConfig(
            environment=self.environment,
            model=self.model,
            training=self.training,
            evaluation=self.evaluation,
            checkpoints=self.checkpoints,
            name="full_vit_training",
            metadata={
                "levels": list(LEVELS),
                "every": self.every,
                "monitor_every": self.monitor_every,
                "monitor_plays_per_level": self.monitor_plays_per_level,
                "warmup_steps": self.warmup_steps,
                "weight_power": self.weight_power,
            },
        ).to_dict()


def build_model(config: TrainConfig) -> FullVisionTransformer:
    model = config.model
    return FullVisionTransformer(
        dim=model.hidden_dim,
        depth=model.depth,
        heads=model.heads,
        patch_size=model.patch_size or 16,
        drop=model.dropout,
        refine_dim=int(model.metadata["refine_dim"]),
    )


def frame_stream(seed: int, every: int, refusals: Optional[Counter] = None):
    """Endless labelled frames: random training levels, each played with a fresh seed."""
    rng = random.Random(seed)
    while True:
        yield from labelled_frames(
            rng.choice(LEVELS),
            frames=PLAY_FRAMES,
            seed=rng.randrange(2**31),
            every=every,
            refusals=refusals,
        )


class FrameStreamDataset(IterableDataset):
    """Each worker plays its own seeded episodes and shuffles them in a buffer."""

    def __init__(self, seed, *, every, buffer=1024):
        self.seed, self.every, self.buffer = seed, every, buffer

    def __iter__(self):
        worker = get_worker_info()
        seed = self.seed * 1_000_003 + (worker.id if worker is not None else 0)
        rng = random.Random(seed)
        frames = frame_stream(seed, self.every)
        pool = list(islice(frames, self.buffer))
        for frame in frames:
            index = rng.randrange(len(pool))
            out, pool[index] = pool[index], frame
            yield torch.from_numpy(out.image.copy()), frame_targets(out.scene)


def _play(level: str, seed: int, every: int) -> tuple[list, list, Counter]:
    refusals = Counter()
    frames = list(
        labelled_frames(level, frames=PLAY_FRAMES, seed=seed, every=every, refusals=refusals)
    )
    return [frame.image for frame in frames], [frame.scene for frame in frames], refusals


def level_frames(seed: int, plays_per_level: int, every: int, workers: int):
    """Frames from fresh plays of every training level, played in parallel.

    Returns (images, scene labels, refusals) with images [N, 240, 256, 3] and
    one smb_scene_labels.SceneLabels per image, in a fixed order for a given seed.
    """
    rng = random.Random(seed)
    plays = [
        (level, rng.randrange(2**31), every)
        for level in LEVELS
        for _ in range(plays_per_level)
    ]
    refusals = Counter()
    images, labels = [], []
    with ProcessPoolExecutor(workers) as pool:
        for play_images, play_labels, play_refusals in pool.map(_play, *zip(*plays)):
            images += play_images
            labels += play_labels
            refusals += play_refusals
    return np.stack(images), labels, refusals


def monitor_frames(config: TrainConfig) -> tuple[np.ndarray, dict, Counter]:
    """A fixed held-out set: separate plays of every training level."""
    images, labels, refusals = level_frames(
        config.evaluation.seed, config.monitor_plays_per_level, config.monitor_every, config.workers
    )
    order = random.Random(config.evaluation.seed).sample(
        range(len(images)), min(config.evaluation.samples, len(images))
    )
    return images[order], stack_targets([frame_targets(labels[i]) for i in order]), refusals


def save_checkpoint(path, model, epoch: int, metrics: dict, config: TrainConfig) -> None:
    save_scene_vision_checkpoint(
        path,
        model,
        stage=config.environment.stage,
        metrics=metrics,
        config=config.to_dict(),
        epoch=epoch,
        trainer="scripts.vit.train_full_vit",
    )


def train(config: TrainConfig, device_name: Optional[str] = None) -> dict:
    seed = config.training.seed
    seeded(seed)
    device = select_device(device_name or config.training.device)
    output = Path(config.checkpoints.output_path or DEFAULT_OUTPUT)
    validate_stage_spec(FULL_SMB_SPEC, context="Full ViT startup stage")
    model = build_model(config)
    if config.model.name != model.spec.name:
        raise ValueError(f"model name {config.model.name!r} is not {model.spec.name!r}")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}; learned numbers: {parameters:,}", flush=True)

    started = time.time()
    monitor_images, monitor_targets, monitor_refusals = monitor_frames(config)
    # Type weights from one more play of every training level.
    _, sample_labels, sample_refusals = level_frames(
        seed + 999, 1, config.monitor_every, config.workers
    )
    weights = pixel_type_weights(
        np.stack([labels.types for labels in sample_labels]), power=config.weight_power
    )
    print(
        f"Levels: {', '.join(LEVELS)}; {len(monitor_images)} monitor frames "
        f"({time.time() - started:.0f}s)\n"
        f"Frames refused (not fully explained by memory): monitor {sum(monitor_refusals.values())}, "
        f"weight sample {sum(sample_refusals.values())} (of {len(sample_labels)} kept)\n"
        "Pixel weights: "
        + ", ".join(f"{name}={w:.2f}" for name, w in zip(PIXEL_TYPES, weights.tolist())),
        flush=True,
    )

    batch_size = config.training.batch_size
    steps_per_epoch = -(-config.training.samples_per_epoch // batch_size)
    loader = DataLoader(
        FrameStreamDataset(seed, every=config.every),
        batch_size=batch_size,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=4,
    )

    def on_epoch(epoch: int, metrics: dict, improved: bool) -> None:
        metrics["frames_seen"] = epoch * steps_per_epoch * batch_size
        print(epoch_line(metrics), flush=True)
        if improved:
            metrics["learned_numbers"] = parameters
            save_checkpoint(output, model, epoch, metrics, config)
            print(f"Saved checkpoint: {output}", flush=True)

    return train_scene_vision(
        model,
        loader,
        epochs=config.training.epochs,
        steps_per_epoch=steps_per_epoch,
        held_out_images=monitor_images,
        held_out_targets=monitor_targets,
        pixel_weights=weights,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.warmup_steps,
        gradient_clip_norm=config.training.gradient_clip_norm,
        on_epoch=on_epoch,
    )


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=defaults.training.epochs)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=defaults.training.samples_per_epoch
    )
    parser.add_argument("--monitor-frames", type=int, default=defaults.evaluation.samples)
    parser.add_argument("--batch-size", type=int, default=defaults.training.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.training.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.training.weight_decay)
    parser.add_argument("--weight-power", type=float, default=defaults.weight_power)
    parser.add_argument("--dim", type=int, default=defaults.model.hidden_dim)
    parser.add_argument("--depth", type=int, default=defaults.model.depth)
    parser.add_argument("--heads", type=int, default=defaults.model.heads)
    parser.add_argument(
        "--refine-dim", type=int, default=int(defaults.model.metadata["refine_dim"])
    )
    parser.add_argument("--every", type=int, default=defaults.every)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = TrainConfig()
    config = replace(
        defaults,
        environment=replace(defaults.environment, seed=args.seed),
        model=replace(
            defaults.model,
            hidden_dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            metadata={"refine_dim": args.refine_dim},
        ),
        training=replace(
            defaults.training,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
        ),
        evaluation=replace(
            defaults.evaluation, samples=args.monitor_frames, seed=args.seed + 1_000
        ),
        checkpoints=replace(defaults.checkpoints, output_path=args.output),
        every=args.every,
        weight_power=args.weight_power,
        workers=args.workers,
    )
    train(config)


if __name__ == "__main__":
    main()
