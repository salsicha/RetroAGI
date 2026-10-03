"""Train the Block SMB vision transformer to report the objects drawn on screen.

Frames are what the policy sees: all Monte Carlo families on the train split
at every difficulty, played with teacher, perturbed, delayed and random
routes, plus procedurally generated levels, some with drawn power-ups
(retroagi/stages/block_smb/vision_frames.py). Worker processes play fresh
episodes throughout training, so frames rarely repeat. Labels are the
simulator's own drawing (MarioScenarioEnv.scene_labels()), exact by
construction. The model and the training loop are the shared ones
(retroagi.core.vision.SceneVisionTransformer, retroagi.core.scene_vision):
pixel types, objects, enemy kinds and Mario's facing and support, trained
together. Progress is monitored on held-out train-split layouts; the
validation split is kept for scripts/vision/evaluate_block_vision.py.

Example:
    python scripts/vit/train_block_vit.py --epochs 40 --samples-per-epoch 40000
"""

import argparse
import random
import sys
import time
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

from retroagi.core.config import (
    CheckpointConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from retroagi.core.devices import select_device
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
from retroagi.stages.block_smb.vision import DEFAULT_BLOCK_VIT_CHECKPOINT, BlockVisionTransformer
from retroagi.stages.block_smb.vision_frames import family_layouts, frame_stream, layout_frames

DEFAULT_OUTPUT = PROJECT_ROOT / DEFAULT_BLOCK_VIT_CHECKPOINT
DEFAULT_SEED = 7


@dataclass(frozen=True)
class TrainConfig:
    environment: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            stage="block_smb", seed=DEFAULT_SEED, rollout_steps=320
        )
    )
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="block_smb_vit",
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
    # Train layouts per family and difficulty; held-out monitor layouts likewise.
    layout_repeats: int = 12
    monitor_repeats: int = 2
    # Share of an episode's frames kept, and share of episodes on generated levels.
    keep: float = 0.25
    generated_share: float = 0.15
    workers: int = 10
    warmup_steps: int = 500
    weight_power: float = 0.5

    def __post_init__(self) -> None:
        if self.training.samples_per_epoch is None or self.evaluation.samples is None:
            raise ValueError("samples_per_epoch and evaluation samples must be set")
        if min(self.layout_repeats, self.monitor_repeats) <= 0:
            raise ValueError("layout repeats must be positive")
        if not 0 < self.keep <= 1 or not 0 <= self.generated_share < 1:
            raise ValueError("keep must be in (0, 1] and generated_share in [0, 1)")

    def to_dict(self) -> dict:
        return ExperimentConfig(
            environment=self.environment,
            model=self.model,
            training=self.training,
            evaluation=self.evaluation,
            checkpoints=self.checkpoints,
            name="block_vit_training",
            metadata={
                "layout_repeats": self.layout_repeats,
                "monitor_repeats": self.monitor_repeats,
                "keep": self.keep,
                "generated_share": self.generated_share,
                "warmup_steps": self.warmup_steps,
                "weight_power": self.weight_power,
            },
        ).to_dict()


def build_model(config: TrainConfig) -> BlockVisionTransformer:
    model = config.model
    return BlockVisionTransformer(
        dim=model.hidden_dim,
        depth=model.depth,
        heads=model.heads,
        patch_size=model.patch_size or 16,
        drop=model.dropout,
        refine_dim=int(model.metadata["refine_dim"]),
    )


class FrameStreamDataset(IterableDataset):
    """Each worker plays its own seeded episodes and shuffles them in a buffer."""

    def __init__(self, layouts, seed, *, keep, generated_share, buffer=1024):
        self.layouts, self.seed = layouts, seed
        self.keep, self.generated_share, self.buffer = keep, generated_share, buffer

    def __iter__(self):
        worker = get_worker_info()
        seed = self.seed * 1_000_003 + (worker.id if worker is not None else 0)
        rng = random.Random(seed)
        frames = frame_stream(
            self.layouts, seed, keep=self.keep, generated_share=self.generated_share
        )
        pool = list(islice(frames, self.buffer))
        for frame in frames:
            index = rng.randrange(len(pool))
            out, pool[index] = pool[index], frame
            yield torch.from_numpy(out.image.copy()), frame_targets(out.scene)


def monitor_frames(layouts, seed: int, count: int) -> tuple[np.ndarray, dict]:
    """A fixed held-out set: teacher and perturbed routes on unseen layouts."""
    rng = random.Random(seed)
    images, targets = [], []
    for layout in layouts:
        for route in ("teacher", "perturbed"):
            for frame in layout_frames(layout, route, rng, keep=0.08):
                images.append(frame.image.copy())
                targets.append(frame_targets(frame.scene))
    order = rng.sample(range(len(images)), min(count, len(images)))
    return np.stack([images[i] for i in order]), stack_targets([targets[i] for i in order])


def save_checkpoint(path, model, epoch: int, metrics: dict, config: TrainConfig) -> None:
    save_scene_vision_checkpoint(
        path,
        model,
        stage=config.environment.stage,
        metrics=metrics,
        config=config.to_dict(),
        epoch=epoch,
        trainer="scripts.vit.train_block_vit",
    )


def train(config: TrainConfig, device_name: Optional[str] = None) -> dict:
    seed = config.training.seed
    seeded(seed)
    device = select_device(device_name or config.training.device)
    output = Path(config.checkpoints.output_path or DEFAULT_OUTPUT)
    model = build_model(config)
    if config.model.name != model.spec.name:
        raise ValueError(f"model name {config.model.name!r} is not {model.spec.name!r}")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}; learned numbers: {parameters:,}", flush=True)

    started = time.time()
    with ProcessPoolExecutor(config.workers) as pool:
        train_layouts = family_layouts(
            "train", config.environment.seed, config.layout_repeats, executor=pool
        )
        held_out = family_layouts(
            "train", config.evaluation.seed, config.monitor_repeats, executor=pool
        )
    monitor_images, monitor_targets = monitor_frames(
        held_out, config.evaluation.seed, config.evaluation.samples
    )
    sample = frame_stream(
        train_layouts, seed + 999, keep=config.keep, generated_share=config.generated_share
    )
    weights = pixel_type_weights(
        np.stack([frame.scene.types for frame in islice(sample, 4_000)]),
        power=config.weight_power,
    )
    print(
        f"Layouts: {len(train_layouts)} train, {len(held_out)} held out; "
        f"{len(monitor_images)} monitor frames ({time.time() - started:.0f}s)\n"
        "Pixel weights: "
        + ", ".join(f"{name}={w:.2f}" for name, w in zip(PIXEL_TYPES, weights.tolist())),
        flush=True,
    )

    batch_size = config.training.batch_size
    steps_per_epoch = -(-config.training.samples_per_epoch // batch_size)
    loader = DataLoader(
        FrameStreamDataset(
            train_layouts, seed, keep=config.keep, generated_share=config.generated_share
        ),
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
    parser.add_argument("--layout-repeats", type=int, default=defaults.layout_repeats)
    parser.add_argument("--monitor-repeats", type=int, default=defaults.monitor_repeats)
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
        layout_repeats=args.layout_repeats,
        monitor_repeats=args.monitor_repeats,
        weight_power=args.weight_power,
        workers=args.workers,
    )
    train(config)


if __name__ == "__main__":
    main()
