"""Train the block-SMB Vision Transformer on live procedural pygame frames.

The environment's deterministic palette provides exact semantic masks and
Mario positions, so no external annotation dataset is required.

Example:
    python scripts/vit/train_block_vit.py --epochs 20 --samples-per-epoch 2048
"""

import argparse
import random
from dataclasses import asdict, dataclass, field
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
    build_checkpoint,
    is_versioned_checkpoint,
    save_checkpoint as save_versioned_checkpoint,
    select_device,
    validate_checkpoint_compatibility,
    validate_model_vision_compatibility,
    validate_stage_spec,
)
from retroagi.stages.block_smb import BLOCK_SMB_SPEC, BlockVisionTransformer, MarioScenarioEnv


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "block_vit" / "block_vit.pth"
DEFAULT_EPOCHS = 20
DEFAULT_SAMPLES_PER_EPOCH = 2048
DEFAULT_VAL_SAMPLES = 512
DEFAULT_ROLLOUT_STEPS = 32
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_POSITION_WEIGHT = 2.0
DEFAULT_SUPPORT_WEIGHT = 1.0
DEFAULT_DIM = 64
DEFAULT_DEPTH = 2
DEFAULT_HEADS = 4
DEFAULT_PATCH_SIZE = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_SEED = 7
DEFAULT_GRADIENT_CLIP_NORM = 1.0


@dataclass(frozen=True)
class TrainConfig:
    environment: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            stage="block_smb",
            seed=DEFAULT_SEED,
            rollout_steps=DEFAULT_ROLLOUT_STEPS,
        )
    )
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="block_smb_vit",
            hidden_dim=DEFAULT_DIM,
            depth=DEFAULT_DEPTH,
            heads=DEFAULT_HEADS,
            patch_size=DEFAULT_PATCH_SIZE,
            dropout=DEFAULT_DROPOUT,
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            epochs=DEFAULT_EPOCHS,
            samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
            batch_size=DEFAULT_BATCH_SIZE,
            learning_rate=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            seed=DEFAULT_SEED,
            gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM,
        )
    )
    evaluation: EvaluationConfig = field(
        default_factory=lambda: EvaluationConfig(
            samples=DEFAULT_VAL_SAMPLES,
            seed=DEFAULT_SEED + 1_000_000,
            metrics=(
                "loss",
                "semantic_loss",
                "position_loss",
                "support_loss",
                "accuracy",
                "foreground_accuracy",
                "mean_iou",
                "support_accuracy",
            ),
        )
    )
    checkpoints: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(
            output_path=DEFAULT_OUTPUT,
            best_metric="mean_iou",
            best_mode="max",
        )
    )
    position_weight: float = DEFAULT_POSITION_WEIGHT
    support_weight: float = DEFAULT_SUPPORT_WEIGHT

    def __post_init__(self) -> None:
        if self.position_weight <= 0:
            raise ValueError("position_weight must be positive")
        if self.support_weight <= 0:
            raise ValueError("support_weight must be positive")
        if self.training.samples_per_epoch is None:
            raise ValueError("training.samples_per_epoch must be set for Block ViT training")
        if self.evaluation.samples is None:
            raise ValueError("evaluation.samples must be set for Block ViT training")

    def to_dict(self) -> dict:
        experiment = ExperimentConfig(
            environment=self.environment,
            model=self.model,
            training=self.training,
            evaluation=self.evaluation,
            checkpoints=self.checkpoints,
            name="block_vit_training",
            metadata={
                "position_weight": self.position_weight,
                "support_weight": self.support_weight,
            },
        )
        return experiment.to_dict()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_action(rng: random.Random) -> int:
    # Forward motion and jumps expose scrolling scenery while retaining some
    # stationary and leftward examples.
    return rng.choices((0, 1, 2, 3, 4, 5), weights=(5, 25, 35, 2, 3, 10), k=1)[0]


def collect_procedural_frames(
    num_samples: int,
    seed: int,
    rollout_steps: int = 32,
    show_progress: bool = False,
) -> torch.Tensor:
    """Run procedural block-SMB episodes and return uint8 NHWC frames."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")

    rng = random.Random(seed)
    env = MarioScenarioEnv()
    frames = []
    scenario_index = 0
    try:
        while len(frames) < num_samples:
            scenario_seed = seed + scenario_index
            scenario = MarioScenarioEnv.generate_scenario(
                num_screens=rng.randint(1, 3),
                enemy_density=rng.uniform(0.25, 0.9),
                moving_platform_chance=rng.uniform(0.1, 0.5),
                seed=scenario_seed,
            )
            observation, _ = env.reset(scenario=scenario, seed=scenario_seed)
            frames.append(observation.copy())

            for _ in range(rollout_steps - 1):
                if len(frames) >= num_samples:
                    break
                observation, _, terminated, truncated, _ = env.step(_sample_action(rng))
                frames.append(observation.copy())
                if terminated or truncated:
                    break

            scenario_index += 1
            if show_progress and (len(frames) == num_samples or scenario_index % 10 == 0):
                print(f"Collected {len(frames):5d}/{num_samples} frames", flush=True)
    finally:
        env.close()

    return torch.from_numpy(np.stack(frames[:num_samples])).to(torch.uint8)


@torch.no_grad()
def build_ground_truth(
    model: BlockVisionTransformer,
    frames: torch.Tensor,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create exact patch labels, support labels, and normalized positions."""
    labels = []
    positions = []
    supports = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size].to(model.pos_embed.device)
        batch_labels = model.patch_targets(batch)
        labels.append(batch_labels.cpu())
        positions.append(model.position_targets(batch).cpu())
        support_targets = model.support_targets_from_labels(batch_labels)
        if support_targets is None:
            raise ValueError("could not infer Block ViT support targets")
        supports.append(support_targets.cpu())
    return torch.cat(labels), torch.cat(positions), torch.cat(supports)


def make_loader(
    frames: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    supports: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    images = frames.permute(0, 3, 1, 2).float().div_(255.0)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels, positions, supports),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def class_weights(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels.flatten(), minlength=num_classes).float()
    weights = counts.sum().clamp_min(1) / counts.clamp_min(1)
    weights = weights.sqrt()
    return (weights / weights.mean()).to(device)


def compute_loss(
    model: BlockVisionTransformer,
    images: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    supports: torch.Tensor,
    weights: torch.Tensor,
    position_weight: float,
    support_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(images)
    semantic_loss = F.cross_entropy(output.semantic_logits, labels, weight=weights)
    position_loss = F.mse_loss(output.position, positions)
    support_loss = F.cross_entropy(output.support_logits, supports)
    total = semantic_loss + position_weight * position_loss + support_weight * support_loss
    return total, semantic_loss, position_loss, support_loss


@torch.no_grad()
def evaluate(
    model: BlockVisionTransformer,
    loader: DataLoader,
    weights: torch.Tensor,
    position_weight: float,
    support_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = semantic_loss = position_loss = support_loss = 0.0
    correct = foreground_correct = foreground_total = samples = patches = 0
    support_correct = support_total = 0
    intersections = torch.zeros(model.spec.num_classes, dtype=torch.float64)
    unions = torch.zeros(model.spec.num_classes, dtype=torch.float64)

    for images, labels, positions, supports in loader:
        images = images.to(device)
        labels = labels.to(device)
        positions = positions.to(device)
        supports = supports.to(device)
        output = model(images)
        semantic = F.cross_entropy(output.semantic_logits, labels, weight=weights)
        position = F.mse_loss(output.position, positions)
        support = F.cross_entropy(output.support_logits, supports)
        total = semantic + position_weight * position + support_weight * support
        batch_size = images.shape[0]
        total_loss += total.item() * batch_size
        semantic_loss += semantic.item() * batch_size
        position_loss += position.item() * batch_size
        support_loss += support.item() * batch_size
        samples += batch_size

        prediction = output.semantic_ids
        support_prediction = output.support_ids
        correct += (prediction == labels).sum().item()
        support_correct += (support_prediction == supports).sum().item()
        support_total += supports.numel()
        patches += labels.numel()
        foreground = labels != 0
        foreground_correct += (prediction[foreground] == labels[foreground]).sum().item()
        foreground_total += foreground.sum().item()
        for class_id in range(model.spec.num_classes):
            predicted = prediction == class_id
            target = labels == class_id
            intersections[class_id] += (predicted & target).sum().cpu()
            unions[class_id] += (predicted | target).sum().cpu()

    valid = unions > 0
    mean_iou = (intersections[valid] / unions[valid]).mean().item() if valid.any() else 0.0
    return {
        "loss": total_loss / samples,
        "semantic_loss": semantic_loss / samples,
        "position_loss": position_loss / samples,
        "support_loss": support_loss / samples,
        "accuracy": correct / patches,
        "foreground_accuracy": foreground_correct / max(foreground_total, 1),
        "mean_iou": mean_iou,
        "support_accuracy": support_correct / max(support_total, 1),
    }


def save_checkpoint(
    path: Path,
    model: BlockVisionTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: TrainConfig,
) -> None:
    config_data = config.to_dict()
    checkpoint = build_checkpoint(
        stage=config.environment.stage,
        model_name=config.model.name,
        checkpoint_kind="vision_encoder",
        epoch=epoch,
        metrics=metrics,
        config=config_data,
        specs={"vision": asdict(model.spec)},
        states={
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        metadata={"trainer": "scripts.vit.train_block_vit"},
    )
    save_versioned_checkpoint(path, checkpoint)


def train(
    config: TrainConfig,
    output: Optional[Path] = None,
    device_name: Optional[str] = None,
    resume: Optional[Path] = None,
) -> dict[str, float]:
    seed_everything(config.training.seed)
    device = select_device(device_name or config.training.device)
    output = output or config.checkpoints.output_path or DEFAULT_OUTPUT
    resume = resume if resume is not None else config.checkpoints.resume_path
    validate_stage_spec(BLOCK_SMB_SPEC, context="Block ViT startup stage")
    if config.environment.stage != BLOCK_SMB_SPEC.name:
        raise ValueError(
            f"environment stage {config.environment.stage!r} does not match "
            f"{BLOCK_SMB_SPEC.name!r}"
        )

    model = BlockVisionTransformer(
        dim=config.model.hidden_dim,
        depth=config.model.depth,
        heads=config.model.heads,
        patch_size=config.model.patch_size or DEFAULT_PATCH_SIZE,
        drop=config.model.dropout,
    ).to(device)
    validate_model_vision_compatibility(
        config.model, model.spec, context="Block ViT startup model"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if is_versioned_checkpoint(checkpoint):
            checkpoint = validate_checkpoint_compatibility(
                checkpoint,
                stage=BLOCK_SMB_SPEC,
                model=config.model,
                vision=model.spec,
                checkpoint_kind="vision_encoder",
                required_states=("model", "optimizer"),
                context=f"resume checkpoint {resume}",
            )
            states = checkpoint["states"]
            model.load_compatible_state_dict(states["model"])
            optimizer.load_state_dict(states["optimizer"])
        else:
            model.load_compatible_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Generating fixed validation rollout...")
    val_frames = collect_procedural_frames(
        config.evaluation.samples,
        config.evaluation.seed,
        config.environment.rollout_steps,
        show_progress=True,
    )
    val_labels, val_positions, val_supports = build_ground_truth(model, val_frames)
    val_loader = make_loader(
        val_frames,
        val_labels,
        val_positions,
        val_supports,
        config.training.batch_size,
        False,
        config.evaluation.seed,
    )

    best_iou = -1.0
    final_metrics = {}
    for epoch in range(start_epoch, config.training.epochs):
        train_frames = collect_procedural_frames(
            config.training.samples_per_epoch,
            config.environment.seed + epoch * 10_000,
            config.environment.rollout_steps,
            show_progress=True,
        )
        train_labels, train_positions, train_supports = build_ground_truth(model, train_frames)
        train_loader = make_loader(
            train_frames,
            train_labels,
            train_positions,
            train_supports,
            config.training.batch_size,
            True,
            config.training.seed + epoch,
        )
        weights = class_weights(train_labels, model.spec.num_classes, device)

        model.train()
        running = 0.0
        for images, labels, positions, supports in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            positions = positions.to(device)
            supports = supports.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = compute_loss(
                model,
                images,
                labels,
                positions,
                supports,
                weights,
                config.position_weight,
                config.support_weight,
            )
            loss.backward()
            if config.training.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            optimizer.step()
            running += loss.item() * images.shape[0]

        final_metrics = evaluate(
            model,
            val_loader,
            weights,
            config.position_weight,
            config.support_weight,
            device,
        )
        train_loss = running / len(train_loader.dataset)
        print(
            f"Epoch {epoch + 1:03d}/{config.training.epochs:03d} "
            f"train={train_loss:.4f} val={final_metrics['loss']:.4f} "
            f"fg_acc={final_metrics['foreground_accuracy'] * 100:5.1f}% "
            f"mIoU={final_metrics['mean_iou'] * 100:5.1f}% "
            f"pos_mse={final_metrics['position_loss']:.5f} "
            f"support_acc={final_metrics['support_accuracy'] * 100:5.1f}%"
        )
        if final_metrics["mean_iou"] >= best_iou:
            best_iou = final_metrics["mean_iou"]
            save_checkpoint(output, model, optimizer, epoch, final_metrics, config)
            print(f"Saved checkpoint: {output}")

    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_PER_EPOCH)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--position-weight", type=float, default=DEFAULT_POSITION_WEIGHT)
    parser.add_argument("--support-weight", type=float, default=DEFAULT_SUPPORT_WEIGHT)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        environment=EnvironmentConfig(
            stage="block_smb",
            seed=args.seed,
            rollout_steps=args.rollout_steps,
        ),
        model=ModelConfig(
            name="block_smb_vit",
            hidden_dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            patch_size=args.patch_size,
            dropout=args.dropout,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            gradient_clip_norm=DEFAULT_GRADIENT_CLIP_NORM,
        ),
        evaluation=EvaluationConfig(
            samples=args.val_samples,
            seed=args.seed + 1_000_000,
            metrics=(
                "loss",
                "semantic_loss",
                "position_loss",
                "support_loss",
                "accuracy",
                "foreground_accuracy",
                "mean_iou",
                "support_accuracy",
            ),
        ),
        checkpoints=CheckpointConfig(
            output_path=args.output,
            resume_path=args.resume,
            best_metric="mean_iou",
            best_mode="max",
        ),
        position_weight=args.position_weight,
        support_weight=args.support_weight,
    )
    train(config)


if __name__ == "__main__":
    main()
