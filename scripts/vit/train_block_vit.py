"""Train the block-SMB Vision Transformer on live procedural pygame frames.

The environment's deterministic palette provides exact semantic masks and
Mario positions, so no external annotation dataset is required.

Example:
    python scripts/vit/train_block_vit.py --epochs 20 --samples-per-epoch 2048
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retroagi.stages.block_smb import BlockVisionTransformer, MarioScenarioEnv


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "block_vit" / "block_vit.pth"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    samples_per_epoch: int = 2048
    val_samples: int = 512
    rollout_steps: int = 32
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    position_weight: float = 2.0
    dim: int = 64
    depth: int = 2
    heads: int = 4
    patch_size: int = 16
    dropout: float = 0.1
    seed: int = 7


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create exact patch labels and normalized positions for collected frames."""
    labels = []
    positions = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size].to(model.pos_embed.device)
        labels.append(model.patch_targets(batch).cpu())
        positions.append(model.position_targets(batch).cpu())
    return torch.cat(labels), torch.cat(positions)


def make_loader(
    frames: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    images = frames.permute(0, 3, 1, 2).float().div_(255.0)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels, positions),
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
    weights: torch.Tensor,
    position_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(images)
    semantic_loss = F.cross_entropy(output.semantic_logits, labels, weight=weights)
    position_loss = F.mse_loss(output.position, positions)
    total = semantic_loss + position_weight * position_loss
    return total, semantic_loss, position_loss


@torch.no_grad()
def evaluate(
    model: BlockVisionTransformer,
    loader: DataLoader,
    weights: torch.Tensor,
    position_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = semantic_loss = position_loss = 0.0
    correct = foreground_correct = foreground_total = samples = patches = 0
    intersections = torch.zeros(model.spec.num_classes, dtype=torch.float64)
    unions = torch.zeros(model.spec.num_classes, dtype=torch.float64)

    for images, labels, positions in loader:
        images, labels, positions = images.to(device), labels.to(device), positions.to(device)
        output = model(images)
        semantic = F.cross_entropy(output.semantic_logits, labels, weight=weights)
        position = F.mse_loss(output.position, positions)
        total = semantic + position_weight * position
        batch_size = images.shape[0]
        total_loss += total.item() * batch_size
        semantic_loss += semantic.item() * batch_size
        position_loss += position.item() * batch_size
        samples += batch_size

        prediction = output.semantic_ids
        correct += (prediction == labels).sum().item()
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
        "accuracy": correct / patches,
        "foreground_accuracy": foreground_correct / max(foreground_total, 1),
        "mean_iou": mean_iou,
    }


def save_checkpoint(
    path: Path,
    model: BlockVisionTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(config),
            "vision_spec": asdict(model.spec),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"epoch": epoch, "metrics": metrics, "config": asdict(config)}, indent=2) + "\n"
    )


def train(
    config: TrainConfig,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "auto",
    resume: Optional[Path] = None,
) -> dict[str, float]:
    seed_everything(config.seed)
    device = select_device(device_name)
    model = BlockVisionTransformer(
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        patch_size=config.patch_size,
        drop=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Generating fixed validation rollout...")
    val_frames = collect_procedural_frames(
        config.val_samples, config.seed + 1_000_000, config.rollout_steps, show_progress=True
    )
    val_labels, val_positions = build_ground_truth(model, val_frames)
    val_loader = make_loader(
        val_frames, val_labels, val_positions, config.batch_size, False, config.seed
    )

    best_iou = -1.0
    final_metrics = {}
    for epoch in range(start_epoch, config.epochs):
        train_frames = collect_procedural_frames(
            config.samples_per_epoch,
            config.seed + epoch * 10_000,
            config.rollout_steps,
            show_progress=True,
        )
        train_labels, train_positions = build_ground_truth(model, train_frames)
        train_loader = make_loader(
            train_frames, train_labels, train_positions, config.batch_size, True, config.seed + epoch
        )
        weights = class_weights(train_labels, model.spec.num_classes, device)

        model.train()
        running = 0.0
        for images, labels, positions in train_loader:
            images, labels, positions = images.to(device), labels.to(device), positions.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = compute_loss(
                model, images, labels, positions, weights, config.position_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * images.shape[0]

        final_metrics = evaluate(model, val_loader, weights, config.position_weight, device)
        train_loss = running / len(train_loader.dataset)
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs:03d} "
            f"train={train_loss:.4f} val={final_metrics['loss']:.4f} "
            f"fg_acc={final_metrics['foreground_accuracy'] * 100:5.1f}% "
            f"mIoU={final_metrics['mean_iou'] * 100:5.1f}% "
            f"pos_mse={final_metrics['position_loss']:.5f}"
        )
        if final_metrics["mean_iou"] >= best_iou:
            best_iou = final_metrics["mean_iou"]
            save_checkpoint(output, model, optimizer, epoch, final_metrics, config)
            print(f"Saved checkpoint: {output}")

    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--samples-per-epoch", type=int, default=TrainConfig.samples_per_epoch)
    parser.add_argument("--val-samples", type=int, default=TrainConfig.val_samples)
    parser.add_argument("--rollout-steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--position-weight", type=float, default=TrainConfig.position_weight)
    parser.add_argument("--dim", type=int, default=TrainConfig.dim)
    parser.add_argument("--depth", type=int, default=TrainConfig.depth)
    parser.add_argument("--heads", type=int, default=TrainConfig.heads)
    parser.add_argument("--patch-size", type=int, default=TrainConfig.patch_size)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        position_weight=args.position_weight,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        patch_size=args.patch_size,
        dropout=args.dropout,
        seed=args.seed,
    )
    train(config, output=args.output, device_name=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
