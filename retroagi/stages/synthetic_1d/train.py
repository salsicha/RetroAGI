import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from retroagi.core import (
    AgentWorldModelCritic,
    StageSpec,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint as save_versioned_checkpoint,
    select_device,
    to_plain_data,
)

SYNTHETIC_1D_SPEC = StageSpec(
    name="synthetic_1d",
    observation_kind="procedural one-dimensional sequences",
    action_kind="continuous controller targets",
    seq_len_a=8,
    ratio_ab=2,
    ratio_bc=4,
    vocab_size=20,
)

SyntheticDataset = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
SyntheticLosses = dict[str, torch.Tensor]
SyntheticMetrics = dict[str, float]
SyntheticHistory = dict[str, list[float]]

SYNTHETIC_LOSS_KEYS = (
    "loss_actor_pass1",
    "loss_actor_pass2",
    "loss_world_model",
    "loss_critic",
    "loss_total",
)
SYNTHETIC_METRIC_KEYS = (
    "controller_mse",
    "controller_mae",
    "controller_rmse",
    "error_B",
    "accuracy_A",
)
SYNTHETIC_MODEL_NAME = "synthetic_1d_actor_world_model_critic"
SYNTHETIC_CHECKPOINT_KIND = "stage1_trainer"


@dataclass(frozen=True)
class SyntheticSplitSeeds:
    train: int = 10_001
    validation: int = 20_001
    test: int = 30_001

    def __post_init__(self) -> None:
        if len({self.train, self.validation, self.test}) != 3:
            raise ValueError("train, validation, and test seeds must be distinct")


@dataclass(frozen=True)
class SyntheticSplitSizes:
    train: int = 1_000
    validation: int = 200
    test: int = 200

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} split size must be positive")


@dataclass(frozen=True)
class SyntheticDatasetSplits:
    train: SyntheticDataset
    validation: SyntheticDataset
    test: SyntheticDataset
    seeds: SyntheticSplitSeeds
    sizes: SyntheticSplitSizes


@dataclass(frozen=True)
class SyntheticBaselinePredictions:
    actions_c: torch.Tensor
    logits_a: torch.Tensor
    w_b: torch.Tensor
    b_b: torch.Tensor


@dataclass(frozen=True)
class SyntheticTrainingConfig:
    seed: int = 0
    split_sizes: SyntheticSplitSizes = field(default_factory=SyntheticSplitSizes)
    split_seeds: Optional[SyntheticSplitSeeds] = None
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-3
    critic_loss_weight: float = 0.0
    tau_start: float = 5.0
    tau_end: float = 0.1
    device: str = "auto"
    deterministic: bool = True
    checkpoint_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    save_checkpoints: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.critic_loss_weight < 0:
            raise ValueError("critic_loss_weight must be non-negative")
        if self.tau_start <= 0:
            raise ValueError("tau_start must be positive")
        if self.tau_end <= 0:
            raise ValueError("tau_end must be positive")
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.save_checkpoints and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required when save_checkpoints is true")

    @property
    def resolved_split_seeds(self) -> SyntheticSplitSeeds:
        if self.split_seeds is not None:
            return self.split_seeds
        return SyntheticSplitSeeds(
            train=self.seed + 10_001,
            validation=self.seed + 20_001,
            test=self.seed + 30_001,
        )

    def train_permutation_seed(self, epoch: int) -> int:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        return self.seed + 1_000_003 + epoch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch RNGs, including available accelerators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def make_train_permutation(
    num_samples: int, config: SyntheticTrainingConfig, epoch: int
) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    generator = torch.Generator()
    generator.manual_seed(config.train_permutation_seed(epoch))
    return torch.randperm(num_samples, generator=generator)


def make_empty_history() -> SyntheticHistory:
    return {key: [] for key in (*SYNTHETIC_LOSS_KEYS, *SYNTHETIC_METRIC_KEYS)}


def compute_synthetic_training_losses(
    actions1: torch.Tensor,
    next_state_pred: torch.Tensor,
    criticism: torch.Tensor,
    actions2: torch.Tensor,
    batch_xc_in: torch.Tensor,
    batch_yc_target: torch.Tensor,
    criterion: nn.Module,
    critic_loss_weight: float = 0.0,
) -> SyntheticLosses:
    if critic_loss_weight < 0:
        raise ValueError("critic_loss_weight must be non-negative")

    true_next_state = batch_xc_in + actions1.detach()
    loss_world_model = criterion(next_state_pred, true_next_state)
    loss_actor_pass1 = criterion(actions1, batch_yc_target)
    loss_actor_pass2 = criterion(actions2, batch_yc_target)
    loss_critic = criticism.pow(2).mean()
    loss_total = (
        loss_actor_pass1
        + loss_actor_pass2
        + loss_world_model
        + critic_loss_weight * loss_critic
    )
    return {
        "loss_actor_pass1": loss_actor_pass1,
        "loss_actor_pass2": loss_actor_pass2,
        "loss_world_model": loss_world_model,
        "loss_critic": loss_critic,
        "loss_total": loss_total,
    }


def append_epoch_history(
    history: SyntheticHistory,
    losses: dict[str, float],
    metrics: dict[str, float],
) -> None:
    for key in SYNTHETIC_LOSS_KEYS:
        history[key].append(losses[key])
    for key in SYNTHETIC_METRIC_KEYS:
        history[key].append(metrics[key])


def compute_synthetic_evaluation_metrics(
    predictions: SyntheticBaselinePredictions,
    dataset: SyntheticDataset,
) -> SyntheticMetrics:
    _xa, ya, _xb, yb, _xc, yc = dataset
    if predictions.actions_c.shape != yc.shape:
        raise ValueError("actions_c shape must match target C shape")
    if predictions.logits_a.shape != (*ya.shape, SYNTHETIC_1D_SPEC.vocab_size):
        raise ValueError("logits_a shape must be (batch, seq_len_a, vocab_size)")
    if predictions.w_b.shape != yb.shape or predictions.b_b.shape != yb.shape:
        raise ValueError("w_b and b_b shapes must match target B shape")

    controller_mse = F.mse_loss(predictions.actions_c, yc).item()
    controller_mae = F.l1_loss(predictions.actions_c, yc).item()
    predicted_a = predictions.logits_a.argmax(dim=-1)
    accuracy_a = (predicted_a == ya).float().mean().item() * 100.0

    w_true = torch.sin(yb.float())
    b_true = torch.cos(yb.float())
    error_b = ((F.mse_loss(predictions.w_b, w_true) + F.mse_loss(predictions.b_b, b_true)) / 2).item()

    return {
        "controller_mse": controller_mse,
        "controller_mae": controller_mae,
        "controller_rmse": controller_mse ** 0.5,
        "error_B": error_b,
        "accuracy_A": accuracy_a,
    }


class RandomSyntheticBaseline:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def predict(
        self, dataset: SyntheticDataset, spec: StageSpec = SYNTHETIC_1D_SPEC
    ) -> SyntheticBaselinePredictions:
        _xa, ya, _xb, yb, _xc, yc = dataset
        generator = torch.Generator().manual_seed(self.seed)
        return SyntheticBaselinePredictions(
            actions_c=torch.randn(yc.shape, generator=generator, dtype=yc.dtype),
            logits_a=torch.randn(
                (*ya.shape, spec.vocab_size), generator=generator, dtype=torch.float32
            ),
            w_b=torch.randn(yb.shape, generator=generator, dtype=torch.float32),
            b_b=torch.randn(yb.shape, generator=generator, dtype=torch.float32),
        )


@dataclass(frozen=True)
class MeanSyntheticBaseline:
    action_mean: torch.Tensor
    a_mode: torch.Tensor
    w_mean: torch.Tensor
    b_mean: torch.Tensor

    @classmethod
    def fit(cls, dataset: SyntheticDataset) -> "MeanSyntheticBaseline":
        _xa, ya, _xb, yb, _xc, yc = dataset
        return cls(
            action_mean=yc.mean(dim=0),
            a_mode=torch.mode(ya, dim=0).values,
            w_mean=torch.sin(yb.float()).mean(dim=0),
            b_mean=torch.cos(yb.float()).mean(dim=0),
        )

    def predict(
        self, dataset: SyntheticDataset, spec: StageSpec = SYNTHETIC_1D_SPEC
    ) -> SyntheticBaselinePredictions:
        _xa, ya, _xb, yb, _xc, yc = dataset
        batch_size = ya.size(0)
        logits_a = torch.zeros((batch_size, ya.size(1), spec.vocab_size), dtype=torch.float32)
        token_indices = self.a_mode.view(1, -1, 1).expand(batch_size, -1, -1)
        logits_a.scatter_(dim=-1, index=token_indices, value=1.0)
        return SyntheticBaselinePredictions(
            actions_c=self.action_mean.view(1, -1).expand_as(yc).clone(),
            logits_a=logits_a,
            w_b=self.w_mean.view(1, -1).expand_as(yb).clone(),
            b_b=self.b_mean.view(1, -1).expand_as(yb).clone(),
        )


def evaluate_synthetic_baselines(
    train_dataset: SyntheticDataset,
    eval_dataset: SyntheticDataset,
    seed: int = 0,
    spec: StageSpec = SYNTHETIC_1D_SPEC,
) -> dict[str, SyntheticMetrics]:
    baselines = {
        "random": RandomSyntheticBaseline(seed=seed),
        "simple": MeanSyntheticBaseline.fit(train_dataset),
    }
    return {
        name: compute_synthetic_evaluation_metrics(model.predict(eval_dataset, spec), eval_dataset)
        for name, model in baselines.items()
    }


def synthetic_spec_metadata(spec: StageSpec = SYNTHETIC_1D_SPEC) -> dict[str, Any]:
    return {
        "stage": {
            "name": spec.name,
            "seq_len_a": spec.seq_len_a,
            "seq_len_b": spec.seq_len_b,
            "seq_len_c": spec.seq_len_c,
            "ratio_ab": spec.ratio_ab,
            "ratio_bc": spec.ratio_bc,
            "vocab_size": spec.vocab_size,
        }
    }


def save_synthetic_checkpoint(
    path: Path,
    model: AgentWorldModelCritic,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    metrics: Optional[Mapping[str, float]] = None,
    config: Optional[SyntheticTrainingConfig] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    checkpoint = build_checkpoint(
        stage=SYNTHETIC_1D_SPEC.name,
        model_name=SYNTHETIC_MODEL_NAME,
        checkpoint_kind=SYNTHETIC_CHECKPOINT_KIND,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics or {},
        config=to_plain_data(config) if config is not None else {},
        specs=synthetic_spec_metadata(),
        states={
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        },
        metadata=metadata or {},
    )
    save_versioned_checkpoint(path, checkpoint)
    return checkpoint


def restore_synthetic_checkpoint(
    path: Path,
    model: AgentWorldModelCritic,
    optimizer: Optional[optim.Optimizer] = None,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint["stage"] != SYNTHETIC_1D_SPEC.name:
        raise ValueError(
            f"checkpoint stage {checkpoint['stage']!r} does not match {SYNTHETIC_1D_SPEC.name!r}"
        )
    if checkpoint["model_name"] != SYNTHETIC_MODEL_NAME:
        raise ValueError(
            f"checkpoint model {checkpoint['model_name']!r} does not match {SYNTHETIC_MODEL_NAME!r}"
        )
    if checkpoint["checkpoint_kind"] != SYNTHETIC_CHECKPOINT_KIND:
        raise ValueError(
            f"checkpoint kind {checkpoint['checkpoint_kind']!r} does not match "
            f"{SYNTHETIC_CHECKPOINT_KIND!r}"
        )

    states = checkpoint["states"]
    if "torch_rng" in states:
        torch.set_rng_state(states["torch_rng"].cpu())
    if "python_rng" in states:
        random.setstate(states["python_rng"])
    if "numpy_rng" in states:
        np.random.set_state(states["numpy_rng"])
    model.load_state_dict(states["model"])
    if optimizer is not None:
        if "optimizer" not in states:
            raise ValueError("checkpoint is missing optimizer state")
        optimizer.load_state_dict(states["optimizer"])
    return checkpoint

def train_synthetic_epoch(
    model: AgentWorldModelCritic,
    optimizer: optim.Optimizer,
    train_dataset: SyntheticDataset,
    config: SyntheticTrainingConfig,
    epoch: int,
    criterion: nn.Module,
    *,
    device: torch.device,
) -> dict[str, float]:
    train_xa, train_ya, train_xb, train_yb, train_xc, train_yc = train_dataset
    del train_ya, train_yb
    model.train()
    permutation = make_train_permutation(train_xa.size(0), config, epoch).to(device)
    epoch_losses = {key: 0.0 for key in SYNTHETIC_LOSS_KEYS}
    batch_count = 0
    tau = max(
        config.tau_end,
        config.tau_start
        - (config.tau_start - config.tau_end) * (epoch / max(1, config.epochs - 1)),
    )

    for start in range(0, train_xa.size(0), config.batch_size):
        indices = permutation[start : start + config.batch_size]
        batch_xa = train_xa[indices]
        batch_xb = train_xb[indices]
        batch_xc_in = train_xc[indices]
        batch_yc_target = train_yc[indices]

        optimizer.zero_grad()
        actions1, next_state_pred, criticism, actions2, _logits_a, _w_b, _b_b = model(
            batch_xa, batch_xb, batch_xc_in, tau=tau
        )
        losses = compute_synthetic_training_losses(
            actions1,
            next_state_pred,
            criticism,
            actions2,
            batch_xc_in,
            batch_yc_target,
            criterion,
            critic_loss_weight=config.critic_loss_weight,
        )
        losses["loss_total"].backward()
        optimizer.step()

        for key in SYNTHETIC_LOSS_KEYS:
            epoch_losses[key] += losses[key].item()
        batch_count += 1

    return {key: value / batch_count for key, value in epoch_losses.items()}


def evaluate_synthetic_model(
    model: AgentWorldModelCritic,
    dataset: SyntheticDataset,
    *,
    tau: float,
) -> SyntheticMetrics:
    eval_xa, eval_ya, eval_xb, eval_yb, eval_xc, eval_yc = dataset
    model.eval()
    with torch.no_grad():
        _actions1, _next_state, _criticism, actions2, logits_a, w_b, b_b = model(
            eval_xa, eval_xb, eval_xc, tau=tau
        )
    return compute_synthetic_evaluation_metrics(
        SyntheticBaselinePredictions(
            actions_c=actions2,
            logits_a=logits_a,
            w_b=w_b,
            b_b=b_b,
        ),
        (eval_xa, eval_ya, eval_xb, eval_yb, eval_xc, eval_yc),
    )

# --------------------------------------------------------
# 1. Synthetic Data Generator
# --------------------------------------------------------
def generate_hierarchical_data(
    num_samples, seq_len_a, ratio_ab, ratio_bc, vocab_size, seed: Optional[int] = None
):
    """
    Generates three levels of data:
    Seq A: Slow discrete sequence.
    Seq B: Medium discrete sequence.
    Seq C: Fast continuous sequence (controlled system).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)

    seq_len_b = seq_len_a * ratio_ab

    X_A, Y_A = [], []
    X_B, Y_B = [], []
    X_C_in, Y_C_target = [], []
    
    for _ in range(num_samples):
        # Sequence A: standard progression
        start_a = rng.integers(0, vocab_size)
        seq_a = [(start_a + i) % vocab_size for i in range(seq_len_a + 1)]
        X_A.append(seq_a[:-1])
        Y_A.append(seq_a[1:])
        
        # Sequence B: faster timescale progression
        start_b = rng.integers(0, vocab_size)
        seq_b = [(start_b + i) % vocab_size for i in range(seq_len_b + 1)]
        X_B.append(seq_b[:-1])
        
        # Target B (implicit, not directly trained, used to generate C's parameters)
        y_b = []
        c_in_seq = []
        c_target_seq = []
        for j in range(seq_len_b):
            i = j // ratio_ab # Index of latest A token
            # Combine A and B to get a target concept
            concept = (seq_b[j+1] + seq_a[i+1]) % vocab_size
            y_b.append(concept)
            
            # The concept defines the true parameters for the controller
            w_true = np.sin(concept)
            b_true = np.cos(concept)
            
            # Sequence C: Generate fast inputs and targets for the controller
            for k in range(ratio_bc):
                x_val = rng.standard_normal()
                y_val = w_true * x_val + b_true
                c_in_seq.append(x_val)
                c_target_seq.append(y_val)
                
        Y_B.append(y_b)
        X_C_in.append(c_in_seq)
        Y_C_target.append(c_target_seq)
        
    return (torch.tensor(X_A, dtype=torch.long), torch.tensor(Y_A, dtype=torch.long),
            torch.tensor(X_B, dtype=torch.long), torch.tensor(Y_B, dtype=torch.long),
            torch.tensor(X_C_in, dtype=torch.float), torch.tensor(Y_C_target, dtype=torch.float))


def generate_dataset_splits(
    spec: StageSpec = SYNTHETIC_1D_SPEC,
    sizes: SyntheticSplitSizes = SyntheticSplitSizes(),
    seeds: SyntheticSplitSeeds = SyntheticSplitSeeds(),
) -> SyntheticDatasetSplits:
    """Create deterministic train, validation, and test splits from fixed seeds."""
    train = generate_hierarchical_data(
        sizes.train,
        spec.seq_len_a,
        spec.ratio_ab,
        spec.ratio_bc,
        spec.vocab_size,
        seed=seeds.train,
    )
    validation = generate_hierarchical_data(
        sizes.validation,
        spec.seq_len_a,
        spec.ratio_ab,
        spec.ratio_bc,
        spec.vocab_size,
        seed=seeds.validation,
    )
    test = generate_hierarchical_data(
        sizes.test,
        spec.seq_len_a,
        spec.ratio_ab,
        spec.ratio_bc,
        spec.vocab_size,
        seed=seeds.test,
    )
    return SyntheticDatasetSplits(
        train=train,
        validation=validation,
        test=test,
        seeds=seeds,
        sizes=sizes,
    )

# --------------------------------------------------------
# 2. Testing and Visualization System
# --------------------------------------------------------
def train_and_evaluate(config: Optional[SyntheticTrainingConfig] = None):
    config = config or SyntheticTrainingConfig()
    seed_everything(config.seed, deterministic=config.deterministic)

    # Hyperparameters
    vocab_size = SYNTHETIC_1D_SPEC.vocab_size
    seq_len_A = SYNTHETIC_1D_SPEC.seq_len_a
    ratio_AB = SYNTHETIC_1D_SPEC.ratio_ab
    ratio_BC = SYNTHETIC_1D_SPEC.ratio_bc
    batch_size = config.batch_size
    epochs = config.epochs
    tau_start = config.tau_start
    tau_end = config.tau_end

    device = select_device(config.device)

    print(f"Using device: {device}")
    
    # Create deterministic datasets
    splits = generate_dataset_splits(
        sizes=config.split_sizes,
        seeds=config.resolved_split_seeds,
    )
    train_XA, train_YA, train_XB, train_YB, train_XC, train_YC = splits.train
    val_XA, val_YA, val_XB, val_YB, val_XC, val_YC = splits.validation
    # Keep the test split fixed and untouched during training; later P2 metrics use it.
    _test_split = splits.test
    baseline_metrics = evaluate_synthetic_baselines(
        splits.train, splits.validation, seed=config.seed, spec=SYNTHETIC_1D_SPEC
    )
    print(
        "Validation baselines - "
        f"random C MSE: {baseline_metrics['random']['controller_mse']:.4f}, "
        f"simple C MSE: {baseline_metrics['simple']['controller_mse']:.4f}"
    )

    train_XA, train_YA = train_XA.to(device), train_YA.to(device)
    train_XB, train_YB = train_XB.to(device), train_YB.to(device)
    train_XC, train_YC = train_XC.to(device), train_YC.to(device)

    val_XA, val_YA = val_XA.to(device), val_YA.to(device)
    val_XB, val_YB = val_XB.to(device), val_YB.to(device)
    val_XC, val_YC = val_XC.to(device), val_YC.to(device)

    seq_len_C = seq_len_A * ratio_AB * ratio_BC
    
    # Initialize model, loss, optimizer
    model = AgentWorldModelCritic(
        vocab_size=vocab_size, seq_len_a=seq_len_A, seq_len_c=seq_len_C, ratio_bc=ratio_BC
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    history = make_empty_history()
    start_epoch = 0
    global_step = 0
    if config.resume_path is not None:
        checkpoint = restore_synthetic_checkpoint(
            config.resume_path, model, optimizer, map_location=device
        )
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

    print("\nStarting Hierarchical Adaptive Controller Training...")
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Epochs"):
        avg_losses = train_synthetic_epoch(
            model,
            optimizer,
            (train_XA, train_YA, train_XB, train_YB, train_XC, train_YC),
            config,
            epoch,
            criterion,
            device=device,
        )
        global_step += (train_XA.size(0) + batch_size - 1) // batch_size
        metrics = evaluate_synthetic_model(
            model,
            (val_XA, val_YA, val_XB, val_YB, val_XC, val_YC),
            tau=tau_end,
        )
        append_epoch_history(history, avg_losses, metrics)

        if config.save_checkpoints and config.checkpoint_path is not None:
            save_synthetic_checkpoint(
                config.checkpoint_path,
                model,
                optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                metrics=metrics,
                config=config,
            )

        tau = max(tau_end, tau_start - (tau_start - tau_end) * (epoch / max(1, epochs - 1)))
        if (epoch + 1) % 5 == 0:
            tqdm.write(
                f"Epoch {epoch+1:02d}/{epochs} [Tau: {tau:.2f}] - "
                f"Actor P1: {avg_losses['loss_actor_pass1']:.4f} -> "
                f"P2: {avg_losses['loss_actor_pass2']:.4f} | "
                f"WM: {avg_losses['loss_world_model']:.4f} | "
                f"Critic: {avg_losses['loss_critic']:.4f} | "
                f"Total: {avg_losses['loss_total']:.4f} | "
                f"C MSE: {metrics['controller_mse']:.4f} | "
                f"Param Err B: {metrics['error_B']:.4f} | Acc A: {metrics['accuracy_A']:.1f}%"
            )
            with torch.no_grad():
                _a1, _ns, _crit, val_actions2, _la, _w, _b = model(
                    val_XA, val_XB, val_XC, tau=tau_end
                )
            target_c = val_YC[0].cpu().numpy()
            pred_c = val_actions2[0].cpu().numpy()
            tqdm.write(f"   [Controller Target C] : {target_c[:5].round(2)}...")
            tqdm.write(f"   [Controller Pred C]   : {pred_c[:5].round(2)}...\n")

    # Visualization
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(24, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['loss_actor_pass1'], label='Pass 1 (No Critic)', color='red', alpha=0.6)
    plt.plot(history['loss_actor_pass2'], label='Pass 2 (With Critic)', color='blue')
    plt.title('Layer C: Controller Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(history['loss_world_model'], label='World Model Loss', color='purple')
    plt.title('World Model Dynamics Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(history['loss_critic'], label='Critic Signal Loss', color='orange')
    plt.title('Critic Feedback Magnitude')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Criticism')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(history['loss_total'], label='Total Loss', color='black')
    plt.title('Optimized Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(history['error_B'], label='Parameter Error B', color='orange')
    plt.title('Layer B: Param Prediction Error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE (w, b)')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(history['accuracy_A'], label='Accuracy A', color='green', linestyle='--')
    plt.title('Layer A: Concept Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    return history

if __name__ == "__main__":
    train_and_evaluate()
