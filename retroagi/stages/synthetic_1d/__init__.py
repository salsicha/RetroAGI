"""Stage 1: one-dimensional procedural validation."""

from .train import (
    SYNTHETIC_1D_SPEC,
    SYNTHETIC_LOSS_KEYS,
    SYNTHETIC_METRIC_KEYS,
    MeanSyntheticBaseline,
    RandomSyntheticBaseline,
    SyntheticBaselinePredictions,
    SyntheticTrainingConfig,
    compute_synthetic_evaluation_metrics,
    compute_synthetic_training_losses,
    evaluate_synthetic_baselines,
    generate_hierarchical_data,
    make_empty_history,
    make_train_permutation,
    seed_everything,
    train_and_evaluate,
)

__all__ = [
    "SYNTHETIC_1D_SPEC",
    "SYNTHETIC_LOSS_KEYS",
    "SYNTHETIC_METRIC_KEYS",
    "MeanSyntheticBaseline",
    "RandomSyntheticBaseline",
    "SyntheticBaselinePredictions",
    "SyntheticTrainingConfig",
    "compute_synthetic_evaluation_metrics",
    "compute_synthetic_training_losses",
    "evaluate_synthetic_baselines",
    "generate_hierarchical_data",
    "make_empty_history",
    "make_train_permutation",
    "seed_everything",
    "train_and_evaluate",
]
