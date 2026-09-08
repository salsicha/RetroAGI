"""A full restart trains one shared core without preliminary family models."""

from dataclasses import dataclass

import pytest
import torch

from scripts import smb_composable_training as training


@dataclass
class TinyData:
    action: torch.Tensor


def test_streamed_bootstrap_preserves_model_budget_data_and_episode_offsets(monkeypatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = dict(
        seed=42,
        families=["flat_run", "single_gap"],
        demonstration_layouts_per_family=7,
        demonstration_chunk_layouts_per_family=3,
        bootstrap_updates=5,
    )
    seen, fits, events = [], [], []

    def samples(config, split, count, *, offset):
        cases = [(f, split, offset + i) for f in config["families"] for i in range(count)]
        seen.extend(cases)
        return cases

    def collect(candidate, cases, vision, *, log):
        assert candidate is model
        return TinyData(torch.arange(len(cases))), [
            dict(start=i, length=1) for i in range(len(cases))
        ]

    def fit(candidate, optim, data, *, steps, **kwargs):
        assert candidate is model and optim is optimizer
        fits.append((steps, len(data.action)))
        return 0.1

    monkeypatch.setattr(training, "samples", samples)
    monkeypatch.setattr(training, "collect", collect)
    monkeypatch.setattr(training, "fit_demonstrations", fit)
    data, episodes = training.bootstrap_shared(model, optimizer, config, None, log=events.append)
    assert fits == [(1, 6), (2, 12), (2, 14)]
    assert len(data.action) == 14
    assert [e["start"] for e in episodes] == list(range(14))
    assert len(set(seen)) == 14
    for family in config["families"]:
        assert {i for f, _, i in seen if f == family} == set(range(10000, 10007))
    assert events[-1]["updates"] == 5


def test_low_family_scores_do_not_block_the_30_shared_epochs(monkeypatch, tmp_path):
    from scripts import smb_physics_audit

    config = dict(
        seed=42,
        device="cpu",
        hidden_dim=8,
        learning_rate=0.001,
        families=["flat_run", "single_gap"],
        epochs=30,
        perception_layouts_per_family=1,
        perception_validation_per_family=1,
        perception_updates=1,
        demonstration_layouts_per_family=6,
        demonstration_chunk_layouts_per_family=3,
        bootstrap_updates=5,
        validation_layouts_per_family=1,
        train_layouts_per_family_per_epoch=1,
        rehearsal_updates=2,
        family_gate=0.99,
    )
    models, fit_models, update_counts, exports = [], [], [], []

    def make_model(**kwargs):
        model = torch.nn.Linear(1, 1)
        models.append(model)
        return model

    def fit(model, optimizer, data, *, steps, **kwargs):
        fit_models.append(model)
        update_counts.append(steps)
        # Simulate updates, preserving a single model across bootstrap and epochs.
        with torch.no_grad():
            model.weight.add_(1)
        return 0.1

    monkeypatch.setattr(smb_physics_audit, "audit", lambda: {"exact_motion_gate": True})
    monkeypatch.setattr(training, "samples", lambda *args, **kwargs: [])
    monkeypatch.setattr(training, "block_clips", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        training, "train_perception", lambda *args, **kwargs: (None, {"qualified": True})
    )
    monkeypatch.setattr(training, "make_model", make_model)
    monkeypatch.setattr(
        training,
        "collect",
        lambda *args, **kwargs: (TinyData(torch.tensor([1])), [dict(start=0, length=1)]),
    )
    monkeypatch.setattr(training, "save_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "fit_demonstrations", fit)
    monkeypatch.setattr(
        training,
        "evaluate",
        lambda *args, **kwargs: {"minimum": 0.1, "rates": {"single_gap": {"hard": 0.1}}},
    )
    monkeypatch.setattr(
        training, "export_bundle", lambda model, path, **kwargs: exports.append(path.name)
    )
    strict = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # Poor scores can block emulator promotion after training, not its start.
        with pytest.raises(training.QualificationFailure, match="Shared Block held-out"):
            training.run(config, tmp_path / "run")
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn_only)
    assert len(models) == 1
    assert all(model is models[0] for model in fit_models)
    assert sum(update_counts) == 5 + 30 * 2
    assert exports == [f"epoch_{i:02d}" for i in range(1, 31)]
    assert not (tmp_path / "run" / "family_learning.json").exists()
    assert (tmp_path / "run" / "block_test.json").is_file()
