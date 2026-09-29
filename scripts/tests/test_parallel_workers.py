"""CPU worker pools reproduce the in-process evaluation and demonstration paths."""

from dataclasses import fields, replace

import pytest
import torch

from retroagi.stages.block_smb.demonstrations import build_balanced_demonstrations
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_parameter_sweep
from retroagi.stages.block_smb.parallel import BlockSMBWorkerPool
from retroagi.stages.block_smb.train import (
    _includes_test_split,
    _move_training_state,
    evaluate_block_smb,
    evaluate_block_smb_monte_carlo,
    make_block_smb_model,
    make_block_smb_optimizer,
)
from scripts.tests.test_block_smb_training import StaticBlockVision, tiny_config


@pytest.fixture(scope="module")
def pool():
    with BlockSMBWorkerPool(2, StaticBlockVision()) as workers:
        yield workers


def test_pooled_evaluation_matches_in_process_and_keeps_its_layouts(pool):
    config = tiny_config(evaluation_max_steps=12, monte_carlo_validate_reachability=False)
    model = make_block_smb_model(config).eval()
    kwargs = dict(
        split="validation",
        sample_count=6,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
        stratified_repeats_per_difficulty=0,
    )
    expected = evaluate_block_smb_monte_carlo(model, config, **kwargs)
    assert evaluate_block_smb_monte_carlo(model, config, pool=pool, **kwargs) == expected
    assert len(pool.sample_sets) == 1
    # Later evaluations reuse the layouts and see updated weights.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05)
    updated = evaluate_block_smb_monte_carlo(model, config, **kwargs)
    assert evaluate_block_smb_monte_carlo(model, config, pool=pool, **kwargs) == updated
    assert len(pool.sample_sets) == 1


def test_pooled_demonstrations_match_in_process(pool):
    config = replace(tiny_config(), demonstration_layouts_per_family=3)
    families = ("flat_run", "bridge_mount", "stomp_mount")
    expected = build_balanced_demonstrations(config, StaticBlockVision, families=families)
    actual = build_balanced_demonstrations(config, StaticBlockVision, families=families, pool=pool)
    for field in fields(expected):
        assert torch.equal(getattr(actual, field.name), getattr(expected, field.name)), field.name


def test_pooled_layout_sweep_matches_sequential(pool):
    kwargs = dict(
        split="validation",
        seed=11,
        repeats_per_difficulty=1,
        families=("flat_run", "enemy_stomp", "piranha_avoidance"),
    )
    expected = sample_block_smb_monte_carlo_parameter_sweep(**kwargs)
    assert sample_block_smb_monte_carlo_parameter_sweep(executor=pool, **kwargs) == expected


def test_test_split_runs_every_interval_and_after_the_last_epoch():
    config = tiny_config(epochs=7, monte_carlo_test_interval_epochs=3)
    assert [epoch for epoch in range(1, 8) if _includes_test_split(config, epoch)] == [3, 6, 7]
    config = tiny_config(
        fixed_scenarios=(),
        monte_carlo_test_samples=1,
        monte_carlo_validation_repeats_per_difficulty=0,
        monte_carlo_validate_reachability=False,
    )
    model = make_block_smb_model(config)
    kwargs = dict(device=torch.device("cpu"), vision_factory=StaticBlockVision)
    assert "monte_carlo_test" in evaluate_block_smb(model, config, **kwargs)
    assert "monte_carlo_test" not in evaluate_block_smb(model, config, include_test=False, **kwargs)


def test_training_run_shares_one_pool_with_demonstrations_and_evaluation(monkeypatch):
    from retroagi.stages.block_smb import demonstrations
    from retroagi.stages.block_smb.demonstrations import collect_demonstrations
    from retroagi.stages.block_smb.train import train_and_evaluate_block_smb
    from scripts.block_smb_family_learning import samples

    base = tiny_config(generated_scenarios=0)
    config = replace(
        base,
        ablation=replace(base.ablation, recurrent_state_enabled=False),
        demonstration_bootstrap_updates=1,
        demonstration_rehearsal_updates=1,
        mastery_gated_schedule=True,
        monte_carlo_validation_samples=2,
        monte_carlo_validation_repeats_per_difficulty=0,
        monte_carlo_validate_reachability=False,
        parallel_workers=2,
        online_training_device="cpu",
    )
    data = collect_demonstrations(
        [(0, samples("flat_run", 7, "train", 1)[0])], config, StaticBlockVision
    )
    pools = []

    def build(*args, pool=None, **kwargs):
        pools.append(pool)
        return data

    monkeypatch.setattr(demonstrations, "build_balanced_demonstrations", build)
    result = train_and_evaluate_block_smb(config, vision_factory=StaticBlockVision)
    assert len(pools) == 1 and isinstance(pools[0], BlockSMBWorkerPool)
    assert result["evaluation"]["monte_carlo_validation"]["sample_count"] == 2
    assert result["history"][0]["demonstration_rehearsal_updates"] == 1


def test_parallel_settings_are_validated():
    with pytest.raises(ValueError, match="parallel_workers"):
        tiny_config(parallel_workers=-1)
    with pytest.raises(ValueError, match="monte_carlo_test_interval_epochs"):
        tiny_config(monte_carlo_test_interval_epochs=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_training_state_moves_with_its_optimizer_moments():
    config = tiny_config()
    model = make_block_smb_model(config).to("cuda")
    optimizer = make_block_smb_optimizer(model, config)

    def step():
        optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()

    step()
    _move_training_state(torch.device("cpu"), model, optimizer)
    moments = [
        value for state in optimizer.state.values() for key, value in state.items() if key != "step"
    ]
    assert moments and all(value.device.type == "cpu" for value in moments)
    step()
    _move_training_state(torch.device("cuda"), model, optimizer)
    step()
    assert all(parameter.is_cuda for parameter in model.parameters())
