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
    config = tiny_config(evaluation_max_steps=12)
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


def _holds_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_holds_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item) for item in value)
    return False


def test_worker_results_are_copied_not_shared_tensors(pool):
    # Tensors cross processes as file descriptors; thousands of pending
    # results exceed a systemd service's 1,024-descriptor soft limit.
    from retroagi.stages.block_smb.demonstrations import _encode_demonstration_task
    from retroagi.stages.block_smb.train import _monte_carlo_sample_task
    from scripts.block_smb_family_learning import samples

    config = tiny_config(evaluation_max_steps=12)
    sample = samples("flat_run", 7, "train", 1)[0]
    [episode] = pool.map(_encode_demonstration_task, [(0, sample, config, 32)])
    path, version = pool.publish_policy(make_block_smb_model(config))
    [outcome] = pool.map(
        _monte_carlo_sample_task, [(config, path, version, sample, "validation", None)]
    )
    assert episode["action"] and not _holds_tensor(episode)
    assert outcome["actions"] and not _holds_tensor(outcome)


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


def _rollout(model, config, sample, *, seed, demonstration_actions=None):
    import copy

    from retroagi.stages.block_smb.adapter import BlockSMBObservationConfig
    from retroagi.stages.block_smb.train import (
        BlockSMBStage,
        MarioScenarioEnv,
        block_smb_policy_scenario,
        collect_trajectory,
    )

    stage = BlockSMBStage(
        env=MarioScenarioEnv(reward_config=config.reward_config),
        scenario=block_smb_policy_scenario(
            copy.deepcopy(sample.scenario), config.autonomous_policy
        ),
        vision=StaticBlockVision(),
        observation_config=BlockSMBObservationConfig(),
    )
    torch.manual_seed(seed)
    try:
        trajectory = collect_trajectory(
            model,
            stage,
            sample.scenario_id,
            rollout_steps=60,
            seed=seed,
            deterministic=demonstration_actions is not None,
            device=torch.device("cpu"),
            ablation=config.ablation,
            skill_goal_conditioning=config.skill_goal_conditioning,
            demonstration_actions=demonstration_actions,
            record_policy_inputs=True,
        )
    finally:
        stage.env.close()
    return trajectory, stage.scenario


def test_recomputed_policy_terms_match_the_rollout_graph():
    # Without dropout both passes are deterministic, so a worker trajectory's
    # recomputed terms, losses and gradients equal the in-process rollout's.
    from dataclasses import replace as replace_dataclass

    from retroagi.stages.block_smb.rollout_workers import (
        pack_trajectory,
        recompute_policy_terms,
        unpack_trajectory,
    )
    from retroagi.stages.block_smb.train import compute_block_smb_losses
    from scripts.block_smb_family_learning import samples

    config = tiny_config(autonomous_policy=True)
    torch.manual_seed(3)
    model = make_block_smb_model(config).eval()
    cases = [
        (samples("stomp_mount", 5, "train", 1)[0], None),
        (samples("bridge_wait", 5, "train", 1)[0], None),
        (samples("stair_climb", 5, "train", 1)[0], None),
    ]
    cases.append((cases[2][0], list(cases[2][0].oracle["actions"])))
    names = (
        "log_prob",
        "entropy",
        "primitive_aux_loss",
        "expected_hold",
        "release_logit",
        "next_state_pred",
        "criticism",
        "logits_a",
        "hold_duration_logits",
        "tactic_logits",
        "memory_prediction",
        "objective_logits",
        "strategy_logits",
    )
    for seed, (sample, demonstration) in enumerate(cases):
        trajectory, scenario = _rollout(
            model, config, sample, seed=seed, demonstration_actions=demonstration
        )
        rebuilt, rebuilt_scenario = unpack_trajectory(pack_trajectory(trajectory, scenario))
        assert rebuilt_scenario == scenario
        recompute_policy_terms(model, [rebuilt], config, torch.device("cpu"))
        for original, recomputed in zip(trajectory.transitions, rebuilt.transitions):
            for name in names:
                a, b = getattr(original, name), getattr(recomputed, name)
                assert (a is None) == (b is None), name
                if a is not None:
                    assert a.shape == b.shape, name
                    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), name
        gradients = []
        for transitions in (trajectory.transitions, rebuilt.transitions):
            model.zero_grad(set_to_none=True)
            losses = compute_block_smb_losses(
                model,
                transitions,
                config,
                torch.device("cpu"),
                trajectories=[replace_dataclass(trajectory, transitions=transitions)],
            )
            losses["loss_total"].backward(retain_graph=True)
            gradients.append(
                torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
            )
        assert torch.allclose(gradients[0], gradients[1], atol=1e-5, rtol=1e-3)


def test_pooled_training_epoch_keeps_update_batches_and_bookkeeping(pool):
    from retroagi.stages.block_smb.train import (
        BlockSMBSuccessReplay,
        make_block_smb_optimizer,
        train_block_smb_epoch,
    )
    from scripts.block_smb_family_learning import samples

    config = tiny_config(
        autonomous_policy=True,
        rollout_steps=24,
        update_batch_episodes=3,
        success_replay_rehearsals_per_epoch=4,
        retention_imitation_weight=0.1,
        policy_recovery_samples_per_bin=1,
    )
    curriculum = [
        (sample.scenario_id, sample.scenario)
        for family in ("flat_run", "stomp_mount", "stair_climb")
        for sample in samples(family, 11, "train", 3)
    ]
    torch.manual_seed(0)
    model = make_block_smb_model(config)
    records = []
    metrics, _ = train_block_smb_epoch(
        model,
        make_block_smb_optimizer(model, config),
        curriculum,
        config,
        0,
        device=torch.device("cpu"),
        vision_factory=StaticBlockVision,
        success_replay=BlockSMBSuccessReplay(max_episodes_per_family=4, seed=0),
        recovery_records=records,
        pool=pool,
    )
    # Every update batch holds three episodes played with that batch's weights.
    assert metrics["episodes"] >= len(curriculum)
    assert metrics["optimizer_updates"] == -(-metrics["episodes"] // 3)
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
    assert metrics["train_total_actions"] > 0
    assert all(record["actions"] and record["scenario"] for record in records)


def test_pooled_training_layouts_match_sequential(pool):
    from retroagi.stages.block_smb.train import (
        build_adaptive_monte_carlo_replay_curriculum,
        build_mastery_monte_carlo_curriculum,
        initial_block_smb_mastery_state,
    )

    config = tiny_config(
        monte_carlo_train_samples_per_epoch=6,
        monte_carlo_failure_replay_samples_per_epoch=4,
    )
    state = initial_block_smb_mastery_state()
    assert build_mastery_monte_carlo_curriculum(
        config, state, phase=2, pool=pool
    ) == build_mastery_monte_carlo_curriculum(config, state, phase=2)
    bins = {"enemy_stomp:medium": {"failure_count": 2}, "piranha_avoidance": {"failure_count": 1}}
    assert build_adaptive_monte_carlo_replay_curriculum(
        config, bins, epoch=3, pool=pool
    ) == build_adaptive_monte_carlo_replay_curriculum(config, bins, epoch=3)


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
