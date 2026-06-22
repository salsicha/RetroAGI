"""Tests for deterministic Synthetic 1D dataset splits and training seeds."""

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from retroagi.core import (
    AgentWorldModelCritic,
    build_checkpoint,
    checkpoint_summary_path,
    load_checkpoint,
    save_checkpoint,
)
from retroagi.stages.synthetic_1d.train import (
    SYNTHETIC_1D_SPEC,
    SYNTHETIC_LOSS_KEYS,
    SYNTHETIC_CHECKPOINT_KIND,
    SYNTHETIC_METRIC_KEYS,
    SYNTHETIC_MODEL_NAME,
    MeanSyntheticBaseline,
    RandomSyntheticBaseline,
    SyntheticBaselinePredictions,
    SyntheticSplitSeeds,
    SyntheticSplitSizes,
    SyntheticTrainingConfig,
    compute_synthetic_evaluation_metrics,
    compute_synthetic_training_losses,
    evaluate_synthetic_baselines,
    generate_dataset_splits,
    generate_hierarchical_data,
    make_empty_history,
    make_train_permutation,
    restore_synthetic_checkpoint,
    save_synthetic_checkpoint,
    seed_everything,
    train_and_evaluate,
    train_synthetic_epoch,
)


class TestSynthetic1DDatasets(unittest.TestCase):
    def assert_dataset_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for left_tensor, right_tensor in zip(left, right):
            torch.testing.assert_close(left_tensor, right_tensor)

    def test_seeded_generation_is_reproducible(self):
        args = (
            5,
            SYNTHETIC_1D_SPEC.seq_len_a,
            SYNTHETIC_1D_SPEC.ratio_ab,
            SYNTHETIC_1D_SPEC.ratio_bc,
            SYNTHETIC_1D_SPEC.vocab_size,
        )

        first = generate_hierarchical_data(*args, seed=123)
        second = generate_hierarchical_data(*args, seed=123)
        different = generate_hierarchical_data(*args, seed=124)

        self.assert_dataset_equal(first, second)
        self.assertFalse(torch.equal(first[0], different[0]))

    def test_fixed_split_seeds_are_reproducible_and_independent(self):
        sizes = SyntheticSplitSizes(train=7, validation=5, test=3)
        seeds = SyntheticSplitSeeds(train=11, validation=22, test=33)

        first = generate_dataset_splits(sizes=sizes, seeds=seeds)
        second = generate_dataset_splits(sizes=sizes, seeds=seeds)

        self.assert_dataset_equal(first.train, second.train)
        self.assert_dataset_equal(first.validation, second.validation)
        self.assert_dataset_equal(first.test, second.test)
        self.assertFalse(torch.equal(first.train[0][:3], first.validation[0][:3]))
        self.assertFalse(torch.equal(first.validation[0][:3], first.test[0]))

    def test_split_shapes_follow_stage_spec_and_requested_sizes(self):
        sizes = SyntheticSplitSizes(train=7, validation=5, test=3)
        splits = generate_dataset_splits(sizes=sizes)
        seq_len_b = SYNTHETIC_1D_SPEC.seq_len_b
        seq_len_c = SYNTHETIC_1D_SPEC.seq_len_c

        expected = {
            "train": sizes.train,
            "validation": sizes.validation,
            "test": sizes.test,
        }
        for name, sample_count in expected.items():
            with self.subTest(split=name):
                xa, ya, xb, yb, xc, yc = getattr(splits, name)
                self.assertEqual(xa.shape, (sample_count, SYNTHETIC_1D_SPEC.seq_len_a))
                self.assertEqual(ya.shape, (sample_count, SYNTHETIC_1D_SPEC.seq_len_a))
                self.assertEqual(xb.shape, (sample_count, seq_len_b))
                self.assertEqual(yb.shape, (sample_count, seq_len_b))
                self.assertEqual(xc.shape, (sample_count, seq_len_c))
                self.assertEqual(yc.shape, (sample_count, seq_len_c))

    def test_split_definitions_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "seeds must be distinct"):
            SyntheticSplitSeeds(train=1, validation=1, test=2)
        with self.assertRaisesRegex(ValueError, "validation split size"):
            SyntheticSplitSizes(train=1, validation=0, test=1)
        with self.assertRaisesRegex(ValueError, "num_samples"):
            generate_hierarchical_data(0, 8, 2, 4, 20, seed=1)


class TestSynthetic1DTrainingReproducibility(unittest.TestCase):
    def test_training_config_derives_split_seeds_from_run_seed(self):
        config = SyntheticTrainingConfig(
            seed=123,
            split_sizes=SyntheticSplitSizes(train=4, validation=3, test=2),
        )

        self.assertEqual(config.resolved_split_seeds, SyntheticSplitSeeds(10_124, 20_124, 30_124))

        first = generate_dataset_splits(
            sizes=config.split_sizes,
            seeds=config.resolved_split_seeds,
        )
        second = generate_dataset_splits(
            sizes=config.split_sizes,
            seeds=config.resolved_split_seeds,
        )

        TestSynthetic1DDatasets().assert_dataset_equal(first.train, second.train)
        TestSynthetic1DDatasets().assert_dataset_equal(first.validation, second.validation)
        TestSynthetic1DDatasets().assert_dataset_equal(first.test, second.test)

    def test_seed_everything_reproduces_python_numpy_and_torch_streams(self):
        seed_everything(77, deterministic=True)
        first = (
            random.random(),
            np.random.random(3),
            torch.rand(3),
        )

        seed_everything(77, deterministic=True)
        second = (
            random.random(),
            np.random.random(3),
            torch.rand(3),
        )

        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_train_permutation_is_deterministic_by_config_and_epoch(self):
        config = SyntheticTrainingConfig(seed=19)

        first = make_train_permutation(16, config, epoch=0)
        second = make_train_permutation(16, config, epoch=0)
        next_epoch = make_train_permutation(16, config, epoch=1)

        generator = torch.Generator().manual_seed(config.train_permutation_seed(0))
        expected = torch.randperm(16, generator=generator)

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, expected)
        self.assertFalse(torch.equal(first, next_epoch))

    def test_training_config_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            SyntheticTrainingConfig(batch_size=0)
        with self.assertRaisesRegex(ValueError, "epochs"):
            SyntheticTrainingConfig(epochs=0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            SyntheticTrainingConfig(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "epoch"):
            SyntheticTrainingConfig().train_permutation_seed(-1)
        with self.assertRaisesRegex(ValueError, "num_samples"):
            make_train_permutation(0, SyntheticTrainingConfig(), epoch=0)


class TestSynthetic1DLossTracking(unittest.TestCase):
    def test_empty_history_contains_all_loss_and_metric_keys(self):
        history = make_empty_history()

        self.assertEqual(set(SYNTHETIC_LOSS_KEYS), {
            "loss_actor_pass1",
            "loss_actor_pass2",
            "loss_world_model",
            "loss_critic",
            "loss_total",
        })
        self.assertEqual(set(history), {*SYNTHETIC_LOSS_KEYS, *SYNTHETIC_METRIC_KEYS})
        self.assertTrue(all(values == [] for values in history.values()))

    def test_training_loss_helper_tracks_named_components_and_total(self):
        actions1 = torch.tensor([[1.0, 2.0]])
        actions2 = torch.tensor([[1.5, 2.5]])
        batch_xc_in = torch.tensor([[0.0, 1.0]])
        batch_yc_target = torch.tensor([[1.0, 3.0]])
        next_state_pred = torch.tensor([[0.5, 4.0]])
        criticism = torch.tensor([[[1.0, -1.0], [3.0, -3.0]]])

        losses = compute_synthetic_training_losses(
            actions1,
            next_state_pred,
            criticism,
            actions2,
            batch_xc_in,
            batch_yc_target,
            nn.MSELoss(),
            critic_loss_weight=0.25,
        )

        expected_world_model = torch.tensor(0.625)
        expected_actor_pass1 = torch.tensor(0.5)
        expected_actor_pass2 = torch.tensor(0.25)
        expected_critic = torch.tensor(5.0)
        expected_total = (
            expected_world_model
            + expected_actor_pass1
            + expected_actor_pass2
            + 0.25 * expected_critic
        )

        self.assertEqual(tuple(losses), SYNTHETIC_LOSS_KEYS)
        torch.testing.assert_close(losses["loss_world_model"], expected_world_model)
        torch.testing.assert_close(losses["loss_actor_pass1"], expected_actor_pass1)
        torch.testing.assert_close(losses["loss_actor_pass2"], expected_actor_pass2)
        torch.testing.assert_close(losses["loss_critic"], expected_critic)
        torch.testing.assert_close(losses["loss_total"], expected_total)

    def test_loss_tracking_rejects_negative_critic_weight(self):
        with self.assertRaisesRegex(ValueError, "critic_loss_weight"):
            SyntheticTrainingConfig(critic_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "critic_loss_weight"):
            compute_synthetic_training_losses(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2, 1),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                nn.MSELoss(),
                critic_loss_weight=-0.1,
            )


class TestSynthetic1DEvaluationMetricsAndBaselines(unittest.TestCase):
    def test_perfect_predictions_have_zero_controller_and_parameter_error(self):
        dataset = generate_hierarchical_data(4, 3, 2, 2, 7, seed=5)
        _xa, ya, _xb, yb, _xc, yc = dataset
        logits_a = torch.zeros((*ya.shape, SYNTHETIC_1D_SPEC.vocab_size), dtype=torch.float32)
        logits_a.scatter_(dim=-1, index=ya.unsqueeze(-1), value=1.0)
        predictions = SyntheticBaselinePredictions(
            actions_c=yc.clone(),
            logits_a=logits_a,
            w_b=torch.sin(yb.float()),
            b_b=torch.cos(yb.float()),
        )

        metrics = compute_synthetic_evaluation_metrics(predictions, dataset)

        self.assertEqual(tuple(metrics), SYNTHETIC_METRIC_KEYS)
        self.assertEqual(metrics["controller_mse"], 0.0)
        self.assertEqual(metrics["controller_mae"], 0.0)
        self.assertEqual(metrics["controller_rmse"], 0.0)
        self.assertEqual(metrics["error_B"], 0.0)
        self.assertEqual(metrics["accuracy_A"], 100.0)

    def test_evaluation_metrics_reject_incompatible_prediction_shapes(self):
        dataset = generate_hierarchical_data(2, 3, 2, 2, 7, seed=5)
        _xa, ya, _xb, yb, _xc, yc = dataset
        predictions = SyntheticBaselinePredictions(
            actions_c=yc[:, :-1],
            logits_a=torch.zeros((*ya.shape, SYNTHETIC_1D_SPEC.vocab_size)),
            w_b=torch.zeros_like(yb, dtype=torch.float32),
            b_b=torch.zeros_like(yb, dtype=torch.float32),
        )

        with self.assertRaisesRegex(ValueError, "actions_c shape"):
            compute_synthetic_evaluation_metrics(predictions, dataset)

    def test_random_baseline_is_seeded_and_matches_prediction_shapes(self):
        dataset = generate_hierarchical_data(3, 3, 2, 2, 7, seed=5)
        first = RandomSyntheticBaseline(seed=11).predict(dataset)
        second = RandomSyntheticBaseline(seed=11).predict(dataset)
        different = RandomSyntheticBaseline(seed=12).predict(dataset)
        _xa, ya, _xb, yb, _xc, yc = dataset

        self.assertEqual(first.actions_c.shape, yc.shape)
        self.assertEqual(first.logits_a.shape, (*ya.shape, SYNTHETIC_1D_SPEC.vocab_size))
        self.assertEqual(first.w_b.shape, yb.shape)
        torch.testing.assert_close(first.actions_c, second.actions_c)
        torch.testing.assert_close(first.logits_a, second.logits_a)
        self.assertFalse(torch.equal(first.actions_c, different.actions_c))

    def test_simple_baseline_uses_training_marginals(self):
        train = generate_hierarchical_data(5, 3, 2, 2, 7, seed=5)
        validation = generate_hierarchical_data(4, 3, 2, 2, 7, seed=6)
        baseline = MeanSyntheticBaseline.fit(train)
        predictions = baseline.predict(validation)
        _xa, train_ya, _xb, train_yb, _xc, train_yc = train
        _vxa, val_ya, _vxb, val_yb, _vxc, val_yc = validation

        torch.testing.assert_close(baseline.action_mean, train_yc.mean(dim=0))
        torch.testing.assert_close(baseline.w_mean, torch.sin(train_yb.float()).mean(dim=0))
        self.assertEqual(predictions.actions_c.shape, val_yc.shape)
        self.assertEqual(predictions.logits_a.shape, (*val_ya.shape, SYNTHETIC_1D_SPEC.vocab_size))
        self.assertEqual(predictions.w_b.shape, val_yb.shape)
        self.assertTrue(torch.equal(predictions.logits_a.argmax(dim=-1)[0], torch.mode(train_ya, dim=0).values))

    def test_baseline_evaluation_returns_random_and_simple_metrics(self):
        train = generate_hierarchical_data(5, 3, 2, 2, 7, seed=5)
        validation = generate_hierarchical_data(4, 3, 2, 2, 7, seed=6)

        metrics = evaluate_synthetic_baselines(train, validation, seed=13)

        self.assertEqual(set(metrics), {"random", "simple"})
        for baseline_metrics in metrics.values():
            self.assertEqual(tuple(baseline_metrics), SYNTHETIC_METRIC_KEYS)
            self.assertGreaterEqual(baseline_metrics["controller_mse"], 0.0)
            self.assertGreaterEqual(baseline_metrics["accuracy_A"], 0.0)
            self.assertLessEqual(baseline_metrics["accuracy_A"], 100.0)


class TestSynthetic1DCPUSmoke(unittest.TestCase):
    def test_tiny_cpu_training_step_has_finite_gradients_and_decreasing_loss(self):
        seed_everything(2_024, deterministic=True)
        xa, _ya, xb, _yb, xc, yc = generate_hierarchical_data(
            4,
            SYNTHETIC_1D_SPEC.seq_len_a,
            SYNTHETIC_1D_SPEC.ratio_ab,
            SYNTHETIC_1D_SPEC.ratio_bc,
            SYNTHETIC_1D_SPEC.vocab_size,
            seed=303,
        )
        model = AgentWorldModelCritic(
            vocab_size=SYNTHETIC_1D_SPEC.vocab_size,
            seq_len_a=SYNTHETIC_1D_SPEC.seq_len_a,
            seq_len_c=SYNTHETIC_1D_SPEC.seq_len_c,
            ratio_bc=SYNTHETIC_1D_SPEC.ratio_bc,
            d_model=8,
        ).cpu()
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.MSELoss()

        def total_loss():
            torch.manual_seed(404)
            actions1, next_state_pred, criticism, actions2, _logits_a, _w_b, _b_b = model(
                xa, xb, xc, tau=1.0
            )
            losses = compute_synthetic_training_losses(
                actions1,
                next_state_pred,
                criticism,
                actions2,
                xc,
                yc,
                criterion,
            )
            return losses["loss_total"]

        initial_loss = total_loss().item()
        saw_nonzero_gradient = False
        for _ in range(8):
            optimizer.zero_grad()
            loss = total_loss()
            loss.backward()

            gradients = [param.grad for param in model.parameters() if param.grad is not None]
            self.assertTrue(gradients)
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all().item())
                saw_nonzero_gradient = saw_nonzero_gradient or bool(torch.count_nonzero(gradient).item())
            optimizer.step()

        final_loss = total_loss().item()
        self.assertTrue(saw_nonzero_gradient)
        self.assertTrue(torch.isfinite(torch.tensor(final_loss)).item())
        self.assertLess(final_loss, initial_loss)


class TestSynthetic1DCheckpointing(unittest.TestCase):
    def make_model_and_optimizer(self):
        model = AgentWorldModelCritic(
            vocab_size=SYNTHETIC_1D_SPEC.vocab_size,
            seq_len_a=SYNTHETIC_1D_SPEC.seq_len_a,
            seq_len_c=SYNTHETIC_1D_SPEC.seq_len_c,
            ratio_bc=SYNTHETIC_1D_SPEC.ratio_bc,
            d_model=8,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return model, optimizer

    def step_optimizer_once(self, model, optimizer):
        xa, _ya, xb, _yb, xc, yc = generate_hierarchical_data(
            2,
            SYNTHETIC_1D_SPEC.seq_len_a,
            SYNTHETIC_1D_SPEC.ratio_ab,
            SYNTHETIC_1D_SPEC.ratio_bc,
            SYNTHETIC_1D_SPEC.vocab_size,
            seed=707,
        )
        optimizer.zero_grad()
        torch.manual_seed(808)
        actions1, next_state_pred, criticism, actions2, _logits_a, _w_b, _b_b = model(
            xa, xb, xc, tau=1.0
        )
        losses = compute_synthetic_training_losses(
            actions1, next_state_pred, criticism, actions2, xc, yc, nn.MSELoss()
        )
        losses["loss_total"].backward()
        optimizer.step()

    def test_saves_shared_schema_checkpoint_payload(self):
        model, optimizer = self.make_model_and_optimizer()
        config = SyntheticTrainingConfig(seed=17, batch_size=4, epochs=2)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synthetic.pth"
            save_synthetic_checkpoint(
                path,
                model,
                optimizer,
                epoch=2,
                global_step=8,
                metrics={"controller_mse": 0.75},
                config=config,
                metadata={"unit": True},
            )
            loaded = load_checkpoint(path)
            summary = json.loads(checkpoint_summary_path(path).read_text(encoding="utf-8"))

        self.assertEqual(loaded["stage"], SYNTHETIC_1D_SPEC.name)
        self.assertEqual(loaded["model_name"], SYNTHETIC_MODEL_NAME)
        self.assertEqual(loaded["checkpoint_kind"], SYNTHETIC_CHECKPOINT_KIND)
        self.assertEqual(loaded["epoch"], 2)
        self.assertEqual(loaded["global_step"], 8)
        self.assertEqual(loaded["metrics"]["controller_mse"], 0.75)
        self.assertEqual(loaded["config"]["seed"], 17)
        self.assertEqual(loaded["specs"]["stage"]["seq_len_c"], SYNTHETIC_1D_SPEC.seq_len_c)
        self.assertTrue(loaded["metadata"]["unit"])
        self.assertIn("code_revision", loaded["metadata"])
        self.assertIn("runtime_environment", loaded["metadata"])
        self.assertEqual(summary["stage"], SYNTHETIC_1D_SPEC.name)
        self.assertEqual(summary["metrics"]["controller_mse"], 0.75)
        self.assertEqual(summary["config"]["seed"], 17)
        self.assertIn("optimizer", summary["state_keys"])
        self.assertIn("model", loaded["states"])
        self.assertIn("optimizer", loaded["states"])

    def test_restores_model_and_optimizer_state(self):
        seed_everything(909, deterministic=True)
        model, optimizer = self.make_model_and_optimizer()
        self.step_optimizer_once(model, optimizer)
        expected_params = {name: value.detach().clone() for name, value in model.state_dict().items()}

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "synthetic.pth"
            save_synthetic_checkpoint(
                path, model, optimizer, epoch=1, global_step=1, metrics={"controller_mse": 1.0}
            )
            restored_model, restored_optimizer = self.make_model_and_optimizer()
            restored = restore_synthetic_checkpoint(path, restored_model, restored_optimizer)

        self.assertEqual(restored["epoch"], 1)
        for name, value in restored_model.state_dict().items():
            torch.testing.assert_close(value, expected_params[name])
        self.assertTrue(restored_optimizer.state_dict()["state"])


    def test_resume_continuation_matches_uninterrupted_training(self):
        config = SyntheticTrainingConfig(
            seed=515,
            split_sizes=SyntheticSplitSizes(train=8, validation=2, test=2),
            split_seeds=SyntheticSplitSeeds(train=616, validation=717, test=818),
            batch_size=4,
            epochs=2,
            learning_rate=0.01,
        )
        splits = generate_dataset_splits(
            sizes=config.split_sizes,
            seeds=config.resolved_split_seeds,
        )
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        seed_everything(config.seed, deterministic=True)
        uninterrupted_model, uninterrupted_optimizer = self.make_model_and_optimizer()
        train_synthetic_epoch(
            uninterrupted_model,
            uninterrupted_optimizer,
            splits.train,
            config,
            0,
            criterion,
            device=device,
        )
        train_synthetic_epoch(
            uninterrupted_model,
            uninterrupted_optimizer,
            splits.train,
            config,
            1,
            criterion,
            device=device,
        )

        seed_everything(config.seed, deterministic=True)
        interrupted_model, interrupted_optimizer = self.make_model_and_optimizer()
        train_synthetic_epoch(
            interrupted_model,
            interrupted_optimizer,
            splits.train,
            config,
            0,
            criterion,
            device=device,
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "resume.pth"
            save_synthetic_checkpoint(
                path,
                interrupted_model,
                interrupted_optimizer,
                epoch=1,
                global_step=2,
                config=config,
            )
            resumed_model, resumed_optimizer = self.make_model_and_optimizer()
            restore_synthetic_checkpoint(path, resumed_model, resumed_optimizer)
            train_synthetic_epoch(
                resumed_model,
                resumed_optimizer,
                splits.train,
                config,
                1,
                criterion,
                device=device,
            )

        for name, value in resumed_model.state_dict().items():
            torch.testing.assert_close(value, uninterrupted_model.state_dict()[name])
        self.assertEqual(
            resumed_optimizer.state_dict()["state"].keys(),
            uninterrupted_optimizer.state_dict()["state"].keys(),
        )

    def test_train_and_evaluate_saves_and_resumes_from_checkpoint(self):
        with TemporaryDirectory() as tmpdir, patch("matplotlib.pyplot.show"):
            path = Path(tmpdir) / "synthetic_resume.pth"
            first_config = SyntheticTrainingConfig(
                seed=919,
                split_sizes=SyntheticSplitSizes(train=4, validation=2, test=2),
                split_seeds=SyntheticSplitSeeds(train=920, validation=921, test=922),
                batch_size=2,
                epochs=1,
                learning_rate=0.01,
                checkpoint_path=path,
                save_checkpoints=True,
                device="cpu",
            )
            first_history = train_and_evaluate(first_config)
            first_checkpoint = load_checkpoint(path)

            resumed_config = SyntheticTrainingConfig(
                seed=919,
                split_sizes=SyntheticSplitSizes(train=4, validation=2, test=2),
                split_seeds=SyntheticSplitSeeds(train=920, validation=921, test=922),
                batch_size=2,
                epochs=2,
                learning_rate=0.01,
                checkpoint_path=path,
                resume_path=path,
                save_checkpoints=True,
                device="cpu",
            )
            resumed_history = train_and_evaluate(resumed_config)
            resumed_checkpoint = load_checkpoint(path)

        self.assertEqual(len(first_history["loss_total"]), 1)
        self.assertEqual(len(resumed_history["loss_total"]), 1)
        self.assertEqual(first_checkpoint["epoch"], 1)
        self.assertEqual(resumed_checkpoint["epoch"], 2)
        self.assertEqual(first_checkpoint["global_step"], 2)
        self.assertEqual(resumed_checkpoint["global_step"], 4)

    def test_restore_rejects_incompatible_stage_checkpoint(self):
        model, optimizer = self.make_model_and_optimizer()
        bad_checkpoint = build_checkpoint(
            stage="block_smb",
            model_name=SYNTHETIC_MODEL_NAME,
            checkpoint_kind=SYNTHETIC_CHECKPOINT_KIND,
            states={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pth"
            save_checkpoint(path, bad_checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoint stage"):
                restore_synthetic_checkpoint(path, model, optimizer)


class TestSynthetic1DBaselineComparison(unittest.TestCase):
    def test_trained_policy_beats_declared_baselines_on_held_out_controller_mse(self):
        seed_everything(123, deterministic=True)
        splits = generate_dataset_splits(
            sizes=SyntheticSplitSizes(train=128, validation=32, test=64),
            seeds=SyntheticSplitSeeds(train=101, validation=202, test=303),
        )
        train_xa, _train_ya, train_xb, _train_yb, train_xc, train_yc = splits.train
        test_xa, _test_ya, test_xb, _test_yb, test_xc, _test_yc = splits.test
        model = AgentWorldModelCritic(
            vocab_size=SYNTHETIC_1D_SPEC.vocab_size,
            seq_len_a=SYNTHETIC_1D_SPEC.seq_len_a,
            seq_len_c=SYNTHETIC_1D_SPEC.seq_len_c,
            ratio_bc=SYNTHETIC_1D_SPEC.ratio_bc,
            d_model=16,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        batch_size = 32

        model.train()
        for epoch in range(60):
            permutation = torch.randperm(
                train_xa.size(0), generator=torch.Generator().manual_seed(500 + epoch)
            )
            for start in range(0, train_xa.size(0), batch_size):
                indices = permutation[start : start + batch_size]
                torch.manual_seed(10_000 + epoch * 10 + start)
                optimizer.zero_grad()
                actions1, next_state_pred, criticism, actions2, _logits_a, _w_b, _b_b = model(
                    train_xa[indices], train_xb[indices], train_xc[indices], tau=1.0
                )
                losses = compute_synthetic_training_losses(
                    actions1,
                    next_state_pred,
                    criticism,
                    actions2,
                    train_xc[indices],
                    train_yc[indices],
                    criterion,
                )
                losses["loss_total"].backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            torch.manual_seed(999)
            _actions1, _next_state, _criticism, actions2, logits_a, w_b, b_b = model(
                test_xa, test_xb, test_xc, tau=0.1
            )
        trained_metrics = compute_synthetic_evaluation_metrics(
            SyntheticBaselinePredictions(
                actions_c=actions2,
                logits_a=logits_a,
                w_b=w_b,
                b_b=b_b,
            ),
            splits.test,
        )
        baseline_metrics = evaluate_synthetic_baselines(splits.train, splits.test, seed=123)

        self.assertLess(
            trained_metrics["controller_mse"],
            baseline_metrics["random"]["controller_mse"],
        )
        self.assertLess(
            trained_metrics["controller_mse"],
            baseline_metrics["simple"]["controller_mse"],
        )
        self.assertLess(
            trained_metrics["controller_mse"],
            0.75 * baseline_metrics["simple"]["controller_mse"],
        )


if __name__ == "__main__":
    unittest.main()
