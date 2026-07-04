"""Tests for transferring Block SMB checkpoints into Full SMB."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    BASELINE_ARCHITECTURE_SPEC,
    ArchitectureSpec,
    build_architecture,
    build_checkpoint,
    get_architecture,
    load_checkpoint,
    register_architecture,
    save_checkpoint,
)
from retroagi.stages.block_smb import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
    BLOCK_SMB_SPEC,
    BlockSMBTrainingConfig,
)
from retroagi.stages.block_smb.train import make_block_smb_model, save_block_smb_checkpoint
from retroagi.stages.full_smb import (
    FULL_SMB_SPEC,
    FullSMBObservationConfig,
    FullSMBStage,
    FullSMBVisionTransformer,
    build_full_smb_vit_checkpoint,
)
from retroagi.stages.full_smb.compare import (
    FullSMBPolicyComparisonConfig,
    FullSMBPolicySuiteComparisonConfig,
    compare_full_smb_policy_suite,
    compare_transferred_checkpoint_with_scratch,
    save_full_smb_policy_comparison,
)
from retroagi.stages.full_smb.transfer import (
    FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    FULL_SMB_TRANSFER_MODEL_NAME,
    block_smb_checkpoint_transfer_source_gate,
    load_transferred_full_smb_policy,
    make_full_smb_policy_model,
    select_transferred_full_smb_action,
    transfer_block_smb_checkpoint_to_full_smb,
)


class TinyFullSMBEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self):
        self.closed = False
        self.seed = 0
        self.step_count = 0

    def reset(self, seed=None):
        self.seed = 0 if seed is None else int(seed)
        self.step_count = 0
        return self._observation(0), {
            "x_pos": self.seed,
            "y_pos": 96,
            "score": 0,
            "coins": 0,
            "lives": 3,
        }

    def step(self, action):
        button_vector = np.asarray(action, dtype=np.int8)
        action_value = int(np.flatnonzero(button_vector).sum())
        self.step_count += 1
        terminated = self.step_count >= 8
        return (
            self._observation(action_value),
            float(action_value + self.step_count),
            terminated,
            False,
            {
                "x_pos": self.seed + self.step_count,
                "y_pos": 96,
                "score": self.step_count * 10,
                "coins": self.step_count % 3,
                "lives": 3,
                "level_complete": terminated,
            },
        )

    def close(self):
        self.closed = True

    def _observation(self, action_value):
        base = np.arange(16 * 20, dtype=np.uint16).reshape(16, 20)
        return np.stack(
            (
                (base + self.seed + self.step_count + action_value) % 256,
                (base * 2 + self.seed + self.step_count) % 256,
                (base * 3 + self.seed + action_value) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)


def tiny_block_config(**overrides):
    values = dict(
        seed=3,
        epochs=1,
        episodes_per_epoch=1,
        rollout_steps=1,
        hidden_dim=8,
        controller_schedule="linear",
        fixed_scenarios=("level_1_flat.json",),
        evaluation_episodes=1,
        evaluation_max_steps=1,
        monte_carlo_validation_samples=12,
        device="cpu",
    )
    values.update(overrides)
    return BlockSMBTrainingConfig(**values)


def transfer_ready_metrics(**overrides):
    metrics = {
        "loss_total": 0.25,
        "eval_threshold_pass_rate": 1.0,
        "semantic_prediction_gate_met": 1.0,
        "eval_monte_carlo_validation_success_rate": 1.0,
        "eval_monte_carlo_validation_gate_met": 1.0,
        "eval_fixed_action_count_0": 1.0,
        "eval_fixed_action_count_1": 8.0,
        "eval_fixed_action_count_2": 2.0,
        "eval_fixed_action_count_3": 0.0,
        "eval_fixed_action_count_4": 0.0,
        "eval_fixed_action_count_5": 0.0,
        "eval_fixed_all_noop_action_collapse": 0.0,
        "eval_monte_carlo_validation_action_count_0": 2.0,
        "eval_monte_carlo_validation_action_count_1": 16.0,
        "eval_monte_carlo_validation_action_count_2": 4.0,
        "eval_monte_carlo_validation_action_count_3": 0.0,
        "eval_monte_carlo_validation_action_count_4": 0.0,
        "eval_monte_carlo_validation_action_count_5": 0.0,
        "eval_monte_carlo_validation_all_noop_action_collapse": 0.0,
    }
    metrics.update(overrides)
    return metrics


def write_full_smb_vision_checkpoint(path: Path) -> None:
    model = FullSMBVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
    checkpoint = build_full_smb_vit_checkpoint(
        model,
        epoch=1,
        metrics={"mean_iou": 1.0},
        config={
            "model": {
                "hidden_dim": 16,
                "depth": 1,
                "heads": 4,
                "patch_size": 16,
                "dropout": 0.0,
            }
        },
    )
    save_checkpoint(path, checkpoint)


def write_block_policy_checkpoint(path: Path):
    config = tiny_block_config()
    model = make_block_smb_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_(0.01 * (index + 1))
    save_block_smb_checkpoint(
        path,
        model,
        optimizer,
        epoch=2,
        global_step=5,
        config=config,
        metrics=transfer_ready_metrics(),
    )
    return model, config


def incompatible_transfer_architecture() -> ArchitectureSpec:
    name = "unit_test_incompatible_full_smb_transfer_architecture"
    try:
        return get_architecture(name)
    except KeyError:
        spec = ArchitectureSpec(
            name=name,
            factory=BASELINE_ARCHITECTURE_SPEC.factory,
            supported_stage_names=("block_smb", "full_smb"),
            checkpoint_model_name=name,
            checkpoint_compatibility_policy=(
                BASELINE_ARCHITECTURE_SPEC.checkpoint_compatibility_policy
            ),
            output_contract=f"{name}.forward.v1",
            configurable_hyperparameters=BASELINE_ARCHITECTURE_SPEC.configurable_hyperparameters,
        )
        register_architecture(spec)
        return spec


class TestFullSMBTransfer(unittest.TestCase):
    def test_block_policy_transfers_to_full_smb_checkpoint_and_stage_batch(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            output_path = tmp / "full_smb_transfer.pth"
            source_model, source_config = write_block_policy_checkpoint(source_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)

            with patch(
                "retroagi.stages.full_smb.transfer.build_architecture",
                wraps=build_architecture,
            ) as build_model:
                result = transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    output_checkpoint=output_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )
                loaded = load_transferred_full_smb_policy(
                    output_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    device="cpu",
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(result.checkpoint["stage"], FULL_SMB_SPEC.name)
            self.assertEqual(result.checkpoint["model_name"], FULL_SMB_TRANSFER_MODEL_NAME)
            self.assertEqual(
                result.checkpoint["checkpoint_kind"],
                FULL_SMB_TRANSFER_CHECKPOINT_KIND,
            )
            self.assertEqual(result.checkpoint["epoch"], 2)
            self.assertEqual(result.checkpoint["global_step"], 5)
            self.assertEqual(
                result.checkpoint["config"]["model"]["controller_schedule"],
                source_config.controller_schedule,
            )
            self.assertEqual(
                result.checkpoint["config"]["architecture_name"],
                BASELINE_ARCHITECTURE_NAME,
            )
            self.assertEqual(
                result.checkpoint["config"]["architecture_config"],
                source_config.architecture_config,
            )
            self.assertEqual(
                result.checkpoint["specs"]["architecture"]["name"],
                BASELINE_ARCHITECTURE_NAME,
            )
            self.assertEqual(
                result.checkpoint["specs"]["architecture_config"],
                source_config.architecture_config,
            )
            self.assertEqual(
                result.checkpoint["metadata"]["architecture"]["name"],
                BASELINE_ARCHITECTURE_NAME,
            )
            self.assertEqual(
                result.source_checkpoint["architecture"]["name"],
                BASELINE_ARCHITECTURE_NAME,
            )
            self.assertTrue(result.source_transfer_gate["transfer_source_gate_met"])
            self.assertTrue(
                result.checkpoint["metadata"]["source_transfer_gate"][
                    "transfer_source_gate_met"
                ]
            )
            self.assertEqual(
                result.checkpoint["architecture"]["name"],
                BASELINE_ARCHITECTURE_NAME,
            )
            self.assertTrue(loaded.source_transfer_gate["transfer_source_gate_met"])
            self.assertEqual(
                loaded.checkpoint["architecture"]["output_contract"],
                BASELINE_ARCHITECTURE_SPEC.output_contract,
            )
            self.assertEqual(result.full_smb_vision_path, full_vision_path)
            self.assertEqual(loaded.output_path, output_path)
            self.assertGreaterEqual(build_model.call_count, 2)
            self.assertEqual(
                build_model.call_args_list[0].args,
                (
                    BASELINE_ARCHITECTURE_NAME,
                    FULL_SMB_SPEC,
                    source_config.architecture_config,
                ),
            )

            source_state = source_model.state_dict()
            transferred_state = result.model.state_dict()
            for key, source_value in source_state.items():
                torch.testing.assert_close(transferred_state[key], source_value)

            stage = FullSMBStage(
                env=TinyFullSMBEnv(),
                vision=result.vision,
                observation_config=FullSMBObservationConfig(
                    frame_skip=1,
                    frame_stack=2,
                    resize_shape=(16, 20),
                ),
            )
            try:
                observation = stage.reset(seed=4)
                batch = stage.encode_observation(observation)
                selection = select_transferred_full_smb_action(
                    result.model,
                    batch,
                    deterministic=True,
                    device="cpu",
                )
            finally:
                stage.close()

            self.assertGreaterEqual(selection.action, 0)
            self.assertLess(selection.action, 6)
            self.assertEqual(selection.logits.shape, (1, 6))
            self.assertTrue(stage.env.closed)

    def test_transfer_rejects_non_block_policy_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "wrong_stage.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            model = make_block_smb_model(tiny_block_config())
            write_full_smb_vision_checkpoint(full_vision_path)
            checkpoint = build_checkpoint(
                stage="full_smb",
                model_name=BLOCK_SMB_MODEL_NAME,
                checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
                states={"model": model.state_dict()},
                specs={
                    "stage": {
                        "name": BLOCK_SMB_SPEC.name,
                        "seq_len_a": BLOCK_SMB_SPEC.seq_len_a,
                        "seq_len_b": BLOCK_SMB_SPEC.seq_len_b,
                        "seq_len_c": BLOCK_SMB_SPEC.seq_len_c,
                        "ratio_bc": BLOCK_SMB_SPEC.ratio_bc,
                        "vocab_size": BLOCK_SMB_SPEC.vocab_size,
                    }
                },
            )
            save_checkpoint(source_policy_path, checkpoint)

            with self.assertRaisesRegex(ValueError, "source policy stage"):
                transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )

    def test_transfer_rejects_block_checkpoint_without_promotion_gate_metrics(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "ungated_block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            config = tiny_block_config(monte_carlo_validation_samples=0)
            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            write_full_smb_vision_checkpoint(full_vision_path)
            save_block_smb_checkpoint(
                source_policy_path,
                model,
                optimizer,
                epoch=1,
                global_step=1,
                config=config,
                metrics={"loss_total": 0.5},
            )
            checkpoint = load_checkpoint(source_policy_path, map_location="cpu")
            gate = block_smb_checkpoint_transfer_source_gate(checkpoint)

            self.assertFalse(gate["transfer_source_gate_met"])
            with self.assertRaisesRegex(ValueError, "held-out Monte Carlo validation gate"):
                transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )

    def test_transfer_rejects_fixed_deterministic_all_noop_action_collapse(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "fixed_noop_block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            config = tiny_block_config()
            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            write_full_smb_vision_checkpoint(full_vision_path)
            save_block_smb_checkpoint(
                source_policy_path,
                model,
                optimizer,
                epoch=1,
                global_step=1,
                config=config,
                metrics=transfer_ready_metrics(
                    eval_fixed_action_count_0=20.0,
                    eval_fixed_action_count_1=0.0,
                    eval_fixed_action_count_2=0.0,
                    eval_fixed_action_count_3=0.0,
                    eval_fixed_action_count_4=0.0,
                    eval_fixed_action_count_5=0.0,
                    eval_fixed_all_noop_action_collapse=1.0,
                ),
            )
            checkpoint = load_checkpoint(source_policy_path, map_location="cpu")
            gate = block_smb_checkpoint_transfer_source_gate(checkpoint)

            self.assertFalse(gate["transfer_source_gate_met"])
            self.assertTrue(gate["fixed_all_noop_action_collapse"])
            self.assertFalse(gate["fixed_action_collapse_gate_met"])
            with self.assertRaisesRegex(ValueError, "fixed deterministic policy collapsed"):
                transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )

    def test_transfer_rejects_monte_carlo_deterministic_all_noop_action_collapse(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "mc_noop_block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            config = tiny_block_config()
            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            write_full_smb_vision_checkpoint(full_vision_path)
            save_block_smb_checkpoint(
                source_policy_path,
                model,
                optimizer,
                epoch=1,
                global_step=1,
                config=config,
                metrics=transfer_ready_metrics(
                    eval_monte_carlo_validation_action_count_0=64.0,
                    eval_monte_carlo_validation_action_count_1=0.0,
                    eval_monte_carlo_validation_action_count_2=0.0,
                    eval_monte_carlo_validation_action_count_3=0.0,
                    eval_monte_carlo_validation_action_count_4=0.0,
                    eval_monte_carlo_validation_action_count_5=0.0,
                    eval_monte_carlo_validation_all_noop_action_collapse=1.0,
                ),
            )
            checkpoint = load_checkpoint(source_policy_path, map_location="cpu")
            gate = block_smb_checkpoint_transfer_source_gate(checkpoint)

            self.assertFalse(gate["transfer_source_gate_met"])
            self.assertTrue(gate["monte_carlo_validation_all_noop_action_collapse"])
            self.assertFalse(gate["monte_carlo_validation_action_collapse_gate_met"])
            with self.assertRaisesRegex(
                ValueError,
                "Monte Carlo validation deterministic policy collapsed",
            ):
                transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )

    def test_transfer_can_explicitly_bypass_source_gate_for_debugging(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "ungated_block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            output_path = tmp / "full_smb_transfer.pth"
            config = tiny_block_config(monte_carlo_validation_samples=0)
            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            write_full_smb_vision_checkpoint(full_vision_path)
            save_block_smb_checkpoint(
                source_policy_path,
                model,
                optimizer,
                epoch=1,
                global_step=1,
                config=config,
                metrics={"loss_total": 0.5},
            )

            result = transfer_block_smb_checkpoint_to_full_smb(
                source_policy_path,
                output_checkpoint=output_path,
                full_smb_vision_checkpoint=full_vision_path,
                block_vision_checkpoint=None,
                device="cpu",
                require_transfer_source_gate=False,
            )

            self.assertTrue(output_path.exists())
            self.assertFalse(result.source_transfer_gate["transfer_source_gate_met"])

    def test_transfer_rejects_incompatible_architecture_contract_before_state_load(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "bad_architecture_contract.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            architecture = incompatible_transfer_architecture()
            write_full_smb_vision_checkpoint(full_vision_path)
            checkpoint = build_checkpoint(
                stage=BLOCK_SMB_SPEC.name,
                model_name=BLOCK_SMB_MODEL_NAME,
                checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
                config={
                    "architecture_name": architecture.name,
                    "architecture_config": {
                        "hidden_dim": 8,
                        "controller_schedule": "linear",
                    },
                },
                states={"model": {"not_a_policy_weight": torch.tensor([1.0])}},
                specs={
                    "stage": {
                        "name": BLOCK_SMB_SPEC.name,
                        "seq_len_a": BLOCK_SMB_SPEC.seq_len_a,
                        "seq_len_b": BLOCK_SMB_SPEC.seq_len_b,
                        "seq_len_c": BLOCK_SMB_SPEC.seq_len_c,
                        "ratio_bc": BLOCK_SMB_SPEC.ratio_bc,
                        "vocab_size": BLOCK_SMB_SPEC.vocab_size,
                    }
                },
            )
            save_checkpoint(source_policy_path, checkpoint)

            with self.assertRaisesRegex(ValueError, "architecture output contract"):
                transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )

    def test_compares_transferred_policy_with_scratch_baseline(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            transfer_path = tmp / "full_smb_transfer.pth"
            report_path = tmp / "comparison.json"
            write_block_policy_checkpoint(source_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)
            transfer_block_smb_checkpoint_to_full_smb(
                source_policy_path,
                output_checkpoint=transfer_path,
                full_smb_vision_checkpoint=full_vision_path,
                block_vision_checkpoint=None,
                device="cpu",
            )

            with patch(
                "retroagi.stages.full_smb.compare.make_full_smb_policy_model",
                wraps=make_full_smb_policy_model,
            ) as make_scratch_model:
                result = compare_transferred_checkpoint_with_scratch(
                    transfer_path,
                    make_stage=lambda vision: FullSMBStage(
                        env=TinyFullSMBEnv(),
                        vision=vision,
                        observation_config=FullSMBObservationConfig(
                            frame_skip=1,
                            frame_stack=2,
                            resize_shape=(16, 20),
                        ),
                    ),
                    full_smb_vision_checkpoint=full_vision_path,
                    config=FullSMBPolicyComparisonConfig(
                        steps=4,
                        seed=9,
                        scratch_seed=99,
                        device="cpu",
                    ),
                )
                self.assertEqual(
                    make_scratch_model.call_args.kwargs["architecture_name"],
                    BASELINE_ARCHITECTURE_NAME,
                )
                self.assertEqual(
                    make_scratch_model.call_args.kwargs["architecture_config"],
                    {"hidden_dim": 8, "controller_schedule": "linear"},
                )
            save_full_smb_policy_comparison(report_path, result)
            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.requested_steps, 4)
        self.assertEqual(result.evaluated_steps, 4)
        self.assertEqual(sum(result.transfer_action_histogram.values()), 4)
        self.assertEqual(sum(result.scratch_action_histogram.values()), 4)
        self.assertGreaterEqual(result.action_agreement, 0.0)
        self.assertLessEqual(result.action_agreement, 1.0)
        self.assertEqual(result.scratch_source, "scratch_initialization")
        self.assertEqual(saved["evaluated_steps"], 4)
        self.assertEqual(saved["scratch_seed"], 99)

    def test_compares_named_policies_on_seeded_task_streams(self):
        observed_tasks = []

        def task_stage(vision, task):
            observed_tasks.append(task.name if task is not None else None)
            return FullSMBStage(
                env=TinyFullSMBEnv(),
                vision=vision,
                observation_config=FullSMBObservationConfig(
                    frame_skip=1,
                    frame_stack=2,
                    resize_shape=(16, 20),
                ),
            )

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            transfer_path = tmp / "full_smb_transfer.pth"
            report_path = tmp / "suite_comparison.json"
            write_block_policy_checkpoint(source_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)
            transfer_block_smb_checkpoint_to_full_smb(
                source_policy_path,
                output_checkpoint=transfer_path,
                full_smb_vision_checkpoint=full_vision_path,
                block_vision_checkpoint=None,
                device="cpu",
            )

            result = compare_full_smb_policy_suite(
                transfer_path,
                make_stage=task_stage,
                scratch_checkpoint=transfer_path,
                fine_tuned_checkpoint=transfer_path,
                known_good_checkpoint=transfer_path,
                full_smb_vision_checkpoint=full_vision_path,
                config=FullSMBPolicySuiteComparisonConfig(
                    steps=3,
                    seeds=(5, 6),
                    scratch_seed=99,
                    device="cpu",
                ),
                task_names=("benchmark_1_1_start", "benchmark_1_2_start"),
            )
            save_full_smb_policy_comparison(report_path, result)
            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            observed_tasks,
            [
                "benchmark_1_1_start",
                "benchmark_1_1_start",
                "benchmark_1_2_start",
                "benchmark_1_2_start",
            ],
        )
        self.assertEqual(result.seeds, (5, 6))
        self.assertEqual(
            result.task_names,
            ("benchmark_1_1_start", "benchmark_1_2_start"),
        )
        self.assertEqual(result.requested_steps_per_stream, 3)
        self.assertEqual(result.evaluated_steps, 12)
        self.assertEqual(len(result.streams), 4)
        self.assertEqual(
            set(result.policies),
            {"transferred", "scratch_trained", "fine_tuned", "known_good"},
        )
        self.assertEqual(result.policies["scratch_trained"]["source"], "scratch_checkpoint")
        self.assertIn("transferred_vs_scratch_trained", result.aggregate_pairwise)
        self.assertEqual(
            result.aggregate_pairwise["transferred_vs_scratch_trained"]["compared_steps"],
            12,
        )
        self.assertEqual(saved["evaluated_steps"], 12)
        self.assertEqual(saved["streams"][0]["task"]["name"], "benchmark_1_1_start")


if __name__ == "__main__":
    unittest.main()
