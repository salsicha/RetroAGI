"""Tests for the progressive-resolution promotion pipeline."""

import io
import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from retroagi import promotion
from retroagi.core import BASELINE_ARCHITECTURE_NAME, build_game_promotion_plan, get_game_plugin
from scripts.tests.test_full_smb_transfer import write_block_policy_checkpoint


def _fake_experiment_manifest(
    args: Namespace,
    *,
    passed: bool = True,
    metrics: dict[str, float] | None = None,
    write_artifacts: bool = True,
):
    stage = args.stage[0]
    if metrics is None:
        metrics = (
            {"controller_mse": 0.5}
            if stage == "synthetic-1d"
            else {"eval_success_rate": 0.0, "gradient_norm": 1.0}
        )
    summary_path = args.artifacts_dir / "run_summary.json"
    checkpoint_path = args.artifacts_dir / "checkpoint.pth"
    log_path = args.artifacts_dir / "events.jsonl" if stage == "block-smb" else None
    if write_artifacts:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}", encoding="utf-8")
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        if log_path is not None:
            log_path.write_text("", encoding="utf-8")
    return {
        "architecture": {
            "name": args.architecture_name,
            "config": dict(args.architecture_config or ()),
        },
        "seed": args.seed,
        "device": args.device,
        "artifacts_dir": str(args.artifacts_dir),
        "stages": [
            {
                "stage": stage,
                "command": ["retroagi", "train", "--stage", stage],
                "summary_path": str(summary_path),
                "checkpoint_path": str(checkpoint_path),
                "log_path": str(log_path) if log_path is not None else None,
                "exit_code": 0 if passed else 1,
                "config": {},
                "metrics": metrics,
                "gates": [{"metric": next(iter(metrics), "metric"), "passed": passed}],
                "passed": passed,
            }
        ],
        "gates": [],
        "passed": passed,
    }


class TestPromotionPipeline(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with redirect_stdout(stream):
            exit_code = promotion.main(argv)
        return exit_code, json.loads(stream.getvalue())

    def test_interface_smoke_runs_each_baseline_stage(self):
        with TemporaryDirectory() as tmpdir:
            exit_code, manifest = self.run_main(
                [
                    "--rung",
                    "interface-smoke",
                    "--game",
                    "smb",
                    "--output",
                    str(Path(tmpdir) / "promotion.json"),
                    "--artifacts-dir",
                    str(Path(tmpdir) / "artifacts"),
                    "--device",
                    "cpu",
                    "--seed",
                    "3",
                    "--architecture-config",
                    "hidden_dim=8",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(manifest["passed"])
        self.assertEqual(manifest["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(manifest["architecture"]["config"], {"hidden_dim": 8})
        self.assertEqual(manifest["game"]["name"], "smb")
        self.assertEqual(manifest["game"]["family"], "super_mario_bros")
        self.assertEqual(
            [phase["name"] for phase in manifest["game_promotion"]["phases"]],
            [
                "architecture-smoke",
                "game-synthetic",
                "game-block",
                "game-full-smoke",
                "game-transfer",
                "game-full-comparison",
                "game-full-training",
            ],
        )
        self.assertEqual(manifest["budget"]["name"], "small")
        self.assertEqual(manifest["rungs"][0]["name"], "interface-smoke")
        self.assertEqual(manifest["rungs"][0]["status"], "passed")
        self.assertEqual(manifest["rungs"][0]["budget"], {"batch_size": 2, "runtime_seconds": 10.0})
        self.assertTrue(all(gate["passed"] for gate in manifest["rungs"][0]["automatic_gates"]))
        self.assertEqual(
            [stage["stage"] for stage in manifest["rungs"][0]["stages"]],
            ["synthetic_1d", "block_smb", "full_smb"],
        )
        self.assertTrue(all(stage["passed"] for stage in manifest["rungs"][0]["stages"]))

    def test_smb_game_promotion_plan_maps_phases_to_architecture_rungs(self):
        plan = build_game_promotion_plan(get_game_plugin("smb"))

        self.assertEqual(plan.game_name, "smb")
        self.assertEqual(plan.phase("architecture-smoke").architecture_rungs, ("interface-smoke",))
        self.assertEqual(plan.phase("game-synthetic").stage_name, "synthetic")
        self.assertEqual(plan.phase("game-synthetic").architecture_rungs, ("synthetic-concept",))
        self.assertEqual(plan.phase("game-block").stage_name, "block")
        self.assertEqual(plan.phase("game-block").architecture_rungs, ("block-smb-smoke",))
        self.assertEqual(
            plan.phase("game-full-smoke").architecture_rungs,
            ("full-smb-asset-mock-perception", "full-smb-transfer-smoke"),
        )
        self.assertEqual(
            plan.phase("game-full-comparison").architecture_rungs,
            ("full-smb-transfer-vs-scratch",),
        )

    def test_default_pipeline_records_supported_and_future_rungs(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "promotion.json"

            def fake_run_experiment(args):
                return _fake_experiment_manifest(args)

            with (
                patch("retroagi.experiments.run_experiment", side_effect=fake_run_experiment),
                patch(
                    "retroagi.promotion._run_full_smb_asset_mock_perception",
                    return_value={
                        "name": "full-smb-asset-mock-perception",
                        "status": "passed",
                        "passed": True,
                        "budget": {
                            "train_scenes": 8,
                            "validation_scenes": 4,
                            "epochs": 8,
                            "semantic_accuracy_threshold": 0.05,
                            "foreground_accuracy_threshold": 0.01,
                            "mean_iou_threshold": 0.01,
                            "position_within_tolerance_threshold": 0.0,
                            "position_tolerance": 0.35,
                            "runtime_seconds": 120.0,
                        },
                        "runtime_seconds": 0.01,
                        "automatic_gates": [{"name": "mock", "passed": True}],
                    },
                ),
                patch(
                    "retroagi.promotion._run_full_smb_transfer_smoke",
                    return_value={
                        "name": "full-smb-transfer-smoke",
                        "status": "passed",
                        "passed": True,
                        "budget": {
                            "steps": 32,
                            "seeds": 1,
                            "runtime_seconds": 120.0,
                        },
                        "runtime_seconds": 0.01,
                        "automatic_gates": [{"name": "mock", "passed": True}],
                    },
                ),
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--output",
                        str(output),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                        "--architecture-config",
                        "hidden_dim=8",
                    ]
                )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written, manifest)
        self.assertTrue(manifest["passed"])
        rung_statuses = {rung["name"]: rung["status"] for rung in manifest["rungs"]}
        self.assertEqual(rung_statuses["interface-smoke"], "passed")
        self.assertEqual(rung_statuses["synthetic-concept"], "passed")
        self.assertEqual(rung_statuses["block-smb-smoke"], "passed")
        self.assertEqual(rung_statuses["full-smb-asset-mock-perception"], "passed")
        self.assertEqual(rung_statuses["full-smb-transfer-smoke"], "passed")
        self.assertEqual(rung_statuses["full-smb-fine-tuning"], "skipped")
        self.assertIn("reason", manifest["rungs"][-1])
        self.assertEqual(manifest["budget"]["name"], "small")
        self.assertEqual(manifest["budget"]["rungs"]["block-smb-smoke"]["rollout_steps"], 2)
        self.assertEqual(
            manifest["rungs"][-1]["budget"],
            {
                "epochs": 1,
                "rollout_steps": 128,
                "evaluation_episodes": 2,
                "runtime_seconds": 600.0,
            },
        )
        for rung in manifest["rungs"]:
            if rung["status"] == "passed":
                self.assertTrue(all(gate["passed"] for gate in rung["automatic_gates"]))

    def test_full_smb_asset_mock_perception_trains_and_gates_vit_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, manifest = self.run_main(
                [
                    "--rung",
                    "full-smb-asset-mock-perception",
                    "--output",
                    str(root / "promotion.json"),
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--device",
                    "cpu",
                ]
            )
            checkpoint_exists = Path(
                manifest["rungs"][0]["artifacts"]["full_smb_vision_checkpoint_path"]
            ).exists()
            summary_payload = json.loads(Path(manifest["rungs"][0]["summary_path"]).read_text())

        self.assertEqual(exit_code, 0)
        self.assertTrue(manifest["passed"])
        rung = manifest["rungs"][0]
        self.assertEqual(rung["name"], "full-smb-asset-mock-perception")
        self.assertEqual(rung["status"], "passed")
        self.assertGreaterEqual(rung["metrics"]["accuracy"], 0.05)
        self.assertGreaterEqual(rung["metrics"]["foreground_accuracy"], 0.01)
        self.assertGreaterEqual(rung["metrics"]["class_coverage"], 13.0)
        self.assertIn("mock_assets", summary_payload)
        self.assertTrue(all(gate["passed"] for gate in rung["automatic_gates"]))
        self.assertTrue(checkpoint_exists)

    def test_full_smb_transfer_smoke_runs_handoff_and_continued_training(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_checkpoint = root / "artifacts" / "block_smb_smoke" / "checkpoint.pth"
            source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            write_block_policy_checkpoint(source_checkpoint)

            exit_code, manifest = self.run_main(
                [
                    "--rung",
                    "full-smb-transfer-smoke",
                    "--output",
                    str(root / "promotion.json"),
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--device",
                    "cpu",
                    "--architecture-config",
                    "hidden_dim=8",
                ]
            )
            transfer_exists = Path(
                manifest["rungs"][0]["artifacts"]["transfer_checkpoint_path"]
            ).exists()
            continued_exists = Path(
                manifest["rungs"][0]["artifacts"]["continued_checkpoint_path"]
            ).exists()
            summary_exists = Path(manifest["rungs"][0]["summary_path"]).exists()
            summary_payload = json.loads(
                Path(manifest["rungs"][0]["summary_path"]).read_text()
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(manifest["passed"])
        rung = manifest["rungs"][0]
        self.assertEqual(rung["name"], "full-smb-transfer-smoke")
        self.assertEqual(rung["status"], "passed")
        self.assertGreater(rung["metrics"]["continued_global_step"], 0)
        self.assertGreaterEqual(rung["metrics"]["deterministic_action"], 0)
        self.assertGreater(rung["metrics"]["controller_transfer_key_count"], 0)
        self.assertEqual(rung["metrics"]["controller_transfer_max_abs_delta"], 0.0)
        self.assertIn("controller_inference_w_mean", rung["metrics"])
        self.assertGreater(rung["metrics"]["controller_adaptation_key_count"], 0)
        self.assertGreater(rung["metrics"]["controller_adaptation_changed_tensors"], 0)
        self.assertIn("controller", summary_payload["inference"])
        self.assertTrue(all(gate["passed"] for gate in rung["automatic_gates"]))
        self.assertTrue(transfer_exists)
        self.assertTrue(continued_exists)
        self.assertTrue(summary_exists)

    def test_medium_budget_changes_experiment_arguments(self):
        calls = []

        def fake_run_experiment(args):
            calls.append(args)
            return _fake_experiment_manifest(args)

        with TemporaryDirectory() as tmpdir:
            with patch("retroagi.experiments.run_experiment", side_effect=fake_run_experiment):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "synthetic-concept",
                        "--budget",
                        "medium",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["budget"]["name"], "medium")
        self.assertEqual(calls[0].synthetic_epochs, 3)
        self.assertEqual(calls[0].synthetic_train_samples, 128)
        self.assertEqual(calls[0].gate[0].threshold, 5.0)
        self.assertEqual(manifest["rungs"][0]["budget"]["epochs"], 3)

    def test_cli_overrides_budget_values(self):
        calls = []

        def fake_run_experiment(args):
            calls.append(args)
            return _fake_experiment_manifest(args)

        with TemporaryDirectory() as tmpdir:
            with patch("retroagi.experiments.run_experiment", side_effect=fake_run_experiment):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "block-smb-smoke",
                        "--budget",
                        "medium",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                        "--block-smoke-rollout-steps",
                        "3",
                        "--block-smoke-success-rate",
                        "0.25",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls[0].block_rollout_steps, 3)
        self.assertEqual(calls[0].gate[0].threshold, 0.25)
        self.assertEqual(manifest["rungs"][0]["budget"]["rollout_steps"], 3)
        self.assertEqual(manifest["rungs"][0]["budget"]["success_rate_threshold"], 0.25)
        self.assertEqual(manifest["rungs"][0]["budget"]["runtime_seconds"], 180.0)

    def test_game_phase_rung_alias_selects_underlying_architecture_rung(self):
        calls = []

        def fake_run_experiment(args):
            calls.append(args)
            return _fake_experiment_manifest(args)

        with TemporaryDirectory() as tmpdir:
            with patch("retroagi.experiments.run_experiment", side_effect=fake_run_experiment):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "game-synthetic",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual([rung["name"] for rung in manifest["rungs"]], ["synthetic-concept"])
        self.assertEqual(calls[0].stage, ["synthetic-1d"])

    def test_game_full_smoke_phase_alias_runs_perception_and_transfer(self):
        with TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "retroagi.promotion._run_full_smb_asset_mock_perception",
                    return_value={
                        "name": "full-smb-asset-mock-perception",
                        "status": "passed",
                        "passed": True,
                        "budget": {},
                        "automatic_gates": [],
                    },
                ) as perception,
                patch(
                    "retroagi.promotion._run_full_smb_transfer_smoke",
                    return_value={
                        "name": "full-smb-transfer-smoke",
                        "status": "passed",
                        "passed": True,
                        "budget": {},
                        "automatic_gates": [],
                    },
                ) as transfer,
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "game-full-smoke",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [rung["name"] for rung in manifest["rungs"]],
            ["full-smb-asset-mock-perception", "full-smb-transfer-smoke"],
        )
        full_smoke = next(
            phase
            for phase in manifest["game_promotion"]["phases"]
            if phase["name"] == "game-full-smoke"
        )
        self.assertEqual(
            full_smoke["rung_statuses"],
            {
                "full-smb-asset-mock-perception": "passed",
                "full-smb-transfer-smoke": "passed",
            },
        )
        perception.assert_called_once()
        transfer.assert_called_once()

    def test_promotion_passes_architecture_ablations_to_experiment_runner(self):
        calls = []

        def fake_run_experiment(args):
            calls.append(args)
            return _fake_experiment_manifest(args)

        with TemporaryDirectory() as tmpdir:
            with patch("retroagi.experiments.run_experiment", side_effect=fake_run_experiment):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "block-smb-smoke",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                        "--architecture-config",
                        "hidden_dim=8",
                        "--ablation",
                        "controller_schedule=linear",
                        "--ablation",
                        "world_model=off",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            manifest["architecture"]["config"],
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )
        self.assertFalse(manifest["architecture_variant"]["ablation"]["world_model_enabled"])
        self.assertEqual(
            calls[0].architecture_config,
            [("hidden_dim", 8), ("controller_schedule", "linear")],
        )
        self.assertIn(("world_model_enabled", False), calls[0].ablation)

    def test_failed_experiment_rung_fails_promotion(self):
        with TemporaryDirectory() as tmpdir:
            with patch(
                "retroagi.experiments.run_experiment",
                side_effect=lambda args: _fake_experiment_manifest(args, passed=False),
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "synthetic-concept",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertFalse(manifest["passed"])
        self.assertEqual(manifest["rungs"][0]["status"], "failed")

    def test_missing_required_metric_stops_following_rungs(self):
        with TemporaryDirectory() as tmpdir:
            with patch(
                "retroagi.experiments.run_experiment",
                side_effect=lambda args: _fake_experiment_manifest(args, metrics={}),
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "synthetic-concept",
                        "--rung",
                        "block-smb-smoke",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertFalse(manifest["passed"])
        self.assertEqual([rung["status"] for rung in manifest["rungs"]], ["failed", "stopped"])
        required_gate = next(
            gate
            for gate in manifest["rungs"][0]["automatic_gates"]
            if gate["name"] == "required-metric:controller_mse"
        )
        self.assertFalse(required_gate["passed"])
        self.assertIn("synthetic-concept failed", manifest["rungs"][1]["reason"])

    def test_missing_artifact_fails_automatic_gate(self):
        with TemporaryDirectory() as tmpdir:
            with patch(
                "retroagi.experiments.run_experiment",
                side_effect=lambda args: _fake_experiment_manifest(args, write_artifacts=False),
            ):
                exit_code, manifest = self.run_main(
                    [
                        "--rung",
                        "synthetic-concept",
                        "--output",
                        str(Path(tmpdir) / "promotion.json"),
                        "--artifacts-dir",
                        str(Path(tmpdir) / "artifacts"),
                        "--device",
                        "cpu",
                    ]
                )

        self.assertEqual(exit_code, 1)
        artifact_gates = [
            gate for gate in manifest["rungs"][0]["automatic_gates"] if gate["kind"] == "artifact"
        ]
        self.assertTrue(artifact_gates)
        self.assertTrue(any(not gate["passed"] for gate in artifact_gates))


if __name__ == "__main__":
    unittest.main()
