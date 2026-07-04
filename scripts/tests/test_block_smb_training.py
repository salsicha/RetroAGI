"""Tests for Block SMB trainer plumbing."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    VisionOutput,
    VisionSpec,
    build_architecture,
    build_checkpoint,
    checkpoint_summary_path,
    load_checkpoint,
    save_checkpoint,
)
from retroagi.stages.block_smb import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MC_DIFFICULTY_BINS,
    BLOCK_SMB_MC_FAMILIES,
    BLOCK_SMB_MODEL_NAME,
    BLOCK_SMB_SPEC,
    ROUTINE_BLOCK_SMB_MC_REQUIRED_TRAIN_FAMILIES,
    BlockSMBAblationConfig,
    BlockSMBRewardConfig,
    BlockSMBStage,
    BlockSMBTrainingConfig,
    MarioScenarioEnv,
    SequentialBlockSMBVectorEnv,
    build_adaptive_monte_carlo_replay_curriculum,
    build_curriculum,
    build_epoch_curriculum,
    block_smb_monte_carlo_train_sample_count,
    evaluate_block_smb,
    evaluate_block_smb_monte_carlo,
    restore_block_smb_checkpoint,
    routine_block_smb_monte_carlo_train_min_sample_count,
    summarize_block_smb_curriculum,
    train_and_evaluate_block_smb,
)
from retroagi.stages.block_smb.train import (
    apply_block_smb_ablations,
    block_smb_c_stream_slot_spans,
    block_smb_noop_allowed_for_step,
    block_smb_noop_suppression_loss,
    collect_trajectory,
    compute_block_smb_losses,
    compute_imagined_rollout_losses,
    make_block_smb_model,
    make_target_network,
    save_block_smb_checkpoint,
    target_network_parameter_delta,
    train_block_smb_epoch,
    update_target_network,
)


class StaticBlockVision:
    spec = VisionSpec(
        name="static_block_trainer",
        semantic_classes=(
            "background",
            "mario",
            "platform",
            "coin",
            "goal",
            "enemy",
            "moving_platform",
        ),
        token_dim=4,
    )

    def encode(self, observation):
        logits = torch.full((1, self.spec.num_classes, 2, 16), -8.0)
        logits[:, 1, :, 1] = 8.0
        logits[:, 2, :, :] = torch.maximum(logits[:, 2, :, :], torch.tensor(1.0))
        return VisionOutput(
            position=torch.tensor([[0.1, 0.8]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.zeros(1, 240, self.spec.token_dim),
            metadata={},
        )


def static_vision_factory():
    return StaticBlockVision()


def tiny_config(**overrides):
    values = dict(
        seed=7,
        epochs=1,
        episodes_per_epoch=1,
        rollout_steps=2,
        hidden_dim=8,
        evaluation_episodes=1,
        evaluation_max_steps=2,
        fixed_scenarios=("level_1_flat.json",),
        generated_scenarios=1,
        device="cpu",
    )
    values.update(overrides)
    return BlockSMBTrainingConfig(**values)


class TestBlockSMBTraining(unittest.TestCase):
    def test_curriculum_and_sequential_vector_env_are_deterministic(self):
        config = tiny_config(generated_scenarios=2)
        curriculum = build_curriculum(config)
        names = [name for name, _scenario in curriculum]
        self.assertEqual(names[0], "level_1_flat.json")
        self.assertEqual(len(names), 3)
        self.assertTrue(names[1].startswith("block_smb_mc_v1.train.50000.000000."))
        self.assertTrue(names[2].startswith("block_smb_mc_v1.train.50000.000001."))
        summary = summarize_block_smb_curriculum(curriculum)
        self.assertEqual(summary["fixed_scenario_count"], 1)
        self.assertEqual(summary["monte_carlo_sample_count"], 2)
        self.assertEqual(
            summary["monte_carlo"]["family_counts"],
            {BLOCK_SMB_MC_FAMILIES[0]: 1, BLOCK_SMB_MC_FAMILIES[1]: 1},
        )

        vector_env = SequentialBlockSMBVectorEnv(curriculum, num_envs=2)
        try:
            resets = vector_env.reset(seed=11)
            self.assertEqual(len(resets), 2)
            steps = vector_env.step([0, 1])
            self.assertEqual(len(steps), 2)
            for observation, reward, terminated, truncated, info in steps:
                self.assertEqual(observation.shape, (240, 256, 3))
                self.assertIsInstance(float(reward), float)
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIn("state_vec", info)
                self.assertEqual(info["state_vec"].shape, (27,))
        finally:
            vector_env.close()

    def test_explicit_monte_carlo_training_count_covers_routine_chained_families(self):
        config = tiny_config(
            generated_scenarios=0,
            monte_carlo_train_samples_per_epoch=8,
        )
        minimum_count = routine_block_smb_monte_carlo_train_min_sample_count()
        curriculum = build_curriculum(config)
        summary = summarize_block_smb_curriculum(curriculum)

        self.assertGreater(minimum_count, 8)
        self.assertEqual(block_smb_monte_carlo_train_sample_count(config), minimum_count)
        self.assertEqual(summary["monte_carlo_sample_count"], minimum_count)
        for family in ROUTINE_BLOCK_SMB_MC_REQUIRED_TRAIN_FAMILIES:
            self.assertEqual(summary["monte_carlo"]["family_counts"][family], 1)

    def test_weighted_monte_carlo_training_count_preserves_requested_focus(self):
        config = tiny_config(
            generated_scenarios=0,
            monte_carlo_train_samples_per_epoch=4,
            monte_carlo_family_weights={"flat_run": 1.0},
        )
        curriculum = build_curriculum(config)
        summary = summarize_block_smb_curriculum(curriculum)

        self.assertEqual(block_smb_monte_carlo_train_sample_count(config), 4)
        self.assertEqual(summary["monte_carlo_sample_count"], 4)
        self.assertEqual(summary["monte_carlo"]["family_counts"], {"flat_run": 4})

    def test_controller_schedule_configures_block_smb_model(self):
        config = tiny_config(controller_schedule="linear")
        with patch(
            "retroagi.stages.block_smb.train.build_architecture",
            wraps=build_architecture,
        ) as build_model:
            model = make_block_smb_model(config)

        self.assertEqual(model.agent.controller.schedule, "linear")
        self.assertEqual(config.architecture_name, BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            config.architecture_config,
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )
        build_model.assert_called_once_with(
            BASELINE_ARCHITECTURE_NAME,
            BLOCK_SMB_SPEC,
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )

        with self.assertRaisesRegex(ValueError, "controller_schedule"):
            tiny_config(controller_schedule="quadratic")
        with self.assertRaisesRegex(ValueError, "architecture_name"):
            tiny_config(architecture_name="")
        with self.assertRaisesRegex(ValueError, "architecture_config"):
            tiny_config(architecture_config={"": 8})

        with self.assertRaisesRegex(ValueError, "target_network_mode"):
            tiny_config(target_network_mode="sometimes")
        with self.assertRaisesRegex(ValueError, "target_network_tau"):
            tiny_config(target_network_tau=1.5)

    def test_collect_trajectory_records_episode_masks(self):
        config = tiny_config(generated_scenarios=0)
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=2,
                seed=3,
                deterministic=True,
                device=torch.device("cpu"),
                record_frames=True,
            )
        finally:
            stage.env.close()

        self.assertGreaterEqual(len(trajectory.transitions), 1)
        self.assertEqual(len(trajectory.frames), len(trajectory.transitions) + 1)
        for step in trajectory.transitions:
            self.assertIn(step.episode_mask, (0.0, 1.0))
            self.assertEqual(step.batch.src_c.shape, (1, BLOCK_SMB_SPEC.seq_len_c))

    def test_noop_allowed_is_step_local_for_wait_scenarios(self):
        wait_scenario = {
            "metadata": {
                "block_smb_monte_carlo": {
                    "oracle": {"actions": [0, 0, 1, 2]},
                },
            },
        }

        self.assertTrue(
            block_smb_noop_allowed_for_step("level_12_wait_bridge.json", {}, 19)
        )
        self.assertFalse(
            block_smb_noop_allowed_for_step("level_12_wait_bridge.json", {}, 20)
        )
        self.assertTrue(block_smb_noop_allowed_for_step("mc.wait_timing", wait_scenario, 1))
        self.assertFalse(block_smb_noop_allowed_for_step("mc.wait_timing", wait_scenario, 2))

    def test_noop_suppression_loss_penalizes_non_wait_noop_logits(self):
        config = tiny_config(generated_scenarios=0)
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=1,
                seed=3,
                deterministic=True,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()
        step = trajectory.transitions[0]
        step.logits_a = torch.full_like(step.logits_a, -4.0)
        step.logits_a[:, -1, 0] = 6.0
        step.noop_allowed = False

        loss = block_smb_noop_suppression_loss(step, device=torch.device("cpu"))
        self.assertGreater(loss.item(), 1.0)

        step.noop_allowed = True
        exempt_loss = block_smb_noop_suppression_loss(step, device=torch.device("cpu"))
        self.assertEqual(exempt_loss.item(), 0.0)

    def test_imagined_rollout_loss_unrolls_within_trajectory(self):
        config = tiny_config(
            generated_scenarios=0,
            rollout_steps=3,
            imagined_rollout_horizon=2,
            imagined_rollout_weight=0.2,
        )
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=3,
                seed=4,
                deterministic=True,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()

        imagined = compute_imagined_rollout_losses(
            model,
            [trajectory],
            config,
            torch.device("cpu"),
        )
        losses = compute_block_smb_losses(
            model,
            trajectory.transitions,
            config,
            torch.device("cpu"),
            trajectories=[trajectory],
        )

        self.assertGreater(imagined["imagined_rollout_steps"].item(), 0.0)
        for key in (
            "loss_imagined_dynamics",
            "loss_imagined_reward",
            "loss_imagined_rollout",
            "imagined_rollout_steps",
        ):
            self.assertTrue(torch.isfinite(imagined[key]).item())
            self.assertTrue(torch.isfinite(losses[key]).item())
        for slot_name in (
            "position",
            "semantic_probabilities",
            "support_state",
            "state",
            "terminal_outcome",
            "patch_tokens",
        ):
            self.assertIn(f"loss_dynamics_{slot_name}", losses)
            self.assertIn(f"dynamics_{slot_name}_rmse", losses)
            self.assertIn(f"dynamics_{slot_name}_mae", losses)
            self.assertGreaterEqual(losses[f"loss_dynamics_{slot_name}"].item(), 0.0)
        self.assertIn("dynamics_semantic_prediction_accuracy", losses)
        self.assertIn("dynamics_semantic_prediction_gate_met", losses)
        self.assertGreaterEqual(losses["dynamics_semantic_prediction_accuracy"].item(), 0.0)
        self.assertLessEqual(losses["dynamics_semantic_prediction_accuracy"].item(), 1.0)
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())

    def test_world_model_ablation_bypasses_dynamics_and_imagination(self):
        config = tiny_config(
            generated_scenarios=0,
            rollout_steps=2,
            imagined_rollout_horizon=2,
            imagined_rollout_weight=0.5,
            world_model_weight=10.0,
            ablation=BlockSMBAblationConfig(world_model_enabled=False),
        )
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=2,
                seed=6,
                deterministic=True,
                device=torch.device("cpu"),
                ablation=config.ablation,
            )
        finally:
            stage.env.close()

        self.assertGreaterEqual(len(trajectory.transitions), 1)
        for step in trajectory.transitions:
            torch.testing.assert_close(step.next_state_pred, step.batch.src_c)
        losses = compute_block_smb_losses(
            model,
            trajectory.transitions,
            config,
            torch.device("cpu"),
            trajectories=[trajectory],
        )

        self.assertEqual(losses["imagined_rollout_steps"].item(), 0.0)
        self.assertEqual(losses["loss_imagined_rollout"].item(), 0.0)
        self.assertGreaterEqual(losses["loss_dynamics"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss_total"]).item())

    def test_train_epoch_clears_replay_tensors_before_returning(self):
        config = tiny_config(generated_scenarios=0, rollout_steps=1)
        model = make_block_smb_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        curriculum = build_curriculum(config)

        metrics, replay = train_block_smb_epoch(
            model,
            optimizer,
            curriculum,
            config,
            epoch=0,
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
        )

        self.assertEqual(metrics["episodes"], 1.0)
        self.assertEqual(replay.trajectories, [])
        self.assertEqual(replay.transitions(), [])

    def test_target_network_auto_activation_and_ema_update(self):
        config = tiny_config(
            generated_scenarios=0,
            target_network_mode="auto",
            target_network_instability_threshold=0.0,
            target_network_tau=0.5,
        )
        model = make_block_smb_model(config)
        target_model = make_target_network(model)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=2,
                seed=5,
                deterministic=True,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()

        losses = compute_block_smb_losses(
            model,
            trajectory.transitions,
            config,
            torch.device("cpu"),
            trajectories=[trajectory],
            target_model=target_model,
        )
        self.assertEqual(losses["target_network_active"].item(), 1.0)
        self.assertGreaterEqual(losses["target_network_instability"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["target_network_drift"]).item())

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
                break
        before = target_network_parameter_delta(model, target_model, torch.device("cpu"))
        update_target_network(target_model, model, tau=0.5)
        after = target_network_parameter_delta(model, target_model, torch.device("cpu"))

        self.assertGreater(before.item(), after.item())

    def test_block_smb_ablations_mask_expected_hierarchy_slots(self):
        config = tiny_config(generated_scenarios=0)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            observation = stage.reset(seed=5)
            batch = stage.encode_observation(observation)
        finally:
            stage.env.close()

        fusion = batch.metadata["vision_fusion"]
        state_start, state_end = fusion["c_state"]
        visual = apply_block_smb_ablations(
            batch,
            BlockSMBAblationConfig(vision_enabled=False),
        )
        hierarchy = apply_block_smb_ablations(
            batch,
            BlockSMBAblationConfig(hierarchy_enabled=False),
        )

        self.assertTrue(torch.equal(visual.src_a, torch.zeros_like(batch.src_a)))
        self.assertTrue(torch.equal(visual.src_b, torch.zeros_like(batch.src_b)))
        for slot in (
            "c_position",
            "c_semantic_probabilities",
            "c_support_state",
            "c_patch_tokens",
        ):
            start, end = fusion[slot]
            torch.testing.assert_close(
                visual.src_c[:, start:end],
                torch.zeros_like(batch.src_c[:, start:end]),
            )
        torch.testing.assert_close(
            visual.src_c[:, state_start:state_end],
            batch.src_c[:, state_start:state_end],
        )
        self.assertTrue(torch.equal(hierarchy.src_a, torch.zeros_like(batch.src_a)))
        self.assertTrue(torch.equal(hierarchy.src_b, torch.zeros_like(batch.src_b)))
        torch.testing.assert_close(hierarchy.src_c, batch.src_c)
        self.assertFalse(visual.metadata["ablation"]["vision_enabled"])
        self.assertFalse(hierarchy.metadata["ablation"]["hierarchy_enabled"])

    def test_terminal_death_transition_is_encoded_as_lstm_target(self):
        stage = BlockSMBStage(
            scenario={
                "mario": [20, 200],
                "platforms": [[0, 220, 256, 20]],
                "enemies": [[20, 206, 20, 20, 0]],
                "world_width": 256,
            },
            vision=StaticBlockVision(),
            env=MarioScenarioEnv(
                reward_config=BlockSMBRewardConfig(fall_death=0.0, enemy_hit=0.0),
            ),
        )
        try:
            observation = stage.reset(seed=5)
            next_observation, _reward, terminated, truncated, info = stage.step(0)
            next_batch = stage.encode_observation(next_observation, info)
        finally:
            stage.env.close()

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["death"])
        self.assertEqual(info["state_vec"][-3:].tolist(), [1.0, 1.0, 0.0])
        spans = block_smb_c_stream_slot_spans(next_batch)
        terminal_start, terminal_end = spans["terminal_outcome"]
        torch.testing.assert_close(
            next_batch.src_c[:, terminal_start:terminal_end],
            torch.tensor([[1.0, 1.0, 0.0]], dtype=next_batch.src_c.dtype),
        )
        self.assertEqual(next_batch.metadata["episode"]["mask"].item(), 0.0)

    def test_train_evaluate_checkpoint_and_recording_smoke(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            checkpoint = tmp / "block_smb.pth"
            video_dir = tmp / "videos"
            config = tiny_config(
                checkpoint_path=checkpoint,
                save_checkpoints=True,
                video_dir=video_dir,
                record_videos=True,
                reward_config=BlockSMBRewardConfig(
                    progress_per_pixel=0.08,
                    coin=8.0,
                    enemy_stomp=4.0,
                    goal=70.0,
                    fall_death=-12.0,
                    enemy_hit=-12.0,
                    frame_penalty=-0.02,
                ),
            )

            result = train_and_evaluate_block_smb(config, vision_factory=static_vision_factory)

            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["stage"], BLOCK_SMB_SPEC.name)
            self.assertEqual(saved["model_name"], BLOCK_SMB_MODEL_NAME)
            self.assertEqual(saved["checkpoint_kind"], BLOCK_SMB_CHECKPOINT_KIND)
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(saved["global_step"], 1)
            self.assertEqual(saved["config"]["reward_config"]["goal"], 70.0)
            self.assertEqual(saved["config"]["architecture_name"], BASELINE_ARCHITECTURE_NAME)
            self.assertEqual(
                saved["config"]["architecture_config"],
                {"hidden_dim": 8, "controller_schedule": "constant"},
            )
            self.assertEqual(saved["specs"]["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
            self.assertEqual(
                saved["specs"]["architecture_config"],
                {"hidden_dim": 8, "controller_schedule": "constant"},
            )
            self.assertEqual(result["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
            self.assertEqual(
                result["architecture"]["config"],
                {"hidden_dim": 8, "controller_schedule": "constant"},
            )
            self.assertEqual(result["curriculum_summary"]["monte_carlo_sample_count"], 1)
            evaluation = result["evaluation"]
            self.assertIn("level_1_flat.json", evaluation["fixed_scenarios"])
            self.assertIn("tuning_metrics", evaluation)
            self.assertIn("action_counts", evaluation)
            self.assertIn("action_collapse", evaluation)
            self.assertFalse(evaluation["success_thresholds_met"])
            level_result = evaluation["fixed_scenarios"]["level_1_flat.json"]
            self.assertIn("threshold", level_result)
            self.assertIn("threshold_diagnostics", level_result)
            self.assertIn("action_counts", level_result)
            self.assertFalse(level_result["threshold_met"])
            self.assertTrue((video_dir / "level_1_flat.json_episode0.npz").exists())
            for key in (
                "loss_representation",
                "loss_dynamics",
                "loss_reward",
                "loss_value",
                "loss_policy",
                "loss_noop",
                "loss_critic_feedback",
                "loss_imagined_dynamics",
                "loss_imagined_reward",
                "loss_imagined_rollout",
                "imagined_rollout_steps",
                "target_network_active",
                "target_network_instability",
                "target_network_drift",
                "target_network_tau",
                "loss_actor_pass1",
                "loss_actor_pass2",
                "loss_world_model",
                "loss_critic",
                "loss_total",
                "gradient_norm",
                "eval_threshold_pass_rate",
                "eval_tuning_score",
                "eval_fixed_action_count_0",
                "eval_fixed_action_count_1",
                "eval_fixed_action_count_2",
                "eval_fixed_action_count_3",
                "eval_fixed_action_count_4",
                "eval_fixed_action_count_5",
                "eval_fixed_all_noop_action_collapse",
            ):
                self.assertTrue(torch.isfinite(torch.tensor(result["metrics"][key])).item())

            resumed_config = tiny_config(
                epochs=2,
                resume_path=checkpoint,
                checkpoint_path=checkpoint,
                save_checkpoints=True,
            )
            resumed = train_and_evaluate_block_smb(
                resumed_config, vision_factory=static_vision_factory
            )
            resumed_checkpoint = load_checkpoint(checkpoint)
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            self.assertEqual(resumed_checkpoint["global_step"], 2)
            self.assertEqual(len(resumed["history"]), 1)

            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            restored = restore_block_smb_checkpoint(checkpoint, model, optimizer)
            self.assertEqual(restored["epoch"], 2)

    def test_monte_carlo_evaluation_reports_coverage_bins_and_gates(self):
        config = tiny_config(
            generated_scenarios=0,
            monte_carlo_validation_samples=len(BLOCK_SMB_MC_FAMILIES),
            monte_carlo_pass_rate_gate=0.1,
            monte_carlo_family_pass_rate_gate=0.1,
        )
        model = make_block_smb_model(config)

        evaluation = evaluate_block_smb_monte_carlo(
            model,
            config,
            split="validation",
            sample_count=len(BLOCK_SMB_MC_FAMILIES),
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
        )

        self.assertEqual(evaluation["sample_count"], len(BLOCK_SMB_MC_FAMILIES))
        self.assertEqual(set(evaluation["families"]), set(BLOCK_SMB_MC_FAMILIES))
        self.assertFalse(evaluation["coverage"]["missing_families"])
        self.assertIn("action_counts", evaluation)
        self.assertIn("action_collapse", evaluation)
        self.assertIn("failure_bins", evaluation)
        self.assertIn("gates", evaluation)
        self.assertFalse(evaluation["gates"]["gate_met"])
        for family in BLOCK_SMB_MC_FAMILIES:
            self.assertIn("success_rate", evaluation["families"][family])
            self.assertIn("action_counts", evaluation["families"][family])

        full_evaluation = evaluate_block_smb(
            model,
            config,
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
        )
        self.assertIn("monte_carlo_validation", full_evaluation)
        self.assertEqual(
            full_evaluation["monte_carlo_validation"]["sample_count"],
            len(BLOCK_SMB_MC_FAMILIES),
        )

    def test_monte_carlo_evaluation_can_use_full_parameter_sweep(self):
        config = tiny_config(
            generated_scenarios=0,
            monte_carlo_parameter_sweep=True,
            monte_carlo_sweep_repeats_per_difficulty=1,
            monte_carlo_validation_samples=0,
        )
        model = make_block_smb_model(config)

        evaluation = evaluate_block_smb_monte_carlo(
            model,
            config,
            split="validation",
            sample_count=0,
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
        )

        self.assertTrue(evaluation["parameter_sweep"])
        self.assertEqual(
            evaluation["sample_count"],
            len(BLOCK_SMB_MC_FAMILIES) * len(BLOCK_SMB_MC_DIFFICULTY_BINS),
        )
        self.assertEqual(set(evaluation["families"]), set(BLOCK_SMB_MC_FAMILIES))
        self.assertFalse(evaluation["coverage"]["missing_families"])

    def test_adaptive_monte_carlo_replay_samples_recent_failure_families(self):
        config = tiny_config(
            generated_scenarios=2,
            monte_carlo_failure_replay_samples_per_epoch=3,
        )
        base_curriculum = build_curriculum(config)
        replay = build_adaptive_monte_carlo_replay_curriculum(
            config,
            {
                "enemy_gap:hard": {"failure_count": 2},
                "wait_timing:medium": {"failure_count": 1},
            },
            epoch=1,
        )
        epoch_curriculum = build_epoch_curriculum(base_curriculum, replay)
        replay_names = [name for name, _scenario in replay]

        self.assertEqual(len(replay), 3)
        self.assertTrue(all(".enemy_gap" in name or ".wait_timing" in name for name in replay_names))
        self.assertEqual(epoch_curriculum[0][0], "level_1_flat.json")
        self.assertEqual(epoch_curriculum[1:4], replay)

    def test_periodic_evaluation_writes_structured_log(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "block_smb.jsonl"
            config = tiny_config(
                epochs=3,
                generated_scenarios=0,
                evaluation_interval_epochs=2,
                log_path=log_path,
            )

            result = train_and_evaluate_block_smb(config, vision_factory=static_vision_factory)
            events = [
                json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(result["history"]), 3)
        self.assertNotIn("eval_mean_return", result["history"][0])
        self.assertIn("eval_mean_return", result["history"][1])
        self.assertIn("eval_mean_return", result["history"][2])
        self.assertEqual([record["epoch"] for record in result["evaluations"]], [2, 3])
        self.assertEqual(events[0]["event"], "run_started")
        self.assertEqual(events[-1]["event"], "run_finished")
        self.assertEqual(events[0]["config"]["evaluation_interval_epochs"], 2)
        self.assertEqual(
            [event["epoch"] for event in events if event["event"] == "train_epoch"],
            [1, 2, 3],
        )
        self.assertEqual(
            [event["epoch"] for event in events if event["event"] == "deterministic_evaluation"],
            [2, 3],
        )
        for event in events:
            self.assertEqual(event["stage"], BLOCK_SMB_SPEC.name)

    def test_optional_tracker_receives_training_and_evaluation_metrics(self):
        class RecordingTracker:
            def __init__(self):
                self.configs = []
                self.metrics = []
                self.closed = False

            def log_config(self, config):
                self.configs.append(config)

            def log_metrics(self, metrics, *, step, prefix=None):
                self.metrics.append((prefix, step, dict(metrics)))

            def close(self):
                self.closed = True

        tracker = RecordingTracker()
        with TemporaryDirectory() as tmpdir:
            config = tiny_config(
                generated_scenarios=0,
                tracking_backend="tensorboard",
                tracking_log_dir=Path(tmpdir) / "tb",
                tracking_project="retroagi-test",
                tracking_run_name="unit",
            )
            with patch(
                "retroagi.stages.block_smb.train.make_experiment_tracker",
                return_value=tracker,
            ) as make_tracker:
                train_and_evaluate_block_smb(config, vision_factory=static_vision_factory)

        tracker_config = make_tracker.call_args.args[0]
        self.assertEqual(tracker_config.backend, "tensorboard")
        self.assertEqual(tracker_config.project, "retroagi-test")
        self.assertEqual(tracker_config.run_name, "unit")
        self.assertEqual(tracker.configs[0]["tracking_backend"], "tensorboard")
        self.assertTrue(any(prefix == "train" for prefix, _step, _metrics in tracker.metrics))
        self.assertTrue(any(prefix == "eval" for prefix, _step, _metrics in tracker.metrics))
        self.assertTrue(tracker.closed)

    def test_train_evaluate_smoke_with_all_block_smb_ablations_disabled(self):
        config = tiny_config(
            generated_scenarios=0,
            ablation=BlockSMBAblationConfig(
                vision_enabled=False,
                world_model_enabled=False,
                critic_feedback_enabled=False,
                hierarchy_enabled=False,
                recurrent_state_enabled=False,
                checkpoint_transfer_enabled=False,
            ),
        )

        result = train_and_evaluate_block_smb(config, vision_factory=static_vision_factory)

        self.assertEqual(
            result["model"].training,
            False,
        )
        self.assertFalse(config.ablation.vision_enabled)
        self.assertFalse(config.ablation.world_model_enabled)
        self.assertFalse(config.ablation.critic_feedback_enabled)
        for key in ("loss_total", "eval_mean_return", "eval_success_rate"):
            self.assertTrue(torch.isfinite(torch.tensor(result["metrics"][key])).item())

    def test_restore_accepts_checkpoint_without_separated_objective_heads(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy_block_smb.pth"
            config = tiny_config()
            source_model = make_block_smb_model(config)
            source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=config.learning_rate)
            legacy_state = {
                key: value
                for key, value in source_model.state_dict().items()
                if not key.startswith(
                    (
                        "transition_representation_head.",
                        "reward_head.",
                        "value_head.",
                    )
                )
            }
            checkpoint = build_checkpoint(
                stage=BLOCK_SMB_SPEC.name,
                model_name=BLOCK_SMB_MODEL_NAME,
                checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
                states={
                    "model": legacy_state,
                    "optimizer": source_optimizer.state_dict(),
                },
                config={"legacy": True},
            )
            save_checkpoint(checkpoint_path, checkpoint)

            restored_model = make_block_smb_model(config)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=config.learning_rate
            )
            restored = restore_block_smb_checkpoint(
                checkpoint_path, restored_model, restored_optimizer
            )

        self.assertEqual(restored["model_name"], BLOCK_SMB_MODEL_NAME)

    def test_restore_rejects_incompatible_architecture_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "bad_architecture_block_smb.pth"
            config = tiny_config()
            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            checkpoint = build_checkpoint(
                stage=BLOCK_SMB_SPEC.name,
                model_name=BLOCK_SMB_MODEL_NAME,
                checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
                config={
                    "architecture_name": "other_architecture",
                    "architecture_config": config.architecture_config,
                },
                states={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
            )
            save_checkpoint(checkpoint_path, checkpoint)

            with self.assertRaisesRegex(ValueError, "checkpoint architecture"):
                restore_block_smb_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    architecture_name=config.architecture_name,
                    architecture_config=config.architecture_config,
                )

    def test_restore_rejects_incompatible_architecture_config_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "bad_architecture_config_block_smb.pth"
            config = tiny_config()
            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            checkpoint = build_checkpoint(
                stage=BLOCK_SMB_SPEC.name,
                model_name=BLOCK_SMB_MODEL_NAME,
                checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
                config={
                    "architecture_name": config.architecture_name,
                    "architecture_config": {
                        "hidden_dim": 16,
                        "controller_schedule": "constant",
                    },
                },
                states={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
            )
            save_checkpoint(checkpoint_path, checkpoint)

            with self.assertRaisesRegex(ValueError, "checkpoint architecture config"):
                restore_block_smb_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    architecture_name=config.architecture_name,
                    architecture_config=config.architecture_config,
                )

    def test_checkpoint_round_trips_target_network_state(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "target_block_smb.pth"
            config = tiny_config(target_network_mode="on")
            model = make_block_smb_model(config)
            target_model = make_target_network(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            with torch.no_grad():
                for parameter in target_model.parameters():
                    parameter.add_(0.5)
                    break

            save_block_smb_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=1,
                global_step=2,
                config=config,
                metrics={"loss_total": 1.0},
                target_model=target_model,
            )
            summary = json.loads(
                checkpoint_summary_path(checkpoint_path).read_text(encoding="utf-8")
            )
            restored_model = make_block_smb_model(config)
            restored_target = make_target_network(restored_model)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=config.learning_rate
            )
            restored = restore_block_smb_checkpoint(
                checkpoint_path,
                restored_model,
                restored_optimizer,
                target_model=restored_target,
            )

        self.assertIn("target_model", restored["states"])
        self.assertEqual(summary["stage"], BLOCK_SMB_SPEC.name)
        self.assertEqual(summary["metrics"]["loss_total"], 1.0)
        self.assertEqual(summary["config"]["architecture_name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(summary["specs"]["architecture"]["name"], BASELINE_ARCHITECTURE_NAME)
        self.assertIn("target_model", summary["state_keys"])
        self.assertIn("code_revision", summary)
        self.assertIn("environment", summary)
        for original, restored_parameter in zip(
            target_model.parameters(), restored_target.parameters()
        ):
            torch.testing.assert_close(original, restored_parameter)


if __name__ == "__main__":
    unittest.main()
