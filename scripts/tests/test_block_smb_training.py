"""Tests for Block SMB trainer plumbing."""

import json
import math
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
    block_smb_monte_carlo_oracle_actions,
    block_smb_monte_carlo_train_sample_count,
    build_adaptive_monte_carlo_replay_curriculum,
    build_curriculum,
    build_epoch_curriculum,
    default_block_smb_failure_focus_monte_carlo_family_weights,
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
    block_smb_oracle_action_loss,
    block_smb_oracle_actions_for_rollout,
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
        with self.assertRaisesRegex(ValueError, "action_gate_min_distinct_actions"):
            tiny_config(action_gate_min_distinct_actions=0)
        with self.assertRaisesRegex(ValueError, "action_gate_max_dominant_fraction"):
            tiny_config(action_gate_max_dominant_fraction=1.5)
        with self.assertRaisesRegex(ValueError, "action_gate_required_actions"):
            tiny_config(action_gate_required_actions=(99,))

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

        self.assertTrue(block_smb_noop_allowed_for_step("level_12_wait_bridge.json", {}, 19))
        self.assertFalse(block_smb_noop_allowed_for_step("level_12_wait_bridge.json", {}, 20))
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

    def test_unlabeled_rollout_does_not_self_label_primitive_auxiliary_loss(self):
        config = tiny_config(generated_scenarios=0, rollout_steps=1)
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=1,
                seed=4,
                deterministic=False,
                device=torch.device("cpu"),
                use_oracle_actions=True,
            )
        finally:
            stage.env.close()

        step = trajectory.transitions[0]
        self.assertIsNone(step.oracle_action)
        self.assertEqual(step.primitive_aux_loss.item(), 0.0)
        self.assertEqual(
            block_smb_oracle_action_loss(step, device=torch.device("cpu")).item(),
            0.0,
        )

    def test_monte_carlo_oracle_actions_supervise_rollout_and_losses(self):
        config = tiny_config(generated_scenarios=1, rollout_steps=2)
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[1]
        expected_actions = block_smb_monte_carlo_oracle_actions(
            scenario,
            max_steps=config.rollout_steps,
        )
        self.assertEqual(
            block_smb_oracle_actions_for_rollout(
                scenario,
                rollout_steps=config.rollout_steps,
            ),
            tuple(expected_actions),
        )
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=config.rollout_steps,
                seed=5,
                deterministic=False,
                device=torch.device("cpu"),
                use_oracle_actions=True,
            )
        finally:
            stage.env.close()

        self.assertGreaterEqual(len(trajectory.transitions), 1)
        for index, step in enumerate(trajectory.transitions):
            self.assertEqual(step.oracle_action, expected_actions[index])
            self.assertEqual(step.action, expected_actions[index])
            self.assertTrue(torch.isfinite(step.primitive_aux_loss).item())

        losses = compute_block_smb_losses(
            model,
            trajectory.transitions,
            config,
            torch.device("cpu"),
            trajectories=[trajectory],
        )

        self.assertEqual(
            losses["oracle_action_supervised_steps"].item(),
            float(len(trajectory.transitions)),
        )
        self.assertGreater(losses["loss_oracle_action"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss_action_aux"]).item())

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
        for key in (
            "train_action_count_0",
            "train_action_count_1",
            "train_action_count_2",
            "train_action_count_3",
            "train_action_count_4",
            "train_action_count_5",
            "train_total_actions",
            "train_distinct_actions",
            "train_dominant_action_fraction",
            "train_distribution_gate_met",
        ):
            self.assertIn(key, metrics)

    def test_train_epoch_covers_full_monte_carlo_curriculum_by_default(self):
        config = tiny_config(generated_scenarios=2, rollout_steps=1)
        model = make_block_smb_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        curriculum = build_curriculum(config)

        metrics, _replay = train_block_smb_epoch(
            model,
            optimizer,
            curriculum,
            config,
            epoch=0,
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
        )

        self.assertEqual(len(curriculum), 3)
        self.assertEqual(metrics["episodes"], 3.0)

    def test_train_epoch_can_preserve_sampled_episode_count_for_smoke_runs(self):
        config = tiny_config(
            generated_scenarios=2,
            rollout_steps=1,
            cover_curriculum_per_epoch=False,
        )
        model = make_block_smb_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        metrics, _replay = train_block_smb_epoch(
            model,
            optimizer,
            build_curriculum(config),
            config,
            epoch=0,
            device=torch.device("cpu"),
            vision_factory=static_vision_factory,
        )

        self.assertEqual(metrics["episodes"], 1.0)

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
            stage.reset(seed=5)
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
            self.assertEqual(saved["global_step"], 2)
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
                "train_distinct_actions",
                "train_dominant_action_fraction",
                "train_distribution_gate_met",
                "eval_fixed_distinct_actions",
                "eval_fixed_dominant_action_fraction",
                "eval_fixed_distribution_gate_met",
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
            self.assertEqual(resumed_checkpoint["global_step"], 4)
            self.assertEqual(len(resumed["history"]), 1)

            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            restored = restore_block_smb_checkpoint(checkpoint, model, optimizer)
            self.assertEqual(restored["epoch"], 2)

    def test_monte_carlo_evaluation_reports_coverage_bins_and_gates(self):
        config = tiny_config(
            generated_scenarios=0,
            monte_carlo_family_weights=default_block_smb_failure_focus_monte_carlo_family_weights(),
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
        self.assertTrue(
            all(".enemy_gap" in name or ".wait_timing" in name for name in replay_names)
        )
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


class TestBlockSMBMasterySchedule(unittest.TestCase):
    def test_initial_state_covers_every_family_with_nonzero_weight(self):
        from retroagi.stages.block_smb.monte_carlo import BLOCK_SMB_MC_FAMILIES
        from retroagi.stages.block_smb.train import (
            block_smb_mastery_family_weights,
            initial_block_smb_mastery_state,
        )

        state = initial_block_smb_mastery_state()
        self.assertEqual(set(state), set(BLOCK_SMB_MC_FAMILIES))
        weights = block_smb_mastery_family_weights(
            state, family_pass_rate_gate=0.9, retention_weight=0.25
        )
        # No family may be excluded from training: the static failure-focus
        # weights previously gave several gated families zero train samples.
        self.assertTrue(all(weight > 0 for weight in weights.values()))
        self.assertEqual(set(weights), set(BLOCK_SMB_MC_FAMILIES))
        for family in BLOCK_SMB_MC_FAMILIES:
            self.assertEqual(state[family]["unlocked_difficulties"], ["easy"])
            self.assertFalse(state[family]["mastered"])

    def test_update_masters_families_and_unlocks_difficulties(self):
        from retroagi.stages.block_smb.train import (
            block_smb_mastery_family_weights,
            initial_block_smb_mastery_state,
            update_block_smb_mastery_state,
        )

        state = initial_block_smb_mastery_state()
        evaluation = {
            "families": {
                "flat_run": {"success_rate": 1.0},
                "tall_pipe_jump": {"success_rate": 0.4},
            },
            "difficulty_bins": {
                "tall_pipe_jump:easy": {"success_rate": 1.0},
            },
        }
        state = update_block_smb_mastery_state(
            state, evaluation, family_pass_rate_gate=0.9
        )
        self.assertTrue(state["flat_run"]["mastered"])
        self.assertFalse(state["tall_pipe_jump"]["mastered"])
        self.assertEqual(
            state["tall_pipe_jump"]["unlocked_difficulties"], ["easy", "medium"]
        )
        weights = block_smb_mastery_family_weights(
            state, family_pass_rate_gate=0.9, retention_weight=0.25
        )
        # A freshly mastered family keeps elevated practice (graduated
        # retention: first mastered eval of a 3-eval grace ramp); unmastered
        # families weigh 1 + deficit so the furthest-from-mastery draw the
        # most samples.
        self.assertAlmostEqual(weights["flat_run"], 0.75)
        self.assertAlmostEqual(weights["tall_pipe_jump"], 1.5)
        self.assertAlmostEqual(weights["single_gap"], 1.9)

    def test_graduated_retention_ramps_down_and_resets_on_regression(self):
        from retroagi.stages.block_smb.train import (
            block_smb_mastery_family_weights,
            initial_block_smb_mastery_state,
            update_block_smb_mastery_state,
        )

        def weight(state):
            return block_smb_mastery_family_weights(
                state, family_pass_rate_gate=0.9, retention_weight=0.25
            )["flat_run"]

        passing = {"families": {"flat_run": {"success_rate": 1.0}}, "difficulty_bins": {}}
        failing = {"families": {"flat_run": {"success_rate": 0.0}}, "difficulty_bins": {}}

        state = initial_block_smb_mastery_state()
        state = update_block_smb_mastery_state(state, passing, family_pass_rate_gate=0.9)
        self.assertEqual(state["flat_run"]["mastered_evals"], 1)
        self.assertAlmostEqual(weight(state), 0.75)
        state = update_block_smb_mastery_state(state, passing, family_pass_rate_gate=0.9)
        self.assertAlmostEqual(weight(state), 0.5)
        state = update_block_smb_mastery_state(state, passing, family_pass_rate_gate=0.9)
        self.assertAlmostEqual(weight(state), 0.25)
        # Long-mastered: stays at the floor.
        state = update_block_smb_mastery_state(state, passing, family_pass_rate_gate=0.9)
        self.assertAlmostEqual(weight(state), 0.25)
        # Regression: counter resets, family returns to full focus...
        state = update_block_smb_mastery_state(state, failing, family_pass_rate_gate=0.9)
        self.assertEqual(state["flat_run"]["mastered_evals"], 0)
        self.assertAlmostEqual(weight(state), 1.9)
        # ...and re-mastering restarts the ramp rather than dropping to the floor.
        state = update_block_smb_mastery_state(state, passing, family_pass_rate_gate=0.9)
        self.assertEqual(state["flat_run"]["mastered_evals"], 1)
        self.assertAlmostEqual(weight(state), 0.75)
        # Grace 0 preserves the old instant-floor behavior.
        instant = block_smb_mastery_family_weights(
            state,
            family_pass_rate_gate=0.9,
            retention_weight=0.25,
            retention_grace_evals=0,
        )
        self.assertAlmostEqual(instant["flat_run"], 0.25)

    def test_difficulty_unlocks_are_monotonic_across_regressions(self):
        from retroagi.stages.block_smb.train import (
            initial_block_smb_mastery_state,
            update_block_smb_mastery_state,
        )

        state = initial_block_smb_mastery_state()
        unlock = {
            "families": {"single_gap": {"success_rate": 0.5}},
            "difficulty_bins": {"single_gap:easy": {"success_rate": 1.0}},
        }
        state = update_block_smb_mastery_state(state, unlock, family_pass_rate_gate=0.9)
        self.assertIn("medium", state["single_gap"]["unlocked_difficulties"])
        regression = {
            "families": {"single_gap": {"success_rate": 0.0}},
            "difficulty_bins": {"single_gap:easy": {"success_rate": 0.0}},
        }
        state = update_block_smb_mastery_state(
            state, regression, family_pass_rate_gate=0.9
        )
        # A later regression must not re-lock medium: the training mix must not
        # thrash between difficulty distributions.
        self.assertIn("medium", state["single_gap"]["unlocked_difficulties"])

    def test_mastery_curriculum_is_deterministic_and_respects_unlocks(self):
        from retroagi.stages.block_smb.monte_carlo import (
            block_smb_monte_carlo_metadata,
        )
        from retroagi.stages.block_smb.train import (
            BlockSMBTrainingConfig,
            build_mastery_monte_carlo_curriculum,
            initial_block_smb_mastery_state,
        )

        config = BlockSMBTrainingConfig(
            monte_carlo_train_samples_per_epoch=8,
            mastery_gated_schedule=True,
        )
        state = initial_block_smb_mastery_state()
        first = build_mastery_monte_carlo_curriculum(config, state, phase=3)
        second = build_mastery_monte_carlo_curriculum(config, state, phase=3)
        self.assertEqual([name for name, _ in first], [name for name, _ in second])
        self.assertEqual(len(first), 8)
        for _name, scenario in first:
            metadata = block_smb_monte_carlo_metadata(scenario)
            # Nothing is unlocked beyond easy at the initial state.
            self.assertEqual(metadata["parameters"]["difficulty_bin"], "easy")
        shifted = build_mastery_monte_carlo_curriculum(config, state, phase=4)
        self.assertNotEqual(
            [name for name, _ in first], [name for name, _ in shifted]
        )

    def test_config_validates_mastery_fields(self):
        from retroagi.stages.block_smb.train import BlockSMBTrainingConfig

        with self.assertRaises(ValueError):
            BlockSMBTrainingConfig(mastery_retention_weight=0.0)
        config = BlockSMBTrainingConfig(mastery_gated_schedule=True)
        self.assertTrue(config.mastery_gated_schedule)

    def test_oracle_actions_default_off_and_cli_toggle(self):
        from retroagi.stages.block_smb import cli
        from retroagi.stages.block_smb.train import BlockSMBTrainingConfig

        # Ranked-candidate search replaces oracle forcing as the default
        # in-rollout skill channel; oracle demonstrations are opt-in.
        self.assertFalse(BlockSMBTrainingConfig().use_oracle_actions)
        with self.assertRaises(TypeError):
            BlockSMBTrainingConfig(use_oracle_actions=1)

        def resolve(extra):
            args = cli.build_parser().parse_args(
                [
                    "train",
                    "--vision-checkpoint",
                    "data/block_vit/block_vit.pth",
                    *extra,
                ]
            )
            return cli._make_train_config(args).use_oracle_actions

        self.assertFalse(resolve([]))
        self.assertFalse(resolve(["--no-oracle-actions"]))
        self.assertTrue(resolve(["--use-oracle-actions"]))

    def test_ranked_candidate_search_default_on_and_wired_to_model(self):
        from retroagi.stages.block_smb import cli
        from retroagi.stages.block_smb.train import (
            BlockSMBTrainingConfig,
            make_block_smb_model,
        )

        config = BlockSMBTrainingConfig()
        self.assertTrue(config.ranked_candidate_search)
        with self.assertRaises(TypeError):
            BlockSMBTrainingConfig(ranked_candidate_search="yes")
        model = make_block_smb_model(config)
        self.assertTrue(model.ranked_candidate_search)
        disabled = make_block_smb_model(
            BlockSMBTrainingConfig(ranked_candidate_search=False)
        )
        self.assertFalse(disabled.ranked_candidate_search)

        def resolve(extra):
            args = cli.build_parser().parse_args(
                [
                    "train",
                    "--vision-checkpoint",
                    "data/block_vit/block_vit.pth",
                    *extra,
                ]
            )
            return cli._make_train_config(args).ranked_candidate_search

        self.assertTrue(resolve([]))
        self.assertFalse(resolve(["--no-ranked-candidate-search"]))

    def test_deterministic_critic_gates_wired_and_layout_matches_projector(self):
        from retroagi.stages.block_smb import cli
        from retroagi.stages.block_smb.adapter import (
            block_smb_deterministic_critic_slots,
        )
        from retroagi.stages.block_smb.train import (
            BlockSMBTrainingConfig,
            make_block_smb_model,
        )
        from retroagi.stages.block_smb.vision import BlockVisionTransformer

        slots = block_smb_deterministic_critic_slots()
        # Drift guard: the static indices must land inside the projector's
        # runtime c_state span at the goal-distance and death state dims. Use
        # the real Block ViT (fresh weights; only output shapes matter) because
        # the production fusion includes its support-state softmax, which
        # simplified test stubs omit.
        stage = BlockSMBStage(
            scenario={
                "mario": [20, 200],
                "platforms": [[0, 220, 256, 20]],
                "world_width": 256,
            },
            vision=BlockVisionTransformer(),
            env=MarioScenarioEnv(),
        )
        try:
            observation = stage.reset(seed=3)
            batch = stage.encode_observation(observation)
        finally:
            stage.env.close()
        spans = block_smb_c_stream_slot_spans(batch)
        state_start, state_end = spans["state"]
        self.assertEqual(slots["goal_distance"], state_start + 17)
        self.assertEqual(slots["death"], state_start + 24)
        self.assertLess(slots["death"], state_end)
        terminal_start, terminal_end = spans["terminal_outcome"]
        self.assertEqual(slots["death"], terminal_start)

        config = BlockSMBTrainingConfig()
        self.assertTrue(config.deterministic_critic_gates)
        model = make_block_smb_model(config)
        self.assertEqual(model.deterministic_critic_slots, slots)
        disabled = make_block_smb_model(
            BlockSMBTrainingConfig(deterministic_critic_gates=False)
        )
        self.assertIsNone(disabled.deterministic_critic_slots)

        def resolve(extra):
            args = cli.build_parser().parse_args(
                [
                    "train",
                    "--vision-checkpoint",
                    "data/block_vit/block_vit.pth",
                    *extra,
                ]
            )
            return cli._make_train_config(args).deterministic_critic_gates

        self.assertTrue(resolve([]))
        self.assertFalse(resolve(["--no-deterministic-critic-gates"]))

    def test_goal_reached_label_under_goal_on_stomp(self):
        # Regression: the rollout success label comes from _goal_reached().
        # Under goal_on_stomp the goal rect is a proxy riding the enemy, so
        # Mario overlaps it on the very frame he walks into the enemy and
        # dies — positional overlap must not label that death a success.
        from retroagi.stages.block_smb.train import _goal_reached

        scenario = {
            "world_width": 256,
            "mario": [20, 204],
            "platforms": [[0, 220, 256, 20]],
            "enemies": [[30, 206, 30, 30, 0.0]],
            "coins": [],
            "goal": [28, 186, 16, 20],
            "goal_on_stomp": True,
        }
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario), seed=0)
            terminated = False
            for _ in range(20):
                _obs, _reward, terminated, _truncated, info = env.step(1)
                if terminated:
                    break
            self.assertTrue(terminated)
            self.assertTrue(info["death"])
            self.assertFalse(_goal_reached(env))
        finally:
            env.close()
        # A genuine stomp credits the goal and flips the label.
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario), seed=0)
            env.mario["x"] = 30.0
            env.mario["y"] = 186.0
            env.mario["vy"] = 8.0
            _obs, _reward, terminated, _truncated, info = env.step(0)
            self.assertTrue(terminated)
            self.assertFalse(info["death"])
            self.assertGreater(info["reward_terms"]["goal"], 0.0)
            self.assertTrue(_goal_reached(env))
        finally:
            env.close()

    def test_goal_distance_shaping_rewards_rising_toward_elevated_goal(self):
        scenario = {
            "world_width": 256,
            "mario": [90, 200],
            "platforms": [[0, 220, 256, 20], [120, 168, 30, 52]],
            "coins": [],
            "goal": [127, 148, 16, 20],
            "reward_goal_distance_shaping": 2.0,
        }
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario), seed=0)
            total = 0.0
            for action in [2] * 12 + [1] * 20:
                _obs, _reward, terminated, truncated, info = env.step(action)
                total += info["reward_terms"]["goal_distance"]
                if terminated or truncated:
                    break
            self.assertGreater(total, 0.0)
            self.assertGreater(info["reward_terms"]["goal"], 0.0)
        finally:
            env.close()
        # Without the opt-in key the term stays exactly zero.
        control = {k: v for k, v in scenario.items() if k != "reward_goal_distance_shaping"}
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=control, seed=0)
            total = 0.0
            for action in [2] * 12 + [1] * 10:
                _obs, _reward, terminated, truncated, info = env.step(action)
                total += info["reward_terms"]["goal_distance"]
                if terminated or truncated:
                    break
            self.assertEqual(total, 0.0)
        finally:
            env.close()

    def test_energy_regulator_charges_held_jump_frames_only_when_opted_in(self):
        scenario = {
            "world_width": 256,
            "mario": [20, 200],
            "platforms": [[0, 220, 256, 20]],
            "coins": [],
            "goal": [230, 200, 16, 20],
            "reward_energy_jump": -0.15,
        }
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=dict(scenario), seed=0)
            energy = 0.0
            for action in [2] * 10 + [1] * 5:
                _obs, _reward, terminated, truncated, info = env.step(action)
                energy += info["reward_terms"]["energy"]
                if terminated or truncated:
                    break
            # Ten held jump frames at -0.15 each; walking frames are free.
            self.assertAlmostEqual(energy, -1.5, places=6)
        finally:
            env.close()
        control = {k: v for k, v in scenario.items() if k != "reward_energy_jump"}
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=control, seed=0)
            energy = 0.0
            for action in [2] * 10:
                _obs, _reward, terminated, truncated, info = env.step(action)
                energy += info["reward_terms"]["energy"]
                if terminated or truncated:
                    break
            self.assertEqual(energy, 0.0)
        finally:
            env.close()
        # Success-conditioned energy: an attempt that ends in death refunds the
        # accumulated charge, so giving up never beats trying.
        doomed = {
            "world_width": 256,
            "mario": [20, 200],
            "platforms": [[0, 220, 100, 20]],
            "coins": [],
            "goal": [230, 200, 16, 20],
            "reward_energy_jump": -0.15,
        }
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=doomed, seed=0)
            energy = 0.0
            for action in [2] * 6 + [1] * 80:
                _obs, _reward, terminated, truncated, info = env.step(action)
                energy += info["reward_terms"]["energy"]
                if terminated or truncated:
                    break
            self.assertTrue(info["death"])
            self.assertAlmostEqual(energy, 0.0, places=6)
        finally:
            env.close()

    def test_pipe_mount_rollout_forces_a_level_action_with_free_b_duration(self):
        from retroagi.stages.block_smb.monte_carlo import (
            sample_block_smb_monte_carlo_scenario,
        )
        from retroagi.stages.block_smb.train import (
            block_smb_forced_action_for_rollout,
            collect_trajectory,
        )
        from retroagi.stages.block_smb.vision import BlockVisionTransformer

        sample = sample_block_smb_monte_carlo_scenario(
            split="train",
            seed=3,
            sample_index=0,
            family="pipe_mount",
            difficulty="easy",
        )
        self.assertEqual(block_smb_forced_action_for_rollout(sample.scenario), 2)
        # Fixed scenarios carry no intent.
        self.assertIsNone(block_smb_forced_action_for_rollout({"mario": [20, 200]}))

        config = tiny_config()
        model = make_block_smb_model(config)
        model.eval()
        stage = BlockSMBStage(
            env=MarioScenarioEnv(),
            scenario=dict(sample.scenario),
            vision=BlockVisionTransformer(),
        )
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                "pipe_mount_forced",
                rollout_steps=8,
                seed=0,
                deterministic=False,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()
        # The given A-level action drives the rollout from the first step; the
        # executor (model duration head) owns the hold, so the executed action
        # is the forced jump rather than a sampled/searched token.
        self.assertEqual(trajectory.transitions[0].action, 2)

    def test_primitive_outcome_loss_pushes_expected_hold_against_error(self):
        # The quadratic regression toward the hindsight hold target must push
        # the long-duration bin down when the target is below the current
        # expectation (overshoot relabel) and up when above (undershoot
        # relabel). Uniform logits give an expected fraction of ~0.41.
        from types import SimpleNamespace

        from retroagi.core import SMBPrimitiveExecution
        from retroagi.stages.block_smb.train import _smb_expected_hold_fraction

        for target, expected_gradient_sign in ((0.15, 1.0), (0.9, -1.0)):
            logits = torch.zeros(1, 1, 8, requires_grad=True)
            motor = SimpleNamespace(
                hold_duration_logits=logits,
                duration_bin_values=torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
                ),
            )
            execution = SMBPrimitiveExecution(action=2, started=True, active=True)
            fraction = _smb_expected_hold_fraction(
                motor,
                execution,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            self.assertIsNotNone(fraction)
            ((fraction - target) ** 2).backward()
            long_bin_gradient = float(logits.grad[0, -1, -1])
            self.assertGreater(long_bin_gradient * expected_gradient_sign, 0.0)
        # No jump engaged -> no supervision tensor.
        idle = SMBPrimitiveExecution(action=1)
        self.assertIsNone(
            _smb_expected_hold_fraction(
                SimpleNamespace(
                    hold_duration_logits=torch.zeros(1, 1, 8),
                    duration_bin_values=torch.tensor([1.0] * 8),
                ),
                idle,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )

    def test_landed_jump_span_ends_as_success_not_interruption(self):
        # Regression: the landing frame reports active=False with landed=True,
        # so the early-close heuristic must not fire one frame before it —
        # that mislabeled every landed jump as an interruption and zeroed the
        # HSP1 landing metrics and HSP2 relabels.
        from retroagi.stages.block_smb.temporal_spans import (
            build_block_smb_temporal_spans,
        )

        base = dict(
            released=False,
            landed=False,
            cancelled=False,
            death=False,
            goal=False,
            terminated=False,
            truncated=False,
            y_before=200.0,
            y_after=200.0,
        )
        records = [
            dict(base, action=2, started=True, active=True, x_before=0.0, x_after=2.0),
            dict(base, action=2, started=False, active=True, x_before=2.0, x_after=5.0),
            dict(
                base,
                action=1,
                started=False,
                active=True,
                released=True,
                x_before=5.0,
                x_after=8.0,
            ),
            dict(
                base,
                action=1,
                started=False,
                active=False,
                released=True,
                landed=True,
                x_before=8.0,
                x_after=10.0,
            ),
            dict(base, action=1, started=False, active=False, x_before=10.0, x_after=12.0),
            dict(
                base,
                action=1,
                started=False,
                active=False,
                truncated=True,
                x_before=12.0,
                x_after=14.0,
            ),
        ]
        spans = build_block_smb_temporal_spans(
            records, episode_id="e", scenario_id="s", seed=0
        )
        jump = [s for s in spans if s.command.get("primitive") == "jump"]
        self.assertEqual(len(jump), 1)
        self.assertEqual(jump[0].termination_reason, "success")
        self.assertEqual(jump[0].end_frame, 3)

    def test_hsp0_rollout_spans_reconstruct_with_full_coverage(self):
        # HSP0 exit gate: rollouts emit temporal spans that tile every frame
        # with unambiguous end reasons, for both policy and scripted play.
        from retroagi.core.temporal import reconstruct_episodes
        from retroagi.stages.block_smb.monte_carlo import (
            sample_block_smb_monte_carlo_scenario,
        )
        from retroagi.stages.block_smb.train import collect_trajectory
        from retroagi.stages.block_smb.vision import BlockVisionTransformer

        sample = sample_block_smb_monte_carlo_scenario(
            split="train",
            seed=3,
            sample_index=0,
            family="stomp_mount",
            difficulty="easy",
        )
        config = tiny_config()
        model = make_block_smb_model(config)
        for use_oracle in (False, True):
            stage = BlockSMBStage(
                env=MarioScenarioEnv(),
                scenario=dict(sample.scenario),
                vision=BlockVisionTransformer(),
            )
            try:
                trajectory = collect_trajectory(
                    model,
                    stage,
                    "hsp0_probe",
                    rollout_steps=30,
                    seed=1,
                    deterministic=False,
                    device=torch.device("cpu"),
                    use_oracle_actions=use_oracle,
                )
            finally:
                stage.env.close()
            self.assertGreater(len(trajectory.spans), 1)
            expected_source = "scripted" if use_oracle else "real"
            for span in trajectory.spans:
                self.assertEqual(span.source, expected_source)
            (report,) = reconstruct_episodes(trajectory.spans)
            self.assertTrue(report.valid, report.problems)
            self.assertEqual(report.frame_count, len(trajectory.transitions))
            self.assertIn("skill", report.levels)
            skill = trajectory.spans[0]
            self.assertEqual(skill.level, "skill")
            self.assertEqual(skill.goal["goal_type"], "stomp_mount")

    def test_stomp_rollout_backfills_primitive_outcomes(self):
        # A stomp_mount rollout (forced jump, goal riding the enemy) must
        # produce jump frames carrying both the graph-attached expected hold
        # and a shared hindsight landing error, and the loss assembly must
        # surface a finite loss_primitive_outcome.
        from retroagi.stages.block_smb.monte_carlo import (
            sample_block_smb_monte_carlo_scenario,
        )
        from retroagi.stages.block_smb.train import (
            collect_trajectory,
            compute_block_smb_losses,
        )
        from retroagi.stages.block_smb.vision import BlockVisionTransformer

        sample = sample_block_smb_monte_carlo_scenario(
            split="train",
            seed=3,
            sample_index=0,
            family="stomp_mount",
            difficulty="easy",
        )
        config = tiny_config()
        model = make_block_smb_model(config)
        stage = BlockSMBStage(
            env=MarioScenarioEnv(),
            scenario=dict(sample.scenario),
            vision=BlockVisionTransformer(),
        )
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                "stomp_outcome",
                rollout_steps=40,
                seed=0,
                deterministic=False,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()
        supervised = [
            step
            for step in trajectory.transitions
            if step.expected_hold is not None
            and step.info.get("primitive_outcome_target") is not None
        ]
        self.assertGreater(len(supervised), 0)
        targets = {float(step.info["primitive_outcome_target"]) for step in supervised}
        for target in targets:
            self.assertTrue(math.isfinite(target))
            self.assertGreaterEqual(target, 1.0 / 16.0)
            self.assertLessEqual(target, 1.0)
        losses = compute_block_smb_losses(
            model,
            trajectory.transitions,
            config,
            torch.device("cpu"),
            trajectories=[trajectory],
        )
        self.assertTrue(torch.isfinite(losses["loss_primitive_outcome"]).item())
        self.assertGreater(
            float(losses["primitive_outcome_supervised_steps"].item()), 0.0
        )
        # World model targets: committed-primitive frames predict the OUTCOME
        # state (at completion), and all frames of one span share it.
        outcome_frames = [
            step
            for step in trajectory.transitions
            if step.info.get("primitive_outcome_batch") is not None
        ]
        self.assertGreater(len(outcome_frames), 0)
        shared = {id(step.info["primitive_outcome_batch"]) for step in outcome_frames}
        self.assertLessEqual(len(shared), len(supervised))
        # HSP1: release timing supervised from the same spans — frame index
        # and hindsight hold backfilled, release logits captured, loss finite.
        self.assertTrue(torch.isfinite(losses["loss_release_timing"]).item())
        release_supervised = [
            step
            for step in trajectory.transitions
            if step.release_logit is not None
            and step.info.get("primitive_frame_index") is not None
            and step.info.get("primitive_target_hold") is not None
        ]
        self.assertGreater(len(release_supervised), 0)


if __name__ == "__main__":
    unittest.main()


class TestBlockSMBSuccessReplay(unittest.TestCase):
    @staticmethod
    def _trajectory(success):
        from types import SimpleNamespace

        info = {"goal_reached": True} if success else {}
        return SimpleNamespace(
            success=success,
            transitions=[SimpleNamespace(info=info)],
        )

    def test_buffer_stores_solved_scenarios_deduped_and_capped(self):
        from retroagi.stages.block_smb.train import BlockSMBSuccessReplay

        buffer = BlockSMBSuccessReplay(max_episodes_per_family=2, seed=0)
        scenario = {"mario": [40, 200], "goal": [96, 186, 16, 20]}
        buffer.add(self._trajectory(False), "pit_leap", "s0", scenario)
        self.assertEqual(len(buffer), 0)
        buffer.add(self._trajectory(True), "", "s0", scenario)
        buffer.add(self._trajectory(True), "pit_leap", None, scenario)
        self.assertEqual(len(buffer), 0)
        # Re-solving the same scenario refreshes rather than duplicates.
        for _ in range(3):
            buffer.add(self._trajectory(True), "pit_leap", "s0", scenario)
        self.assertEqual(len(buffer), 1)
        buffer.add(self._trajectory(True), "pit_leap", "s1", scenario)
        buffer.add(self._trajectory(True), "pit_leap", "s2", scenario)
        # Cap of 2 per family: oldest (s0) evicted.
        self.assertEqual(len(buffer), 2)
        buffer.add(self._trajectory(True), "moving_bridge", "b0", scenario)
        self.assertEqual(buffer.families(), ["moving_bridge", "pit_leap"])

    def test_rehearsal_sampling_is_balanced_and_returns_copies(self):
        from retroagi.stages.block_smb.train import BlockSMBSuccessReplay

        buffer = BlockSMBSuccessReplay(max_episodes_per_family=4, seed=0)
        scenario = {"mario": [40, 200]}
        for index in range(4):
            buffer.add(self._trajectory(True), "pit_leap", f"p{index}", scenario)
        buffer.add(self._trajectory(True), "moving_bridge", "b0", scenario)
        picks = buffer.sample_scenarios(3)
        self.assertEqual(len(picks), 3)
        families = [p["family"] for p in picks]
        # Round-robin without replacement: the single-scenario family is
        # always represented despite the other family holding four.
        self.assertIn("moving_bridge", families)
        ids = [p["scenario_id"] for p in picks]
        self.assertEqual(len(ids), len(set(ids)))
        # Returned scenarios are deep copies: mutating one cannot corrupt
        # the stored layout.
        picks[0]["scenario"]["mario"][0] = 999
        again = buffer.sample_scenarios(5)
        self.assertTrue(all(p["scenario"]["mario"][0] == 40 for p in again))
        # Empty buffer and zero count degrade cleanly.
        self.assertEqual(BlockSMBSuccessReplay().sample_scenarios(3), [])
        self.assertEqual(buffer.sample_scenarios(0), [])


if __name__ == "__main__":
    unittest.main()


class TestWaitCoaching(unittest.TestCase):
    def test_wait_target_is_frames_until_platform_nearest(self):
        from retroagi.stages.block_smb.train import block_smb_wait_target_frames

        def records(platform_positions):
            return [{"platform_x": p} for p in platform_positions]

        # Platform approaches from 100px away, nearest (2px) at offset 20.
        positions = [100.0 - 5.0 * k if k <= 20 else 2.0 + 5.0 * (k - 20) for k in range(40)]
        target = block_smb_wait_target_frames(records(positions), 0, 0.0)
        self.assertEqual(target, 20)
        # Already adjacent: clamps to the 4-frame minimum.
        target = block_smb_wait_target_frames(records([1.0] * 10), 0, 0.0)
        self.assertEqual(target, 4)
        # Never comes close: no coaching target.
        far = block_smb_wait_target_frames(records([500.0] * 40), 0, 0.0)
        self.assertIsNone(far)
        # No moving platform recorded: no target.
        none = block_smb_wait_target_frames([{"platform_x": None}] * 10, 0, 0.0)
        self.assertIsNone(none)
        # A platform that stays distant but clearly approaches still yields
        # a target: the agent may wait well back from the crossing edge.
        approach = [100.0 - 1.5 * k for k in range(30)] + [55.0 + 1.5 * k for k in range(10)]
        target = block_smb_wait_target_frames(records(approach), 0, 0.0)
        self.assertEqual(target, 30)
        # Ceiling: nearest approach beyond 64 frames is out of reach.
        late = [200.0 - 2.0 * k for k in range(100)]
        target = block_smb_wait_target_frames(records(late), 0, 0.0)
        self.assertIsNone(target) if target is None else self.assertLessEqual(target, 64)


class TestScopedForcedAction(unittest.TestCase):
    def test_first_primitive_scope_releases_after_opening_wait(self):
        from retroagi.stages.block_smb.monte_carlo import (
            sample_block_smb_monte_carlo_scenario,
        )
        from retroagi.stages.block_smb.train import (
            block_smb_forced_action_scope,
            collect_trajectory,
        )
        from retroagi.stages.block_smb.vision import BlockVisionTransformer

        sample = sample_block_smb_monte_carlo_scenario(
            split="train",
            seed=3,
            sample_index=0,
            family="bridge_wait",
            difficulty="easy",
        )
        self.assertEqual(block_smb_forced_action_scope(sample.scenario), "first_primitive")
        self.assertEqual(block_smb_forced_action_scope({"mario": [20, 200]}), "episode")

        config = tiny_config()
        model = make_block_smb_model(config)
        stage = BlockSMBStage(
            env=MarioScenarioEnv(),
            scenario=dict(sample.scenario),
            vision=BlockVisionTransformer(),
        )
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                "bridge_wait_probe",
                rollout_steps=160,
                seed=0,
                deterministic=False,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()
        actions = [t.action for t in trajectory.transitions]
        # The given opening primitive is a wait...
        self.assertEqual(actions[0], 0)
        # ...and once it completes the policy owns the episode: the rollout
        # is not NOOP-forced throughout.
        self.assertTrue(any(a != 0 for a in actions))
