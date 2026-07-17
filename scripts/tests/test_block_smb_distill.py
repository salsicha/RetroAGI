"""Tests for Block SMB distillation data collection."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from retroagi.core import (
    DEFAULT_PRIMITIVE_DURATION_BINS,
    SMBAction,
    StageBatch,
    VisionOutput,
    VisionSpec,
)
from retroagi.stages.block_smb import distill as distill_module
from retroagi.stages.block_smb.distill import (
    DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
    BlockSMBDistillationConfig,
    build_block_smb_distillation_scenarios,
    collect_dagger_distillation_examples,
    collect_scripted_distillation_examples,
)
from retroagi.stages.block_smb.distill import (
    main as distill_main,
)


class StaticBlockVision:
    spec = VisionSpec(
        name="static_block_distill",
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

    def encode(self, _observation):
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


class _FixedActionDaggerModel(torch.nn.Module):
    """Stub student policy whose argmax action is always the same."""

    def __init__(self, action: int):
        super().__init__()
        self._action = int(action)

    def forward(self, _src_a, _src_b, src_c, **_kwargs):
        logits = torch.full((src_c.size(0), 1, distill_module.BLOCK_SMB_ACTION_COUNT), -8.0)
        logits[:, :, self._action] = 8.0
        return (None, None, None, None, logits, None, None, None)


class _ScriptReplayDaggerModel(torch.nn.Module):
    """Stub student policy that replays a fixed action script step by step."""

    def __init__(self, actions):
        super().__init__()
        self._actions = [int(action) for action in actions]
        self._calls = 0

    def forward(self, _src_a, _src_b, src_c, **_kwargs):
        action = self._actions[min(self._calls, len(self._actions) - 1)]
        self._calls += 1
        logits = torch.full((src_c.size(0), 1, distill_module.BLOCK_SMB_ACTION_COUNT), -8.0)
        logits[:, :, action] = 8.0
        return (None, None, None, None, logits, None, None, None)


def _gap_dagger_config(**overrides) -> BlockSMBDistillationConfig:
    values = dict(
        fixed_scenarios=("level_2_gap.json",),
        monte_carlo_samples=0,
        required_monte_carlo_families=(),
        rollout_steps=40,
        episodes_per_scenario=1,
        evaluation_episodes=1,
        evaluation_max_steps=40,
        dagger_iterations=1,
        device="cpu",
    )
    values.update(overrides)
    return BlockSMBDistillationConfig(**values)


def _primitive_outcome_batch(
    *,
    x: float,
    support: tuple[float, float, float] = (0.0, 1.0, 0.0),
    death: bool = False,
    terminated: bool = False,
    truncated: bool = False,
) -> StageBatch:
    src_c = torch.zeros(1, 40)
    src_c[:, 0] = float(x)
    src_c[:, 9:12] = torch.tensor([support], dtype=torch.float32)
    src_c[:, 36] = float(death)
    src_c[:, 37] = float(terminated)
    src_c[:, 38] = float(truncated)
    return StageBatch(
        src_a=torch.zeros(1, 8, dtype=torch.long),
        target_a=None,
        src_b=torch.zeros(1, 16, dtype=torch.long),
        target_b=None,
        src_c=src_c,
        target_c=None,
        metadata={
            "vision_fusion": {
                "c_position": (0, 2),
                "c_semantic_probabilities": (2, 9),
                "c_support_state": (9, 12),
                "c_state": (12, 39),
                "c_patch_tokens": (39, 40),
            },
            "episode": {"terminated": terminated, "truncated": truncated},
            "info": {
                "death": death,
                "terminated": terminated,
                "truncated": truncated,
                "reward_terms": {
                    "enemy_hit": -10.0 if death else 0.0,
                    "fall_death": 0.0,
                    "goal": 0.0,
                    "progress": 0.0,
                },
            },
        },
    )


class TestBlockSMBDistillation(unittest.TestCase):
    def test_monte_carlo_oracle_examples_are_collected(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=(),
            monte_carlo_samples=2,
            required_monte_carlo_families=(),
            rollout_steps=24,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=24,
            device="cpu",
        )

        scenarios, scripts, summary = build_block_smb_distillation_scenarios(config)
        examples = collect_scripted_distillation_examples(
            config,
            vision_factory=static_vision_factory,
        )

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(summary["monte_carlo"]["sample_count"], 2)
        self.assertTrue(all(name in scripts for name, _scenario in scenarios))
        self.assertTrue(any(action == 2 for actions in scripts.values() for action in actions))
        self.assertGreater(len(examples), 0)
        self.assertTrue(
            all(example.scenario_name.startswith("block_smb_mc_v1.train.") for example in examples)
        )
        self.assertIn(2, {example.action for example in examples})

    def test_default_warm_start_covers_fixed_chained_and_full_smb_proxy(self):
        config = BlockSMBDistillationConfig(
            monte_carlo_samples=0,
            rollout_steps=24,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=24,
            device="cpu",
        )

        scenarios, scripts, summary = build_block_smb_distillation_scenarios(config)
        family_counts = summary["monte_carlo"]["coverage"]["family_counts"]

        self.assertEqual(summary["fixed_scenario_count"], len(config.fixed_scenarios))
        self.assertEqual(
            tuple(summary["monte_carlo"]["required_families"]),
            DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
        )
        for family in DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES:
            self.assertEqual(family_counts[family], 3)
        self.assertEqual(
            summary["scenario_count"],
            len(config.fixed_scenarios) + 3 * len(DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES),
        )
        self.assertTrue(any(".full_smb_opening_proxy." in name for name, _scenario in scenarios))
        self.assertTrue(all(name in scripts for name, _scenario in scenarios))
        self.assertTrue(any(action == 2 for actions in scripts.values() for action in actions))

    def test_dagger_alignment_rejects_diverged_and_offtimeline_states(self):
        reference = [{"x": 36.5, "y": 204.0, "vx": 3.0, "vy": 0.0, "on_ground": True}]

        def aligned(mario, step_index=0):
            return distill_module._dagger_teacher_alignment(
                reference,
                step_index,
                mario,
                position_tolerance=4.0,
                velocity_tolerance=1.0,
            )

        on_track = {"x": 34.0, "y": 204.0, "vx": 2.5, "vy": 0.0, "on_ground": True}
        self.assertTrue(aligned(on_track))
        self.assertFalse(aligned({**on_track, "x": 20.0}))
        self.assertFalse(aligned({**on_track, "y": 150.0}))
        self.assertFalse(aligned({**on_track, "vx": 0.0}))
        self.assertFalse(aligned({**on_track, "on_ground": False}))
        self.assertFalse(aligned(on_track, step_index=1))

    def test_teacher_reference_states_follow_script_timeline(self):
        config = _gap_dagger_config()
        scenarios, scripts, _summary = build_block_smb_distillation_scenarios(config)
        scenario_name, scenario = scenarios[0]

        states = distill_module._scripted_teacher_reference_states(scenario, scripts[scenario_name])

        self.assertEqual(states[0]["x"], float(scenario["mario"][0]))
        jump_start = scripts[scenario_name].index(int(SMBAction.RIGHT_JUMP))
        self.assertTrue(states[jump_start]["on_ground"])
        self.assertGreater(states[jump_start]["x"], states[0]["x"] + 12.0)

    def test_dagger_diverged_student_gets_no_time_indexed_jump_labels(self):
        config = _gap_dagger_config(dagger_labeler="script")
        model = _FixedActionDaggerModel(int(SMBAction.NOOP))

        examples = collect_dagger_distillation_examples(
            model,
            config,
            vision_factory=static_vision_factory,
            device=torch.device("cpu"),
            iteration=1,
        )

        # The stationary student never reaches the scripted jump window, so the
        # script's step-indexed RIGHT_JUMP labels must not attach to its states.
        self.assertGreater(len(examples), 0)
        self.assertTrue(all(example.action != int(SMBAction.RIGHT_JUMP) for example in examples))
        self.assertTrue(
            all(example.primitive_button_combo != int(SMBAction.RIGHT_JUMP) for example in examples)
        )
        jump_start = 10
        self.assertLess(max(example.step_index for example in examples), jump_start)

    def test_dagger_aligned_student_keeps_time_indexed_jump_labels(self):
        config = _gap_dagger_config(dagger_labeler="script")
        _scenarios, scripts, _summary = build_block_smb_distillation_scenarios(config)
        model = _ScriptReplayDaggerModel(scripts["level_2_gap.json"])

        examples = collect_dagger_distillation_examples(
            model,
            config,
            vision_factory=static_vision_factory,
            device=torch.device("cpu"),
            iteration=1,
        )

        # A student that tracks the teacher's timeline stays aligned, so the
        # rejection filter must not discard its jump-window labels.
        self.assertTrue(any(example.action == int(SMBAction.RIGHT_JUMP) for example in examples))
        self.assertGreaterEqual(max(example.step_index for example in examples), 10)

    def test_dagger_expert_labeler_covers_diverged_states_with_state_labels(self):
        config = _gap_dagger_config(dagger_labeler="geometry_expert", rollout_steps=15)
        model = _FixedActionDaggerModel(int(SMBAction.NOOP))

        examples = collect_dagger_distillation_examples(
            model,
            config,
            vision_factory=static_vision_factory,
            device=torch.device("cpu"),
            iteration=1,
        )

        # The stationary student never advances, so the expert labels every
        # visited state (no rejection) with the action correct for the state
        # itself: run toward the gap, never the script's time-indexed jump
        # from a standstill at spawn.
        self.assertEqual(len(examples), config.rollout_steps)
        self.assertEqual(max(example.step_index for example in examples), config.rollout_steps - 1)
        self.assertTrue(all(example.action == int(SMBAction.RIGHT) for example in examples))
        # Expert-labeled states never inherit the script's primitive masks.
        self.assertTrue(all(example.primitive_button_combo_mask == 0.0 for example in examples))

    def test_scripted_examples_carry_primitive_duration_release_labels(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=("level_5_enemy_hop.json",),
            monte_carlo_samples=0,
            required_monte_carlo_families=(),
            rollout_steps=45,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=45,
            primitive_hazard_weight_multiplier=3.0,
            device="cpu",
        )

        examples = collect_scripted_distillation_examples(
            config,
            vision_factory=static_vision_factory,
        )

        jump_examples = [
            example for example in examples if example.action == int(SMBAction.RIGHT_JUMP)
        ]
        duration_examples = [
            example for example in examples if example.primitive_duration_mask > 0.0
        ]
        release_examples = [example for example in examples if example.primitive_release_mask > 0.0]
        positive_release_examples = [
            example for example in examples if example.primitive_release > 0.0
        ]
        hazard_window_examples = [
            example for example in examples if example.primitive_hazard_window > 0.0
        ]
        positive_cancel_examples = [
            example for example in examples if example.primitive_cancel > 0.0
        ]
        positive_replan_examples = [
            example for example in examples if example.primitive_replan > 0.0
        ]

        self.assertEqual(len(jump_examples), 18)
        self.assertTrue(
            all(example.primitive_button_combo == example.action for example in examples)
        )
        self.assertTrue(all(example.primitive_button_combo_mask == 1.0 for example in examples))
        self.assertEqual(len(duration_examples), 1)
        self.assertEqual(
            duration_examples[0].primitive_duration_bin,
            int(
                torch.abs(
                    torch.as_tensor(DEFAULT_PRIMITIVE_DURATION_BINS, dtype=torch.float32) - 18.0
                )
                .argmin()
                .item()
            ),
        )
        self.assertEqual(len(release_examples), 18)
        self.assertEqual(len(positive_release_examples), 1)
        self.assertEqual(
            {example.primitive_post_release for example in jump_examples},
            {int(SMBAction.RIGHT)},
        )
        self.assertGreater(len(hazard_window_examples), len(jump_examples))
        self.assertEqual(len(positive_cancel_examples), 1)
        self.assertGreaterEqual(len(positive_replan_examples), 3)
        self.assertTrue(all(example.primitive_weight == 3.0 for example in jump_examples))
        summary = distill_module._dataset_summary(examples)
        self.assertEqual(summary["primitive_button_combo_supervision_count"], len(examples))
        self.assertEqual(
            summary["primitive_cancel_positive_count"],
            float(len(positive_cancel_examples)),
        )
        self.assertEqual(
            summary["primitive_hazard_window_positive_count"],
            float(len(hazard_window_examples)),
        )
        self.assertEqual(summary["primitive_outcome_supervision_count"], len(examples))

    def test_primitive_loss_uses_scripted_duration_release_targets(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=("level_5_enemy_hop.json",),
            required_monte_carlo_families=(),
            rollout_steps=45,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=45,
            device="cpu",
        )
        examples = collect_scripted_distillation_examples(
            config,
            vision_factory=static_vision_factory,
        )
        example = next(example for example in examples if example.primitive_duration_mask > 0.0)
        motor_primitives = SimpleNamespace(
            button_combo_logits=torch.zeros(1, 1, len(SMBAction)),
            hold_duration_logits=torch.zeros(1, 1, len(DEFAULT_PRIMITIVE_DURATION_BINS)),
            release_logit=torch.zeros(1, 1),
            cancel_logit=torch.zeros(1, 1),
            interrupt_logit=torch.zeros(1, 1),
            post_release_logits=torch.zeros(1, 1, len(SMBAction)),
        )

        loss = distill_module._block_smb_distillation_primitive_loss(
            motor_primitives,
            [example],
            device=torch.device("cpu"),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_k_step_primitive_outcome_targets_capture_bad_future(self):
        first = distill_module.BlockSMBDistillationExample(
            batch=_primitive_outcome_batch(x=0.0),
            next_batch=_primitive_outcome_batch(x=0.05),
            action=int(SMBAction.RIGHT_JUMP),
            scenario_name="synthetic_gap",
            episode=0,
            step_index=0,
        )
        second = distill_module.BlockSMBDistillationExample(
            batch=_primitive_outcome_batch(x=0.05),
            next_batch=_primitive_outcome_batch(
                x=0.05,
                support=(1.0, 0.0, 0.0),
                death=True,
                terminated=True,
            ),
            action=int(SMBAction.RIGHT_JUMP),
            scenario_name="synthetic_gap",
            episode=0,
            step_index=1,
            primitive_cancel=1.0,
            primitive_replan=1.0,
        )

        annotated = distill_module._annotate_primitive_outcomes(
            [first, second],
            horizon=2,
        )

        self.assertEqual(annotated[0].primitive_outcome_mask, 1.0)
        self.assertGreater(annotated[0].primitive_outcome_progress_delta, 0.0)
        self.assertEqual(annotated[0].primitive_outcome_support_loss, 1.0)
        self.assertEqual(annotated[0].primitive_outcome_collision_death_risk, 1.0)
        self.assertEqual(annotated[0].primitive_outcome_terminal, 1.0)
        self.assertEqual(annotated[0].primitive_outcome_continue, 0.0)
        self.assertEqual(annotated[0].primitive_outcome_cancel, 1.0)
        self.assertEqual(annotated[0].primitive_outcome_replan, 1.0)

    def test_detached_death_rollout_produces_nonzero_collision_death_risk_targets(self):
        def assert_no_tensors(value):
            self.assertNotIsInstance(value, torch.Tensor)
            if isinstance(value, dict):
                for item in value.values():
                    assert_no_tensors(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    assert_no_tensors(item)

        _name, scenario = distill_module.load_fixed_scenarios(("level_2_gap.json",))[0]
        env = distill_module.MarioScenarioEnv()
        stage = distill_module.BlockSMBStage(
            env=env,
            scenario=scenario,
            vision=static_vision_factory(),
        )
        examples = []
        try:
            observation = stage.reset(seed=7)
            for step_index in range(80):
                action = int(SMBAction.RIGHT)  # walk into the gap until Mario dies
                batch = distill_module._detach_batch(stage.encode_observation(observation))
                next_observation, _reward, terminated, truncated, info = stage.step(action)
                next_batch = distill_module._detach_batch(
                    stage.encode_observation(next_observation, dict(info))
                )
                examples.append(
                    distill_module.BlockSMBDistillationExample(
                        batch=batch,
                        next_batch=next_batch,
                        action=action,
                        scenario_name="level_2_gap.json",
                        episode=0,
                        step_index=step_index,
                    )
                )
                observation = next_observation
                if terminated or truncated:
                    break
        finally:
            env.close()

        self.assertTrue(examples[-1].next_batch.metadata["info"]["death"])
        for example in examples:
            for batch in (example.batch, example.next_batch):
                assert_no_tensors(batch.metadata.get("info"))
                assert_no_tensors(batch.metadata.get("episode"))

        annotated = distill_module._annotate_primitive_outcomes(examples, horizon=8)

        self.assertGreater(
            max(example.primitive_outcome_collision_death_risk for example in annotated),
            0.0,
        )
        self.assertGreater(
            max(example.primitive_outcome_terminal for example in annotated),
            0.0,
        )

    def test_primitive_outcome_loss_uses_k_step_targets(self):
        example = distill_module.BlockSMBDistillationExample(
            batch=_primitive_outcome_batch(x=0.0),
            next_batch=_primitive_outcome_batch(x=0.1),
            action=int(SMBAction.RIGHT),
            scenario_name="synthetic_flat",
            episode=0,
            step_index=0,
            primitive_outcome_mask=1.0,
            primitive_outcome_progress_delta=0.1,
            primitive_outcome_support_loss=0.0,
            primitive_outcome_collision_death_risk=0.0,
            primitive_outcome_terminal=0.0,
            primitive_outcome_continue=1.0,
            primitive_outcome_cancel=0.0,
            primitive_outcome_replan=0.0,
        )
        primitive_outcome = SimpleNamespace(
            progress_delta=torch.tensor([0.0]),
            support_loss_logit=torch.tensor([0.0]),
            collision_death_logit=torch.tensor([0.0]),
            terminal_logit=torch.tensor([0.0]),
            continue_logit=torch.tensor([0.0]),
            cancel_logit=torch.tensor([0.0]),
            replan_logit=torch.tensor([0.0]),
        )

        loss = distill_module._block_smb_distillation_primitive_outcome_loss(
            primitive_outcome,
            [example],
            device=torch.device("cpu"),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_cli_passes_monte_carlo_distillation_config(self):
        with patch(
            "retroagi.stages.block_smb.distill.train_distilled_block_smb_policy",
            return_value={"ok": True},
        ) as train:
            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = distill_main(
                    [
                        "--checkpoint",
                        "data/block_smb/distilled.pth",
                        "--vision-checkpoint",
                        "data/pipeline/block_vit.pth",
                        "--epochs",
                        "1",
                        "--primitive-loss-weight",
                        "0.6",
                        "--primitive-hazard-weight-multiplier",
                        "3.5",
                        "--primitive-outcome-loss-weight",
                        "0.7",
                        "--primitive-outcome-horizon",
                        "6",
                        "--monte-carlo-samples",
                        "3",
                        "--monte-carlo-seed",
                        "60002",
                        "--monte-carlo-family-weight",
                        "flat_run=1",
                        "--monte-carlo-parameter-sweep",
                        "--monte-carlo-sweep-repeats-per-difficulty",
                        "2",
                        "--monte-carlo-validation-samples",
                        "4",
                        "--monte-carlo-test-samples",
                        "5",
                        "--monte-carlo-pass-rate-gate",
                        "0.8",
                        "--monte-carlo-family-pass-rate-gate",
                        "0.7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stream.getvalue()), {"ok": True})
        config = train.call_args.args[0]
        self.assertEqual(config.checkpoint_path, Path("data/block_smb/distilled.pth"))
        self.assertEqual(config.vision_checkpoint, Path("data/pipeline/block_vit.pth"))
        self.assertEqual(config.monte_carlo_samples, 3)
        self.assertEqual(config.primitive_loss_weight, 0.6)
        self.assertEqual(config.primitive_hazard_weight_multiplier, 3.5)
        self.assertEqual(config.primitive_outcome_loss_weight, 0.7)
        self.assertEqual(config.primitive_outcome_horizon, 6)
        self.assertEqual(config.monte_carlo_seed, 60002)
        self.assertEqual(config.monte_carlo_family_weights, {"flat_run": 1.0})
        self.assertTrue(config.monte_carlo_parameter_sweep)
        self.assertEqual(config.monte_carlo_sweep_repeats_per_difficulty, 2)
        self.assertEqual(
            config.required_monte_carlo_families,
            DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
        )
        self.assertEqual(config.required_monte_carlo_repeats_per_difficulty, 1)
        self.assertEqual(config.monte_carlo_validation_samples, 4)
        self.assertEqual(config.monte_carlo_test_samples, 5)
        self.assertEqual(config.monte_carlo_pass_rate_gate, 0.8)
        self.assertEqual(config.monte_carlo_family_pass_rate_gate, 0.7)

    def test_cli_defaults_to_failure_focused_distillation_weights(self):
        with patch(
            "retroagi.stages.block_smb.distill.train_distilled_block_smb_policy",
            return_value={"ok": True},
        ) as train:
            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = distill_main(
                    [
                        "--checkpoint",
                        "data/block_smb/distilled.pth",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stream.getvalue()), {"ok": True})
        config = train.call_args.args[0]
        self.assertEqual(
            config.monte_carlo_family_weights,
            distill_module.default_block_smb_failure_focus_monte_carlo_family_weights(),
        )

    def test_training_config_preserves_distillation_vision_checkpoint(self):
        config = BlockSMBDistillationConfig(vision_checkpoint=Path("data/pipeline/block_vit.pth"))

        training_config = distill_module._training_config_from_distillation(config)

        self.assertEqual(
            training_config.vision_checkpoint_path,
            Path("data/pipeline/block_vit.pth"),
        )
        self.assertEqual(
            training_config.monte_carlo_train_samples_per_epoch,
            distill_module.DEFAULT_BLOCK_SMB_MC_TRAIN_SAMPLES,
        )
        self.assertEqual(
            training_config.monte_carlo_validation_samples,
            distill_module.DEFAULT_BLOCK_SMB_MC_VALIDATION_SAMPLES,
        )
        self.assertEqual(
            training_config.monte_carlo_test_samples,
            distill_module.DEFAULT_BLOCK_SMB_MC_TEST_SAMPLES,
        )

    def test_monte_carlo_samples_are_total_distillation_volume(self):
        config = BlockSMBDistillationConfig(
            fixed_scenarios=(),
            monte_carlo_samples=50,
            required_monte_carlo_families=DEFAULT_BLOCK_SMB_WARM_START_MC_FAMILIES,
            required_monte_carlo_repeats_per_difficulty=1,
            rollout_steps=2,
            episodes_per_scenario=1,
            evaluation_episodes=1,
            evaluation_max_steps=2,
            device="cpu",
        )

        scenarios, _scripts, summary = build_block_smb_distillation_scenarios(config)
        training_config = distill_module._training_config_from_distillation(config)

        self.assertEqual(len(scenarios), 50)
        self.assertEqual(summary["monte_carlo"]["requested_sample_count"], 50)
        self.assertEqual(summary["monte_carlo"]["sample_count"], 50)
        self.assertEqual(training_config.monte_carlo_train_samples_per_epoch, 50)


if __name__ == "__main__":
    unittest.main()
