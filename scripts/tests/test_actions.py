"""Tests for the action vocabulary shared by Block SMB and Full SMB."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from retroagi.core import (
    SMB_ACTION_SPECS,
    SMB_ACTIONS,
    ActionSpec,
    ContinuousControlSpec,
    SMBAction,
    SMBJumpActionTerminator,
    SMBParameterizedPrimitiveExecutor,
    SMBWalkActionLimiter,
    VisionOutput,
    action_backend_id,
    action_button_vector,
    block_smb_action,
    coerce_action_spec,
    coerce_smb_action,
    full_smb_action,
    is_smb_walk_action,
    smb_action_spec,
    smb_jump_release_action,
)


class TestSMBActionVocabulary(unittest.TestCase):
    def test_ids_preserve_block_smb_action_order(self):
        self.assertEqual(
            [(action.name, action.value) for action in SMB_ACTIONS],
            [
                ("NOOP", 0),
                ("RIGHT", 1),
                ("RIGHT_JUMP", 2),
                ("LEFT", 3),
                ("LEFT_JUMP", 4),
                ("JUMP", 5),
            ],
        )
        self.assertEqual(block_smb_action(SMBAction.LEFT_JUMP), 4)
        self.assertIs(coerce_smb_action(2), SMBAction.RIGHT_JUMP)
        self.assertEqual(action_backend_id(smb_action_spec(SMBAction.LEFT_JUMP)), 4)

    def test_full_smb_mapping_uses_button_names_not_positions(self):
        buttons = ("A", "RIGHT", "B", "LEFT", "START")
        mapped = full_smb_action(SMBAction.RIGHT_JUMP, buttons)

        np.testing.assert_array_equal(mapped, np.array([1, 1, 0, 0, 0], dtype=np.int8))

    def test_noop_releases_every_full_smb_button(self):
        mapped = full_smb_action(SMBAction.NOOP, ("LEFT", "RIGHT", "A", "B"))
        np.testing.assert_array_equal(mapped, np.zeros(4, dtype=np.int8))
        self.assertTrue(smb_action_spec(SMBAction.NOOP).is_noop)

    def test_generic_action_specs_support_buttons_release_and_continuous_axes(self):
        throttle = ActionSpec(
            name="throttle",
            stable_id=0,
            kind="continuous",
            continuous_controls=(ContinuousControlSpec("x", 0.75),),
        )
        jump = ActionSpec(name="jump", stable_id=1, buttons=("A",), backend_action_id=7)
        release = ActionSpec(name="release", stable_id=2, release_all=True)
        action_space = (throttle, jump, release)

        self.assertIs(coerce_action_spec(action_space, "jump"), jump)
        self.assertEqual(action_backend_id(jump), 7)
        np.testing.assert_array_equal(
            action_button_vector(jump, ("LEFT", "A")),
            np.array([0, 1], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            action_button_vector(release, ("LEFT", "A")),
            np.zeros(2, dtype=np.int8),
        )
        with self.assertRaisesRegex(ValueError, "continuous action"):
            action_backend_id(throttle)

    def test_invalid_action_and_button_layout_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "invalid SMB action"):
            coerce_smb_action(99)
        with self.assertRaisesRegex(ValueError, "missing.*A"):
            full_smb_action(SMBAction.JUMP, ("LEFT", "RIGHT", "B"))
        with self.assertRaisesRegex(ValueError, "not in this action space"):
            coerce_action_spec(SMB_ACTION_SPECS[:1], SMB_ACTION_SPECS[-1])
        with self.assertRaisesRegex(ValueError, "discrete action"):
            ActionSpec(
                name="bad",
                stable_id=0,
                continuous_controls=(ContinuousControlSpec("x", 0.5),),
            )

    def test_jump_release_actions_preserve_horizontal_intent(self):
        self.assertIs(smb_jump_release_action(SMBAction.RIGHT_JUMP), SMBAction.RIGHT)
        self.assertIs(smb_jump_release_action(SMBAction.LEFT_JUMP), SMBAction.LEFT)
        self.assertIs(smb_jump_release_action(SMBAction.JUMP), SMBAction.NOOP)

    def test_walk_limiter_defaults_to_no_walk_cap(self):
        limiter = SMBWalkActionLimiter(actions_per_second=2.0)

        self.assertTrue(is_smb_walk_action(SMBAction.RIGHT))
        self.assertTrue(is_smb_walk_action(SMBAction.LEFT))
        self.assertFalse(is_smb_walk_action(SMBAction.RIGHT_JUMP))
        for _ in range(5):
            self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))

    def test_walk_limiter_can_be_configured_explicitly(self):
        limiter = SMBWalkActionLimiter(max_walk_seconds=1.0, actions_per_second=2.0)

        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.NOOP))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))

    def test_walk_limiter_resets_on_non_walk_or_direction_change(self):
        limiter = SMBWalkActionLimiter(max_walk_seconds=1.0, actions_per_second=2.0)

        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.JUMP), int(SMBAction.JUMP))
        self.assertEqual(limiter.filter_action(SMBAction.RIGHT), int(SMBAction.RIGHT))
        self.assertEqual(limiter.filter_action(SMBAction.LEFT), int(SMBAction.LEFT))
        self.assertEqual(limiter.filter_action(SMBAction.LEFT), int(SMBAction.LEFT))
        self.assertEqual(limiter.filter_action(SMBAction.LEFT), int(SMBAction.NOOP))

    def test_jump_terminator_releases_after_vit_support_landing(self):
        terminator = SMBJumpActionTerminator()

        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT_JUMP),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(0)),
            ),
            int(SMBAction.RIGHT_JUMP),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(2)),
            ),
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.RIGHT_JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.RIGHT_JUMP),
        )

    def test_jump_terminator_releases_on_vit_enemy_contact(self):
        terminator = SMBJumpActionTerminator()
        self.assertEqual(
            terminator.filter_action(
                SMBAction.JUMP,
                batch=self._batch_with_vision(self._support_vision(1)),
            ),
            int(SMBAction.JUMP),
        )
        self.assertEqual(
            terminator.filter_action(
                SMBAction.JUMP,
                batch=self._batch_with_vision(self._support_vision(0)),
            ),
            int(SMBAction.JUMP),
        )

        labels = torch.zeros(1, 5, 5, dtype=torch.long)
        labels[0, 2, 2] = 1
        labels[0, 3, 2] = 2
        vision = self._support_vision(
            0,
            semantic_ids=labels,
            semantic_classes=("background", "mario", "enemy"),
        )

        self.assertEqual(
            terminator.filter_action(SMBAction.JUMP, batch=self._batch_with_vision(vision)),
            int(SMBAction.NOOP),
        )

    def test_parameterized_primitive_executor_releases_after_learned_hold(self):
        executor = SMBParameterizedPrimitiveExecutor()
        motor = SimpleNamespace(
            hold_duration_logits=torch.tensor([[[0.0, 4.0, -1.0]]]),
            duration_bin_values=torch.tensor([1.0, 2.0, 4.0]),
            cancel_logit=torch.tensor([[-4.0]]),
        )

        first = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=motor,
            batch=self._batch_with_vision(self._support_vision(1)),
        )
        second = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )
        third = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )
        repeated = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )

        self.assertTrue(first.started)
        self.assertEqual(first.hold_frames, 2)
        self.assertEqual(first.duration_bin_index, 1)
        self.assertEqual(second.action, int(SMBAction.RIGHT_JUMP))
        self.assertEqual(third.action, int(SMBAction.RIGHT))
        self.assertEqual(repeated.action, int(SMBAction.RIGHT))

    def test_support_override_lands_jump_without_vision(self):
        # Engine ground truth drives landing: no vision is consulted at all,
        # and when both are supplied the override wins over the vision
        # support head (which is exactly the misfire being bypassed).
        executor = SMBParameterizedPrimitiveExecutor()
        motor = SimpleNamespace(
            hold_duration_logits=torch.tensor([[[0.0, 4.0, -1.0]]]),
            duration_bin_values=torch.tensor([1.0, 2.0, 4.0]),
            cancel_logit=torch.tensor([[-4.0]]),
        )

        first = executor.execute(
            SMBAction.RIGHT_JUMP, motor_primitives=motor, support_override="ground"
        )
        second = executor.execute(
            SMBAction.RIGHT_JUMP, motor_primitives=motor, support_override="air"
        )
        # Vision still claims airborne; the engine says landed. Engine wins.
        landed = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=motor,
            batch=self._batch_with_vision(self._support_vision(0)),
            support_override="ground",
        )

        self.assertTrue(first.started)
        self.assertTrue(second.active)
        self.assertTrue(landed.landed)
        self.assertTrue(landed.released)

    def test_parameterized_primitive_executor_samples_durations_when_enabled(self):
        motor = SimpleNamespace(
            hold_duration_logits=torch.zeros(1, 1, 8),
            duration_bin_values=torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]),
        )

        # Default (argmax) mode: flat logits always resolve to the same bin.
        argmax_executor = SMBParameterizedPrimitiveExecutor()
        argmax_bins = set()
        for _ in range(12):
            execution = argmax_executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            )
            argmax_bins.add(execution.duration_bin_index)
            argmax_executor.reset()
        self.assertEqual(argmax_bins, {0})

        # Sampling mode: flat logits explore many bins, deterministically per
        # seed so rollouts stay reproducible.
        sampler = SMBParameterizedPrimitiveExecutor(duration_sampling=True, duration_seed=11)
        sampled_bins = []
        for _ in range(24):
            execution = sampler.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            )
            sampled_bins.append(execution.duration_bin_index)
            sampler.reset()
        self.assertGreater(len(set(sampled_bins)), 3)
        replay = SMBParameterizedPrimitiveExecutor(duration_sampling=True, duration_seed=11)
        replay_bins = []
        for _ in range(24):
            execution = replay.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            )
            replay_bins.append(execution.duration_bin_index)
            replay.reset()
        self.assertEqual(sampled_bins, replay_bins)

        with self.assertRaises(ValueError):
            SMBParameterizedPrimitiveExecutor(duration_temperature=0.0)

    def test_parameterized_primitive_executor_commits_through_interrupt_spikes(self):
        # Regression for cancel-head truncation: spiking cancel/release logits
        # mid-flight must NOT abort a committed hold. Only landing, enemy
        # contact, or hold exhaustion end a started jump.
        executor = SMBParameterizedPrimitiveExecutor()
        hold_motor = SimpleNamespace(
            hold_duration_logits=torch.tensor([[[-1.0, 0.0, 4.0]]]),
            duration_bin_values=torch.tensor([1.0, 2.0, 4.0]),
            release_logit=torch.tensor([[-4.0]]),
            cancel_logit=torch.tensor([[-4.0]]),
        )
        interrupt_motor = SimpleNamespace(
            hold_duration_logits=torch.tensor([[[-1.0, 0.0, 4.0]]]),
            duration_bin_values=torch.tensor([1.0, 2.0, 4.0]),
            release_logit=torch.tensor([[4.0]]),
            cancel_logit=torch.tensor([[4.0]]),
        )

        first = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=hold_motor,
            batch=self._batch_with_vision(self._support_vision(1)),
        )
        second = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=interrupt_motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )
        third = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=interrupt_motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )
        fourth = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=interrupt_motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )
        fifth = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=interrupt_motor,
            batch=self._batch_with_vision(self._support_vision(0)),
        )

        self.assertTrue(first.started)
        self.assertEqual(first.hold_frames, 4)
        # Interrupt spikes are ignored: the hold keeps emitting the jump.
        self.assertEqual(second.action, int(SMBAction.RIGHT_JUMP))
        self.assertTrue(second.active)
        self.assertFalse(second.released)
        self.assertEqual(third.action, int(SMBAction.RIGHT_JUMP))
        self.assertEqual(fourth.action, int(SMBAction.RIGHT_JUMP))
        # The hold ends only when its committed duration is exhausted.
        self.assertEqual(fifth.action, int(SMBAction.RIGHT))

    @staticmethod
    def _duration_motor(logits: list[float]) -> SimpleNamespace:
        return SimpleNamespace(
            hold_duration_logits=torch.tensor([[logits]]),
            duration_bin_values=torch.tensor([2.0, 4.0, 16.0]),
        )

    def test_adaptive_duration_extends_hold_when_belief_rises(self):
        # B-level re-parameterizes the jump mid-air: when its duration belief
        # rises after initiation (a moving target drifted away), the tracked
        # setpoint climbs and the arc extends beyond the initiation hold.
        executor = SMBParameterizedPrimitiveExecutor()
        short_belief = self._duration_motor([0.0, 6.0, -6.0])  # E[hold] ~ 4
        long_belief = self._duration_motor([-6.0, 0.0, 6.0])  # E[hold] ~ 16

        first = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=short_belief,
            batch=self._batch_with_vision(self._support_vision(1)),
        )
        self.assertTrue(first.started)
        self.assertEqual(first.hold_frames, 4)

        results = [
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=long_belief,
                batch=self._batch_with_vision(self._support_vision(0)),
            )
            for _ in range(16)
        ]
        # A locked snapshot would release on the 5th call; the tracked
        # setpoint slews upward 1 frame/frame and stays ahead of the counter
        # until it saturates near the new 16-frame belief.
        self.assertTrue(all(result.active and not result.released for result in results[:15]))
        self.assertTrue(results[15].released)
        # Initiation logging/credit is untouched by in-flight adaptation.
        self.assertEqual(results[0].hold_frames, 4)

    def test_adaptive_duration_shortens_hold_when_belief_drops(self):
        # The converse interception move: the target closed in, B's duration
        # belief collapses, and the hold releases earlier than committed —
        # but only as fast as the slew limit allows.
        executor = SMBParameterizedPrimitiveExecutor()
        long_belief = self._duration_motor([-6.0, 0.0, 6.0])
        short_belief = self._duration_motor([0.0, 6.0, -6.0])

        first = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=long_belief,
            batch=self._batch_with_vision(self._support_vision(1)),
        )
        self.assertTrue(first.started)
        self.assertEqual(first.hold_frames, 16)

        results = [
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=short_belief,
                batch=self._batch_with_vision(self._support_vision(0)),
            )
            for _ in range(8)
        ]
        # Setpoint decays 1 frame/frame from 16 while the counter climbs, so
        # they meet in the middle: 8 held frames, not 16 and not an instant 4.
        self.assertTrue(all(result.active and not result.released for result in results[:7]))
        self.assertTrue(results[7].released)

    def test_adaptive_duration_ignores_single_frame_noise_spike(self):
        # The old cancel head died for this: one noisy frame must never
        # truncate a committed hold. The slew clamp bounds a single spike to
        # a 1-frame setpoint dip that recovers on the next frame.
        executor = SMBParameterizedPrimitiveExecutor()
        steady = self._duration_motor([0.0, 6.0, -6.0])  # E[hold] ~ 4
        spike = self._duration_motor([6.0, 0.0, -6.0])  # E[hold] ~ 2

        first = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=steady,
            batch=self._batch_with_vision(self._support_vision(1)),
        )
        self.assertTrue(first.started)
        motors = [spike, steady, steady, steady]
        results = [
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(0)),
            )
            for motor in motors
        ]
        # Same release frame as an unperturbed 4-frame hold.
        self.assertTrue(all(result.active and not result.released for result in results[:3]))
        self.assertTrue(results[3].released)
        self.assertEqual(results[3].action, int(SMBAction.RIGHT))

    def test_adaptive_duration_preserves_sampled_exploration(self):
        # With stationary logits the setpoint tracks *changes* in belief, so
        # a sampled exploratory bin is never eroded toward the distribution
        # mean: adaptive and locked executors release on the same frame.
        motor = SimpleNamespace(
            hold_duration_logits=torch.zeros(1, 1, 8),
            duration_bin_values=torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]),
        )
        release_frames = []
        for adaptive in (True, False):
            executor = SMBParameterizedPrimitiveExecutor(
                duration_sampling=True,
                duration_seed=123,
                adaptive_duration=adaptive,
            )
            first = executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            )
            self.assertTrue(first.started)
            for frame in range(2, 40):
                result = executor.execute(
                    SMBAction.RIGHT_JUMP,
                    motor_primitives=motor,
                    batch=self._batch_with_vision(self._support_vision(0)),
                )
                if result.released:
                    release_frames.append(frame)
                    break
        self.assertEqual(len(release_frames), 2)
        self.assertEqual(release_frames[0], release_frames[1])

    def test_parameterized_primitive_executor_requires_non_jump_after_landing(self):
        executor = SMBParameterizedPrimitiveExecutor()
        motor = SimpleNamespace(
            hold_duration_logits=torch.tensor([[[6.0, -1.0]]]),
            duration_bin_values=torch.tensor([1.0, 4.0]),
            cancel_logit=torch.tensor([[-4.0]]),
        )

        self.assertEqual(
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            ).action,
            int(SMBAction.RIGHT_JUMP),
        )
        self.assertEqual(
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(0)),
            ).action,
            int(SMBAction.RIGHT),
        )
        landed = executor.execute(
            SMBAction.RIGHT_JUMP,
            motor_primitives=motor,
            batch=self._batch_with_vision(self._support_vision(1)),
        )
        self.assertTrue(landed.landed)
        self.assertEqual(landed.action, int(SMBAction.RIGHT))
        self.assertEqual(
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            ).action,
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            executor.execute(
                SMBAction.RIGHT,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            ).action,
            int(SMBAction.RIGHT),
        )
        self.assertEqual(
            executor.execute(
                SMBAction.RIGHT_JUMP,
                motor_primitives=motor,
                batch=self._batch_with_vision(self._support_vision(1)),
            ).action,
            int(SMBAction.RIGHT_JUMP),
        )

    def test_parameterized_primitive_executor_times_out_without_landing_signal(self):
        executor = SMBParameterizedPrimitiveExecutor(default_hold_frames=1, max_hold_frames=2)

        first = executor.execute(SMBAction.RIGHT_JUMP)
        self.assertTrue(first.started)
        for _ in range(executor.max_active_frames):
            step = executor.execute(SMBAction.RIGHT_JUMP)
            self.assertFalse(step.started)
        restarted = executor.execute(SMBAction.RIGHT_JUMP)
        self.assertTrue(restarted.started)

        executor.reset()
        executor.execute(SMBAction.RIGHT_JUMP)
        for _ in range(executor.max_active_frames):
            executor.execute(SMBAction.RIGHT_JUMP)
        recovered = executor.execute(SMBAction.LEFT)
        self.assertEqual(recovered.action, int(SMBAction.LEFT))
        self.assertFalse(executor.active)

    @staticmethod
    def _batch_with_vision(vision: VisionOutput):
        return SimpleNamespace(metadata={"vision": vision})

    @staticmethod
    def _support_vision(
        support_id: int,
        *,
        semantic_ids: torch.Tensor | None = None,
        semantic_classes: tuple[str, ...] = ("background", "mario", "enemy"),
    ) -> VisionOutput:
        if semantic_ids is None:
            semantic_ids = torch.zeros(1, 5, 5, dtype=torch.long)
            semantic_ids[0, 2, 2] = 1
        semantic_logits = torch.zeros(
            semantic_ids.shape[0],
            len(semantic_classes),
            semantic_ids.shape[1],
            semantic_ids.shape[2],
        )
        semantic_logits.scatter_(1, semantic_ids.unsqueeze(1), 1.0)
        return VisionOutput(
            position=torch.zeros(1, 2),
            semantic_logits=semantic_logits,
            semantic_ids=semantic_ids,
            tokens=torch.zeros(1, 1, 4),
            metadata={
                "semantic_classes": semantic_classes,
                "support_classes": ("air", "ground", "platform"),
            },
            support_logits=torch.nn.functional.one_hot(
                torch.tensor([support_id]),
                num_classes=3,
            ).float(),
            support_ids=torch.tensor([support_id]),
        )


if __name__ == "__main__":
    unittest.main()


class TestSteadyPrimitives(unittest.TestCase):
    @staticmethod
    def _motor(logits):
        return SimpleNamespace(
            hold_duration_logits=torch.tensor([[logits]]),
            duration_bin_values=torch.tensor([2.0, 4.0, 16.0]),
        )

    def _air(self):
        return None  # no vision: no support/enemy signals

    def test_walk_commits_for_selected_duration_and_marks_completion(self):
        executor = SMBParameterizedPrimitiveExecutor()
        motor = self._motor([0.0, 6.0, -6.0])  # argmax bin -> 4 frames
        first = executor.execute(SMBAction.RIGHT, motor_primitives=motor)
        self.assertTrue(first.started)
        self.assertEqual(first.action, int(SMBAction.RIGHT))
        self.assertEqual(first.hold_frames, 4)
        # Policy asks for something else mid-walk: the commitment holds.
        second = executor.execute(SMBAction.LEFT, motor_primitives=motor)
        third = executor.execute(SMBAction.LEFT, motor_primitives=motor)
        self.assertEqual(second.action, int(SMBAction.RIGHT))
        self.assertTrue(second.active)
        self.assertFalse(second.released)
        self.assertEqual(third.action, int(SMBAction.RIGHT))
        fourth = executor.execute(SMBAction.LEFT, motor_primitives=motor)
        self.assertEqual(fourth.action, int(SMBAction.RIGHT))
        self.assertTrue(fourth.released)
        # Commitment satisfied: the next call is a fresh decision.
        fifth = executor.execute(SMBAction.LEFT, motor_primitives=motor)
        self.assertTrue(fifth.started)
        self.assertEqual(fifth.action, int(SMBAction.LEFT))

    def test_wait_is_a_first_class_primitive(self):
        # Wait primitives scale the duration menu by 4: bin value 2 -> an
        # 8-frame committed wait, and the longest bin reaches the 64-frame
        # ceiling a moving bridge's cycle can demand.
        executor = SMBParameterizedPrimitiveExecutor()
        motor = self._motor([6.0, 0.0, -6.0])  # argmax bin -> 2 -> 8 frames
        first = executor.execute(SMBAction.NOOP, motor_primitives=motor)
        self.assertTrue(first.started)
        self.assertEqual(first.hold_frames, 8)
        held = 1
        result = first
        for _ in range(10):
            result = executor.execute(SMBAction.RIGHT, motor_primitives=motor)
            self.assertEqual(result.action, int(SMBAction.NOOP))
            held += 1
            if result.released:
                break
        self.assertTrue(result.released)
        self.assertEqual(held, 8)
        long_motor = self._motor([-6.0, 0.0, 6.0])  # argmax bin -> 16 -> 64
        start = executor.execute(SMBAction.NOOP, motor_primitives=long_motor)
        self.assertEqual(start.hold_frames, 64)
        # Walks are unscaled: bin 4 stays a 4-frame walk.
        executor2 = SMBParameterizedPrimitiveExecutor()
        walk = executor2.execute(
            SMBAction.RIGHT, motor_primitives=self._motor([0.0, 6.0, -6.0])
        )
        self.assertEqual(walk.hold_frames, 4)

    def test_steady_disabled_passes_actions_through(self):
        executor = SMBParameterizedPrimitiveExecutor(steady_primitives=False)
        motor = self._motor([0.0, 6.0, -6.0])
        result = executor.execute(SMBAction.RIGHT, motor_primitives=motor)
        self.assertFalse(result.started)
        self.assertEqual(result.action, int(SMBAction.RIGHT))

    def test_adaptive_belief_drop_shortens_walk(self):
        executor = SMBParameterizedPrimitiveExecutor()
        long_belief = self._motor([-6.0, 0.0, 6.0])  # 16 frames
        short_belief = self._motor([6.0, 0.0, -6.0])  # ~2
        first = executor.execute(SMBAction.RIGHT, motor_primitives=long_belief)
        self.assertEqual(first.hold_frames, 16)
        frames = 1
        for _ in range(20):
            result = executor.execute(SMBAction.RIGHT, motor_primitives=short_belief)
            frames += 1
            if result.released:
                break
        # Setpoint decays 1/frame from 16 while the counter climbs: meets in
        # the middle rather than running the full 16 or stopping instantly.
        self.assertLess(frames, 16)
        self.assertGreater(frames, 4)
