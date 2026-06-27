"""Tests for shared actor/world-model/critic model contracts."""

import unittest

import torch
import torch.nn as nn

from retroagi.core import (
    AdaptiveController,
    AgentWorldModelCritic,
    HierarchicalAdaptiveModel,
    MotorPrimitiveController,
    WorldModel,
    WorldModelState,
)


class RecordingAgent(nn.Module):
    def __init__(self, seq_len_a, seq_len_b, vocab_size):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.seq_len_b = seq_len_b
        self.vocab_size = vocab_size
        self.criticisms = []

    def forward(self, src_a, src_b, src_c, criticism=None, tau=1.0):
        self.criticisms.append(None if criticism is None else criticism.detach().clone())
        batch_size = src_a.size(0)
        device = src_a.device
        logits = torch.zeros(batch_size, self.seq_len_a, self.vocab_size, device=device)
        w_b = torch.ones(batch_size, self.seq_len_b, device=device)
        b_b = torch.zeros(batch_size, self.seq_len_b, device=device)
        if criticism is None:
            actions = src_c.clone()
        else:
            actions = src_c + criticism.mean(dim=(1, 2)).unsqueeze(1)
        return logits, actions, w_b, b_b


class EchoWorldModel(nn.Module):
    def forward(self, state, action, w_context, b_context):
        return state + action


class FailingWorldModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("world model should not be called")


class FixedCritic(nn.Module):
    def __init__(self, criticism):
        super().__init__()
        self.register_buffer("criticism", criticism)

    def forward(self, next_state_pred):
        return self.criticism[: next_state_pred.size(0)]


class FixedControllerAgent(nn.Module):
    def __init__(self, seq_len_a, vocab_size, w_b, b_b, schedule="linear"):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.vocab_size = vocab_size
        self.controller = AdaptiveController(schedule=schedule)
        self.register_buffer("w_b", w_b)
        self.register_buffer("b_b", b_b)

    def forward(self, src_a, src_b, src_c, criticism=None, tau=1.0):
        batch_size = src_a.size(0)
        logits = torch.zeros(
            batch_size,
            self.seq_len_a,
            self.vocab_size,
            device=src_a.device,
        )
        actions = src_c.clone()
        return (
            logits,
            actions,
            self.w_b[:batch_size].to(src_a.device),
            self.b_b[:batch_size].to(src_a.device),
        )


class RecordingWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_context = None
        self.b_context = None

    def forward(self, state, action, w_context, b_context):
        self.w_context = w_context.detach().clone()
        self.b_context = b_context.detach().clone()
        return state + action


class TestAdaptiveControllerSchedules(unittest.TestCase):
    def test_constant_schedule_repeats_b_level_gains(self):
        controller = AdaptiveController(schedule="constant")
        x_c = torch.arange(1, 7, dtype=torch.float32).view(1, 6)
        w_b = torch.tensor([[2.0, 4.0]])
        b_b = torch.tensor([[1.0, -1.0]])

        w_context, b_context = controller.expand_context(x_c, w_b, b_b)
        output = controller(x_c, w_b, b_b)

        torch.testing.assert_close(
            w_context,
            torch.tensor([[2.0, 2.0, 2.0, 4.0, 4.0, 4.0]]),
        )
        torch.testing.assert_close(
            b_context,
            torch.tensor([[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]]),
        )
        torch.testing.assert_close(output, w_context * x_c + b_context)

    def test_linear_schedule_interpolates_between_b_level_gains(self):
        controller = AdaptiveController(schedule="linear")
        x_c = torch.ones(1, 6)
        w_b = torch.tensor([[1.0, 4.0]])
        b_b = torch.tensor([[0.0, 6.0]])

        w_context, b_context = controller.expand_context(x_c, w_b, b_b)
        output = controller(x_c, w_b, b_b)

        torch.testing.assert_close(
            w_context,
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 4.0]]),
        )
        torch.testing.assert_close(
            b_context,
            torch.tensor([[0.0, 2.0, 4.0, 6.0, 6.0, 6.0]]),
        )
        torch.testing.assert_close(
            output,
            torch.tensor([[1.0, 4.0, 7.0, 10.0, 10.0, 10.0]]),
        )

    def test_schedule_rejects_invalid_lengths_and_modes(self):
        with self.assertRaisesRegex(ValueError, "controller schedule"):
            AdaptiveController(schedule="quadratic")
        controller = AdaptiveController(schedule="linear")
        with self.assertRaisesRegex(ValueError, "divisible"):
            controller.expand_context(
                torch.zeros(1, 5),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
            )


class TestMotorPrimitiveController(unittest.TestCase):
    def test_decodes_b_stream_primitives_with_lstm_prediction_context(self):
        controller = MotorPrimitiveController(ratio_ab=2, ratio_bc=2, max_hold_duration=5.0)
        logits_a = torch.tensor(
            [
                [
                    [0.0, 1.0, -1.0],
                    [2.0, -2.0, 0.5],
                ]
            ],
            dtype=torch.float32,
        )
        w_pred = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
        b_pred = torch.tensor([[0.0, -1.0, 1.0, 0.25]])
        current = torch.zeros(1, 8)
        next_state = torch.zeros(1, 8)
        next_state[:, 4:] = 2.0

        output = controller(
            logits_a,
            w_pred,
            b_pred,
            current_state=current,
            next_state_pred=next_state,
        )

        self.assertEqual(output.button_combo_logits.shape, (1, 4, 3))
        torch.testing.assert_close(output.button_combo_logits[:, 0], logits_a[:, 0])
        torch.testing.assert_close(output.button_combo_logits[:, 1], logits_a[:, 0])
        torch.testing.assert_close(output.button_combo_logits[:, 2], logits_a[:, 1])
        torch.testing.assert_close(output.button_combo_logits[:, 3], logits_a[:, 1])
        self.assertEqual(output.hold_duration.shape, (1, 4))
        self.assertTrue(torch.all(output.hold_duration >= 1.0))
        self.assertTrue(torch.all(output.hold_duration <= 5.0))
        self.assertEqual(output.release_logit.shape, (1, 4))
        self.assertEqual(output.cancel_logit.shape, (1, 4))
        self.assertEqual(output.confidence.shape, (1, 4))
        self.assertEqual(output.interrupt_logit.shape, (1, 4))
        self.assertEqual(output.replan_probability.shape, (1, 4))
        self.assertGreater(
            output.replan_probability[0, 0].item(),
            output.replan_probability[0, 1].item(),
        )


class TestCriticFeedbackContract(unittest.TestCase):
    def test_critic_feedback_is_additive_a_stream_residual(self):
        model = HierarchicalAdaptiveModel(vocab_size=5, d_model=4, nhead=2, num_layers=1)
        encoded_a = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]],
            ]
        )
        criticism = torch.tensor(
            [
                [[0.5, 0.0, -0.5, 1.0], [1.5, 0.0, -1.5, 2.0]],
                [[-0.5, 0.0, 0.5, -1.0], [-1.5, 0.0, 1.5, -2.0]],
            ]
        )

        refined = model.apply_critic_feedback(encoded_a, criticism)

        torch.testing.assert_close(refined, encoded_a + criticism)

    def test_critic_feedback_rejects_shape_mismatch(self):
        model = HierarchicalAdaptiveModel(vocab_size=5, d_model=4, nhead=2, num_layers=1)

        with self.assertRaisesRegex(ValueError, "criticism shape"):
            model.apply_critic_feedback(torch.zeros(2, 3, 4), torch.zeros(2, 4, 4))

    def test_critic_feedback_rejects_non_floating_tensor(self):
        model = HierarchicalAdaptiveModel(vocab_size=5, d_model=4, nhead=2, num_layers=1)

        with self.assertRaisesRegex(TypeError, "floating-point"):
            model.apply_critic_feedback(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4, dtype=torch.long))

    def test_agent_second_pass_receives_exact_critic_output(self):
        seq_len_a = 2
        seq_len_b = 4
        seq_len_c = 8
        d_model = 4
        vocab_size = 7
        model = AgentWorldModelCritic(
            vocab_size=vocab_size,
            seq_len_a=seq_len_a,
            seq_len_c=seq_len_c,
            ratio_bc=2,
            d_model=d_model,
        )
        fixed_criticism = torch.arange(2 * seq_len_a * d_model, dtype=torch.float32).view(
            2, seq_len_a, d_model
        )
        recording_agent = RecordingAgent(seq_len_a, seq_len_b, vocab_size)
        model.agent = recording_agent
        model.world_model = EchoWorldModel()
        model.critic = FixedCritic(fixed_criticism)

        src_a = torch.zeros(2, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(2, seq_len_b, dtype=torch.long)
        src_c = torch.ones(2, seq_len_c)

        _actions1, _next_state, criticism, actions2, _logits_a, _w_b, _b_b = model(src_a, src_b, src_c)

        self.assertEqual(len(recording_agent.criticisms), 2)
        self.assertIsNone(recording_agent.criticisms[0])
        torch.testing.assert_close(recording_agent.criticisms[1], fixed_criticism)
        torch.testing.assert_close(criticism, fixed_criticism)
        torch.testing.assert_close(actions2, src_c + fixed_criticism.mean(dim=(1, 2)).unsqueeze(1))

    def test_agent_second_pass_can_disable_critic_feedback(self):
        seq_len_a = 2
        seq_len_b = 4
        seq_len_c = 8
        d_model = 4
        vocab_size = 7
        model = AgentWorldModelCritic(
            vocab_size=vocab_size,
            seq_len_a=seq_len_a,
            seq_len_c=seq_len_c,
            ratio_bc=2,
            d_model=d_model,
        )
        fixed_criticism = torch.ones(2, seq_len_a, d_model)
        recording_agent = RecordingAgent(seq_len_a, seq_len_b, vocab_size)
        model.agent = recording_agent
        model.world_model = EchoWorldModel()
        model.critic = FixedCritic(fixed_criticism)

        src_a = torch.zeros(2, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(2, seq_len_b, dtype=torch.long)
        src_c = torch.ones(2, seq_len_c)

        _actions1, _next_state, criticism, actions2, _logits_a, _w_b, _b_b = model(
            src_a,
            src_b,
            src_c,
            critic_feedback_enabled=False,
        )

        self.assertEqual(len(recording_agent.criticisms), 2)
        self.assertIsNone(recording_agent.criticisms[0])
        self.assertIsNone(recording_agent.criticisms[1])
        torch.testing.assert_close(criticism, fixed_criticism)
        torch.testing.assert_close(actions2, src_c)

    def test_world_model_receives_scheduled_controller_context(self):
        seq_len_a = 2
        seq_len_c = 6
        vocab_size = 7
        model = AgentWorldModelCritic(
            vocab_size=vocab_size,
            seq_len_a=seq_len_a,
            seq_len_c=seq_len_c,
            ratio_bc=3,
            d_model=4,
            controller_schedule="linear",
        )
        recording_world_model = RecordingWorldModel()
        model.agent = FixedControllerAgent(
            seq_len_a,
            vocab_size,
            w_b=torch.tensor([[1.0, 4.0]]),
            b_b=torch.tensor([[0.0, 6.0]]),
            schedule="linear",
        )
        model.world_model = recording_world_model
        model.critic = FixedCritic(torch.zeros(1, seq_len_a, 4))

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, 2, dtype=torch.long)
        src_c = torch.ones(1, seq_len_c)

        model(src_a, src_b, src_c)

        torch.testing.assert_close(
            recording_world_model.w_context,
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 4.0]]),
        )
        torch.testing.assert_close(
            recording_world_model.b_context,
            torch.tensor([[0.0, 2.0, 4.0, 6.0, 6.0, 6.0]]),
        )

    def test_world_model_can_be_disabled_for_ablation(self):
        seq_len_a = 2
        seq_len_b = 4
        seq_len_c = 8
        d_model = 4
        vocab_size = 7
        model = AgentWorldModelCritic(
            vocab_size=vocab_size,
            seq_len_a=seq_len_a,
            seq_len_c=seq_len_c,
            ratio_bc=2,
            d_model=d_model,
        )
        fixed_criticism = torch.ones(1, seq_len_a, d_model)
        model.agent = RecordingAgent(seq_len_a, seq_len_b, vocab_size)
        model.world_model = FailingWorldModel()
        model.critic = FixedCritic(fixed_criticism)

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, seq_len_b, dtype=torch.long)
        src_c = torch.arange(seq_len_c, dtype=torch.float32).view(1, seq_len_c)

        (
            _actions1,
            next_state_pred,
            criticism,
            _actions2,
            _logits_a,
            _w_b,
            _b_b,
            next_world_model_state,
        ) = model(
            src_a,
            src_b,
            src_c,
            return_world_model_state=True,
            world_model_enabled=False,
        )

        torch.testing.assert_close(next_state_pred, src_c)
        torch.testing.assert_close(criticism, fixed_criticism)
        self.assertIsNone(next_world_model_state)

    def test_auxiliary_objective_heads_produce_scalar_reward_value_and_representation(self):
        model = AgentWorldModelCritic(
            vocab_size=7,
            seq_len_a=2,
            seq_len_c=8,
            ratio_bc=2,
            d_model=4,
        )
        state = torch.ones(3, 8)

        representation = model.transition_representation(state)
        reward = model.predict_reward(state)
        value = model.predict_value(state)

        self.assertEqual(representation.shape, (3, 4))
        self.assertEqual(reward.shape, (3,))
        self.assertEqual(value.shape, (3,))
        self.assertTrue(torch.isfinite(representation).all())
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(torch.isfinite(value).all())


class TestWorldModelRecurrentBoundaries(unittest.TestCase):
    def test_episode_mask_resets_initial_recurrent_state(self):
        torch.manual_seed(123)
        model = WorldModel(hidden_size=4, num_freqs=0, ratio_bc=2)
        state = torch.zeros(1, 4)
        action = torch.ones(1, 4)
        w_context = torch.ones(1, 4)
        b_context = torch.zeros(1, 4)
        carried = WorldModelState(
            hidden=torch.randn(1, 1, 4),
            cell=torch.randn(1, 1, 4),
        )

        reset_prediction, reset_state = model(
            state,
            action,
            w_context,
            b_context,
            initial_state=carried,
            episode_mask=torch.tensor([0.0]),
            return_state=True,
        )
        fresh_prediction, fresh_state = model(
            state, action, w_context, b_context, return_state=True
        )
        kept_prediction = model(
            state,
            action,
            w_context,
            b_context,
            initial_state=carried,
            episode_mask=torch.tensor([1.0]),
        )

        torch.testing.assert_close(reset_prediction, fresh_prediction)
        torch.testing.assert_close(reset_state.hidden, fresh_state.hidden)
        torch.testing.assert_close(reset_state.cell, fresh_state.cell)
        self.assertGreater(
            torch.max(torch.abs(kept_prediction - fresh_prediction)).item(),
            1e-6,
        )

    def test_chunk_episode_mask_resets_state_inside_sequence(self):
        torch.manual_seed(456)
        model = WorldModel(hidden_size=4, num_freqs=0, ratio_bc=2)
        state = torch.arange(4, dtype=torch.float32).view(1, 4)
        action = torch.ones(1, 4)
        w_context = torch.ones(1, 4)
        b_context = torch.zeros(1, 4)
        carried = WorldModelState(
            hidden=torch.randn(1, 1, 4),
            cell=torch.randn(1, 1, 4),
        )

        chunk_reset_prediction = model(
            state,
            action,
            w_context,
            b_context,
            initial_state=carried,
            episode_mask=torch.tensor([[1.0, 0.0]]),
        )
        first_chunk = model(
            state[:, :2],
            action[:, :2],
            w_context[:, :2],
            b_context[:, :2],
            initial_state=carried,
        )
        second_chunk = model(
            state[:, 2:],
            action[:, 2:],
            w_context[:, 2:],
            b_context[:, 2:],
        )

        torch.testing.assert_close(chunk_reset_prediction[:, :2], first_chunk)
        torch.testing.assert_close(chunk_reset_prediction[:, 2:], second_chunk)


if __name__ == "__main__":
    unittest.main()
