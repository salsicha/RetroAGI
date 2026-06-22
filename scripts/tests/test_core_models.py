"""Tests for shared actor/world-model/critic model contracts."""

import unittest

import torch
import torch.nn as nn

from retroagi.core import (
    AgentWorldModelCritic,
    HierarchicalAdaptiveModel,
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


class FixedCritic(nn.Module):
    def __init__(self, criticism):
        super().__init__()
        self.register_buffer("criticism", criticism)

    def forward(self, next_state_pred):
        return self.criticism[: next_state_pred.size(0)]


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
