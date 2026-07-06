"""Tests for shared actor/world-model/critic model contracts."""

import unittest

import torch
import torch.nn as nn

from retroagi.core import (
    ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    DEFAULT_PRIMITIVE_DURATION_BINS,
    AdaptiveController,
    AgentWorldModelCritic,
    CriticActionEvaluation,
    HierarchicalAdaptiveModel,
    LevelBPrimitiveParameters,
    MotorPrimitiveController,
    SMBAction,
    WorldModel,
    WorldModelState,
    action_level_world_model_state_dict,
)


class RecordingAgent(nn.Module):
    uses_world_model_context = True

    def __init__(self, seq_len_a, seq_len_b, vocab_size):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.seq_len_b = seq_len_b
        self.vocab_size = vocab_size
        self.criticisms = []
        self.world_model_contexts = []

    def forward(
        self,
        src_a,
        src_b,
        src_c,
        criticism=None,
        tau=1.0,
        *,
        world_model_context=None,
    ):
        self.criticisms.append(None if criticism is None else criticism.detach().clone())
        self.world_model_contexts.append(
            None if world_model_context is None else world_model_context.detach().clone()
        )
        batch_size = src_a.size(0)
        device = src_a.device
        logits = torch.zeros(batch_size, self.seq_len_a, self.vocab_size, device=device)
        w_b = torch.ones(batch_size, self.seq_len_b, device=device)
        b_b = torch.zeros(batch_size, self.seq_len_b, device=device)
        if criticism is None:
            actions = src_c.clone()
        else:
            actions = src_c + criticism.mean(dim=(1, 2)).unsqueeze(1)
        if world_model_context is not None:
            actions = actions + world_model_context.mean(dim=(1, 2)).unsqueeze(1)
        return logits, actions, w_b, b_b


class ScriptedRefinementAgent(nn.Module):
    def __init__(self, seq_len_a, seq_len_b, vocab_size, action_ids, action_values):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.seq_len_b = seq_len_b
        self.vocab_size = vocab_size
        self.action_ids = tuple(int(action_id) for action_id in action_ids)
        self.action_values = tuple(float(value) for value in action_values)
        self.criticisms = []
        self.calls = 0

    def forward(self, src_a, src_b, src_c, criticism=None, tau=1.0):
        self.criticisms.append(None if criticism is None else criticism.detach().clone())
        index = min(self.calls, len(self.action_ids) - 1)
        self.calls += 1
        batch_size = src_a.size(0)
        device = src_a.device
        logits = torch.full(
            (batch_size, self.seq_len_a, self.vocab_size),
            -10.0,
            device=device,
        )
        logits[:, -1, self.action_ids[index]] = 10.0
        actions = torch.full_like(src_c, self.action_values[index])
        w_b = torch.ones(batch_size, self.seq_len_b, device=device)
        b_b = torch.zeros(batch_size, self.seq_len_b, device=device)
        return logits, actions, w_b, b_b


class EchoWorldModel(nn.Module):
    def forward(self, state, action, w_context, b_context):
        return state + action


class FailingWorldModel(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("world model should not be called")


class FixedCritic(nn.Module):
    def __init__(self, criticism, progress_scores=(1.0,), death_risks=(0.0,)):
        super().__init__()
        self.register_buffer("criticism", criticism)
        self.progress_scores = tuple(float(score) for score in progress_scores)
        self.death_risks = tuple(float(risk) for risk in death_risks)
        self.evaluations = 0

    def forward(self, next_state_pred):
        return self.criticism[: next_state_pred.size(0)]

    def evaluate_action(
        self,
        next_state_pred,
        current_state=None,
        *,
        progress_threshold=0.0,
        death_threshold=0.75,
    ):
        del current_state
        index = min(self.evaluations, len(self.progress_scores) - 1)
        risk_index = min(self.evaluations, len(self.death_risks) - 1)
        self.evaluations += 1
        progress_score = torch.full(
            (next_state_pred.size(0),),
            self.progress_scores[index],
            dtype=next_state_pred.dtype,
            device=next_state_pred.device,
        )
        death_risk = torch.full(
            (next_state_pred.size(0),),
            self.death_risks[risk_index],
            dtype=next_state_pred.dtype,
            device=next_state_pred.device,
        )
        return CriticActionEvaluation(
            feedback=self.forward(next_state_pred),
            progress_score=progress_score,
            death_risk=death_risk,
            would_progress=progress_score >= progress_threshold,
            predicts_death=death_risk >= death_threshold,
        )


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
        self.calls = 0

    def forward(self, state, action, w_context, b_context):
        self.calls += 1
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
        self.assertIsNone(output.hold_duration_logits)
        self.assertIsNone(output.post_release_logits)

    def test_decodes_explicit_level_b_primitive_heads(self):
        controller = MotorPrimitiveController(ratio_ab=2, ratio_bc=2)
        logits_a = torch.zeros(1, 2, len(SMBAction))
        w_pred = torch.zeros(1, 4)
        b_pred = torch.zeros(1, 4)
        hold_logits = torch.full((1, 4, len(DEFAULT_PRIMITIVE_DURATION_BINS)), -8.0)
        hold_logits[:, :, 3] = 8.0
        primitive_params = LevelBPrimitiveParameters(
            hold_duration_logits=hold_logits,
            release_logit=torch.ones(1, 4),
            cancel_logit=torch.full((1, 4), -2.0),
            replan_logit=torch.zeros(1, 4),
            post_release_logits=torch.zeros(1, 4, len(SMBAction)),
        )

        output = controller(
            logits_a,
            w_pred,
            b_pred,
            primitive_params=primitive_params,
        )

        torch.testing.assert_close(
            output.hold_duration,
            torch.full((1, 4), DEFAULT_PRIMITIVE_DURATION_BINS[3]),
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(output.release_logit, torch.ones(1, 4))
        torch.testing.assert_close(output.cancel_logit, torch.full((1, 4), -2.0))
        self.assertIs(output.hold_duration_logits, hold_logits)
        self.assertIs(output.post_release_logits, primitive_params.post_release_logits)

    def test_lstm_prediction_risk_biases_cancel_release_and_replan(self):
        controller = MotorPrimitiveController(ratio_ab=2, ratio_bc=2)
        logits_a = torch.zeros(1, 2, len(SMBAction))
        w_pred = torch.zeros(1, 4)
        b_pred = torch.zeros(1, 4)
        hold_logits = torch.full((1, 4, len(DEFAULT_PRIMITIVE_DURATION_BINS)), -8.0)
        hold_logits[:, :, 2] = 8.0
        primitive_params = LevelBPrimitiveParameters(
            hold_duration_logits=hold_logits,
            release_logit=torch.full((1, 4), -4.0),
            cancel_logit=torch.full((1, 4), -4.0),
            replan_logit=torch.full((1, 4), -4.0),
            post_release_logits=torch.zeros(1, 4, len(SMBAction)),
        )
        current_state = torch.zeros(1, 64)
        current_state[:, 9:12] = torch.tensor([[0.0, 1.0, 0.0]])
        next_good = current_state.clone()
        next_good[:, 0] = 0.03
        next_bad = current_state.clone()
        next_bad[:, 0] = -0.02
        next_bad[:, 9:12] = torch.tensor([[1.0, 0.0, 0.0]])
        next_bad[:, 36:39] = torch.tensor([[1.0, 1.0, 0.0]])

        good = controller(
            logits_a,
            w_pred,
            b_pred,
            current_state=current_state,
            next_state_pred=next_good,
            primitive_params=primitive_params,
        )
        bad = controller(
            logits_a,
            w_pred,
            b_pred,
            current_state=current_state,
            next_state_pred=next_bad,
            primitive_params=primitive_params,
        )

        self.assertTrue(torch.all(bad.cancel_logit > good.cancel_logit))
        self.assertTrue(torch.all(bad.release_logit > good.release_logit))
        self.assertTrue(torch.all(bad.interrupt_logit > good.interrupt_logit))
        self.assertTrue(torch.all(bad.replan_probability > good.replan_probability))
        self.assertTrue(torch.all(bad.predicted_support_risk > 0.0))
        self.assertTrue(torch.all(bad.predicted_terminal_risk > 0.0))
        self.assertTrue(torch.all(bad.prediction_replan_bias > good.prediction_replan_bias))

    def test_leaves_walk_primitives_uncapped_by_default(self):
        controller = MotorPrimitiveController(
            ratio_ab=2,
            ratio_bc=2,
            max_hold_duration=5.0,
        )
        logits_a = torch.full((1, 3, len(SMBAction)), -10.0)
        logits_a[0, 0, int(SMBAction.RIGHT)] = 10.0
        logits_a[0, 1, int(SMBAction.JUMP)] = 10.0
        logits_a[0, 2, int(SMBAction.LEFT)] = 10.0
        w_pred = torch.full((1, 6), 10.0)
        b_pred = torch.zeros(1, 6)

        output = controller(logits_a, w_pred, b_pred)

        self.assertTrue(torch.all(output.hold_duration > 4.9))

    def test_can_cap_walk_primitives_when_configured(self):
        controller = MotorPrimitiveController(
            ratio_ab=2,
            ratio_bc=2,
            max_hold_duration=5.0,
            max_walk_action_duration=1.0,
            walk_action_ids=(SMBAction.RIGHT, SMBAction.LEFT),
        )
        logits_a = torch.full((1, 3, len(SMBAction)), -10.0)
        logits_a[0, 0, int(SMBAction.RIGHT)] = 10.0
        logits_a[0, 1, int(SMBAction.JUMP)] = 10.0
        logits_a[0, 2, int(SMBAction.LEFT)] = 10.0
        w_pred = torch.full((1, 6), 10.0)
        b_pred = torch.zeros(1, 6)

        output = controller(logits_a, w_pred, b_pred)

        torch.testing.assert_close(
            output.hold_duration[:, (0, 1, 4, 5)],
            torch.ones(1, 4),
        )
        self.assertTrue(torch.all(output.hold_duration[:, (2, 3)] > 4.9))

    def test_rejects_invalid_walk_primitive_limits(self):
        with self.assertRaisesRegex(ValueError, "max_walk_action_duration"):
            MotorPrimitiveController(
                ratio_ab=1,
                ratio_bc=1,
                max_walk_action_duration=0.5,
                walk_action_ids=(SMBAction.RIGHT,),
            )
        with self.assertRaisesRegex(ValueError, "walk_action_ids"):
            MotorPrimitiveController(
                ratio_ab=1,
                ratio_bc=1,
                walk_action_ids=(SMBAction.RIGHT, SMBAction.RIGHT),
            )


class TestCriticFeedbackContract(unittest.TestCase):
    def test_actor_action_head_can_be_smaller_than_semantic_vocab(self):
        model = HierarchicalAdaptiveModel(
            vocab_size=20,
            action_vocab_size=len(SMBAction),
            d_model=8,
            nhead=2,
            num_layers=1,
        )
        src_a = torch.tensor([[0, 5]], dtype=torch.long)
        src_b = torch.tensor([[0, 19, 5, 18]], dtype=torch.long)
        src_c = torch.ones(1, 8)

        logits, actions, w_b, b_b = model(src_a, src_b, src_c, tau=1.0)

        self.assertEqual(model.embedding.num_embeddings, 20)
        self.assertEqual(model.action_embedding.num_embeddings, len(SMBAction))
        self.assertEqual(logits.shape, (1, 2, len(SMBAction)))
        self.assertEqual(actions.shape, (1, 8))
        self.assertEqual(w_b.shape, (1, 4))
        self.assertEqual(b_b.shape, (1, 4))
        self.assertIsNotNone(model.last_level_b_primitives)
        self.assertEqual(
            model.last_level_b_primitives.post_release_logits.shape,
            (1, 4, len(SMBAction)),
        )

    def test_policy_checkpoint_migration_skips_obsolete_action_head_shapes(self):
        old_model = AgentWorldModelCritic(
            vocab_size=20,
            seq_len_a=2,
            seq_len_c=8,
            ratio_bc=2,
            d_model=4,
        )
        new_model = AgentWorldModelCritic(
            vocab_size=20,
            action_vocab_size=len(SMBAction),
            seq_len_a=2,
            seq_len_c=8,
            ratio_bc=2,
            d_model=4,
        )

        migrated, skipped = action_level_world_model_state_dict(
            new_model,
            old_model.state_dict(),
        )
        load_result = new_model.load_state_dict(migrated, strict=False)

        self.assertIn("agent.action_embedding.weight", skipped)
        self.assertIn("agent.fc_out_A.weight", skipped)
        self.assertIn("agent.fc_out_A.bias", skipped)
        self.assertIn("agent.fc_primitive_post_release.weight", skipped)
        self.assertIn("agent.fc_primitive_post_release.bias", skipped)
        self.assertEqual(tuple(load_result.unexpected_keys), ())
        self.assertEqual(set(load_result.missing_keys), set(skipped))

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
            model.apply_critic_feedback(
                torch.zeros(2, 3, 4), torch.zeros(2, 3, 4, dtype=torch.long)
            )

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
        model.critic = FixedCritic(
            fixed_criticism,
            progress_scores=(-1.0, 1.0),
            death_risks=(0.0, 0.0),
        )

        src_a = torch.zeros(2, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(2, seq_len_b, dtype=torch.long)
        src_c = torch.ones(2, seq_len_c)

        _actions1, _next_state, criticism, actions2, _logits_a, _w_b, _b_b = model(
            src_a, src_b, src_c
        )

        self.assertEqual(len(recording_agent.criticisms), 2)
        self.assertIsNone(recording_agent.criticisms[0])
        self.assertIsNone(recording_agent.world_model_contexts[0])
        torch.testing.assert_close(recording_agent.criticisms[1], fixed_criticism)
        self.assertIsNone(recording_agent.world_model_contexts[1])
        torch.testing.assert_close(criticism, fixed_criticism)
        torch.testing.assert_close(actions2, src_c + fixed_criticism.mean(dim=(1, 2)).unsqueeze(1))
        self.assertIsNotNone(model.last_action_refinement)
        self.assertEqual(model.last_action_refinement.iterations, 2)
        self.assertTrue(model.last_action_refinement.accepted)

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

        self.assertEqual(len(recording_agent.criticisms), 1)
        self.assertIsNone(recording_agent.criticisms[0])
        self.assertIsNone(recording_agent.world_model_contexts[0])
        torch.testing.assert_close(criticism, fixed_criticism)
        torch.testing.assert_close(actions2, src_c)

    def test_world_model_state_conditions_initial_actor_decision(self):
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
            max_action_refinement_passes=1,
        )
        projection = nn.Linear(model.world_model.hidden_size * 2, d_model, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:, 0] = 1.0
        model.world_model_actor_context = projection
        recording_agent = RecordingAgent(seq_len_a, seq_len_b, vocab_size)
        model.agent = recording_agent
        model.critic = FixedCritic(torch.zeros(1, seq_len_a, d_model))
        carried_state = WorldModelState(
            hidden=torch.zeros(model.world_model.num_layers, 1, model.world_model.hidden_size),
            cell=torch.zeros(model.world_model.num_layers, 1, model.world_model.hidden_size),
        )
        carried_state.hidden[:, :, 0] = 2.0

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, seq_len_b, dtype=torch.long)
        src_c = torch.ones(1, seq_len_c)

        actions1, _next_state, _criticism, actions2, _logits_a, _w_b, _b_b = model(
            src_a,
            src_b,
            src_c,
            world_model_state=carried_state,
        )

        self.assertEqual(len(recording_agent.world_model_contexts), 1)
        torch.testing.assert_close(
            recording_agent.world_model_contexts[0],
            torch.full((1, seq_len_a, d_model), 2.0),
        )
        torch.testing.assert_close(
            model.last_actor_world_model_context, torch.full((1, seq_len_a, d_model), 2.0)
        )
        torch.testing.assert_close(actions1, src_c + 2.0)
        torch.testing.assert_close(actions2, src_c + 2.0)

    def test_episode_mask_resets_actor_lstm_context(self):
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
            max_action_refinement_passes=1,
        )
        projection = nn.Linear(model.world_model.hidden_size * 2, d_model, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:, 0] = 1.0
        model.world_model_actor_context = projection
        recording_agent = RecordingAgent(seq_len_a, seq_len_b, vocab_size)
        model.agent = recording_agent
        model.critic = FixedCritic(torch.zeros(1, seq_len_a, d_model))
        carried_state = WorldModelState(
            hidden=torch.zeros(model.world_model.num_layers, 1, model.world_model.hidden_size),
            cell=torch.zeros(model.world_model.num_layers, 1, model.world_model.hidden_size),
        )
        carried_state.hidden[:, :, 0] = 2.0

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, seq_len_b, dtype=torch.long)
        src_c = torch.ones(1, seq_len_c)

        actions1, _next_state, _criticism, _actions2, _logits_a, _w_b, _b_b = model(
            src_a,
            src_b,
            src_c,
            world_model_state=carried_state,
            episode_mask=torch.tensor([0.0]),
        )

        torch.testing.assert_close(
            recording_agent.world_model_contexts[0],
            torch.zeros(1, seq_len_a, d_model),
        )
        torch.testing.assert_close(actions1, src_c)

    def test_action_refinement_retries_death_and_non_progress_predictions(self):
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
            max_action_refinement_passes=4,
        )
        fixed_criticism = torch.ones(1, seq_len_a, d_model)
        recording_agent = RecordingAgent(seq_len_a, seq_len_b, vocab_size)
        recording_world_model = RecordingWorldModel()
        model.agent = recording_agent
        model.world_model = recording_world_model
        model.critic = FixedCritic(
            fixed_criticism,
            progress_scores=(1.0, -1.0, 1.0),
            death_risks=(0.95, 0.0, 0.0),
        )

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, seq_len_b, dtype=torch.long)
        src_c = torch.ones(1, seq_len_c)

        _actions1, _next_state, criticism, actions2, _logits_a, _w_b, _b_b = model(
            src_a,
            src_b,
            src_c,
        )

        self.assertEqual(recording_world_model.calls, 3)
        self.assertEqual(len(recording_agent.criticisms), 3)
        self.assertIsNone(recording_agent.criticisms[0])
        self.assertIsNone(recording_agent.world_model_contexts[0])
        torch.testing.assert_close(recording_agent.criticisms[1], fixed_criticism)
        self.assertIsNone(recording_agent.world_model_contexts[1])
        torch.testing.assert_close(recording_agent.criticisms[2], fixed_criticism)
        self.assertIsNone(recording_agent.world_model_contexts[2])
        torch.testing.assert_close(criticism, fixed_criticism)
        torch.testing.assert_close(actions2, src_c + 1.0)
        self.assertIsNotNone(model.last_action_refinement)
        self.assertEqual(model.last_action_refinement.iterations, 3)
        self.assertEqual(model.last_action_refinement.selected_iteration, 3)
        self.assertTrue(model.last_action_refinement.accepted)

    def test_action_refinement_retries_non_pause_no_motion_predictions(self):
        seq_len_a = 2
        seq_len_b = 4
        seq_len_c = 8
        d_model = 4
        vocab_size = len(SMBAction)
        model = AgentWorldModelCritic(
            vocab_size=vocab_size,
            seq_len_a=seq_len_a,
            seq_len_c=seq_len_c,
            ratio_bc=2,
            d_model=d_model,
            max_action_refinement_passes=3,
            pause_action_ids=(int(SMBAction.NOOP),),
            motion_position_dims=2,
            critic_motion_threshold=0.01,
        )
        fixed_criticism = torch.ones(1, seq_len_a, d_model)
        recording_agent = ScriptedRefinementAgent(
            seq_len_a,
            seq_len_b,
            vocab_size,
            action_ids=(SMBAction.RIGHT, SMBAction.RIGHT_JUMP),
            action_values=(0.0, 0.25),
        )
        recording_world_model = RecordingWorldModel()
        model.agent = recording_agent
        model.world_model = recording_world_model
        model.critic = FixedCritic(
            fixed_criticism,
            progress_scores=(1.0, 1.0),
            death_risks=(0.0, 0.0),
        )

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, seq_len_b, dtype=torch.long)
        src_c = torch.ones(1, seq_len_c)

        _actions1, _next_state, criticism, actions2, _logits_a, _w_b, _b_b = model(
            src_a,
            src_b,
            src_c,
        )

        self.assertEqual(recording_world_model.calls, 2)
        self.assertEqual(len(recording_agent.criticisms), 2)
        torch.testing.assert_close(criticism, fixed_criticism)
        torch.testing.assert_close(actions2, torch.full_like(src_c, 0.25))
        self.assertIsNotNone(model.last_action_refinement)
        self.assertEqual(model.last_action_refinement.iterations, 2)
        self.assertEqual(model.last_action_refinement.selected_iteration, 2)
        self.assertTrue(model.last_action_refinement.accepted)
        self.assertFalse(bool(model.last_action_refinement.predicts_no_motion.item()))
        self.assertFalse(bool(model.last_action_refinement.is_pause_action.item()))
        torch.testing.assert_close(
            model.last_action_refinement.motion_score,
            torch.tensor([0.25]),
        )

    def test_action_refinement_allows_pause_action_without_motion(self):
        seq_len_a = 2
        seq_len_b = 4
        seq_len_c = 8
        d_model = 4
        vocab_size = len(SMBAction)
        model = AgentWorldModelCritic(
            vocab_size=vocab_size,
            seq_len_a=seq_len_a,
            seq_len_c=seq_len_c,
            ratio_bc=2,
            d_model=d_model,
            max_action_refinement_passes=3,
            pause_action_ids=(int(SMBAction.NOOP),),
            motion_position_dims=2,
            critic_motion_threshold=0.01,
        )
        fixed_criticism = torch.ones(1, seq_len_a, d_model)
        recording_agent = ScriptedRefinementAgent(
            seq_len_a,
            seq_len_b,
            vocab_size,
            action_ids=(SMBAction.NOOP,),
            action_values=(0.0,),
        )
        recording_world_model = RecordingWorldModel()
        model.agent = recording_agent
        model.world_model = recording_world_model
        model.critic = FixedCritic(
            fixed_criticism,
            progress_scores=(1.0,),
            death_risks=(0.0,),
        )

        src_a = torch.zeros(1, seq_len_a, dtype=torch.long)
        src_b = torch.zeros(1, seq_len_b, dtype=torch.long)
        src_c = torch.ones(1, seq_len_c)

        _actions1, _next_state, _criticism, actions2, _logits_a, _w_b, _b_b = model(
            src_a,
            src_b,
            src_c,
        )

        self.assertEqual(recording_world_model.calls, 1)
        torch.testing.assert_close(actions2, torch.zeros_like(src_c))
        self.assertIsNotNone(model.last_action_refinement)
        self.assertEqual(model.last_action_refinement.iterations, 1)
        self.assertTrue(model.last_action_refinement.accepted)
        self.assertFalse(bool(model.last_action_refinement.predicts_no_motion.item()))
        self.assertTrue(bool(model.last_action_refinement.is_pause_action.item()))
        torch.testing.assert_close(
            model.last_action_refinement.motion_score,
            torch.tensor([0.0]),
        )

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
    def test_migrates_token_level_world_model_checkpoint_keys(self):
        model = AgentWorldModelCritic(
            vocab_size=7,
            seq_len_a=2,
            seq_len_c=8,
            ratio_bc=2,
            d_model=4,
        )
        old_state = dict(model.state_dict())
        for key in tuple(old_state):
            if key.startswith(("critic.progress_head.", "critic.death_head.")):
                del old_state[key]
        old_state["world_model.lstm.weight_ih_l0"] = torch.zeros(128, 12)
        old_state["world_model.fc.weight"] = torch.zeros(1, 32)
        old_state["world_model.fc.bias"] = torch.zeros(1)

        migrated, skipped = action_level_world_model_state_dict(model, old_state)
        load_result = model.load_state_dict(migrated, strict=False)
        unsupported_missing = tuple(
            key
            for key in load_result.missing_keys
            if not key.startswith(ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES)
        )

        self.assertIn("world_model.lstm.weight_ih_l0", skipped)
        self.assertIn("world_model.fc.weight", skipped)
        self.assertIn("world_model.fc.bias", skipped)
        self.assertTrue(
            any(key.startswith("critic.progress_head.") for key in load_result.missing_keys)
        )
        self.assertTrue(
            any(key.startswith("critic.death_head.") for key in load_result.missing_keys)
        )
        self.assertEqual(unsupported_missing, ())
        self.assertEqual(tuple(load_result.unexpected_keys), ())

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

    def test_returned_recurrent_state_is_detached(self):
        torch.manual_seed(321)
        model = WorldModel(hidden_size=4, num_freqs=0, ratio_bc=2)
        state = torch.zeros(1, 4, requires_grad=True)
        action = torch.ones(1, 4)
        w_context = torch.ones(1, 4)
        b_context = torch.zeros(1, 4)

        prediction, recurrent_state = model(
            state,
            action,
            w_context,
            b_context,
            return_state=True,
        )

        self.assertTrue(prediction.requires_grad)
        self.assertFalse(recurrent_state.hidden.requires_grad)
        self.assertFalse(recurrent_state.cell.requires_grad)
        prediction.sum().backward()
        self.assertIsNotNone(state.grad)

    def test_world_model_conditions_on_structured_b_primitive_context(self):
        torch.manual_seed(654)
        model = WorldModel(hidden_size=4, num_freqs=0, ratio_bc=2)
        state = torch.zeros(1, 4)
        action = torch.ones(1, 4)
        w_context = torch.ones(1, 4)
        b_context = torch.zeros(1, 4)
        primitive_a = torch.zeros(1, 2, 9)
        primitive_b = torch.ones(1, 2, 9)

        prediction_a = model(
            state,
            action,
            w_context,
            b_context,
            primitive_context=primitive_a,
        )
        outcome_a = model.last_primitive_outcome
        prediction_b = model(
            state,
            action,
            w_context,
            b_context,
            primitive_context=primitive_b,
        )
        outcome_b = model.last_primitive_outcome

        self.assertIsNotNone(outcome_a)
        self.assertIsNotNone(outcome_b)
        self.assertTrue(outcome_a.progress_delta.requires_grad)
        self.assertTrue(outcome_b.progress_delta.requires_grad)
        self.assertEqual(outcome_a.progress_delta.shape, (1,))
        self.assertEqual(outcome_a.cancel_logit.shape, (1,))
        self.assertGreater(torch.max(torch.abs(prediction_a - prediction_b)).item(), 1e-6)
        self.assertGreater(
            torch.max(torch.abs(outcome_a.progress_delta - outcome_b.progress_delta)).item(),
            1e-6,
        )

    def test_agent_builds_b_primitive_context_for_world_model(self):
        model = AgentWorldModelCritic(
            vocab_size=len(SMBAction),
            seq_len_a=2,
            seq_len_c=8,
            ratio_bc=2,
            d_model=4,
        )
        logits_a = torch.zeros(1, 2, len(SMBAction))
        logits_a[:, :, int(SMBAction.RIGHT_JUMP)] = 4.0
        w_b = torch.zeros(1, 4)
        hold_logits = torch.full((1, 4, len(DEFAULT_PRIMITIVE_DURATION_BINS)), -8.0)
        hold_logits[:, :, 4] = 8.0
        primitive_params = LevelBPrimitiveParameters(
            hold_duration_logits=hold_logits,
            release_logit=torch.ones(1, 4),
            cancel_logit=torch.zeros(1, 4),
            replan_logit=torch.full((1, 4), -1.0),
            post_release_logits=torch.zeros(1, 4, len(SMBAction)),
        )

        context = model._level_b_primitive_context(logits_a, primitive_params, w_b)

        self.assertEqual(context.shape, (1, 4, 9))
        self.assertTrue(torch.all(context[:, :, 3] > 0.0))
        self.assertTrue(torch.all(context[:, :, 4] > 0.5))

    def test_chunk_episode_masks_are_not_supported_for_action_level_lstm(self):
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

        with self.assertRaisesRegex(ValueError, "episode_mask"):
            model(
                state,
                action,
                w_context,
                b_context,
                initial_state=carried,
                episode_mask=torch.tensor([[1.0, 0.0]]),
            )


if __name__ == "__main__":
    unittest.main()
