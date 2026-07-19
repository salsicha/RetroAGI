"""Reusable hierarchical actor, world-model, and critic components."""

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_CONTROLLER_SCHEDULES = ("constant", "linear")
ACTION_LEVEL_WORLD_MODEL_ALLOWED_MISSING_PREFIXES = (
    "world_model.decoder.",
    "world_model.lstm.weight_ih",
    "world_model.primitive_encoder.",
    "world_model.primitive_outcome_head.",
)
ACTION_LEVEL_WORLD_MODEL_OBSOLETE_PREFIXES = ("world_model.fc.",)
ACTION_REFINEMENT_ALLOWED_MISSING_PREFIXES = (
    "critic.progress_head.",
    "critic.death_head.",
)
LEVEL_B_PRIMITIVE_ALLOWED_MISSING_PREFIXES = (
    "agent.fc_primitive_hold_duration.",
    "agent.fc_primitive_release.",
    "agent.fc_primitive_cancel.",
    "agent.fc_primitive_replan.",
    "agent.fc_primitive_post_release.",
)
ACTION_HEAD_ALLOWED_MISSING_PREFIXES = (
    "agent.action_embedding.",
    "agent.fc_out_A.",
    "agent.fc_primitive_post_release.",
)
ACTOR_WORLD_MODEL_CONTEXT_ALLOWED_MISSING_PREFIXES = ("world_model_actor_context.",)
ACTOR_C_STATE_CONTEXT_ALLOWED_MISSING_PREFIXES = ("agent.c_state_context.",)
ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES = (
    *ACTION_LEVEL_WORLD_MODEL_ALLOWED_MISSING_PREFIXES,
    *ACTION_REFINEMENT_ALLOWED_MISSING_PREFIXES,
    *LEVEL_B_PRIMITIVE_ALLOWED_MISSING_PREFIXES,
    *ACTION_HEAD_ALLOWED_MISSING_PREFIXES,
    *ACTOR_WORLD_MODEL_CONTEXT_ALLOWED_MISSING_PREFIXES,
    *ACTOR_C_STATE_CONTEXT_ALLOWED_MISSING_PREFIXES,
)
DEFAULT_PRIMITIVE_DURATION_BINS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
DEFAULT_ACTION_MOTION_THRESHOLD = 1.0e-4
WORLD_MODEL_PRIMITIVE_FEATURE_DIM = 9
WORLD_MODEL_PRIMITIVE_EMBEDDING_DIM = 16
WORLD_MODEL_PRIMITIVE_OUTCOME_DIM = 7
MOTOR_PRIMITIVE_PROGRESS_MIN_DELTA = 0.005
MOTOR_PRIMITIVE_LOW_PROGRESS_SCALE = 10.0
MOTOR_PRIMITIVE_SUPPORT_RISK_SCALE = 2.0
MOTOR_PRIMITIVE_TERMINAL_RISK_SCALE = 4.0
_MOTOR_PRIMITIVE_SUPPORT_SPANS = ((9, 12), (15, 18))
_MOTOR_PRIMITIVE_TERMINAL_SPANS = ((24, 27), (36, 39))
_MOTOR_PRIMITIVE_PROGRESS_SLOTS = (0, 18)


def action_level_world_model_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Drop obsolete tensors before loading older policy checkpoints."""

    current = model.state_dict()
    migrated: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if key.startswith(ACTION_LEVEL_WORLD_MODEL_OBSOLETE_PREFIXES):
            skipped.append(key)
            continue
        expected = current.get(key)
        if (
            expected is not None
            and key.startswith("world_model.")
            and tuple(value.shape) != tuple(expected.shape)
        ):
            skipped.append(key)
            continue
        if (
            expected is not None
            and key.startswith(ACTION_HEAD_ALLOWED_MISSING_PREFIXES)
            and tuple(value.shape) != tuple(expected.shape)
        ):
            skipped.append(key)
            continue
        migrated[key] = value
    return migrated, tuple(skipped)


@dataclass(frozen=True)
class WorldModelState:
    """LSTM state carried between world-model calls."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def detach(self):
        return WorldModelState(self.hidden.detach(), self.cell.detach())


@dataclass(frozen=True)
class PrimitiveOutcomePrediction:
    """K-step outcome predicted from a structured B-level primitive."""

    progress_delta: torch.Tensor
    support_loss_logit: torch.Tensor
    collision_death_logit: torch.Tensor
    terminal_logit: torch.Tensor
    continue_logit: torch.Tensor
    cancel_logit: torch.Tensor
    replan_logit: torch.Tensor

    def detach(self):
        return PrimitiveOutcomePrediction(
            progress_delta=self.progress_delta.detach(),
            support_loss_logit=self.support_loss_logit.detach(),
            collision_death_logit=self.collision_death_logit.detach(),
            terminal_logit=self.terminal_logit.detach(),
            continue_logit=self.continue_logit.detach(),
            cancel_logit=self.cancel_logit.detach(),
            replan_logit=self.replan_logit.detach(),
        )


@dataclass(frozen=True)
class MotorPrimitiveOutput:
    """B-stream motor primitive parameters decoded for action selection."""

    button_combo_logits: torch.Tensor
    hold_duration: torch.Tensor
    release_logit: torch.Tensor
    cancel_logit: torch.Tensor
    confidence: torch.Tensor
    interrupt_logit: torch.Tensor
    replan_probability: torch.Tensor
    hold_duration_logits: torch.Tensor | None = None
    duration_bin_values: torch.Tensor | None = None
    post_release_logits: torch.Tensor | None = None
    predicted_progress_delta: torch.Tensor | None = None
    predicted_support_risk: torch.Tensor | None = None
    predicted_terminal_risk: torch.Tensor | None = None
    prediction_replan_bias: torch.Tensor | None = None

    def detach(self):
        return MotorPrimitiveOutput(
            button_combo_logits=self.button_combo_logits.detach(),
            hold_duration=self.hold_duration.detach(),
            release_logit=self.release_logit.detach(),
            cancel_logit=self.cancel_logit.detach(),
            confidence=self.confidence.detach(),
            interrupt_logit=self.interrupt_logit.detach(),
            replan_probability=self.replan_probability.detach(),
            hold_duration_logits=(
                self.hold_duration_logits.detach()
                if self.hold_duration_logits is not None
                else None
            ),
            duration_bin_values=(
                self.duration_bin_values.detach() if self.duration_bin_values is not None else None
            ),
            post_release_logits=(
                self.post_release_logits.detach() if self.post_release_logits is not None else None
            ),
            predicted_progress_delta=(
                self.predicted_progress_delta.detach()
                if self.predicted_progress_delta is not None
                else None
            ),
            predicted_support_risk=(
                self.predicted_support_risk.detach()
                if self.predicted_support_risk is not None
                else None
            ),
            predicted_terminal_risk=(
                self.predicted_terminal_risk.detach()
                if self.predicted_terminal_risk is not None
                else None
            ),
            prediction_replan_bias=(
                self.prediction_replan_bias.detach()
                if self.prediction_replan_bias is not None
                else None
            ),
        )


@dataclass(frozen=True)
class LevelBPrimitiveParameters:
    """Explicit primitive-control heads emitted by the B-level transformer."""

    hold_duration_logits: torch.Tensor
    release_logit: torch.Tensor
    cancel_logit: torch.Tensor
    replan_logit: torch.Tensor
    post_release_logits: torch.Tensor


@dataclass(frozen=True)
class CriticActionEvaluation:
    """Critic decision signals for one imagined action candidate."""

    feedback: torch.Tensor
    progress_score: torch.Tensor
    death_risk: torch.Tensor
    would_progress: torch.Tensor
    predicts_death: torch.Tensor
    motion_score: torch.Tensor | None = None
    predicts_no_motion: torch.Tensor | None = None
    is_pause_action: torch.Tensor | None = None


@dataclass(frozen=True)
class ActionRefinementTrace:
    """Diagnostics for the actor/world-model/critic refinement loop."""

    iterations: int
    accepted: bool
    selected_iteration: int
    progress_score: torch.Tensor
    death_risk: torch.Tensor
    motion_score: torch.Tensor | None = None
    predicts_no_motion: torch.Tensor | None = None
    is_pause_action: torch.Tensor | None = None


@dataclass(frozen=True)
class _ActionCandidate:
    logits_a: torch.Tensor
    actions: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor
    primitive_params: LevelBPrimitiveParameters | None
    next_state_pred: torch.Tensor
    primitive_outcome: PrimitiveOutcomePrediction | None
    criticism: torch.Tensor
    evaluation: CriticActionEvaluation
    next_world_model_state: WorldModelState | None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class AdaptiveController(nn.Module):
    """Applies B-level controller parameters at C-level resolution."""

    def __init__(self, schedule="constant"):
        super().__init__()
        if schedule not in SUPPORTED_CONTROLLER_SCHEDULES:
            raise ValueError(
                "controller schedule must be one of "
                f"{SUPPORTED_CONTROLLER_SCHEDULES}, got {schedule!r}"
            )
        self.schedule = schedule

    def expand_context(self, x_c, w, b):
        if x_c.ndim != 2 or w.ndim != 2 or b.ndim != 2:
            raise ValueError("x_c, w, and b must all have shape [batch, length]")
        if w.shape != b.shape:
            raise ValueError(f"w and b shapes must match, got {w.shape} and {b.shape}")
        if x_c.size(0) != w.size(0):
            raise ValueError(
                "x_c, w, and b batch sizes must match, got " f"{x_c.size(0)} and {w.size(0)}"
            )
        if x_c.size(1) % w.size(1) != 0:
            raise ValueError(
                "C length must be divisible by B length, got " f"{x_c.size(1)} and {w.size(1)}"
            )
        ratio_bc = x_c.size(1) // w.size(1)
        if self.schedule == "constant":
            return (
                w.repeat_interleave(ratio_bc, dim=1),
                b.repeat_interleave(ratio_bc, dim=1),
            )

        phase = torch.arange(ratio_bc, device=x_c.device, dtype=x_c.dtype) / float(ratio_bc)
        w = w.to(device=x_c.device, dtype=x_c.dtype)
        b = b.to(device=x_c.device, dtype=x_c.dtype)
        w_next = torch.cat((w[:, 1:], w[:, -1:]), dim=1)
        b_next = torch.cat((b[:, 1:], b[:, -1:]), dim=1)
        w_context = (w.unsqueeze(-1) + (w_next - w).unsqueeze(-1) * phase).reshape(
            x_c.size(0), x_c.size(1)
        )
        b_context = (b.unsqueeze(-1) + (b_next - b).unsqueeze(-1) * phase).reshape(
            x_c.size(0), x_c.size(1)
        )
        return w_context, b_context

    def forward(self, x_c, w, b):
        w_context, b_context = self.expand_context(x_c, w, b)
        return w_context * x_c + b_context


class MotorPrimitiveController(nn.Module):
    """Decodes B-stream controller outputs into motor primitive controls.

    The existing B stream still emits the two controller channels used by the
    C-stream adaptive controller. This decoder interprets those same channels
    as primitive timing and replan controls so old checkpoints remain loadable.
    """

    def __init__(
        self,
        *,
        ratio_ab,
        ratio_bc,
        max_hold_duration=8.0,
        max_walk_action_duration=None,
        walk_action_ids: Iterable[int] = (),
        duration_bins: Iterable[float] = DEFAULT_PRIMITIVE_DURATION_BINS,
    ):
        super().__init__()
        if int(ratio_ab) <= 0:
            raise ValueError("ratio_ab must be positive")
        if int(ratio_bc) <= 0:
            raise ValueError("ratio_bc must be positive")
        if float(max_hold_duration) < 1.0:
            raise ValueError("max_hold_duration must be at least 1")
        if max_walk_action_duration is not None and float(max_walk_action_duration) < 1.0:
            raise ValueError("max_walk_action_duration must be at least 1 when set")
        resolved_walk_action_ids = tuple(int(action_id) for action_id in walk_action_ids)
        if any(action_id < 0 for action_id in resolved_walk_action_ids):
            raise ValueError("walk_action_ids must be non-negative")
        if len(set(resolved_walk_action_ids)) != len(resolved_walk_action_ids):
            raise ValueError("walk_action_ids must be unique")
        resolved_duration_bins = tuple(float(value) for value in duration_bins)
        if not resolved_duration_bins:
            raise ValueError("duration_bins must not be empty")
        if any(value < 1.0 or not math.isfinite(value) for value in resolved_duration_bins):
            raise ValueError("duration_bins must contain finite values >= 1")
        if tuple(sorted(resolved_duration_bins)) != resolved_duration_bins:
            raise ValueError("duration_bins must be sorted")
        self.ratio_ab = int(ratio_ab)
        self.ratio_bc = int(ratio_bc)
        self.max_hold_duration = float(max_hold_duration)
        self.max_walk_action_duration = (
            None if max_walk_action_duration is None else float(max_walk_action_duration)
        )
        self.walk_action_ids = resolved_walk_action_ids
        self.register_buffer(
            "duration_bin_values",
            torch.tensor(resolved_duration_bins, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        logits_a,
        w_pred,
        b_pred,
        *,
        next_state_pred=None,
        current_state=None,
        primitive_params: LevelBPrimitiveParameters | None = None,
    ):
        if logits_a.ndim != 3:
            raise ValueError("logits_a must have shape [batch, seq_len_a, vocab]")
        if w_pred.ndim != 2 or b_pred.ndim != 2:
            raise ValueError("w_pred and b_pred must have shape [batch, seq_len_b]")
        if w_pred.shape != b_pred.shape:
            raise ValueError(
                "w_pred and b_pred shapes must match, got " f"{w_pred.shape} and {b_pred.shape}"
            )
        if logits_a.size(0) != w_pred.size(0):
            raise ValueError(
                "logits_a and B-stream controller batch sizes must match, got "
                f"{logits_a.size(0)} and {w_pred.size(0)}"
            )
        if w_pred.size(1) != logits_a.size(1) * self.ratio_ab:
            raise ValueError(
                "B-stream length must equal A-stream length times ratio_ab, got "
                f"{w_pred.size(1)} and {logits_a.size(1)} * {self.ratio_ab}"
            )

        button_combo_logits = logits_a.repeat_interleave(self.ratio_ab, dim=1)
        confidence = torch.sigmoid(w_pred.abs() + b_pred.abs())
        prediction_signals = self._prediction_control_signals(
            next_state_pred,
            current_state,
            w_pred,
        )
        motion = prediction_signals["motion"]
        prediction_replan_bias = prediction_signals["replan_bias"]
        hold_duration_logits = None
        post_release_logits = None
        if primitive_params is None:
            hold_duration = 1.0 + (self.max_hold_duration - 1.0) * torch.sigmoid(w_pred)
            release_logit = -b_pred
            cancel_logit = b_pred - w_pred + prediction_replan_bias
            release_logit = release_logit + 0.5 * prediction_replan_bias
            interrupt_logit = cancel_logit + (0.05 - motion) + prediction_replan_bias
        else:
            self._validate_primitive_params(primitive_params, w_pred, logits_a)
            hold_duration_logits = primitive_params.hold_duration_logits
            post_release_logits = primitive_params.post_release_logits
            duration_values = self.duration_bin_values.to(
                device=hold_duration_logits.device,
                dtype=hold_duration_logits.dtype,
            )
            hold_probabilities = F.softmax(hold_duration_logits, dim=-1)
            hold_duration = (hold_probabilities * duration_values.view(1, 1, -1)).sum(dim=-1)
            release_logit = primitive_params.release_logit + 0.5 * prediction_replan_bias
            cancel_logit = primitive_params.cancel_logit + prediction_replan_bias
            interrupt_logit = (
                primitive_params.replan_logit + (0.05 - motion) + prediction_replan_bias
            )
        hold_duration = self._cap_walk_hold_duration(hold_duration, button_combo_logits)
        replan_probability = torch.sigmoid(interrupt_logit)
        return MotorPrimitiveOutput(
            button_combo_logits=button_combo_logits,
            hold_duration=hold_duration,
            release_logit=release_logit,
            cancel_logit=cancel_logit,
            confidence=confidence,
            interrupt_logit=interrupt_logit,
            replan_probability=replan_probability,
            hold_duration_logits=hold_duration_logits,
            duration_bin_values=self.duration_bin_values.to(
                device=hold_duration.device,
                dtype=hold_duration.dtype,
            ),
            post_release_logits=post_release_logits,
            predicted_progress_delta=prediction_signals["progress_delta"],
            predicted_support_risk=prediction_signals["support_risk"],
            predicted_terminal_risk=prediction_signals["terminal_risk"],
            prediction_replan_bias=prediction_replan_bias,
        )

    def _validate_primitive_params(
        self,
        primitive_params: LevelBPrimitiveParameters,
        reference_b: torch.Tensor,
        logits_a: torch.Tensor,
    ) -> None:
        expected_b = tuple(reference_b.shape)
        for name in ("release_logit", "cancel_logit", "replan_logit"):
            value = getattr(primitive_params, name)
            if tuple(value.shape) != expected_b:
                raise ValueError(
                    f"primitive {name} must have shape {expected_b}, got {tuple(value.shape)}"
                )
        expected_hold = (*expected_b, int(self.duration_bin_values.numel()))
        if tuple(primitive_params.hold_duration_logits.shape) != expected_hold:
            raise ValueError(
                "primitive hold_duration_logits must have shape "
                f"{expected_hold}, got {tuple(primitive_params.hold_duration_logits.shape)}"
            )
        expected_release = (reference_b.size(0), reference_b.size(1), logits_a.size(-1))
        if tuple(primitive_params.post_release_logits.shape) != expected_release:
            raise ValueError(
                "primitive post_release_logits must have shape "
                f"{expected_release}, got {tuple(primitive_params.post_release_logits.shape)}"
            )

    def _cap_walk_hold_duration(self, hold_duration, button_combo_logits):
        if self.max_walk_action_duration is None or not self.walk_action_ids:
            return hold_duration
        selected_actions = button_combo_logits.argmax(dim=-1)
        walk_action_ids = torch.as_tensor(
            self.walk_action_ids,
            dtype=selected_actions.dtype,
            device=selected_actions.device,
        )
        walk_mask = (selected_actions.unsqueeze(-1) == walk_action_ids).any(dim=-1)
        cap = torch.as_tensor(
            self.max_walk_action_duration,
            dtype=hold_duration.dtype,
            device=hold_duration.device,
        )
        return torch.where(walk_mask, torch.minimum(hold_duration, cap), hold_duration)

    def _prediction_control_signals(self, next_state_pred, current_state, reference):
        zeros = torch.zeros_like(reference)
        if next_state_pred is None or current_state is None:
            return {
                "motion": zeros,
                "progress_delta": zeros,
                "support_risk": zeros,
                "terminal_risk": zeros,
                "replan_bias": zeros,
            }
        self._validate_prediction_state(next_state_pred, current_state)
        motion = self._predicted_motion(next_state_pred, current_state, reference)
        progress_delta = self._predicted_progress_delta(
            next_state_pred,
            current_state,
            reference,
        )
        support_risk = self._predicted_support_risk(
            next_state_pred,
            current_state,
            reference,
        )
        terminal_risk = self._predicted_terminal_risk(
            next_state_pred,
            current_state,
            reference,
        )
        low_progress = (MOTOR_PRIMITIVE_PROGRESS_MIN_DELTA - progress_delta).clamp_min(
            0.0
        ) * MOTOR_PRIMITIVE_LOW_PROGRESS_SCALE
        replan_bias = (
            low_progress
            + MOTOR_PRIMITIVE_SUPPORT_RISK_SCALE * support_risk
            + MOTOR_PRIMITIVE_TERMINAL_RISK_SCALE * terminal_risk
        ).clamp_min(0.0)
        return {
            "motion": motion,
            "progress_delta": progress_delta,
            "support_risk": support_risk,
            "terminal_risk": terminal_risk,
            "replan_bias": replan_bias,
        }

    def _validate_prediction_state(self, next_state_pred, current_state) -> None:
        if next_state_pred.shape != current_state.shape:
            raise ValueError(
                "next_state_pred and current_state must have the same shape for "
                "motor primitive prediction, got "
                f"{next_state_pred.shape} and {current_state.shape}"
            )
        if next_state_pred.ndim != 2:
            raise ValueError("next_state_pred and current_state must have shape [batch, seq_len_c]")

    def _predicted_motion(self, next_state_pred, current_state, reference):
        self._validate_prediction_state(next_state_pred, current_state)
        usable = reference.size(1) * self.ratio_bc
        if next_state_pred.size(1) < usable:
            raise ValueError(
                "C-stream length is too short for motor primitive grouping: "
                f"{next_state_pred.size(1)} < {usable}"
            )
        delta = next_state_pred[:, :usable] - current_state[:, :usable]
        return delta.reshape(delta.size(0), reference.size(1), self.ratio_bc).abs().mean(dim=-1)

    def _predicted_progress_delta(self, next_state_pred, current_state, reference):
        deltas = []
        for slot in _MOTOR_PRIMITIVE_PROGRESS_SLOTS:
            if next_state_pred.size(1) > slot:
                deltas.append(next_state_pred[:, slot] - current_state[:, slot])
        if not deltas:
            return torch.zeros_like(reference)
        progress_delta = torch.stack(deltas, dim=1).amax(dim=1)
        return progress_delta.unsqueeze(1).expand_as(reference)

    def _predicted_support_risk(self, next_state_pred, current_state, reference):
        risks = []
        for start, end in _MOTOR_PRIMITIVE_SUPPORT_SPANS:
            if next_state_pred.size(1) < end:
                continue
            current_support = current_state[:, start:end].clamp(0.0, 1.0)
            predicted_support = next_state_pred[:, start:end].clamp(0.0, 1.0)
            support_score = (current_support.sum(dim=1) - 1.0).abs() + (
                predicted_support.sum(dim=1) - 1.0
            ).abs()
            current_air = current_support[:, 0]
            predicted_air = predicted_support[:, 0]
            current_stable = current_support[:, 1:].amax(dim=1)
            predicted_stable = predicted_support[:, 1:].amax(dim=1)
            air_risk = (predicted_air - current_air).clamp_min(0.0)
            support_loss = (current_stable - predicted_stable).clamp_min(0.0)
            risk = torch.maximum(air_risk, support_loss)
            risks.append(torch.where(support_score <= 0.5, risk, torch.zeros_like(risk)))
        if not risks:
            return torch.zeros_like(reference)
        support_risk = torch.stack(risks, dim=1).amax(dim=1).clamp(0.0, 1.0)
        return support_risk.unsqueeze(1).expand_as(reference)

    def _predicted_terminal_risk(self, next_state_pred, current_state, reference):
        risks = []
        for start, end in _MOTOR_PRIMITIVE_TERMINAL_SPANS:
            if next_state_pred.size(1) < end:
                continue
            current_terminal = current_state[:, start:end]
            predicted_terminal = next_state_pred[:, start:end]
            current_valid = ((current_terminal >= -0.05) & (current_terminal <= 1.05)).all(dim=1)
            predicted_probability = predicted_terminal.clamp(0.0, 1.0).amax(dim=1)
            rising_probability = (predicted_terminal - current_terminal).clamp_min(0.0).amax(dim=1)
            risk = torch.maximum(predicted_probability, rising_probability).clamp(0.0, 1.0)
            risks.append(torch.where(current_valid, risk, torch.zeros_like(risk)))
        if not risks:
            return torch.zeros_like(reference)
        terminal_risk = torch.stack(risks, dim=1).amax(dim=1).clamp(0.0, 1.0)
        return terminal_risk.unsqueeze(1).expand_as(reference)


class HierarchicalAdaptiveModel(nn.Module):
    """Three-level actor: Transformer A -> Transformer B -> adaptive controller."""

    uses_world_model_context = True

    def __init__(
        self,
        vocab_size,
        action_vocab_size=None,
        d_model=64,
        nhead=4,
        num_layers=2,
        controller_schedule="constant",
        seq_len_c=None,
    ):
        super().__init__()
        vocab_size = int(vocab_size)
        action_vocab_size = vocab_size if action_vocab_size is None else int(action_vocab_size)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if action_vocab_size <= 0:
            raise ValueError("action_vocab_size must be positive")
        if action_vocab_size > vocab_size:
            raise ValueError(
                "action_vocab_size must be less than or equal to vocab_size, got "
                f"{action_vocab_size} and {vocab_size}"
            )
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.action_vocab_size = action_vocab_size
        self.seq_len_c = None if seq_len_c is None else int(seq_len_c)
        if self.seq_len_c is not None and self.seq_len_c <= 0:
            raise ValueError("seq_len_c must be positive when provided")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.action_embedding = nn.Embedding(action_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        rng_state = torch.get_rng_state() if self.seq_len_c is not None else None
        self.c_state_context = (
            None if self.seq_len_c is None else nn.Linear(self.seq_len_c, d_model)
        )
        if self.c_state_context is not None:
            nn.init.zeros_(self.c_state_context.weight)
            nn.init.zeros_(self.c_state_context.bias)
            torch.set_rng_state(rng_state)

        encoder_layers_a = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer_A = nn.TransformerEncoder(encoder_layers_a, num_layers)
        self.fc_out_A = nn.Linear(d_model, action_vocab_size)

        decoder_layers_b = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer_B = nn.TransformerDecoder(decoder_layers_b, num_layers)
        self.fc_controller_params = nn.Linear(d_model, 2)
        self.fc_primitive_hold_duration = nn.Linear(
            d_model,
            len(DEFAULT_PRIMITIVE_DURATION_BINS),
        )
        self.fc_primitive_release = nn.Linear(d_model, 1)
        self.fc_primitive_cancel = nn.Linear(d_model, 1)
        self.fc_primitive_replan = nn.Linear(d_model, 1)
        self.fc_primitive_post_release = nn.Linear(d_model, action_vocab_size)
        self.last_level_b_primitives: LevelBPrimitiveParameters | None = None
        self._initialize_primitive_heads()

        self.controller = AdaptiveController(schedule=controller_schedule)

    def _initialize_primitive_heads(self) -> None:
        nn.init.zeros_(self.fc_primitive_release.weight)
        nn.init.constant_(self.fc_primitive_release.bias, -4.0)
        nn.init.zeros_(self.fc_primitive_cancel.weight)
        nn.init.constant_(self.fc_primitive_cancel.bias, -4.0)
        nn.init.zeros_(self.fc_primitive_replan.weight)
        nn.init.constant_(self.fc_primitive_replan.bias, -1.0)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_cross_causal_mask(self, sz_B, sz_A, ratio):
        """Prevents Stream B from looking at Stream A's future predictions."""
        mask = torch.ones(sz_B, sz_A)
        for i in range(sz_B):
            for j in range(sz_A):
                if j <= i // ratio:
                    mask[i, j] = 0.0
        return mask.masked_fill(mask == 1, float("-inf"))

    def apply_critic_feedback(self, encoded_a, criticism):
        """
        Inject critic feedback into the A stream as an additive residual.

        The critic output must match the encoded A tensor exactly:
        `[batch, seq_len_a, d_model]`. No scaling, gating, normalization, or
        detach is applied here; training losses decide how strongly the critic
        pathway is shaped.
        """
        if criticism is None:
            return encoded_a
        if criticism.shape != encoded_a.shape:
            raise ValueError(
                "criticism shape must match encoded A stream shape "
                f"{tuple(encoded_a.shape)}, got {tuple(criticism.shape)}"
            )
        if criticism.device != encoded_a.device:
            raise ValueError(
                "criticism device must match encoded A stream device "
                f"{encoded_a.device}, got {criticism.device}"
            )
        if not torch.is_floating_point(criticism):
            raise TypeError("criticism must be a floating-point tensor")
        return encoded_a + criticism.to(dtype=encoded_a.dtype)

    def apply_world_model_context(self, encoded_a, world_model_context):
        """Inject carried LSTM dynamics context into the A stream."""

        if world_model_context is None:
            return encoded_a
        if world_model_context.shape != encoded_a.shape:
            raise ValueError(
                "world_model_context shape must match encoded A stream shape "
                f"{tuple(encoded_a.shape)}, got {tuple(world_model_context.shape)}"
            )
        if world_model_context.device != encoded_a.device:
            raise ValueError(
                "world_model_context device must match encoded A stream device "
                f"{encoded_a.device}, got {world_model_context.device}"
            )
        if not torch.is_floating_point(world_model_context):
            raise TypeError("world_model_context must be a floating-point tensor")
        return encoded_a + world_model_context.to(dtype=encoded_a.dtype)

    def forward(
        self,
        src_A,
        src_B,
        src_C,
        criticism=None,
        tau=1.0,
        return_hidden=False,
        *,
        world_model_context=None,
    ):
        seq_len_a = src_A.size(1)
        seq_len_b = src_B.size(1)
        ratio_ab = seq_len_b // seq_len_a

        causal_mask_a = self.generate_causal_mask(seq_len_a).to(src_A.device)
        causal_mask_b = self.generate_causal_mask(seq_len_b).to(src_B.device)

        x_a = self.embedding(src_A) * math.sqrt(self.d_model)
        x_a = self.pos_encoder(x_a)
        if self.c_state_context is not None:
            if src_C.ndim != 2 or src_C.shape != (src_A.size(0), self.seq_len_c):
                raise ValueError(
                    f"src_C must have shape [batch, {self.seq_len_c}], got {tuple(src_C.shape)}"
                )
            c_context = torch.tanh(self.c_state_context(src_C.float()))
            x_a = x_a + c_context.to(dtype=x_a.dtype).unsqueeze(1)
        x_a = self.apply_world_model_context(x_a, world_model_context)
        x_a = self.apply_critic_feedback(x_a, criticism)

        hidden_a = self.transformer_A(x_a, mask=causal_mask_a)
        logits_a = self.fc_out_A(hidden_a)

        if self.training:
            probs_a = F.gumbel_softmax(logits_a, tau=tau, hard=True, dim=-1)
        else:
            probs_a = F.softmax(logits_a, dim=-1)
        pred_emb_a = torch.matmul(probs_a, self.action_embedding.weight)

        x_b = self.embedding(src_B) * math.sqrt(self.d_model)
        x_b = self.pos_encoder(x_b)
        cross_mask = self.generate_cross_causal_mask(seq_len_b, seq_len_a, ratio_ab).to(
            src_B.device
        )
        hidden_b = self.transformer_B(
            tgt=x_b, memory=pred_emb_a, tgt_mask=causal_mask_b, memory_mask=cross_mask
        )

        controller_params = self.fc_controller_params(hidden_b)
        w_pred = controller_params[:, :, 0]
        b_pred = controller_params[:, :, 1]
        self.last_level_b_primitives = LevelBPrimitiveParameters(
            hold_duration_logits=self.fc_primitive_hold_duration(hidden_b),
            release_logit=self.fc_primitive_release(hidden_b).squeeze(-1),
            cancel_logit=self.fc_primitive_cancel(hidden_b).squeeze(-1),
            replan_logit=self.fc_primitive_replan(hidden_b).squeeze(-1),
            post_release_logits=self.fc_primitive_post_release(hidden_b),
        )

        y_hat_c = self.controller(src_C, w_pred, b_pred)
        if return_hidden:
            return logits_a, y_hat_c, w_pred, b_pred, hidden_a
        return logits_a, y_hat_c, w_pred, b_pred


class WorldModel(nn.Module):
    """
    Predicts the state at the end of the selected action.

    The LSTM advances once per actor decision. A position-wise decoder then
    expands that action-level recurrent state back to the C-stream contract.
    """

    uses_primitive_context = True

    def __init__(
        self,
        hidden_size=32,
        num_layers=1,
        num_freqs=4,
        ratio_bc=4,
        primitive_feature_dim=WORLD_MODEL_PRIMITIVE_FEATURE_DIM,
        primitive_embedding_dim=WORLD_MODEL_PRIMITIVE_EMBEDDING_DIM,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_freqs = num_freqs
        self.ratio_bc = ratio_bc
        self.primitive_feature_dim = int(primitive_feature_dim)
        self.primitive_embedding_dim = int(primitive_embedding_dim)
        if self.primitive_feature_dim <= 0:
            raise ValueError("primitive_feature_dim must be positive")
        if self.primitive_embedding_dim <= 0:
            raise ValueError("primitive_embedding_dim must be positive")
        self.primitive_encoder = nn.Sequential(
            nn.Linear(self.primitive_feature_dim * 4, self.primitive_embedding_dim),
            nn.Tanh(),
        )
        input_size = 16 + self.primitive_embedding_dim
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        decoder_input_size = hidden_size + 4 + num_freqs * 2 + self.primitive_embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.primitive_outcome_head = nn.Sequential(
            nn.Linear(hidden_size + self.primitive_embedding_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, WORLD_MODEL_PRIMITIVE_OUTCOME_DIM),
        )
        self.last_primitive_outcome: PrimitiveOutcomePrediction | None = None

    def initial_state(self, batch_size, device, dtype=torch.float32):
        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        cell = torch.zeros_like(hidden)
        return WorldModelState(hidden, cell)

    def _coerce_state(self, state, batch_size, device, dtype):
        if state is None:
            return self.initial_state(batch_size, device, dtype=dtype)
        if isinstance(state, WorldModelState):
            hidden, cell = state.hidden, state.cell
        else:
            hidden, cell = state
        expected = (self.num_layers, batch_size, self.hidden_size)
        if tuple(hidden.shape) != expected or tuple(cell.shape) != expected:
            raise ValueError(
                "world model recurrent state must have hidden and cell shape "
                f"{expected}, got {tuple(hidden.shape)} and {tuple(cell.shape)}"
            )
        return WorldModelState(
            hidden.to(device=device, dtype=dtype),
            cell.to(device=device, dtype=dtype),
        )

    def _normalize_episode_mask(self, episode_mask, batch_size, chunk_count, device, dtype):
        chunk_masks = torch.ones(batch_size, chunk_count, device=device, dtype=dtype)
        if episode_mask is None:
            return chunk_masks

        mask = torch.as_tensor(episode_mask, device=device, dtype=dtype)
        if mask.ndim == 0:
            mask = mask.view(1)
        if mask.ndim == 1:
            if mask.numel() == 1 and batch_size != 1:
                mask = mask.expand(batch_size)
            if tuple(mask.shape) != (batch_size,):
                raise ValueError(
                    "episode_mask must have shape [batch] or [batch, 1], "
                    f"got {tuple(mask.shape)}"
                )
            chunk_masks[:, 0] = mask
        elif mask.ndim == 2:
            if tuple(mask.shape) == (batch_size, 1):
                chunk_masks[:, 0] = mask[:, 0]
            else:
                raise ValueError(
                    "episode_mask must have shape [batch] or [batch, 1], "
                    f"got {tuple(mask.shape)}"
                )
        else:
            raise ValueError(
                "episode_mask must have shape [batch] or [batch, 1], " f"got {tuple(mask.shape)}"
            )
        if not torch.isfinite(chunk_masks).all().item():
            raise ValueError("episode_mask must contain only finite values")
        return chunk_masks

    @staticmethod
    def _mask_state(state, mask):
        mask = mask.view(1, -1, 1)
        return WorldModelState(state.hidden * mask, state.cell * mask)

    def _make_phases(self, seq_len, device, dtype=torch.float32):
        phases = []
        for t in range(seq_len):
            t_norm = t / seq_len
            step_phases = []
            for k in range(1, self.num_freqs + 1):
                step_phases.append(math.sin(2 * math.pi * k * t_norm))
                step_phases.append(math.cos(2 * math.pi * k * t_norm))
            phases.append(step_phases)
        return torch.tensor(phases, dtype=dtype, device=device)

    @staticmethod
    def _summary_features(values):
        return torch.stack(
            (
                values.mean(dim=1),
                values.std(dim=1, unbiased=False),
                values.amin(dim=1),
                values.amax(dim=1),
            ),
            dim=1,
        )

    def _primitive_summary_features(self, primitive_context, batch_size, device, dtype):
        feature_count = self.primitive_feature_dim
        if primitive_context is None:
            return torch.zeros(
                batch_size,
                feature_count * 4,
                device=device,
                dtype=dtype,
            )
        features = torch.as_tensor(primitive_context, device=device, dtype=dtype)
        if features.ndim == 2:
            if tuple(features.shape) != (batch_size, feature_count):
                raise ValueError(
                    "primitive_context must have shape "
                    f"[batch, {feature_count}] or [batch, steps, {feature_count}], "
                    f"got {tuple(features.shape)}"
                )
            # Match the interleaved [f0_mean, f0_std, f0_min, f0_max, f1_mean, ...]
            # layout produced by the [batch, steps, F] branch below.
            zeros = torch.zeros_like(features)
            summary = torch.stack((features, zeros, features, features), dim=-1)
            return summary.reshape(batch_size, feature_count * 4)
        if features.ndim != 3:
            raise ValueError(
                "primitive_context must have shape "
                f"[batch, {feature_count}] or [batch, steps, {feature_count}], "
                f"got {tuple(features.shape)}"
            )
        if features.size(0) != batch_size or features.size(2) != feature_count:
            raise ValueError(
                "primitive_context must have shape "
                f"[batch, steps, {feature_count}], got {tuple(features.shape)}"
            )
        if features.size(1) <= 0:
            raise ValueError("primitive_context step dimension must be positive")
        summary = torch.stack(
            (
                features.mean(dim=1),
                features.std(dim=1, unbiased=False),
                features.amin(dim=1),
                features.amax(dim=1),
            ),
            dim=-1,
        )
        return summary.reshape(batch_size, feature_count * 4)

    @staticmethod
    def _decode_primitive_outcome(raw: torch.Tensor) -> PrimitiveOutcomePrediction:
        return PrimitiveOutcomePrediction(
            progress_delta=raw[:, 0],
            support_loss_logit=raw[:, 1],
            collision_death_logit=raw[:, 2],
            terminal_logit=raw[:, 3],
            continue_logit=raw[:, 4],
            cancel_logit=raw[:, 5],
            replan_logit=raw[:, 6],
        )

    def forward(
        self,
        state,
        action,
        w_context,
        b_context,
        *,
        primitive_context=None,
        initial_state=None,
        episode_mask=None,
        return_state=False,
    ):
        if state.ndim != 2:
            raise ValueError("state must have shape [batch, seq_len_c]")
        batch_size = state.size(0)
        seq_len_c = state.size(1)
        device = state.device
        dtype = state.dtype
        if (
            action.shape != state.shape
            or w_context.shape != state.shape
            or b_context.shape != state.shape
        ):
            raise ValueError(
                "state, action, w_context, and b_context must all have shape "
                f"{tuple(state.shape)}"
            )

        phases = (
            self._make_phases(seq_len_c, device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        primitive_summary = self._primitive_summary_features(
            primitive_context,
            batch_size,
            device,
            dtype,
        )
        primitive_embedding = self.primitive_encoder(primitive_summary)
        action_features = torch.cat(
            (
                self._summary_features(state),
                self._summary_features(action),
                self._summary_features(w_context),
                self._summary_features(b_context),
                primitive_embedding,
            ),
            dim=1,
        ).unsqueeze(1)

        recurrent_state = self._coerce_state(initial_state, batch_size, device, dtype)
        chunk_masks = self._normalize_episode_mask(episode_mask, batch_size, 1, device, dtype)

        recurrent_state = self._mask_state(recurrent_state, chunk_masks[:, 0])
        out, (hidden, cell) = self.lstm(
            action_features,
            (recurrent_state.hidden, recurrent_state.cell),
        )
        recurrent_state = WorldModelState(hidden, cell)
        action_hidden = out[:, -1, :].unsqueeze(1).expand(-1, seq_len_c, -1)
        slot_features = torch.stack([state, action, w_context, b_context], dim=-1)
        primitive_slots = primitive_embedding.unsqueeze(1).expand(-1, seq_len_c, -1)
        decoder_input = torch.cat([slot_features, phases, action_hidden, primitive_slots], dim=-1)
        prediction = self.decoder(decoder_input).squeeze(-1)
        outcome_raw = self.primitive_outcome_head(
            torch.cat((out[:, -1, :], primitive_embedding), dim=1)
        )
        self.last_primitive_outcome = self._decode_primitive_outcome(outcome_raw)
        if return_state:
            return prediction, recurrent_state.detach()
        return prediction


def action_motion_score(
    next_state_pred: torch.Tensor,
    current_state: torch.Tensor | None,
    *,
    motion_position_dims: int | None = None,
) -> torch.Tensor:
    if current_state is None:
        return next_state_pred.new_full((next_state_pred.size(0),), float("inf"))
    if next_state_pred.shape != current_state.shape:
        raise ValueError(
            "next_state_pred and current_state must have the same shape for "
            "critic motion evaluation, got "
            f"{next_state_pred.shape} and {current_state.shape}"
        )
    if next_state_pred.ndim != 2:
        raise ValueError("next_state_pred and current_state must have shape [batch, seq_len_c]")
    if motion_position_dims is None:
        end = next_state_pred.size(1)
    else:
        end = max(0, min(int(motion_position_dims), next_state_pred.size(1)))
        if end == 0:
            end = next_state_pred.size(1)
    delta = next_state_pred[:, :end] - current_state[:, :end]
    return delta.abs().amax(dim=1)


def action_pause_mask(
    is_pause_action: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if is_pause_action is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask = torch.as_tensor(is_pause_action, dtype=torch.bool, device=device)
    if mask.ndim == 0:
        mask = mask.view(1)
    if mask.numel() == 1 and batch_size != 1:
        mask = mask.expand(batch_size)
    if tuple(mask.shape) != (batch_size,):
        raise ValueError(
            "is_pause_action must have shape [batch] or scalar, got "
            f"{tuple(mask.shape)} for batch size {batch_size}"
        )
    return mask


def action_rejection_mask(
    value: torch.Tensor | None,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(reference.size(0), dtype=torch.bool, device=reference.device)
    mask = torch.as_tensor(value, dtype=torch.bool, device=reference.device)
    if mask.ndim == 0:
        mask = mask.view(1)
    if mask.numel() == 1 and reference.size(0) != 1:
        mask = mask.expand(reference.size(0))
    if tuple(mask.shape) != (reference.size(0),):
        raise ValueError(
            "critic rejection mask must have shape [batch] or scalar, got " f"{tuple(mask.shape)}"
        )
    return mask


class Critic(nn.Module):
    """Evaluates predicted C state and returns actor feedback plus outcome gates."""

    def __init__(self, seq_len_c, seq_len_a, d_model):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(seq_len_c, 128), nn.ReLU(), nn.Linear(128, seq_len_a * d_model)
        )
        self.progress_head = nn.Sequential(
            nn.Linear(seq_len_c, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.death_head = nn.Sequential(
            nn.Linear(seq_len_c, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, next_state_pred):
        critique = self.net(next_state_pred)
        return critique.view(-1, self.seq_len_a, self.d_model)

    def action_progress_logit(self, next_state_pred, current_state=None):
        progress_logit = self.progress_head(next_state_pred).squeeze(-1)
        if current_state is None:
            return progress_logit
        if next_state_pred.shape != current_state.shape:
            raise ValueError(
                "next_state_pred and current_state must have the same shape for "
                "critic progress evaluation, got "
                f"{next_state_pred.shape} and {current_state.shape}"
            )
        delta_hint = (next_state_pred - current_state).mean(dim=1)
        return progress_logit + 0.1 * delta_hint

    def action_death_logit(self, next_state_pred):
        return self.death_head(next_state_pred).squeeze(-1)

    def evaluate_action(
        self,
        next_state_pred,
        current_state=None,
        *,
        progress_threshold=0.0,
        death_threshold=0.75,
        motion_threshold=DEFAULT_ACTION_MOTION_THRESHOLD,
        is_pause_action=None,
    ) -> CriticActionEvaluation:
        feedback = self(next_state_pred)
        progress_score = self.action_progress_logit(
            next_state_pred,
            current_state=current_state,
        )
        death_risk = torch.sigmoid(self.action_death_logit(next_state_pred))
        motion_score = action_motion_score(next_state_pred, current_state)
        pause_mask = action_pause_mask(
            is_pause_action,
            batch_size=next_state_pred.size(0),
            device=next_state_pred.device,
        )
        predicts_no_motion = (motion_score <= float(motion_threshold)) & ~pause_mask
        return CriticActionEvaluation(
            feedback=feedback,
            progress_score=progress_score,
            death_risk=death_risk,
            would_progress=progress_score >= float(progress_threshold),
            predicts_death=death_risk >= float(death_threshold),
            motion_score=motion_score,
            predicts_no_motion=predicts_no_motion,
            is_pause_action=pause_mask,
        )


class AgentWorldModelCritic(nn.Module):
    """
    Combines the actor, world model, and critic in a bounded refinement loop.

    The first pass runs the actor with no feedback. Each candidate C action and
    B controller parameter set feeds the action-level LSTM world model. The
    critic maps that predicted C state to A-level feedback plus progress/death
    gates. The actor is rerun with the latest critic feedback until the critic
    predicts progress without predicted death, or until the pass budget is
    exhausted. Feedback is detached between refinement passes so one rollout
    does not retain every earlier candidate graph.
    """

    def __init__(
        self,
        vocab_size,
        seq_len_a,
        seq_len_c,
        ratio_bc,
        action_vocab_size=None,
        d_model=64,
        controller_schedule="constant",
        max_walk_action_duration=None,
        walk_action_ids: Iterable[int] = (),
        pause_action_ids: Iterable[int] = (0,),
        motion_position_dims: int | None = None,
        max_action_refinement_passes=3,
        critic_progress_threshold=0.0,
        critic_death_threshold=0.75,
        critic_motion_threshold=DEFAULT_ACTION_MOTION_THRESHOLD,
        direct_c_state_context=False,
    ):
        super().__init__()
        if int(max_action_refinement_passes) <= 0:
            raise ValueError("max_action_refinement_passes must be positive")
        vocab_size = int(vocab_size)
        action_vocab_size = vocab_size if action_vocab_size is None else int(action_vocab_size)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if action_vocab_size <= 0:
            raise ValueError("action_vocab_size must be positive")
        if action_vocab_size > vocab_size:
            raise ValueError(
                "action_vocab_size must be less than or equal to vocab_size, got "
                f"{action_vocab_size} and {vocab_size}"
            )
        if not math.isfinite(float(critic_progress_threshold)):
            raise ValueError("critic_progress_threshold must be finite")
        if not 0.0 < float(critic_death_threshold) < 1.0:
            raise ValueError("critic_death_threshold must be in (0, 1)")
        if (
            not math.isfinite(float(critic_motion_threshold))
            or float(critic_motion_threshold) < 0.0
        ):
            raise ValueError("critic_motion_threshold must be finite and non-negative")
        resolved_pause_action_ids = tuple(int(action_id) for action_id in pause_action_ids)
        if any(action_id < 0 for action_id in resolved_pause_action_ids):
            raise ValueError("pause_action_ids must be non-negative")
        if any(action_id >= action_vocab_size for action_id in resolved_pause_action_ids):
            raise ValueError("pause_action_ids must be smaller than action_vocab_size")
        if len(set(resolved_pause_action_ids)) != len(resolved_pause_action_ids):
            raise ValueError("pause_action_ids must be unique")
        if motion_position_dims is not None and int(motion_position_dims) < 0:
            raise ValueError("motion_position_dims must be non-negative when set")
        seq_len_b = seq_len_c // ratio_bc
        ratio_ab = seq_len_b // seq_len_a
        self.ratio_bc = ratio_bc
        self.ratio_ab = ratio_ab
        self.vocab_size = vocab_size
        self.action_vocab_size = action_vocab_size
        self.max_action_refinement_passes = int(max_action_refinement_passes)
        self.critic_progress_threshold = float(critic_progress_threshold)
        self.critic_death_threshold = float(critic_death_threshold)
        self.critic_motion_threshold = float(critic_motion_threshold)
        self.pause_action_ids = resolved_pause_action_ids
        self.motion_position_dims = (
            None if motion_position_dims is None else int(motion_position_dims)
        )
        self.agent = HierarchicalAdaptiveModel(
            vocab_size,
            action_vocab_size=action_vocab_size,
            seq_len_c=seq_len_c if direct_c_state_context else None,
            d_model=d_model,
            controller_schedule=controller_schedule,
        )
        self.world_model = WorldModel(ratio_bc=ratio_bc)
        self.critic = Critic(seq_len_c, seq_len_a, d_model)
        self.world_model_actor_context = nn.Sequential(
            nn.Linear(self.world_model.hidden_size * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
        )
        self.motor_controller = MotorPrimitiveController(
            ratio_ab=ratio_ab,
            ratio_bc=ratio_bc,
            max_walk_action_duration=max_walk_action_duration,
            walk_action_ids=walk_action_ids,
        )
        self.last_motor_primitives: MotorPrimitiveOutput | None = None
        self.last_primitive_outcome: PrimitiveOutcomePrediction | None = None
        self.last_actor_world_model_context: torch.Tensor | None = None
        self.last_action_refinement: ActionRefinementTrace | None = None
        self.transition_representation_head = nn.Sequential(
            nn.Linear(seq_len_c, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(seq_len_c, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(seq_len_c, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def transition_representation(self, state):
        return self.transition_representation_head(state)

    def predict_reward(self, next_state_pred):
        return self.reward_head(next_state_pred).squeeze(-1)

    def predict_value(self, state):
        return self.value_head(state).squeeze(-1)

    def predict_action_progress_logit(self, next_state_pred, current_state=None):
        critic = getattr(self, "critic", None)
        if hasattr(critic, "action_progress_logit"):
            return critic.action_progress_logit(
                next_state_pred,
                current_state=current_state,
            )
        if current_state is not None and next_state_pred.shape == current_state.shape:
            return (next_state_pred - current_state).mean(dim=1)
        return next_state_pred.new_zeros((next_state_pred.size(0),))

    def predict_action_death_logit(self, next_state_pred):
        critic = getattr(self, "critic", None)
        if hasattr(critic, "action_death_logit"):
            return critic.action_death_logit(next_state_pred)
        return next_state_pred.new_full((next_state_pred.size(0),), -10.0)

    def initial_world_model_state(self, batch_size, device, dtype=torch.float32):
        return self.world_model.initial_state(batch_size, device, dtype=dtype)

    def controller_context(self, src_c, w, b):
        controller = getattr(self.agent, "controller", None)
        if hasattr(controller, "expand_context"):
            return controller.expand_context(src_c, w, b)
        ratio_bc = src_c.size(1) // w.size(1)
        return (
            w.repeat_interleave(ratio_bc, dim=1),
            b.repeat_interleave(ratio_bc, dim=1),
        )

    def _actor_episode_mask(self, episode_mask, batch_size, device, dtype):
        if episode_mask is None:
            return None
        mask = torch.as_tensor(episode_mask, device=device, dtype=dtype)
        if mask.ndim == 0:
            mask = mask.view(1)
        if mask.ndim == 2 and tuple(mask.shape) == (batch_size, 1):
            mask = mask[:, 0]
        if mask.ndim != 1:
            raise ValueError(
                "episode_mask must have shape [batch] or [batch, 1], " f"got {tuple(mask.shape)}"
            )
        if mask.numel() == 1 and batch_size != 1:
            mask = mask.expand(batch_size)
        if tuple(mask.shape) != (batch_size,):
            raise ValueError(
                "episode_mask must have shape [batch] or [batch, 1], " f"got {tuple(mask.shape)}"
            )
        if not torch.isfinite(mask).all().item():
            raise ValueError("episode_mask must contain only finite values")
        return mask.clamp(0.0, 1.0)

    def _actor_world_model_context(self, world_model_state, src_A, episode_mask=None):
        if world_model_state is None:
            return None
        if isinstance(world_model_state, WorldModelState):
            hidden, cell = world_model_state.hidden, world_model_state.cell
        else:
            hidden, cell = world_model_state
        if hidden.ndim != 3 or cell.ndim != 3:
            raise ValueError("world_model_state hidden and cell must be 3D tensors")
        if tuple(hidden.shape) != tuple(cell.shape):
            raise ValueError(
                "world_model_state hidden and cell shapes must match, got "
                f"{tuple(hidden.shape)} and {tuple(cell.shape)}"
            )
        expected_layers = int(getattr(self.world_model, "num_layers", hidden.size(0)))
        expected_hidden = int(getattr(self.world_model, "hidden_size", hidden.size(2)))
        expected = (expected_layers, src_A.size(0), expected_hidden)
        if tuple(hidden.shape) != expected:
            raise ValueError(
                "world_model_state hidden and cell shape must match actor context "
                f"{expected}, got {tuple(hidden.shape)}"
            )
        parameter = next(self.world_model_actor_context.parameters())
        hidden_last = hidden[-1].to(device=src_A.device, dtype=parameter.dtype)
        cell_last = cell[-1].to(device=src_A.device, dtype=parameter.dtype)
        recurrent_features = torch.cat((hidden_last, cell_last), dim=-1)
        mask = self._actor_episode_mask(
            episode_mask,
            src_A.size(0),
            src_A.device,
            recurrent_features.dtype,
        )
        if mask is not None:
            recurrent_features = recurrent_features * mask.view(-1, 1)
        context = self.world_model_actor_context(recurrent_features)
        return context.unsqueeze(1).expand(-1, src_A.size(1), -1)

    def _agent_forward(
        self,
        src_A,
        src_B,
        src_C,
        *,
        criticism=None,
        tau=1.0,
        world_model_context=None,
    ):
        if bool(getattr(self.agent, "uses_world_model_context", False)):
            return self.agent(
                src_A,
                src_B,
                src_C,
                criticism=criticism,
                tau=tau,
                world_model_context=world_model_context,
            )
        return self.agent(src_A, src_B, src_C, criticism=criticism, tau=tau)

    def _world_model_prediction(
        self,
        src_C,
        actions,
        logits_a,
        w,
        b,
        *,
        primitive_params=None,
        world_model_state=None,
        episode_mask=None,
        return_world_model_state=False,
        world_model_enabled=True,
    ):
        w_context, b_context = self.controller_context(src_C, w, b)
        if not world_model_enabled:
            return src_C.detach(), None, None
        primitive_context = self._level_b_primitive_context(logits_a, primitive_params, w)
        supports_primitive_context = bool(
            getattr(self.world_model, "uses_primitive_context", False)
        )
        if world_model_state is None and episode_mask is None and not return_world_model_state:
            if supports_primitive_context:
                prediction = self.world_model(
                    src_C,
                    actions,
                    w_context,
                    b_context,
                    primitive_context=primitive_context,
                )
            else:
                prediction = self.world_model(src_C, actions, w_context, b_context)
            return prediction, None, getattr(self.world_model, "last_primitive_outcome", None)
        if supports_primitive_context:
            prediction, next_state = self.world_model(
                src_C,
                actions,
                w_context,
                b_context,
                primitive_context=primitive_context,
                initial_state=world_model_state,
                episode_mask=episode_mask,
                return_state=True,
            )
        else:
            prediction, next_state = self.world_model(
                src_C,
                actions,
                w_context,
                b_context,
                initial_state=world_model_state,
                episode_mask=episode_mask,
                return_state=True,
            )
        return prediction, next_state, getattr(self.world_model, "last_primitive_outcome", None)

    def _level_b_primitive_context(
        self,
        logits_a: torch.Tensor,
        primitive_params: LevelBPrimitiveParameters | None,
        w: torch.Tensor,
    ) -> torch.Tensor | None:
        if primitive_params is None:
            return None
        if logits_a.ndim != 3 or w.ndim != 2:
            raise ValueError("logits_a must be [batch, A, vocab] and w must be [batch, B]")
        batch_size, seq_len_b = w.shape
        action_probs = F.softmax(logits_a, dim=-1).repeat_interleave(self.ratio_ab, dim=1)
        if tuple(action_probs.shape[:2]) != (batch_size, seq_len_b):
            raise ValueError(
                "expanded action probabilities must align with B stream, got "
                f"{tuple(action_probs.shape[:2])} and {(batch_size, seq_len_b)}"
            )
        vocab_size = action_probs.size(-1)
        action_ids = torch.linspace(
            0.0,
            1.0,
            vocab_size,
            device=action_probs.device,
            dtype=action_probs.dtype,
        )
        expected_button = (action_probs * action_ids.view(1, 1, -1)).sum(dim=-1)
        button_confidence = action_probs.amax(dim=-1)
        button_entropy = -(action_probs.clamp_min(1e-8).log() * action_probs).sum(
            dim=-1
        ) / math.log(max(vocab_size, 2))

        self.motor_controller._validate_primitive_params(primitive_params, w, logits_a)
        duration_values = self.motor_controller.duration_bin_values.to(
            device=w.device,
            dtype=w.dtype,
        )
        duration_probs = F.softmax(primitive_params.hold_duration_logits, dim=-1)
        hold_duration = (duration_probs * duration_values.view(1, 1, -1)).sum(dim=-1)
        hold_duration = hold_duration / duration_values.amax().clamp_min(1.0)

        post_release_probs = F.softmax(primitive_params.post_release_logits, dim=-1)
        post_vocab_size = post_release_probs.size(-1)
        post_release_ids = torch.linspace(
            0.0,
            1.0,
            post_vocab_size,
            device=post_release_probs.device,
            dtype=post_release_probs.dtype,
        )
        post_release_expected = (post_release_probs * post_release_ids.view(1, 1, -1)).sum(dim=-1)
        post_release_confidence = post_release_probs.amax(dim=-1)

        return torch.stack(
            (
                expected_button,
                button_confidence,
                button_entropy,
                hold_duration,
                torch.sigmoid(primitive_params.release_logit),
                torch.sigmoid(primitive_params.cancel_logit),
                torch.sigmoid(primitive_params.replan_logit),
                post_release_expected,
                post_release_confidence,
            ),
            dim=-1,
        )

    def _critic_evaluation(
        self, next_state_pred, current_state, logits_a, *, motion_gate_enabled=True
    ):
        critic = getattr(self, "critic", None)
        if hasattr(critic, "evaluate_action"):
            evaluation = critic.evaluate_action(
                next_state_pred,
                current_state=current_state,
                progress_threshold=self.critic_progress_threshold,
                death_threshold=self.critic_death_threshold,
            )
            return self._with_action_motion_gate(
                evaluation,
                next_state_pred,
                current_state,
                logits_a,
                gate_enabled=motion_gate_enabled,
            )

        feedback = self.critic(next_state_pred)
        progress_score = self.predict_action_progress_logit(
            next_state_pred,
            current_state=current_state,
        )
        death_risk = torch.sigmoid(self.predict_action_death_logit(next_state_pred))
        evaluation = CriticActionEvaluation(
            feedback=feedback,
            progress_score=progress_score,
            death_risk=death_risk,
            would_progress=progress_score >= self.critic_progress_threshold,
            predicts_death=death_risk >= self.critic_death_threshold,
        )
        return self._with_action_motion_gate(
            evaluation,
            next_state_pred,
            current_state,
            logits_a,
            gate_enabled=motion_gate_enabled,
        )

    def _candidate_pause_mask(self, logits_a: torch.Tensor) -> torch.Tensor:
        if logits_a.ndim != 3:
            raise ValueError("logits_a must have shape [batch, seq_len_a, vocab]")
        if not self.pause_action_ids:
            return torch.zeros(logits_a.size(0), dtype=torch.bool, device=logits_a.device)
        selected_action = logits_a[:, -1, :].argmax(dim=-1)
        pause_ids = torch.as_tensor(
            self.pause_action_ids,
            dtype=selected_action.dtype,
            device=selected_action.device,
        )
        return (selected_action.unsqueeze(-1) == pause_ids).any(dim=-1)

    def _with_action_motion_gate(
        self,
        evaluation: CriticActionEvaluation,
        next_state_pred: torch.Tensor,
        current_state: torch.Tensor,
        logits_a: torch.Tensor,
        *,
        gate_enabled: bool = True,
    ) -> CriticActionEvaluation:
        pause_mask = self._candidate_pause_mask(logits_a)
        if gate_enabled:
            motion_score = action_motion_score(
                next_state_pred,
                current_state,
                motion_position_dims=self.motion_position_dims,
            )
            predicts_no_motion = (motion_score <= self.critic_motion_threshold) & ~pause_mask
        else:
            # With the world model ablated the "prediction" is the current state,
            # so a motion gate would reject every non-pause candidate.
            motion_score = action_motion_score(next_state_pred, None)
            predicts_no_motion = torch.zeros_like(pause_mask)
        return CriticActionEvaluation(
            feedback=evaluation.feedback,
            progress_score=evaluation.progress_score,
            death_risk=evaluation.death_risk,
            would_progress=evaluation.would_progress,
            predicts_death=evaluation.predicts_death,
            motion_score=motion_score,
            predicts_no_motion=predicts_no_motion,
            is_pause_action=pause_mask,
        )

    def _candidate_from_actor_outputs(
        self,
        src_C,
        logits_a,
        actions,
        w,
        b,
        primitive_params=None,
        *,
        world_model_state=None,
        episode_mask=None,
        return_world_model_state=False,
        world_model_enabled=True,
    ) -> _ActionCandidate:
        next_state_pred, next_world_model_state, primitive_outcome = self._world_model_prediction(
            src_C,
            actions,
            logits_a,
            w,
            b,
            primitive_params=primitive_params,
            world_model_state=world_model_state,
            episode_mask=episode_mask,
            return_world_model_state=return_world_model_state,
            world_model_enabled=world_model_enabled,
        )
        evaluation = self._critic_evaluation(
            next_state_pred,
            src_C,
            logits_a,
            motion_gate_enabled=world_model_enabled,
        )
        return _ActionCandidate(
            logits_a=logits_a,
            actions=actions,
            w=w,
            b=b,
            primitive_params=primitive_params,
            next_state_pred=next_state_pred,
            primitive_outcome=primitive_outcome,
            criticism=evaluation.feedback,
            evaluation=evaluation,
            next_world_model_state=next_world_model_state,
        )

    @staticmethod
    def _candidate_is_accepted(candidate: _ActionCandidate) -> bool:
        predicts_no_motion = action_rejection_mask(
            candidate.evaluation.predicts_no_motion,
            reference=candidate.evaluation.progress_score,
        )
        accepted = (
            candidate.evaluation.would_progress
            & ~candidate.evaluation.predicts_death
            & ~predicts_no_motion
        )
        return bool(torch.all(accepted.detach()).cpu().item())

    @staticmethod
    def _candidate_rank(candidate: _ActionCandidate) -> float:
        predicts_no_motion = action_rejection_mask(
            candidate.evaluation.predicts_no_motion,
            reference=candidate.evaluation.progress_score,
        ).to(dtype=candidate.evaluation.progress_score.dtype)
        score = (
            candidate.evaluation.progress_score
            - candidate.evaluation.death_risk
            - predicts_no_motion
        )
        return float(score.detach().mean().cpu().item())

    def _select_fallback_candidate(
        self,
        candidates: list[_ActionCandidate],
    ) -> tuple[_ActionCandidate, int]:
        if not candidates:
            raise ValueError("at least one action candidate is required")
        best_index = max(
            range(len(candidates)), key=lambda index: self._candidate_rank(candidates[index])
        )
        return candidates[best_index], best_index + 1

    def _record_refinement_trace(
        self,
        *,
        candidates: list[_ActionCandidate],
        selected: _ActionCandidate,
        selected_iteration: int,
        accepted: bool,
    ) -> None:
        self.last_action_refinement = ActionRefinementTrace(
            iterations=len(candidates),
            accepted=bool(accepted),
            selected_iteration=int(selected_iteration),
            progress_score=selected.evaluation.progress_score.detach(),
            death_risk=selected.evaluation.death_risk.detach(),
            motion_score=(
                None
                if selected.evaluation.motion_score is None
                else selected.evaluation.motion_score.detach()
            ),
            predicts_no_motion=(
                None
                if selected.evaluation.predicts_no_motion is None
                else selected.evaluation.predicts_no_motion.detach()
            ),
            is_pause_action=(
                None
                if selected.evaluation.is_pause_action is None
                else selected.evaluation.is_pause_action.detach()
            ),
        )

    def forward(
        self,
        src_A,
        src_B,
        src_C,
        tau=1.0,
        *,
        world_model_state=None,
        episode_mask=None,
        return_world_model_state=False,
        critic_feedback_enabled=True,
        world_model_enabled=True,
    ):
        actor_world_model_context = self._actor_world_model_context(
            world_model_state,
            src_A,
            episode_mask=episode_mask,
        )
        self.last_actor_world_model_context = (
            None if actor_world_model_context is None else actor_world_model_context.detach()
        )
        logits_a1, actions1, w_1, b_1 = self._agent_forward(
            src_A,
            src_B,
            src_C,
            criticism=None,
            tau=tau,
            world_model_context=actor_world_model_context,
        )
        primitive_params1 = getattr(self.agent, "last_level_b_primitives", None)
        first_candidate = self._candidate_from_actor_outputs(
            src_C,
            logits_a1,
            actions1,
            w_1,
            b_1,
            primitive_params=primitive_params1,
            world_model_state=world_model_state,
            episode_mask=episode_mask,
            return_world_model_state=return_world_model_state,
            world_model_enabled=world_model_enabled,
        )
        candidates = [first_candidate]
        selected_candidate = first_candidate
        selected_iteration = 1
        accepted = self._candidate_is_accepted(first_candidate)

        if critic_feedback_enabled and not accepted:
            actor_criticism = first_candidate.criticism.detach()
            for _pass_index in range(1, self.max_action_refinement_passes):
                logits_a, actions, w, b = self._agent_forward(
                    src_A,
                    src_B,
                    src_C,
                    criticism=actor_criticism,
                    tau=tau,
                    world_model_context=actor_world_model_context,
                )
                primitive_params = getattr(self.agent, "last_level_b_primitives", None)
                candidate = self._candidate_from_actor_outputs(
                    src_C,
                    logits_a,
                    actions,
                    w,
                    b,
                    primitive_params=primitive_params,
                    world_model_state=world_model_state,
                    episode_mask=episode_mask,
                    return_world_model_state=return_world_model_state,
                    world_model_enabled=world_model_enabled,
                )
                candidates.append(candidate)
                selected_candidate = candidate
                selected_iteration = len(candidates)
                accepted = self._candidate_is_accepted(candidate)
                if accepted:
                    break
                actor_criticism = candidate.criticism.detach()

        if not accepted:
            selected_candidate, selected_iteration = self._select_fallback_candidate(candidates)

        self._record_refinement_trace(
            candidates=candidates,
            selected=selected_candidate,
            selected_iteration=selected_iteration,
            accepted=accepted,
        )
        self.last_motor_primitives = self.motor_controller(
            selected_candidate.logits_a,
            selected_candidate.w,
            selected_candidate.b,
            next_state_pred=selected_candidate.next_state_pred,
            current_state=src_C,
            primitive_params=selected_candidate.primitive_params,
        )
        self.last_primitive_outcome = selected_candidate.primitive_outcome

        outputs = (
            actions1,
            selected_candidate.next_state_pred,
            selected_candidate.criticism,
            selected_candidate.actions,
            selected_candidate.logits_a,
            selected_candidate.w,
            selected_candidate.b,
        )
        if return_world_model_state:
            return (*outputs, selected_candidate.next_world_model_state)
        return outputs
