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
ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES = (
    *ACTION_LEVEL_WORLD_MODEL_ALLOWED_MISSING_PREFIXES,
    *ACTION_REFINEMENT_ALLOWED_MISSING_PREFIXES,
    *LEVEL_B_PRIMITIVE_ALLOWED_MISSING_PREFIXES,
)
DEFAULT_PRIMITIVE_DURATION_BINS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


def action_level_world_model_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Drop obsolete token-level world-model tensors before policy loading."""

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
                self.duration_bin_values.detach()
                if self.duration_bin_values is not None
                else None
            ),
            post_release_logits=(
                self.post_release_logits.detach()
                if self.post_release_logits is not None
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


@dataclass(frozen=True)
class ActionRefinementTrace:
    """Diagnostics for the actor/world-model/critic refinement loop."""

    iterations: int
    accepted: bool
    selected_iteration: int
    progress_score: torch.Tensor
    death_risk: torch.Tensor


@dataclass(frozen=True)
class _ActionCandidate:
    logits_a: torch.Tensor
    actions: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor
    primitive_params: LevelBPrimitiveParameters | None
    next_state_pred: torch.Tensor
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
                "x_c, w, and b batch sizes must match, got "
                f"{x_c.size(0)} and {w.size(0)}"
            )
        if x_c.size(1) % w.size(1) != 0:
            raise ValueError(
                "C length must be divisible by B length, got "
                f"{x_c.size(1)} and {w.size(1)}"
            )
        ratio_bc = x_c.size(1) // w.size(1)
        if self.schedule == "constant":
            return (
                w.repeat_interleave(ratio_bc, dim=1),
                b.repeat_interleave(ratio_bc, dim=1),
            )

        phase = (
            torch.arange(ratio_bc, device=x_c.device, dtype=x_c.dtype)
            / float(ratio_bc)
        )
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
        if (
            max_walk_action_duration is not None
            and float(max_walk_action_duration) < 1.0
        ):
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
            None
            if max_walk_action_duration is None
            else float(max_walk_action_duration)
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
                "w_pred and b_pred shapes must match, got "
                f"{w_pred.shape} and {b_pred.shape}"
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
        motion = self._predicted_motion(next_state_pred, current_state, w_pred)
        hold_duration_logits = None
        post_release_logits = None
        if primitive_params is None:
            hold_duration = 1.0 + (self.max_hold_duration - 1.0) * torch.sigmoid(w_pred)
            release_logit = -b_pred
            cancel_logit = b_pred - w_pred
            interrupt_logit = cancel_logit + (0.05 - motion)
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
            release_logit = primitive_params.release_logit
            cancel_logit = primitive_params.cancel_logit
            interrupt_logit = primitive_params.replan_logit + (0.05 - motion)
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

    def _predicted_motion(self, next_state_pred, current_state, reference):
        if next_state_pred is None or current_state is None:
            return torch.zeros_like(reference)
        if next_state_pred.shape != current_state.shape:
            raise ValueError(
                "next_state_pred and current_state must have the same shape for "
                "motor primitive prediction, got "
                f"{next_state_pred.shape} and {current_state.shape}"
            )
        if next_state_pred.ndim != 2:
            raise ValueError(
                "next_state_pred and current_state must have shape [batch, seq_len_c]"
            )
        usable = reference.size(1) * self.ratio_bc
        if next_state_pred.size(1) < usable:
            raise ValueError(
                "C-stream length is too short for motor primitive grouping: "
                f"{next_state_pred.size(1)} < {usable}"
            )
        delta = next_state_pred[:, :usable] - current_state[:, :usable]
        return delta.reshape(delta.size(0), reference.size(1), self.ratio_bc).abs().mean(dim=-1)


class HierarchicalAdaptiveModel(nn.Module):
    """Three-level actor: Transformer A -> Transformer B -> adaptive controller."""

    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        controller_schedule="constant",
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers_a = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer_A = nn.TransformerEncoder(encoder_layers_a, num_layers)
        self.fc_out_A = nn.Linear(d_model, vocab_size)

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
        self.fc_primitive_post_release = nn.Linear(d_model, vocab_size)
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

    def forward(self, src_A, src_B, src_C, criticism=None, tau=1.0, return_hidden=False):
        seq_len_a = src_A.size(1)
        seq_len_b = src_B.size(1)
        ratio_ab = seq_len_b // seq_len_a

        causal_mask_a = self.generate_causal_mask(seq_len_a).to(src_A.device)
        causal_mask_b = self.generate_causal_mask(seq_len_b).to(src_B.device)

        x_a = self.embedding(src_A) * math.sqrt(self.d_model)
        x_a = self.pos_encoder(x_a)
        x_a = self.apply_critic_feedback(x_a, criticism)

        hidden_a = self.transformer_A(x_a, mask=causal_mask_a)
        logits_a = self.fc_out_A(hidden_a)

        probs_a = F.gumbel_softmax(logits_a, tau=tau, hard=True, dim=-1)
        pred_emb_a = torch.matmul(probs_a, self.embedding.weight)

        x_b = self.embedding(src_B) * math.sqrt(self.d_model)
        x_b = self.pos_encoder(x_b)
        cross_mask = self.generate_cross_causal_mask(seq_len_b, seq_len_a, ratio_ab).to(src_B.device)
        hidden_b = self.transformer_B(tgt=x_b, memory=pred_emb_a, tgt_mask=causal_mask_b, memory_mask=cross_mask)

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

    def __init__(self, hidden_size=32, num_layers=1, num_freqs=4, ratio_bc=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_freqs = num_freqs
        self.ratio_bc = ratio_bc
        input_size = 16
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        decoder_input_size = hidden_size + 4 + num_freqs * 2
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

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
                "episode_mask must have shape [batch] or [batch, 1], "
                f"got {tuple(mask.shape)}"
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

    def forward(
        self,
        state,
        action,
        w_context,
        b_context,
        *,
        initial_state=None,
        episode_mask=None,
        return_state=False,
    ):
        batch_size = state.size(0)
        seq_len_c = state.size(1)
        device = state.device
        dtype = state.dtype
        if state.ndim != 2:
            raise ValueError("state must have shape [batch, seq_len_c]")
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
        action_features = torch.cat(
            (
                self._summary_features(state),
                self._summary_features(action),
                self._summary_features(w_context),
                self._summary_features(b_context),
            ),
            dim=1,
        ).unsqueeze(1)

        recurrent_state = self._coerce_state(initial_state, batch_size, device, dtype)
        chunk_masks = self._normalize_episode_mask(
            episode_mask, batch_size, 1, device, dtype
        )

        recurrent_state = self._mask_state(recurrent_state, chunk_masks[:, 0])
        out, (hidden, cell) = self.lstm(
            action_features,
            (recurrent_state.hidden, recurrent_state.cell),
        )
        recurrent_state = WorldModelState(hidden, cell)
        action_hidden = out[:, -1, :].unsqueeze(1).expand(-1, seq_len_c, -1)
        slot_features = torch.stack([state, action, w_context, b_context], dim=-1)
        decoder_input = torch.cat([slot_features, phases, action_hidden], dim=-1)
        prediction = self.decoder(decoder_input).squeeze(-1)
        if return_state:
            return prediction, recurrent_state
        return prediction


class Critic(nn.Module):
    """Evaluates predicted C state and returns actor feedback plus outcome gates."""

    def __init__(self, seq_len_c, seq_len_a, d_model):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.d_model = d_model
        self.net = nn.Sequential(nn.Linear(seq_len_c, 128), nn.ReLU(), nn.Linear(128, seq_len_a * d_model))
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
    ) -> CriticActionEvaluation:
        feedback = self(next_state_pred)
        progress_score = self.action_progress_logit(
            next_state_pred,
            current_state=current_state,
        )
        death_risk = torch.sigmoid(self.action_death_logit(next_state_pred))
        return CriticActionEvaluation(
            feedback=feedback,
            progress_score=progress_score,
            death_risk=death_risk,
            would_progress=progress_score >= float(progress_threshold),
            predicts_death=death_risk >= float(death_threshold),
        )


class AgentWorldModelCritic(nn.Module):
    """
    Combines the actor, world model, and critic in a bounded refinement loop.

    The first pass runs the actor with no feedback. Each candidate C action and
    B controller parameter set feeds the action-level LSTM world model. The
    critic maps that predicted C state to A-level feedback plus progress/death
    gates. The actor is rerun with the latest critic feedback until the critic
    predicts progress without predicted death, or until the pass budget is
    exhausted.
    """

    def __init__(
        self,
        vocab_size,
        seq_len_a,
        seq_len_c,
        ratio_bc,
        d_model=64,
        controller_schedule="constant",
        max_walk_action_duration=None,
        walk_action_ids: Iterable[int] = (),
        max_action_refinement_passes=3,
        critic_progress_threshold=0.0,
        critic_death_threshold=0.75,
    ):
        super().__init__()
        if int(max_action_refinement_passes) <= 0:
            raise ValueError("max_action_refinement_passes must be positive")
        if not math.isfinite(float(critic_progress_threshold)):
            raise ValueError("critic_progress_threshold must be finite")
        if not 0.0 < float(critic_death_threshold) < 1.0:
            raise ValueError("critic_death_threshold must be in (0, 1)")
        seq_len_b = seq_len_c // ratio_bc
        ratio_ab = seq_len_b // seq_len_a
        self.ratio_bc = ratio_bc
        self.max_action_refinement_passes = int(max_action_refinement_passes)
        self.critic_progress_threshold = float(critic_progress_threshold)
        self.critic_death_threshold = float(critic_death_threshold)
        self.agent = HierarchicalAdaptiveModel(
            vocab_size,
            d_model=d_model,
            controller_schedule=controller_schedule,
        )
        self.world_model = WorldModel(ratio_bc=ratio_bc)
        self.critic = Critic(seq_len_c, seq_len_a, d_model)
        self.motor_controller = MotorPrimitiveController(
            ratio_ab=ratio_ab,
            ratio_bc=ratio_bc,
            max_walk_action_duration=max_walk_action_duration,
            walk_action_ids=walk_action_ids,
        )
        self.last_motor_primitives: MotorPrimitiveOutput | None = None
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

    def _world_model_prediction(
        self,
        src_C,
        actions,
        w,
        b,
        *,
        world_model_state=None,
        episode_mask=None,
        return_world_model_state=False,
        world_model_enabled=True,
    ):
        w_context, b_context = self.controller_context(src_C, w, b)
        if not world_model_enabled:
            return src_C.detach(), None
        if (
            world_model_state is None
            and episode_mask is None
            and not return_world_model_state
        ):
            return self.world_model(src_C, actions, w_context, b_context), None
        return self.world_model(
            src_C,
            actions,
            w_context,
            b_context,
            initial_state=world_model_state,
            episode_mask=episode_mask,
            return_state=True,
        )

    def _critic_evaluation(self, next_state_pred, current_state):
        critic = getattr(self, "critic", None)
        if hasattr(critic, "evaluate_action"):
            return critic.evaluate_action(
                next_state_pred,
                current_state=current_state,
                progress_threshold=self.critic_progress_threshold,
                death_threshold=self.critic_death_threshold,
            )

        feedback = self.critic(next_state_pred)
        progress_score = self.predict_action_progress_logit(
            next_state_pred,
            current_state=current_state,
        )
        death_risk = torch.sigmoid(self.predict_action_death_logit(next_state_pred))
        return CriticActionEvaluation(
            feedback=feedback,
            progress_score=progress_score,
            death_risk=death_risk,
            would_progress=progress_score >= self.critic_progress_threshold,
            predicts_death=death_risk >= self.critic_death_threshold,
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
        next_state_pred, next_world_model_state = self._world_model_prediction(
            src_C,
            actions,
            w,
            b,
            world_model_state=world_model_state,
            episode_mask=episode_mask,
            return_world_model_state=return_world_model_state,
            world_model_enabled=world_model_enabled,
        )
        evaluation = self._critic_evaluation(next_state_pred, src_C)
        return _ActionCandidate(
            logits_a=logits_a,
            actions=actions,
            w=w,
            b=b,
            primitive_params=primitive_params,
            next_state_pred=next_state_pred,
            criticism=evaluation.feedback,
            evaluation=evaluation,
            next_world_model_state=next_world_model_state,
        )

    @staticmethod
    def _candidate_is_accepted(candidate: _ActionCandidate) -> bool:
        accepted = candidate.evaluation.would_progress & ~candidate.evaluation.predicts_death
        return bool(torch.all(accepted.detach()).cpu().item())

    @staticmethod
    def _candidate_rank(candidate: _ActionCandidate) -> float:
        score = candidate.evaluation.progress_score - candidate.evaluation.death_risk
        return float(score.detach().mean().cpu().item())

    def _select_fallback_candidate(
        self,
        candidates: list[_ActionCandidate],
    ) -> tuple[_ActionCandidate, int]:
        if not candidates:
            raise ValueError("at least one action candidate is required")
        best_index = max(range(len(candidates)), key=lambda index: self._candidate_rank(candidates[index]))
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
        logits_a1, actions1, w_1, b_1 = self.agent(src_A, src_B, src_C, criticism=None, tau=tau)
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
            actor_criticism = first_candidate.criticism
            for _pass_index in range(1, self.max_action_refinement_passes):
                logits_a, actions, w, b = self.agent(
                    src_A,
                    src_B,
                    src_C,
                    criticism=actor_criticism,
                    tau=tau,
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
                actor_criticism = candidate.criticism

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
