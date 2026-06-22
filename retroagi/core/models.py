"""Reusable hierarchical actor, world-model, and critic components."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CONTROLLER_SCHEDULES = ("constant", "linear")


@dataclass(frozen=True)
class WorldModelState:
    """LSTM state carried between world-model calls."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def detach(self):
        return WorldModelState(self.hidden.detach(), self.cell.detach())


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

        self.controller = AdaptiveController(schedule=controller_schedule)

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

    def forward(self, src_A, src_B, src_C, criticism=None, tau=1.0):
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

        y_hat_c = self.controller(src_C, w_pred, b_pred)
        return logits_a, y_hat_c, w_pred, b_pred


class WorldModel(nn.Module):
    """
    Predicts the next state using episodic LSTM memory plus multi-frequency
    sinusoidal position features.
    """

    def __init__(self, hidden_size=32, num_layers=1, num_freqs=4, ratio_bc=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_freqs = num_freqs
        self.ratio_bc = ratio_bc
        input_size = 4 + num_freqs * 2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

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
                    "episode_mask must have shape [batch] or "
                    f"[batch, chunks], got {tuple(mask.shape)}"
                )
            chunk_masks[:, 0] = mask
        elif mask.ndim == 2:
            if tuple(mask.shape) == (batch_size, 1):
                chunk_masks[:, 0] = mask[:, 0]
            elif tuple(mask.shape) == (batch_size, chunk_count):
                chunk_masks = mask
            else:
                raise ValueError(
                    "episode_mask must have shape [batch] or "
                    f"[batch, {chunk_count}], got {tuple(mask.shape)}"
                )
        else:
            raise ValueError(
                "episode_mask must have shape [batch] or "
                f"[batch, {chunk_count}], got {tuple(mask.shape)}"
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
        chunk_count = (seq_len_c + self.ratio_bc - 1) // self.ratio_bc

        phases = (
            self._make_phases(seq_len_c, device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        x = torch.stack([state, action, w_context, b_context], dim=-1)
        x = torch.cat([x, phases], dim=-1)

        recurrent_state = self._coerce_state(initial_state, batch_size, device, dtype)
        chunk_masks = self._normalize_episode_mask(
            episode_mask, batch_size, chunk_count, device, dtype
        )

        outputs = []
        for chunk_index, ep_start in enumerate(range(0, seq_len_c, self.ratio_bc)):
            ep_end = ep_start + self.ratio_bc
            recurrent_state = self._mask_state(
                recurrent_state, chunk_masks[:, chunk_index]
            )
            out, (hidden, cell) = self.lstm(
                x[:, ep_start:ep_end, :],
                (recurrent_state.hidden, recurrent_state.cell),
            )
            recurrent_state = WorldModelState(hidden, cell)
            outputs.append(out)

        all_out = torch.cat(outputs, dim=1)
        prediction = self.fc(all_out).squeeze(-1)
        if return_state:
            return prediction, recurrent_state
        return prediction


class Critic(nn.Module):
    """Evaluates predicted C state and returns `[B, L_A, d_model]` feedback."""

    def __init__(self, seq_len_c, seq_len_a, d_model):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.d_model = d_model
        self.net = nn.Sequential(nn.Linear(seq_len_c, 128), nn.ReLU(), nn.Linear(128, seq_len_a * d_model))

    def forward(self, next_state_pred):
        critique = self.net(next_state_pred)
        return critique.view(-1, self.seq_len_a, self.d_model)


class AgentWorldModelCritic(nn.Module):
    """
    Combines the actor, world model, and critic in a two-pass refinement loop.

    Pass one runs the actor with no feedback. Its C actions and B controller
    parameters feed the world model. The critic maps that predicted C state to
    A-level feedback, and pass two reruns the same actor inputs with that
    feedback added to the encoded A stream.
    """

    def __init__(
        self,
        vocab_size,
        seq_len_a,
        seq_len_c,
        ratio_bc,
        d_model=64,
        controller_schedule="constant",
    ):
        super().__init__()
        self.ratio_bc = ratio_bc
        self.agent = HierarchicalAdaptiveModel(
            vocab_size,
            d_model=d_model,
            controller_schedule=controller_schedule,
        )
        self.world_model = WorldModel(ratio_bc=ratio_bc)
        self.critic = Critic(seq_len_c, seq_len_a, d_model)
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
    ):
        logits_a1, actions1, w_1, b_1 = self.agent(src_A, src_B, src_C, criticism=None, tau=tau)

        w_1_context, b_1_context = self.controller_context(src_C, w_1, b_1)
        if (
            world_model_state is None
            and episode_mask is None
            and not return_world_model_state
        ):
            next_state_pred = self.world_model(
                src_C, actions1, w_1_context, b_1_context
            )
            next_world_model_state = None
        else:
            next_state_pred, next_world_model_state = self.world_model(
                src_C,
                actions1,
                w_1_context,
                b_1_context,
                initial_state=world_model_state,
                episode_mask=episode_mask,
                return_state=True,
            )

        criticism = self.critic(next_state_pred)
        actor_criticism = criticism if critic_feedback_enabled else None
        logits_a2, actions2, w_2, b_2 = self.agent(
            src_A, src_B, src_C, criticism=actor_criticism, tau=tau
        )

        outputs = (actions1, next_state_pred, criticism, actions2, logits_a2, w_2, b_2)
        if return_world_model_state:
            return (*outputs, next_world_model_state)
        return outputs
