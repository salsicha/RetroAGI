"""Reusable hierarchical actor, world-model, and critic components."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self):
        super().__init__()

    def forward(self, x_c, w, b):
        ratio_bc = x_c.size(1) // w.size(1)
        w_upsampled = w.repeat_interleave(ratio_bc, dim=1)
        b_upsampled = b.repeat_interleave(ratio_bc, dim=1)
        return w_upsampled * x_c + b_upsampled


class HierarchicalAdaptiveModel(nn.Module):
    """Three-level actor: Transformer A -> Transformer B -> adaptive controller."""

    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
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

        self.controller = AdaptiveController()

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

    def forward(self, src_A, src_B, src_C, criticism=None, tau=1.0):
        seq_len_a = src_A.size(1)
        seq_len_b = src_B.size(1)
        ratio_ab = seq_len_b // seq_len_a

        causal_mask_a = self.generate_causal_mask(seq_len_a).to(src_A.device)
        causal_mask_b = self.generate_causal_mask(seq_len_b).to(src_B.device)

        x_a = self.embedding(src_A) * math.sqrt(self.d_model)
        x_a = self.pos_encoder(x_a)
        if criticism is not None:
            x_a = x_a + criticism

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

    def _make_phases(self, seq_len, device):
        phases = []
        for t in range(seq_len):
            t_norm = t / seq_len
            step_phases = []
            for k in range(1, self.num_freqs + 1):
                step_phases.append(math.sin(2 * math.pi * k * t_norm))
                step_phases.append(math.cos(2 * math.pi * k * t_norm))
            phases.append(step_phases)
        return torch.tensor(phases, dtype=torch.float, device=device)

    def forward(self, state, action, w_context, b_context):
        batch_size = state.size(0)
        seq_len_c = state.size(1)
        device = state.device

        phases = self._make_phases(seq_len_c, device).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.stack([state, action, w_context, b_context], dim=-1)
        x = torch.cat([x, phases], dim=-1)

        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        outputs = []
        for ep_start in range(0, seq_len_c, self.ratio_bc):
            ep_end = ep_start + self.ratio_bc
            out, (h, c) = self.lstm(x[:, ep_start:ep_end, :], (h, c))
            outputs.append(out)

        all_out = torch.cat(outputs, dim=1)
        return self.fc(all_out).squeeze(-1)


class Critic(nn.Module):
    """Evaluates the predicted next state and outputs a criticism vector."""

    def __init__(self, seq_len_c, seq_len_a, d_model):
        super().__init__()
        self.seq_len_a = seq_len_a
        self.d_model = d_model
        self.net = nn.Sequential(nn.Linear(seq_len_c, 128), nn.ReLU(), nn.Linear(128, seq_len_a * d_model))

    def forward(self, next_state_pred):
        critique = self.net(next_state_pred)
        return critique.view(-1, self.seq_len_a, self.d_model)


class AgentWorldModelCritic(nn.Module):
    """Combines the actor, world model, and critic in a two-pass refinement loop."""

    def __init__(self, vocab_size, seq_len_a, seq_len_c, ratio_bc, d_model=64):
        super().__init__()
        self.ratio_bc = ratio_bc
        self.agent = HierarchicalAdaptiveModel(vocab_size, d_model=d_model)
        self.world_model = WorldModel(ratio_bc=ratio_bc)
        self.critic = Critic(seq_len_c, seq_len_a, d_model)

    def forward(self, src_A, src_B, src_C, tau=1.0):
        logits_a1, actions1, w_1, b_1 = self.agent(src_A, src_B, src_C, criticism=None, tau=tau)

        w_1_up = w_1.repeat_interleave(self.ratio_bc, dim=1)
        b_1_up = b_1.repeat_interleave(self.ratio_bc, dim=1)
        next_state_pred = self.world_model(src_C, actions1, w_1_up, b_1_up)

        criticism = self.critic(next_state_pred)
        logits_a2, actions2, w_2, b_2 = self.agent(src_A, src_B, src_C, criticism=criticism, tau=tau)

        return actions1, next_state_pred, criticism, actions2, logits_a2, w_2, b_2

