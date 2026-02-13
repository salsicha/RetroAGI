"""Motor model for controlling the game character."""
import torch
import torch.nn as nn
import math
import torch.distributions

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :].unsqueeze(1)

class MotorLobe(nn.Module):
    """
    The Motor Lobe model (Transformer Decoder).
    Inputs: Parietal latent (Context).
    Outputs: Sequence of Key press tokens.
    """
    def __init__(self, input_dim, action_space, d_model=128, nhead=4, num_layers=2, max_seq_length=20):
        super(MotorLobe, self).__init__()
        self.action_space = action_space
        self.max_seq_length = max_seq_length
        self.d_model = d_model

        # Project parietal latent to transformer dimension
        self.context_projection = nn.Linear(input_dim, d_model)

        # Value Head for Actor-Critic (PPO)
        self.value_head = nn.Linear(d_model, 1)

        # Action Embedding (vocab size = action_space + special tokens)
        # 0: <SOS>, 1: <EOS>, 2..N: Actions
        self.vocab_size = action_space + 2
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length + 2)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, self.vocab_size)

    def forward(self, parietal_latent, target_seq=None, sample=False):
        """
        Args:
            parietal_latent: (batch_size, input_dim)
            target_seq: (seq_len, batch_size) - Optional, for training with teacher forcing
            sample: (bool) - If True, samples from distribution and returns log_probs for RL.
        Returns:
            logits (if target_seq) OR sequence (if not sample) OR (sequence, log_probs) (if sample)
        """
        batch_size = parietal_latent.size(0)
        device = parietal_latent.device

        # Prepare Memory (Context from Parietal)
        # Transformer expects memory shape: (mem_seq_len, batch_size, d_model)
        memory = self.context_projection(parietal_latent).unsqueeze(0) 
        
        # Calculate Value
        value = self.value_head(memory.squeeze(0))

        if target_seq is not None:
            # Training mode (Teacher Forcing)
            tgt_emb = self.embedding(target_seq) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_mask = self.generate_square_subsequent_mask(target_seq.size(0)).to(device)
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            return self.fc_out(output), value
        else:
            # Inference mode (Autoregressive generation)
            seq_output = self.generate(memory, batch_size, device, sample=sample)
            if sample:
                return seq_output[0], seq_output[1], value
            return seq_output, value

    def generate(self, memory, batch_size, device, sample=False):
        input_token = torch.tensor([[0] * batch_size], device=device) # <SOS>
        generated_seq = []
        log_probs = []
        
        for _ in range(self.max_seq_length):
            tgt_emb = self.embedding(input_token) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            output = self.transformer_decoder(tgt_emb, memory)
            logits = self.fc_out(output[-1, :, :])
            
            if sample:
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_token = dist.sample()
                log_probs.append(dist.log_prob(next_token))
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
            
            next_token = next_token.unsqueeze(0)
            generated_seq.append(next_token)
            input_token = torch.cat([input_token, next_token], dim=0)
            
            if next_token.item() == 1: # <EOS>
                break
                
        seq = torch.stack(generated_seq, dim=0).squeeze(1)
        if sample:
            return seq, torch.stack(log_probs)
        return seq

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask