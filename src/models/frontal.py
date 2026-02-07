"""Frontal model for determining long-term goals."""
import torch
import torch.nn as nn

class FrontalLobe(nn.Module):
    """
    The Frontal Lobe model.
    Inputs: Parietal latent.
    Outputs: Long-term goals (text), and latent vector.
    """
    def __init__(self, input_dim, latent_dim=128, vocab_size=100, max_seq_length=10):
        super(FrontalLobe, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder for text (similar to Temporal)
        self.hidden_dim = latent_dim
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, vocab_size)

        # Dummy vocab
        self.vocab = ["<SOS>", "<EOS>", "win", "level", "collect", "coins", "defeat", "bowser"]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}

    def forward(self, parietal_latent):
        latent = self.fc(parietal_latent)
        
        # Generate text goal
        batch_size = latent.size(0)
        hidden = latent.view(1, batch_size, -1)
        input_tok = torch.tensor([self.word_to_idx["<SOS>"]] * batch_size, device=latent.device)
        
        outputs = []
        for _ in range(self.max_seq_length):
            embedded = self.embedding(input_tok).view(1, batch_size, -1)
            output, hidden = self.gru(embedded, hidden)
            output = self.out(output.squeeze(0))
            _, topi = output.topk(1)
            input_tok = topi.squeeze().detach()
            outputs.append(topi)
            
        generated_goals = torch.stack(outputs, dim=1).squeeze()
        return latent, generated_goals

    def sequence_to_text(self, sequence):
        text = []
        # Handle single integer case
        if sequence.dim() == 0:
             sequence = [sequence]
             
        for idx in sequence:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.idx_to_word.get(idx)
            if word == "<EOS>":
                break
            if word:
                text.append(word)
        return " ".join(text)