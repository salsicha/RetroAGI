"""Temporal model for processing sequences of events."""
import torch
import torch.nn as nn

class TemporalLobe(nn.Module):
    """
    The Temporal Lobe model, which is a sequence-to-sequence model.
    It takes in a latent vector from the Occipital Lobe and generates a
    description of the event.
    """

    def __init__(self, latent_dim, hidden_dim, vocab_size, max_seq_length=20):
        super(TemporalLobe, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        
        # A dummy vocabulary for now
        self.vocab = ["<SOS>", "<EOS>", "mario", "jumps", "moves", "right", "left", "on", "a", "platform"]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}

    def forward(self, latent_vector):
        """
        Generate a sequence of words from a latent vector.
        """
        batch_size = latent_vector.size(0)
        hidden = self.init_hidden(latent_vector)

        # Start with the <SOS> token
        input = torch.tensor([self.word_to_idx["<SOS>"]] * batch_size, device=latent_vector.device)

        outputs = []
        for _ in range(self.max_seq_length):
            embedded = self.embedding(input).view(1, batch_size, -1)
            output, hidden = self.gru(embedded, hidden)
            output = self.out(output.squeeze(0))
            _, topi = output.topk(1)
            input = topi.squeeze().detach()
            outputs.append(topi)

        return torch.stack(outputs, dim=1).squeeze()

    def init_hidden(self, latent_vector):
        """
        Initializes the hidden state of the GRU with the latent vector
        from the Occipital Lobe.
        """
        # Reshape the latent vector to be the initial hidden state
        return latent_vector.view(1, latent_vector.size(0), -1)

    def sequence_to_text(self, sequence):
        """
        Convert a sequence of indices to a text string.
        """
        text = []
        for idx in sequence:
            word = self.idx_to_word.get(idx.item())
            if word == "<EOS>":
                break
            if word:
                text.append(word)
        return " ".join(text)
