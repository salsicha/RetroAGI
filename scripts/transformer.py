
import torch
import torch.nn as nn

class MarioTransformer(nn.Module):
    def __init__(self, vocab_size=4, max_seq_len=12, d_model=4, nhead=1, num_layers=3):
        super(MarioTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.register_buffer('pos_encoder', self._get_sinusoidal_encoding(max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.sequences = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], 
                                       [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], 
                                       [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0],
                                       [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]])

        self.vocab_size = vocab_size

        self.custom_vectors = {
            0: [1.0, 0.0, 0.0, 0.0],
            1: [0.0, 1.0, 0.0, 0.0],
            2: [-1.0, 0.0, 0.0, 0.0],
            3: [0.0, -1.0, 0.0, 0.0],
        }


    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1)]
        # x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x)

    def _get_sinusoidal_encoding(self, seq_len, dim):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


    def train_model(self, model, sequences, epochs=200, lr=1e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for epoch in range(epochs):
            for seq in sequences:
                optimizer.zero_grad()
                
                input_seq = seq[:-1].unsqueeze(0)  # [1, seq_len]
                target_seq = seq[1:].unsqueeze(0)  # [1, seq_len]
        
                output = model(input_seq)  # [1, seq_len, vocab_size]
                output = output.view(-1, self.vocab_size)
                target_seq = target_seq.view(-1)
        
                loss = criterion(output, target_seq)
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


    def set_embedding(self, model, index_to_vector):
        """
        Set custom 2D embedding vectors for specified indices.

        Args:
            model: An instance of IntegerTransformer2D.
            index_to_vector: Dict[int, list or tuple of 2 floats], e.g., {1: [0.0, 1.0], 2: [1.0, 1.0]}
        """
        with torch.no_grad():
            for idx, vec in index_to_vector.items():
                # assert len(vec) == 2, f"Vector for index {idx} must be 2-dimensional."
                model.embedding.weight[idx] = torch.tensor(vec, dtype=torch.float32)


    def generate(self, model, start_token, length):
        model.eval()
        tokens = [start_token]
        for _ in range(length - 1):
            inp = torch.tensor(tokens).unsqueeze(0)
            with torch.no_grad():
                output = model(inp)
                next_token = output[0, -1].argmax().item()
            tokens.append(next_token)
        return tokens







