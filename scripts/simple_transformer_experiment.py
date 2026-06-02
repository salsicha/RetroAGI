import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# --------------------------------------------------------
# 1. Synthetic Data Generator
# --------------------------------------------------------
def generate_dual_synthetic_data(num_samples, seq_len_a, ratio, vocab_size):
    """
    Generates two sequences. Seq A is an arithmetic progression.
    Seq B generates `ratio` tokens for every token of Seq A.
    Seq B's target depends on Seq B's input and the *latest* Seq A target.
    """
    seq_len_b = seq_len_a * ratio
    X_A, Y_A, X_B, Y_B = [], [], [], []
    
    for _ in range(num_samples):
        # Sequence A: standard progression
        start_a = np.random.randint(0, vocab_size)
        seq_a = [(start_a + i) % vocab_size for i in range(seq_len_a + 1)]
        X_A.append(seq_a[:-1])
        Y_A.append(seq_a[1:])
        
        # Sequence B: faster timescale progression
        start_b = np.random.randint(0, vocab_size)
        seq_b = [(start_b + i) % vocab_size for i in range(seq_len_b + 1)]
        X_B.append(seq_b[:-1])
        
        # Target B = (Next Token of B + Latest Next Token of A)
        y_b = []
        for j in range(seq_len_b):
            i = j // ratio # Find the index of the latest A token
            y_b.append((seq_b[j+1] + seq_a[i+1]) % vocab_size)
        Y_B.append(y_b)
        
    return (torch.tensor(X_A, dtype=torch.long), torch.tensor(Y_A, dtype=torch.long),
            torch.tensor(X_B, dtype=torch.long), torch.tensor(Y_B, dtype=torch.long))

# --------------------------------------------------------
# 2. Transformer Model Components
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class DualCausalTransformer(nn.Module):
    """
    A two-stream Transformer where Stream B depends on the output of Stream A.
    Stream A predicts a token, passes it to B, and B predicts the final answer.
    """
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Stream A
        encoder_layers_A = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer_A = nn.TransformerEncoder(encoder_layers_A, num_layers)
        
        # Stream B
        encoder_layers_B = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer_B = nn.TransformerEncoder(encoder_layers_B, num_layers)
        
        # Shared Output Projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz):
        """ Prevents the model from looking ahead into future tokens. """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_A, src_B):
        seq_len_A = src_A.size(1)
        seq_len_B = src_B.size(1)
        ratio = seq_len_B // seq_len_A
        
        causal_mask_A = self.generate_causal_mask(seq_len_A).to(src_A.device)
        causal_mask_B = self.generate_causal_mask(seq_len_B).to(src_B.device)
        
        # --- Stream A ---
        x_A = self.embedding(src_A) * math.sqrt(self.d_model)
        x_A = self.pos_encoder(x_A)
        hidden_A = self.transformer_A(x_A, mask=causal_mask_A)
        logits_A = self.fc_out(hidden_A)
        
        # --- The Bridge ---
        # Use Gumbel-Softmax with hard=True to sample a discrete one-hot token 
        # during the forward pass, but maintain a continuous gradient for the backward pass.
        probs_A = F.gumbel_softmax(logits_A, tau=1.0, hard=True, dim=-1)
        pred_emb_A = torch.matmul(probs_A, self.embedding.weight)
        
        # Upsample Stream A's predictions to match Stream B's timescale
        pred_emb_A_upsampled = pred_emb_A.repeat_interleave(ratio, dim=1)
        
        # --- Stream B ---
        x_B = self.embedding(src_B) * math.sqrt(self.d_model)
        x_B = self.pos_encoder(x_B)
        
        # Inject Upsampled Stream A token embeddings into Stream B
        x_B_combined = x_B + pred_emb_A_upsampled
        
        hidden_B = self.transformer_B(x_B_combined, mask=causal_mask_B)
        logits_B = self.fc_out(hidden_B)
        
        return logits_A, logits_B

# --------------------------------------------------------
# 3. Testing and Visualization System
# --------------------------------------------------------
def train_and_evaluate():
    # Hyperparameters
    vocab_size = 20
    seq_len_A = 8
    ratio = 5
    batch_size = 32
    epochs = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    train_XA, train_YA, train_XB, train_YB = generate_dual_synthetic_data(1000, seq_len_A, ratio, vocab_size)
    test_XA, test_YA, test_XB, test_YB = generate_dual_synthetic_data(200, seq_len_A, ratio, vocab_size)
    
    train_XA, train_YA = train_XA.to(device), train_YA.to(device)
    train_XB, train_YB = train_XB.to(device), train_YB.to(device)
    test_XA, test_YA = test_XA.to(device), test_YA.to(device)
    test_XB, test_YB = test_XB.to(device), test_YB.to(device)
    
    # Initialize model, loss, optimizer
    model = DualCausalTransformer(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'accuracy_A': [], 'accuracy_B': []}
    
    print("\nStarting Dual Sequence Training...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        
        # Shuffle batch
        permutation = torch.randperm(train_XA.size()[0])
        epoch_loss = 0
        
        for i in range(0, train_XA.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_xa, batch_ya = train_XA[indices], train_YA[indices]
            batch_xb, batch_yb = train_XB[indices], train_YB[indices]
            
            optimizer.zero_grad()
            logits_A, logits_B = model(batch_xa, batch_xb)
            
            # Core Requirement: First sequence trained by getting error from the second!
            # We ONLY compute loss on Model B. Model A must learn purely through the bridge.
            loss = criterion(logits_B.view(-1, vocab_size), batch_yb.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation step
        model.eval()
        with torch.no_grad():
            test_logits_A, test_logits_B = model(test_XA, test_XB)
            
            preds_A = test_logits_A.argmax(dim=-1)
            preds_B = test_logits_B.argmax(dim=-1)
            
            acc_A = (preds_A == test_YA).sum().item() / test_YA.numel() * 100
            acc_B = (preds_B == test_YB).sum().item() / test_YB.numel() * 100
            
        avg_loss = epoch_loss / (train_XA.size()[0] / batch_size)
        history['loss'].append(avg_loss)
        history['accuracy_A'].append(acc_A)
        history['accuracy_B'].append(acc_B)
        
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch+1:02d}/{epochs} - Loss (B only): {avg_loss:.4f} | Acc A: {acc_A:.1f}% | Acc B: {acc_B:.1f}%")
            
            # Show a sample prediction during training
            sample_xa = test_XA[0].cpu().numpy()
            sample_xb = test_XB[0].cpu().numpy()
            target_b = test_YB[0].cpu().numpy()
            pred_b = preds_B[0].cpu().numpy()
            
            tqdm.write(f"   [Input A]  : {sample_xa} (len {len(sample_xa)})")
            tqdm.write(f"   [Input B]  : {sample_xb[:15]}... (len {len(sample_xb)})")
            tqdm.write(f"   [Target B] : {target_b[:15]}...")
            tqdm.write(f"   [Model B]  : {pred_b[:15]}...\n")

    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss (Stream B)', color='red')
    plt.title('Dual Transformer Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (from B only)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy_A'], label='Accuracy A (Implicit)', color='green', linestyle='--')
    plt.plot(history['accuracy_B'], label='Accuracy B (Explicit)', color='blue')
    plt.title('Transformer Prediction Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()