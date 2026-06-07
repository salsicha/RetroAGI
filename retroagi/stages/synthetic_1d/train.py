import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from retroagi.core import AgentWorldModelCritic, StageSpec

SYNTHETIC_1D_SPEC = StageSpec(
    name="synthetic_1d",
    observation_kind="procedural one-dimensional sequences",
    action_kind="continuous controller targets",
    seq_len_a=8,
    ratio_ab=2,
    ratio_bc=4,
    vocab_size=20,
)

# --------------------------------------------------------
# 1. Synthetic Data Generator
# --------------------------------------------------------
def generate_hierarchical_data(num_samples, seq_len_a, ratio_ab, ratio_bc, vocab_size):
    """
    Generates three levels of data:
    Seq A: Slow discrete sequence.
    Seq B: Medium discrete sequence.
    Seq C: Fast continuous sequence (controlled system).
    """
    seq_len_b = seq_len_a * ratio_ab
    seq_len_c = seq_len_b * ratio_bc
    
    X_A, Y_A = [], []
    X_B, Y_B = [], []
    X_C_in, Y_C_target = [], []
    
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
        
        # Target B (implicit, not directly trained, used to generate C's parameters)
        y_b = []
        c_in_seq = []
        c_target_seq = []
        for j in range(seq_len_b):
            i = j // ratio_ab # Index of latest A token
            # Combine A and B to get a target concept
            concept = (seq_b[j+1] + seq_a[i+1]) % vocab_size
            y_b.append(concept)
            
            # The concept defines the true parameters for the controller
            w_true = np.sin(concept)
            b_true = np.cos(concept)
            
            # Sequence C: Generate fast inputs and targets for the controller
            for k in range(ratio_bc):
                x_val = np.random.randn()
                y_val = w_true * x_val + b_true
                c_in_seq.append(x_val)
                c_target_seq.append(y_val)
                
        Y_B.append(y_b)
        X_C_in.append(c_in_seq)
        Y_C_target.append(c_target_seq)
        
    return (torch.tensor(X_A, dtype=torch.long), torch.tensor(Y_A, dtype=torch.long),
            torch.tensor(X_B, dtype=torch.long), torch.tensor(Y_B, dtype=torch.long),
            torch.tensor(X_C_in, dtype=torch.float), torch.tensor(Y_C_target, dtype=torch.float))

# --------------------------------------------------------
# 2. Testing and Visualization System
# --------------------------------------------------------
def train_and_evaluate():
    # Hyperparameters
    vocab_size = SYNTHETIC_1D_SPEC.vocab_size
    seq_len_A = SYNTHETIC_1D_SPEC.seq_len_a
    ratio_AB = SYNTHETIC_1D_SPEC.ratio_ab
    ratio_BC = SYNTHETIC_1D_SPEC.ratio_bc
    batch_size = 32
    epochs = 60
    tau_start = 5.0
    tau_end = 0.1
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Create datasets
    train_XA, train_YA, train_XB, train_YB, train_XC, train_YC = generate_hierarchical_data(1000, seq_len_A, ratio_AB, ratio_BC, vocab_size)
    test_XA, test_YA, test_XB, test_YB, test_XC, test_YC = generate_hierarchical_data(200, seq_len_A, ratio_AB, ratio_BC, vocab_size)
    
    train_XA, train_YA = train_XA.to(device), train_YA.to(device)
    train_XB, train_YB = train_XB.to(device), train_YB.to(device)
    train_XC, train_YC = train_XC.to(device), train_YC.to(device)
    
    test_XA, test_YA = test_XA.to(device), test_YA.to(device)
    test_XB, test_YB = test_XB.to(device), test_YB.to(device)
    test_XC, test_YC = test_XC.to(device), test_YC.to(device)
    
    seq_len_C = seq_len_A * ratio_AB * ratio_BC
    
    # Initialize model, loss, optimizer
    model = AgentWorldModelCritic(vocab_size=vocab_size, seq_len_a=seq_len_A, seq_len_c=seq_len_C, ratio_bc=ratio_BC).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss_agent1': [], 'loss_agent2': [], 'loss_wm': [], 'error_B': [], 'accuracy_A': []}
    
    print("\nStarting Hierarchical Adaptive Controller Training...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Linearly decay temperature from tau_start to tau_end over the epochs
        tau = max(tau_end, tau_start - (tau_start - tau_end) * (epoch / max(1, epochs - 1)))
        
        model.train()
        
        permutation = torch.randperm(train_XA.size()[0])
        epoch_loss_1 = 0
        epoch_loss_2 = 0
        epoch_loss_wm = 0
        
        for i in range(0, train_XA.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_xa, batch_ya = train_XA[indices], train_YA[indices]
            batch_xb = train_XB[indices]
            batch_xc_in, batch_yc_target = train_XC[indices], train_YC[indices]
            
            optimizer.zero_grad()
            actions1, next_state_pred, criticism, actions2, logits_A2, w_2, b_2 = model(batch_xa, batch_xb, batch_xc_in, tau=tau)
            
            # 1. World Model Loss (Simulate environment transition S' = S + Action)
            true_next_state = batch_xc_in + actions1.detach()
            loss_wm = criterion(next_state_pred, true_next_state)
            
            # 2. Agent Pass 1 Loss (No criticism)
            loss_agent1 = criterion(actions1, batch_yc_target)
            
            # 3. Agent Pass 2 Loss (With criticism)
            # The Critic's gradients come entirely through this loss!
            loss_agent2 = criterion(actions2, batch_yc_target)
            
            # Total Loss
            loss = loss_agent1 + loss_agent2 + loss_wm
            loss.backward()
            optimizer.step()
            
            epoch_loss_1 += loss_agent1.item()
            epoch_loss_2 += loss_agent2.item()
            epoch_loss_wm += loss_wm.item()
            
        # Validation step
        model.eval()
        with torch.no_grad():
            # Evaluate with the lowest temperature to simulate hard discrete choices
            test_actions1, test_next_state, test_crit, test_actions2, test_logits_A2, w_pred, b_pred = model(test_XA, test_XB, test_XC, tau=tau_end)
            
            preds_A = test_logits_A2.argmax(dim=-1)
            acc_A = (preds_A == test_YA).sum().item() / test_YA.numel() * 100
            
            # Calculate Layer B's implicit parameter prediction error
            w_true = torch.sin(test_YB.float())
            b_true = torch.cos(test_YB.float())
            error_B = ((F.mse_loss(w_pred, w_true) + F.mse_loss(b_pred, b_true)) / 2).item()
            
        batches = train_XA.size()[0] / batch_size
        avg_loss_1 = epoch_loss_1 / batches
        avg_loss_2 = epoch_loss_2 / batches
        avg_loss_wm = epoch_loss_wm / batches
        
        history['loss_agent1'].append(avg_loss_1)
        history['loss_agent2'].append(avg_loss_2)
        history['loss_wm'].append(avg_loss_wm)
        history['error_B'].append(error_B)
        history['accuracy_A'].append(acc_A)
        
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch+1:02d}/{epochs} [Tau: {tau:.2f}] - Agent Loss P1: {avg_loss_1:.4f} -> P2: {avg_loss_2:.4f} | WM Loss: {avg_loss_wm:.4f} | Param Err B: {error_B:.4f} | Acc A: {acc_A:.1f}%")
            
            # Show a sample prediction during training
            target_c = test_YC[0].cpu().numpy()
            pred_c = test_actions2[0].cpu().numpy()
            
            tqdm.write(f"   [Controller Target C] : {target_c[:5].round(2)}...")
            tqdm.write(f"   [Controller Pred C]   : {pred_c[:5].round(2)}...\n")

    # Visualization
    import matplotlib.pyplot as plt

    plt.figure(figsize=(24, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['loss_agent1'], label='Pass 1 (No Critic)', color='red', alpha=0.6)
    plt.plot(history['loss_agent2'], label='Pass 2 (With Critic)', color='blue')
    plt.title('Layer C: Controller Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(history['loss_wm'], label='World Model Loss', color='purple')
    plt.title('World Model Dynamics Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(history['error_B'], label='Parameter Error B', color='orange')
    plt.title('Layer B: Param Prediction Error')
    plt.xlabel('Epochs')
    plt.ylabel('MSE (w, b)')
    plt.legend()
    
    plt.subplot(1, 4, 4)
    plt.plot(history['accuracy_A'], label='Accuracy A', color='green', linestyle='--')
    plt.title('Layer A: Concept Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
