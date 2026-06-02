"""
Offline Training Script
Trains all lobes using the synthetic dataset.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.occipital import OccipitalLobe
from src.models.temporal import TemporalLobe
from src.models.parietal import ParietalLobe
from src.models.frontal import FrontalLobe
from src.models.motor import MotorLobe
from src.utils.dataset import RetroAGIDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Configuration
    data_dir = os.path.join(os.getcwd(), 'data', 'synthetic')
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Run scripts/generate_data.py first.")
        return

    batch_size = 8
    epochs = 2 # Keep it small for demo
    lr = 1e-4

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Dataset & Loader
    dataset = RetroAGIDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Models
    # Occipital: latent=128 -> split to 64 (what) + 64 (where)
    dim_what = 64
    dim_where = 64
    dim_temporal_hidden = 128
    dim_frontal_hidden = 128
    dim_parietal_hidden = 128
    
    occipital_lobe = OccipitalLobe(latent_dim=128).to(device)
    temporal_lobe = TemporalLobe(input_dim=dim_what + dim_parietal_hidden, hidden_dim=dim_temporal_hidden, vocab_size=50).to(device)
    frontal_lobe = FrontalLobe(input_dim=dim_parietal_hidden, latent_dim=dim_frontal_hidden, vocab_size=50).to(device)
    parietal_lobe = ParietalLobe(input_dim=dim_where + dim_temporal_hidden + dim_frontal_hidden, latent_dim=dim_parietal_hidden).to(device)
    motor_lobe = MotorLobe(input_dim=dim_frontal_hidden, action_space=9).to(device) # Assuming 9 actions

    # Optimizers
    opt_occipital = optim.Adam(occipital_lobe.parameters(), lr=lr)
    opt_temporal = optim.Adam(temporal_lobe.parameters(), lr=lr)
    opt_parietal = optim.Adam(parietal_lobe.parameters(), lr=lr)
    opt_frontal = optim.Adam(frontal_lobe.parameters(), lr=lr)
    opt_motor = optim.Adam(motor_lobe.parameters(), lr=lr)

    # Losses
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for batch in dataloader:
            images = batch['image'].to(device)
            actions = batch['action'].to(device) # (B, 9)
            parietal_maps = batch['parietal_map'].to(device) # (B, 32, 32) -> needs flatten or check decoder output
            
            # 1. Occipital Pass
            reconstructed, (what, where) = occipital_lobe(images), occipital_lobe.get_latent(images)
            loss_occ = criterion_mse(reconstructed, images)
            
            # 2. Predictive Coding Top-Down Integration
            # We use a 2-pass approach to connect the gradient pathways fully, 
            # replacing the disconnected dummies with actual recurrent context.
            dummy_parietal = torch.zeros(images.size(0), dim_parietal_hidden).to(device)
            dummy_frontal = torch.zeros(images.size(0), dim_frontal_hidden).to(device)

            # Pass 1: Feed-forward to get initial top-down contexts
            _, new_temporal_init = temporal_lobe(what, dummy_parietal)
            parietal_latent_init, _ = parietal_lobe(where, new_temporal_init.squeeze(0), dummy_frontal)
            frontal_latent_init, _, _ = frontal_lobe(parietal_latent_init)
            
            # Pass 2: Full Forward Pass with Feedback (Gradient pathway complete)
            generated_seq, new_temporal = temporal_lobe(what, parietal_latent_init)
            parietal_latent, pred_map = parietal_lobe(where, new_temporal.squeeze(0), frontal_latent_init)
            frontal_latent, goals, goal_map = frontal_lobe(parietal_latent)

            # Predictive Coding Self-Consistency Losses
            # This ensures top-down feedback aligns with bottom-up predictions
            loss_temp = criterion_mse(new_temporal, new_temporal_init.detach())
            loss_par = criterion_mse(pred_map.squeeze(1), parietal_maps)
            loss_front = criterion_mse(frontal_latent, frontal_latent_init.detach())

            # Motor
            # Target: Action One-Hot or Multi-Binary
            # Now accurately flows from the Prefrontal planning lobe
            pred_actions = motor_lobe(frontal_latent)
            # MotorLobe output is Softmax(dim=1). Action target is MultiBinary (from recorder).
            # Convert target to index for CrossEntropy
            # The recorder saved One-Hot-like (Single 1).
            target_indices = torch.argmax(actions, dim=1)
            loss_motor = criterion_ce(pred_actions, target_indices)

            # Backprop
            loss = loss_occ + loss_temp + loss_par + loss_front + loss_motor
            
            opt_occipital.zero_grad()
            opt_temporal.zero_grad()
            opt_parietal.zero_grad()
            opt_frontal.zero_grad()
            opt_motor.zero_grad()
            
            loss.backward()
            
            opt_occipital.step()
            opt_temporal.step()
            opt_parietal.step()
            opt_frontal.step()
            opt_motor.step()
            
            total_loss += loss.item()
            
        print(f"Average Loss: {total_loss / len(dataloader):.4f}")

    print("Training Complete. Saving models...")
    os.makedirs('models', exist_ok=True)
    torch.save(occipital_lobe.state_dict(), 'models/occipital.pth')
    torch.save(motor_lobe.state_dict(), 'models/motor.pth')
    # Save others...
    print("Models saved.")

if __name__ == "__main__":
    train()
