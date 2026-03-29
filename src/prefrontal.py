import torch.nn as nn
import torch

class PrefrontalLobe(nn.Module):
    def __init__(self, planning_mode=0.5):
        super().__init__()
        self.planning_mode = planning_mode # 0: Speedrun, 1: Max Coins
        self.fc = nn.Sequential(
            nn.Linear(256, 128), # Input from Temporal + Hippocampus
            nn.ReLU(),
            nn.Linear(128, 64)   # Latent plan for Motor
        )

    def process(self, latent_t, latent_h):
        # Combine inputs and inject the planning bias
        combined = torch.cat([latent_t, latent_h], dim=-1)
        # planning_mode acts as a contextual bias to the network
        bias = torch.full((combined.size(0), 1), self.planning_mode)
        x = torch.cat([combined, bias], dim=-1)
        
        return self.fc(x)

    def learn(self, signal):
        # Implementation of learning logic based on supervisor signal
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)