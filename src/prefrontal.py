import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

class PrefrontalLobe(nn.Module):
    def __init__(self, planning_mode=0.5):
        super().__init__()
        self.planning_mode = planning_mode # 0: Speedrun, 1: Max Coins
        self.fc = nn.Sequential(
            nn.Linear(256, 128), # Input from Temporal + Hippocampus
            nn.ReLU()
        )
        
        # Actor head: Generates latent plan for Motor
        self.actor_head = nn.Linear(129, 64) # 128 + 1 (bias)
        
        # Critic head: Estimates state value (V) for Advantage Actor-Critic
        self.critic_head = nn.Linear(129, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def process(self, latent_t, latent_h):
        # Combine inputs and inject the planning bias
        combined = torch.cat([latent_t, latent_h], dim=-1)
        hidden = self.fc(combined)
        # planning_mode acts as a contextual bias to the network
        bias = torch.full((hidden.size(0), 1), self.planning_mode, device=hidden.device)
        x = torch.cat([hidden, bias], dim=-1)
        
        plan = self.actor_head(x)
        return plan

    def learn(self, signal):
        # Advantage Actor-Critic (A2C) Logic
        if 'latents_t' not in signal or signal['latents_t'] is None:
            return
            
        latents_t = signal['latents_t']
        latents_h = signal['latents_h']
        rewards = signal['reward']
        
        combined = torch.cat([latents_t, latents_h], dim=-1)
        hidden = self.fc(combined)
        bias = torch.full((hidden.size(0), 1), self.planning_mode, device=hidden.device)
        x = torch.cat([hidden, bias], dim=-1)
        
        # Critic Loss: MSE of Value vs Reward
        values = self.critic_head(x).squeeze(-1)
        critic_loss = F.mse_loss(values, rewards)
        
        # Actor Loss: Push continuous latent plans towards positive advantage
        advantage = rewards - values.detach()
        pred_plans = self.actor_head(x)
        actor_loss = -(advantage.unsqueeze(-1) * pred_plans).mean()
        
        loss = critic_loss + actor_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))