import torch
import torch.nn as nn
import torch.nn.functional as F
from src.brain.universal import encoder

class PrefrontalLobe(nn.Module):
    """
    Prefrontal Lobe acting as the planner and meta-controller.
    Optimizes planning vector representations to correlate with high expected reward.
    """
    def __init__(self, planning_mode=0.5):
        super().__init__()
        self.planning_mode = planning_mode
        self.encoder = encoder

    def forward(self, latent_t, latent_h):
        combined = torch.cat([latent_t, latent_h], dim=-1)
        bias = torch.full((combined.size(0), 1), float(self.planning_mode)).to(combined.device)
        x = torch.cat([combined, bias], dim=-1)
        plan = self.encoder(x, modality='vector')
        return plan

    def process(self, latent_t, latent_h):
        self.eval()
        with torch.no_grad():
            plan = self.forward(latent_t, latent_h)
        return plan

    def learn(self, signal):
        latents_t = signal.get('latents_t')
        latents_h = signal.get('latents_h')
        rewards = signal.get('reward')
        
        if latents_t is None or len(latents_t) == 0:
            return

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()

        plans = self.forward(latents_t.detach(), latents_h.detach())
        
        # Use the magnitude/mean of the plan to predict the expected reward (TD-like objective)
        plan_value = torch.mean(plans, dim=-1)
        loss = F.mse_loss(plan_value, rewards)
        
        loss.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
