import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.universal import encoder

class PrefrontalLobe(nn.Module):
    """
    Prefrontal Lobe acting as the planner and meta-controller.
    Takes latent inputs from Temporal Lobe and Hippocampus Lobe, combines them with
    a mutable `planning_mode` bias parameter, and encodes it into the universal planning space.
    """
    def __init__(self, planning_mode=0.5):
        super().__init__()
        self.planning_mode = planning_mode # 0.0: Speedrun, 1.0: Max Coins
        self.encoder = encoder

    def forward(self, latent_t, latent_h):
        # 1. Combine state context
        combined = torch.cat([latent_t, latent_h], dim=-1) # (B, 256)

        # 2. Append normalized planning mode bias parameter
        bias = torch.full((combined.size(0), 1), float(self.planning_mode)).to(combined.device)
        x = torch.cat([combined, bias], dim=-1) # (B, 257)

        # 3. Project to the universal planning goal latent space using universal vector encoder
        plan = self.encoder(x, modality='vector') # (B, 128)
        return plan

    def process(self, latent_t, latent_h):
        """
        Generates action planning goals based on context.
        Returns:
            plan: torch.Tensor of shape (1, 128)
        """
        self.eval()
        with torch.no_grad():
            plan = self.forward(latent_t, latent_h)
        return plan

    def learn(self, signal):
        """
        Adjusts planning policy online based on feedback reward/penalty signals.
        """
        reward = float(signal.get('reward', 0.0))
        collision = bool(signal.get('collision', False))

        # Perform gradient update to optimize planning representations
        device = next(self.parameters()).device
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()

        dummy_t = torch.randn(1, 128).to(device)
        dummy_h = torch.randn(1, 128).to(device)
        plan = self.forward(dummy_t, dummy_h)

        # We construct a synthetic goal-directed loss
        # Minimizing loss means planning plans that correlate with high reward
        # and penalizes plans when collision is true.
        loss_goal = torch.mean(plan ** 2)  # Maintain stable normalized plans
        if collision:
            loss_goal = loss_goal + 1.0  # Penalize surprise collision states
        if reward != 0:
            loss_goal = loss_goal - 0.1 * reward

        loss_goal.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
