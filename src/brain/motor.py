import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.universal import decoder

class MotorLobe(nn.Module):
    """
    Motor Lobe acting as the policy network.
    Takes the latent plan from PrefrontalLobe, and uses the universal decoder
    to decode it into NES/Joypad keyboard action logits.
    """
    def __init__(self, num_actions=7):
        super().__init__()
        self.num_actions = num_actions
        self.decoder = decoder

    def forward(self, plan):
        """
        x: Latent plan of shape (B, 128)
        Returns: Action logits of shape (B, num_actions)
        """
        return self.decoder(plan, modality='action', target_dim=self.num_actions)

    def decide(self, plan):
        """
        Decides the action index based on action logits.
        plan: torch.Tensor of shape (1, 128)
        Returns: int (action index)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(plan)
            action = torch.argmax(logits, dim=-1).item()
        return action

    def learn(self, signal):
        """
        Online reinforcement update based on reward signals.
        """
        reward = float(signal.get('reward', 0.0))
        collision = bool(signal.get('collision', False))

        device = next(self.parameters()).device
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()

        # Generate a forward pass to optimize representation
        dummy_plan = torch.randn(1, 128).to(device)
        logits = self.forward(dummy_plan)

        # Basic policy gradient-like loss to optimize motor mapping online
        loss = torch.mean(logits ** 2)
        if collision:
            # Shift action logits to prevent repeating same action
            loss = loss + 0.5 * torch.max(logits)
        if reward > 0:
            loss = loss - 0.2 * torch.max(logits)

        loss.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
