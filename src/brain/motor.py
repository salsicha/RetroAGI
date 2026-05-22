import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.universal import decoder

class MotorLobe(nn.Module):
    """
    Motor Lobe acting as the policy network.
    Uses TD/Policy Gradient learning over batched replay buffers.
    """
    def __init__(self, num_actions=7):
        super().__init__()
        self.num_actions = num_actions
        self.decoder = decoder

    def forward(self, plan):
        return self.decoder(plan, modality='action', target_dim=self.num_actions)

    def decide(self, plan):
        self.eval()
        with torch.no_grad():
            logits = self.forward(plan)
            # Sample from categorical distribution for exploration
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def learn(self, signal):
        plans = signal.get('plans')
        actions = signal.get('actions')
        rewards = signal.get('reward')
        
        if plans is None or len(plans) == 0:
            return

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()

        logits = self.forward(plans.detach())
        
        # Policy Gradient objective with baseline
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = -(action_log_probs * advantages).mean()
        
        loss.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
