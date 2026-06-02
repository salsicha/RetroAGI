import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class MotorLobe(nn.Module):
    def __init__(self, input_dim=64, action_space=9):
        super().__init__()
        # Policy Network
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def decide(self, plan):
        probs = self.policy(plan)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def forward(self, plan):
        return self.policy(plan)

    def learn(self, signal):
        # Reinforce with reward baseline from supervisor
        if 'plans' not in signal or signal['plans'] is None:
            return
            
        plans = signal['plans']
        actions = signal['actions']
        rewards = signal['reward']
        
        probs = self.policy(plans.detach())
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Calculate Entropy to encourage exploration
        entropy = dist.entropy()
        
        # Policy Gradient loss with Entropy Regularization
        # The 0.05 multiplier controls the strength of the exploration bonus
        loss = -(log_probs * rewards + 0.05 * entropy).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))