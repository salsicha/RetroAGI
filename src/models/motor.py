"""Motor model for controlling the game character."""
import torch
import torch.nn as nn

class MotorLobe(nn.Module):
    """
    The Motor Lobe model.
    Inputs: Parietal latent.
    Outputs: Key press inputs (Discrete).
    """
    def __init__(self, input_dim, action_space=9): # NES typically has 8 buttons + 1 null? Or combinations.
        super(MotorLobe, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=1)
        )

    def forward(self, parietal_latent):
        action_probs = self.fc(parietal_latent)
        return action_probs