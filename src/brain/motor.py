import torch
import torch.nn as nn

class MotorLobe(nn.Module):
    """
    Architecture: Policy Network.
    Input: Prefrontal Latent.
    Output: Keyboard presses (Action logits).
    """
    def __init__(self, input_dim=64, num_actions=12):
        super().__init__()
        # input_dim matches the output of PrefrontalLobe (64 by default)
        # num_actions matches common NES/Super Mario Bros action space
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        """
        x: Latent plan from PrefrontalLobe
        returns: Action logits for the game environment
        """
        return self.net(x)
