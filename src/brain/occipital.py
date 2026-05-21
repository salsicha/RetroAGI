import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brain.universal import encoder, decoder

class OccipitalLobe(nn.Module):
    """
    Occipital Lobe for spatial perception and frame reconstruction.
    Uses the universal encoder and decoder to map pixel frames to the universal latent space.
    """
    def __init__(self, num_keypoints=32, channels=3, input_size=(64, 64), *args, **kwargs):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.input_size = input_size

        # Register global universal encoder and decoder as submodules
        # This makes their parameters discoverable by occipital.parameters()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def get_latent(self, x):
        return self.encode(x)

    def encode(self, x):
        return self.encoder(x, modality='image')

    def decode(self, z):
        return self.decoder(z, modality='image', target_shape=self.input_size)

    def process(self, state):
        """
        Processes game frames from environment.
        state: np.ndarray or torch.Tensor
        Returns:
            latent_v: torch.Tensor of shape (1, 128)
            reconstruction: torch.Tensor of shape (1, channels, H, W)
        """
        device = next(self.parameters()).device
        if isinstance(state, np.ndarray):
            # Input is (H, W, C) from gym/retro, transpose to (C, H, W)
            if len(state.shape) == 3:
                state = state.transpose(2, 0, 1)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        else:
            state_t = state.to(device)
            if len(state_t.shape) == 3:
                state_t = state_t.unsqueeze(0)
            if state_t.max() > 1.0:
                state_t = state_t / 255.0

        self.eval()
        with torch.no_grad():
            z, recon = self.forward(state_t)
        return z, recon

    def learn(self, signal):
        """
        Online learning driven by predictive coding. Minimizes reconstruction surprise.
        """
        next_state = signal.get('actual_next_state')
        if next_state is not None:
            device = next(self.parameters()).device
            if isinstance(next_state, np.ndarray):
                if len(next_state.shape) == 3:
                    next_state = next_state.transpose(2, 0, 1)
                x = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            else:
                x = next_state.to(device)
                if len(x.shape) == 3:
                    x = x.unsqueeze(0)
                if x.max() > 1.0:
                    x = x / 255.0

            self.train()
            # Set up online optimizer for the shared weights
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            optimizer.zero_grad()
            z, recon = self.forward(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
