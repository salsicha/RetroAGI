import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.brain.universal import encoder, decoder

class OccipitalLobe(nn.Module):
    """
    Occipital Lobe for spatial perception and frame reconstruction.
    """
    def __init__(self, num_keypoints=32, channels=3, input_size=(64, 64), *args, **kwargs):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.channels = channels
        self.input_size = input_size
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
        device = next(self.parameters()).device
        if isinstance(state, np.ndarray):
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
        next_states = signal.get('actual_next_state')
        if next_states is None or len(next_states) == 0:
            return
            
        device = next(self.parameters()).device
        
        if isinstance(next_states[0], np.ndarray):
            processed = []
            for s in next_states:
                if len(s.shape) == 3 and s.shape[-1] == 3: # (H, W, C)
                    s = s.transpose(2, 0, 1)
                processed.append(s)
            x = torch.tensor(np.stack(processed), dtype=torch.float32).to(device) / 255.0
        else:
            x = torch.stack(next_states).to(device)
            if len(x.shape) == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            if x.max() > 1.0:
                x = x / 255.0

        self.train()
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
