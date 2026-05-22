import torch
import torch.nn as nn
from src.brain.universal import decoder

class TemporalLobe(nn.Module):
    """
    Temporal Lobe for sequences and tracking.
    Vectorized representation of Hierarchical Gaussian Filters (HGF) for 128-dim latent space.
    """
    def __init__(self, num_keypoints=16):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.coords_dim = num_keypoints * 2
        self.latent_dim = 128

        self.register_buffer('hgf_mean', torch.zeros(1, self.latent_dim))
        self.register_buffer('hgf_precision', torch.ones(1, self.latent_dim))
        self.hgf_decay = 0.95

        self.decoder = decoder

    def process(self, latent_v):
        z = latent_v.detach()

        prediction = self.hgf_mean * self.hgf_decay
        prediction_error = z - prediction
        self.hgf_mean = prediction + 0.15 * prediction_error

        latent_t = self.hgf_mean.clone()

        with torch.no_grad():
            sprite_pos_pred = self.decoder(latent_t, modality='coords', target_dim=self.coords_dim)

        return latent_t, sprite_pos_pred

    def learn(self, signal):
        x_pos = signal.get('x_pos')
        if x_pos is None:
            return

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()
        
        pred_z = self.hgf_mean.detach()
        pred_coords = self.decoder(pred_z, modality='coords', target_dim=self.coords_dim)
        
        target = torch.zeros_like(pred_coords)
        if isinstance(x_pos, torch.Tensor):
            target[:, 0] = x_pos.float().view(-1) / 100.0
        else:
            target[:, 0] = float(x_pos) / 100.0
            
        loss = nn.functional.mse_loss(pred_coords, target)
        loss.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
