import numpy as np
import torch
import torch.nn as nn
from brain.universal import decoder

try:
    from pyhgf import GaussianRandomWalk
except ImportError:
    class GaussianRandomWalk:
        def __init__(self, mean, precision, decay):
            self.mean = mean
            self.precision = precision
            self.decay = decay
        def update(self, x):
            prediction = self.mean * self.decay
            prediction_error = x - prediction
            self.mean = prediction + 0.15 * prediction_error
            return self.mean

class TemporalLobe(nn.Module):
    """
    Temporal Lobe for sequences and tracking.
    Uses Hierarchical Gaussian Filters (HGF) to model sequential transitions in the 128-dim latent space,
    and the universal decoder to reconstruct sprite positions over time.
    """
    def __init__(self, num_keypoints=16):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.coords_dim = num_keypoints * 2  # 32 coords by default
        self.latent_dim = 128

        # Initialize HGF bank for each of the 128 latent dimensions
        self.nodes = [
            GaussianRandomWalk(mean=0.0, precision=1.0, decay=0.95)
            for _ in range(self.latent_dim)
        ]

        # Register universal decoder as a submodule
        self.decoder = decoder

    def process(self, latent_v):
        """
        Updates the sequence predictions based on current visual latent.
        latent_v: torch.Tensor of shape (1, 128)
        Returns:
            latent_t: torch.Tensor of shape (1, 128) (Predicted next state latent)
            sprite_pos_pred: torch.Tensor of shape (1, 32) (Reconstructed coordinates)
        """
        device = latent_v.device
        z_np = latent_v.detach().cpu().numpy().squeeze(0)

        # Update HGF nodes
        pred_z_np = []
        for i, val in enumerate(z_np):
            pred = self.nodes[i].update(val)
            pred_z_np.append(pred)

        latent_t = torch.tensor(pred_z_np, dtype=torch.float32).unsqueeze(0).to(device)

        # Decode predicted coordinates from the predicted next state latent
        with torch.no_grad():
            sprite_pos_pred = self.decoder(latent_t, modality='coords', target_dim=self.coords_dim)

        return latent_t, sprite_pos_pred

    def learn(self, signal):
        """
        Online predictive coding update for temporal tracking.
        We adjust HGF predictions or decoder parameters.
        """
        # Temporal lobe uses pyhgf online filtering so tracking is inherently online.
        # We can also perform a quick gradient step to align the coordinate decoder to target sprite locations
        # if they are present in the info signal (e.g. from x_pos).
        x_pos = signal.get('x_pos')
        if x_pos is not None:
            # We mock the ground-truth sprite positions or train coordinate head dynamically
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            optimizer.zero_grad()
            
            # Predict coordinates from current HGF predictions
            pred_z = torch.tensor([n.mean for n in self.nodes], dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
            pred_coords = self.decoder(pred_z, modality='coords', target_dim=self.coords_dim)
            
            # Construct dummy target centered around x_pos for testing/bootstrapping
            target = torch.zeros_like(pred_coords)
            target[0, 0] = float(x_pos) / 100.0  # Normalized x-coordinate
            
            loss = nn.functional.mse_loss(pred_coords, target)
            loss.backward()
            optimizer.step()
            self.eval()

    def state_dict(self):
        return {'means': [n.mean for n in self.nodes], 'precisions': [n.precision for n in self.nodes]}

    def load_state_dict(self, state):
        for i, node in enumerate(self.nodes):
            node.mean = state['means'][i]
            node.precision = state['precisions'][i]

    def save(self, path):
        # Save both torch weights (decoder) and HGF states
        save_data = {
            'decoder': self.decoder.state_dict(),
            'hgf': self.state_dict()
        }
        torch.save(save_data, path)

    def load(self, path):
        save_data = torch.load(path)
        if 'decoder' in save_data:
            self.decoder.load_state_dict(save_data['decoder'])
        if 'hgf' in save_data:
            self.load_state_dict(save_data['hgf'])
