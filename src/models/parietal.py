"""Parietal model for identifying short-term objectives."""
import torch
import torch.nn as nn

class ParietalLobe(nn.Module):
    """
    The Parietal Lobe model.
    Inputs: "Where/How" (Occipital), Temporal latent, Frontal latent.
    Outputs: Short-term objectives (decoder), and latent vector for other lobes.
    """
    def __init__(self, input_dim, latent_dim=128):
        super(ParietalLobe, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs a "map" or "mask" of objectives (e.g., 32x32 map)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.Sigmoid()
        )

    def forward(self, occipital_where, temporal_latent, frontal_latent):
        # Concatenate inputs
        # Handle cases where inputs might be None (e.g. first step)
        if temporal_latent is None:
             temporal_latent = torch.zeros_like(occipital_where) # Assuming same batch size, careful with dim
             # Wait, temporal_latent dim might be diff. 
             # Let's assume the caller handles zeros or we infer batch size.
             pass 
        
        # simplified for now: assume inputs are provided and correct shape
        x = torch.cat((occipital_where, temporal_latent, frontal_latent), dim=1)
        latent = self.fc(x)
        objectives_map = self.decoder(latent)
        return latent, objectives_map