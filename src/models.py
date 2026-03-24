"""
src/models.py
Defines the 5 lobes of the RetroAGI brain based on Predictive Coding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Assuming pyhgf is installed. 
# If strictly pyhgf is required for the temporal dynamics:
try:
    from pyhgf import GaussianRandomWalk
except ImportError:
    # Fallback or mock for the sake of the code structure if package is missing
    class GaussianRandomWalk:
        def __init__(self, mean, precision, decay):
            self.mean = mean
            self.precision = precision
            self.decay = decay
        def update(self, x):
            # Simplified Kalman-like update for demonstration
            prediction = self.mean
            prediction_error = x - prediction
            self.mean = self.mean + 0.1 * prediction_error
            return self.mean

class OccipitalLobe(nn.Module):
    """
    Architecture: Spatial Autoencoder.
    Input: Game Frames (Batch, C, H, W)
    Output: Latent Keypoints (feature coordinates)
    Decoder: Reconstructs input frames.
    """
    def __init__(self, num_keypoints=32, channels=3):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Encoder: Extracts feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, num_keypoints, 3, stride=1, padding=1)
        )
        
        # Spatial Softmax: Converts feature maps to (x,y) coordinates
        self.softmax = nn.Softmax(dim=2) # Flattened spatial dimensions handled in forward

        # Decoder: Reconstructs image from keypoints (using Gaussian blobs or deconv)
        self.decoder_fc = nn.Linear(num_keypoints * 2, 1024)
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1), # 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def encode(self, x):
        """Returns normalized (x,y) coordinates for each keypoint."""
        feat = self.encoder(x)
        b, k, h, w = feat.shape
        
        # Spatial Softmax to get coordinates
        # Coordinate grid
        grid_x = torch.linspace(-1, 1, w).to(x.device).view(1, 1, 1, w).expand(b, k, h, w)
        grid_y = torch.linspace(-1, 1, h).to(x.device).view(1, 1, h, 1).expand(b, k, h, w)
        
        softmax_feat = F.softmax(feat.view(b, k, -1), dim=2).view(b, k, h, w)
        
        avg_x = torch.sum(softmax_feat * grid_x, dim=(2, 3))
        avg_y = torch.sum(softmax_feat * grid_y, dim=(2, 3))
        
        # Concatenate [x1, y1, x2, y2, ...]
        z = torch.stack([avg_x, avg_y], dim=2).view(b, -1)
        return z

    def decode(self, z):
        x = self.decoder_fc(z).view(-1, 64, 4, 4) # Reshape to small map
        # Upsample to full size (simplified architecture, output size needs to match input)
        # Assuming input is 64x64 or similar resized. 
        # For actual NES 240x256, we usually resize to 84x84.
        # This decoder structure assumes 3 upsamples of factor 2 -> 32x32. Needs adjusting for input size.
        # Let's assume input is 64x64.
        return self.decoder_net(x)

class TemporalLobe:
    """
    Architecture: Hierarchical Gaussian Filter (pyhgf).
    Input: Latent keypoints from Occipital.
    Function: Tracks movement dynamics of sprites (keypoints).
    """
    def __init__(self, num_keypoints):
        self.num_vars = num_keypoints * 2
        # Create a bank of HGF nodes, one for each coordinate
        # In a real scenario, we might use a multivariate node.
        self.nodes = [
            GaussianRandomWalk(mean=0.0, precision=1.0, decay=0.95)
            for _ in range(self.num_vars)
        ]
        
    def update(self, z_numpy):
        """
        z_numpy: Shape (batch, num_vars). Assuming batch=1 for online play.
        """
        predictions = []
        for i, val in enumerate(z_numpy[0]):
            pred = self.nodes[i].update(val)
            predictions.append(pred)
        return np.array(predictions)

    def predict_next(self):
        # Return current mean as prediction for next step
        return np.array([n.mean for n in self.nodes])

    def state_dict(self):
        return {'means': [n.mean for n in self.nodes], 'precisions': [n.precision for n in self.nodes]}

    def load_state_dict(self, state):
        for i, node in enumerate(self.nodes):
            node.mean = state['means'][i]
            node.precision = state['precisions'][i]

class Hippocampus(nn.Module):
    """
    Architecture: Sparse Distributed Representation (SDR).
    Input: Concatenated Occipital + Temporal latents.
    Function: Learns spatial relationships / map building.
    Decoder: Reconstructs spatial configuration.
    """
    def __init__(self, input_dim, memory_size=1000):
        super().__init__()
        self.memory_keys = nn.Parameter(torch.randn(memory_size, input_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, input_dim)) # Auto-associative
        
        # Spatial Decoder
        self.spatial_decoder_fc = nn.Linear(input_dim, 1024)
        self.spatial_decoder_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 6, 4, stride=2, padding=1)   # 64x64, 6 segmentation classes
        )
        
    def forward(self, x):
        # Simple attention mechanism to retrieve memory
        # x: (batch, input_dim)
        scores = torch.matmul(x, self.memory_keys.t())
        attn = F.softmax(scores, dim=1)
        
        # Retrieve
        retrieved = torch.matmul(attn, self.memory_values)
        
        # Decode spatial semantic map
        spatial_feat = self.spatial_decoder_fc(retrieved).view(-1, 64, 4, 4)
        spatial_map = self.spatial_decoder_net(spatial_feat)
        
        # Reconstruction (Decoder equivalent)
        # In predictive coding, we want to minimize x - retrieved
        return retrieved, spatial_map

class PrefrontalLobe(nn.Module):
    """
    Architecture: Planner / Meta-controller.
    Input: Temporal latent (current state) + Hippocampus (context).
    Output: Latent goal vector for Motor.
    Parameter: 'mode' (0=speed run, 1=max coins).
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + 1, 128), # +1 for the mode parameter
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        self.mode_param = 0.0 # Mutable parameter

    def set_mode(self, mode_value):
        self.mode_param = mode_value

    def forward(self, state_vector):
        # Append mode parameter to state
        mode_tensor = torch.full((state_vector.shape[0], 1), self.mode_param).to(state_vector.device)
        inp = torch.cat([state_vector, mode_tensor], dim=1)
        return self.fc(inp)

class MotorLobe(nn.Module):
    """
    Architecture: Policy Network.
    Input: Prefrontal Latent.
    Output: Keyboard presses (Action logits).
    """
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)
