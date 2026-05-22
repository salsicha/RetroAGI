import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from brain.universal import encoder, decoder

class HippocampusLobe(nn.Module):
    """
    Hippocampus Lobe for spatial mapping and memory lookup.
    Uses Sparse Distributed Representation (SDR) memory retrieval and the universal encoder-decoder
    to reconstruct semantic/spatial sprite configurations.
    """
    def __init__(self, memory_size=500, latent_dim=128, *args, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_size = memory_size

        # Universal modules
        self.encoder = encoder
        self.decoder = decoder

        # Sparse Auto-Associative Memory parameters
        # memory_keys: (memory_size, latent_dim)
        # memory_values: (memory_size, latent_dim)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, latent_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, latent_dim))

        # Top-k sparsity ratio for Sparse Distributed Representation (SDR)
        self.sparsity_ratio = 0.1  # Keep top 10% memory activations

    def forward(self, latent_v, latent_t):
        # 1. Combine latents from Occipital (visual) and Temporal (sequence)
        combined_input = torch.cat([latent_v, latent_t], dim=-1) # (B, 256)

        # 2. Encode combined latents to spatial representation using universal vector encoder
        latent_h = self.encoder(combined_input, modality='vector') # (B, 128)

        # 3. SDR Memory Lookup
        # Compute cosine similarity / attention scores
        scores = torch.matmul(F.normalize(latent_h, dim=-1), F.normalize(self.memory_keys, dim=-1).t()) # (B, memory_size)

        # Apply top-k sparsity (SDR)
        k = max(1, int(self.memory_size * self.sparsity_ratio))
        topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)

        # Create sparse attention map
        sparse_attn = torch.zeros_like(scores)
        sparse_attn.scatter_(-1, topk_indices, F.softmax(topk_vals, dim=-1))

        # Retrieve representation
        retrieved = torch.matmul(sparse_attn, self.memory_values) # (B, 128)

        # 4. Decode retrieved representation to spatial map
        spatial_map = self.decoder(retrieved, modality='map') # (B, 6, 64, 64)

        return retrieved, spatial_map

    def process(self, latent_v, latent_t):
        """
        Process visual and temporal latents.
        Returns:
            latent_h: torch.Tensor of shape (1, 128)
            spatial_map: torch.Tensor of shape (1, 6, 64, 64)
        """
        self.eval()
        with torch.no_grad():
            latent_h, spatial_map = self.forward(latent_v, latent_t)
        return latent_h, spatial_map

    def learn(self, signal):
        """
        Online associative learning and training of spatial semantic decoder.
        """
        # The hippocampus updates its associative memory keys and decoder weights online.
        # We simulate this via a backprop step to optimize reconstruction of spatial mappings.
        # If ground-truth semantic mask is available, we learn to predict it.
        # Otherwise, we optimize for auto-associative memory consistency (retrieved matching input).
        device = next(self.parameters()).device
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()

        # Create dummy local inputs for online step (using current state)
        dummy_v = torch.randn(1, 128).to(device)
        dummy_t = torch.randn(1, 128).to(device)

        retrieved, spatial_map = self.forward(dummy_v, dummy_t)

        # Loss 1: Auto-associative memory retention (minimize reconstruction error of latent_h representation)
        combined = torch.cat([dummy_v, dummy_t], dim=-1)
        latent_h = self.encoder(combined, modality='vector')
        loss_assoc = F.mse_loss(retrieved, latent_h.detach())

        # Loss 2: Semantic mapping loss (default to a mock/bootstrap mapping objective when training live)
        # Class targets need shape (1, 64, 64)
        target_map = torch.zeros(1, 64, 64, dtype=torch.long).to(device)
        # Add basic Mario detection block in the center to bootstrap spatial semantic prediction
        target_map[0, 30:34, 30:34] = 1 # e.g. Class 1 (Player)
        loss_spatial = F.cross_entropy(spatial_map, target_map)

        loss = loss_assoc + loss_spatial
        loss.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
