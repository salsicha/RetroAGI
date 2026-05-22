import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.universal import encoder, decoder

class HippocampusLobe(nn.Module):
    """
    Hippocampus Lobe for spatial mapping and memory lookup.
    """
    def __init__(self, memory_size=500, latent_dim=128, *args, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        self.encoder = encoder
        self.decoder = decoder

        self.memory_keys = nn.Parameter(torch.randn(memory_size, latent_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, latent_dim))
        self.sparsity_ratio = 0.1

    def forward(self, latent_v, latent_t):
        combined_input = torch.cat([latent_v, latent_t], dim=-1)
        latent_h = self.encoder(combined_input, modality='vector')

        scores = torch.matmul(F.normalize(latent_h, dim=-1), F.normalize(self.memory_keys, dim=-1).t())
        k = max(1, int(self.memory_size * self.sparsity_ratio))
        topk_vals, topk_indices = torch.topk(scores, k=k, dim=-1)

        sparse_attn = torch.zeros_like(scores)
        sparse_attn.scatter_(-1, topk_indices, F.softmax(topk_vals, dim=-1))

        retrieved = torch.matmul(sparse_attn, self.memory_values)
        spatial_map = self.decoder(retrieved, modality='map')

        return retrieved, spatial_map

    def process(self, latent_v, latent_t):
        self.eval()
        with torch.no_grad():
            latent_h, spatial_map = self.forward(latent_v, latent_t)
        return latent_h, spatial_map

    def learn(self, signal):
        latents_v = signal.get('latents_v')
        latents_t = signal.get('latents_t')
        
        if latents_v is None or len(latents_v) == 0:
            return

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        optimizer.zero_grad()

        retrieved, spatial_map = self.forward(latents_v.detach(), latents_t.detach())

        combined = torch.cat([latents_v.detach(), latents_t.detach()], dim=-1)
        latent_h = self.encoder(combined, modality='vector')
        loss_assoc = F.mse_loss(retrieved, latent_h.detach())

        target_map = torch.zeros(spatial_map.size(0), 64, 64, dtype=torch.long).to(spatial_map.device)
        target_map[:, 30:34, 30:34] = 1 # Mock bootstrap mapping
        loss_spatial = F.cross_entropy(spatial_map, target_map)

        loss = loss_assoc + loss_spatial
        loss.backward()
        optimizer.step()
        self.eval()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
