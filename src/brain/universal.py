import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalEncoder(nn.Module):
    """
    Universal Encoder mapping multiple modalities (images, semantic maps, coordinates, vectors)
    into a unified latent space of size `latent_dim`.
    """
    def __init__(self, latent_dim=128, channels=3, num_classes=6, coords_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.num_classes = num_classes
        self.coords_dim = coords_dim

        # 1. Image Encoder Head (maps 3x64x64 to latent_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 7, stride=2, padding=3),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),        # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),       # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )

        # 2. Semantic Map Encoder Head (maps num_classesx64x64 to latent_dim)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(num_classes, 32, 7, stride=2, padding=3), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),          # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),         # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )

        # 3. Dynamic Vector / Coordinate Encoder Head
        # We use a modular dictionary to dynamically handle different input vector lengths
        self.vector_encoders = nn.ModuleDict()

    def _get_vector_encoder(self, in_dim):
        key = str(in_dim)
        if key not in self.vector_encoders:
            self.vector_encoders[key] = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.latent_dim)
            )
        return self.vector_encoders[key]

    def forward(self, x, modality='image'):
        """
        x: Input tensor of arbitrary modality.
        modality: 'image', 'map', 'coords', or 'vector'.
        """
        if modality == 'image':
            # Support any input resolution by interpolating to 64x64
            if x.shape[-2:] != (64, 64):
                x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            return self.image_encoder(x)
        
        elif modality == 'map':
            # Handle class-index integer maps (B, H, W) by converting to one-hot floats
            if len(x.shape) == 3 or (len(x.shape) == 4 and x.shape[1] == 1):
                x = x.squeeze(1) if len(x.shape) == 4 else x
                x = F.one_hot(x, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            if x.shape[-2:] != (64, 64):
                x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            return self.map_encoder(x)
        
        elif modality in ['coords', 'vector']:
            # Flatten inputs if multidimensional (e.g. (B, N, 2) to (B, N*2))
            x_flat = x.view(x.shape[0], -1)
            enc = self._get_vector_encoder(x_flat.shape[1])
            return enc(x_flat)
        
        else:
            raise ValueError(f"Unknown encoder modality: {modality}")


class UniversalDecoder(nn.Module):
    """
    Universal Decoder reconstructing original modalities (images, maps, coordinates, actions)
    from a unified latent representation of size `latent_dim`.
    """
    def __init__(self, latent_dim=128, channels=3, num_classes=6, coords_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.num_classes = num_classes
        self.coords_dim = coords_dim

        # 1. Image Decoder Head (maps latent_dim to 3x64x64)
        self.image_decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.image_decoder_net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1), # 64x64
            nn.Sigmoid()
        )

        # 2. Semantic Map Decoder Head (maps latent_dim to num_classesx64x64)
        self.map_decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.map_decoder_net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1) # 64x64
        )

        # 3. Dynamic Vector / Coordinate Decoder Head
        self.vector_decoders = nn.ModuleDict()

    def _get_vector_decoder(self, out_dim):
        key = str(out_dim)
        if key not in self.vector_decoders:
            self.vector_decoders[key] = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim)
            )
        return self.vector_decoders[key]

    def forward(self, z, modality='image', target_shape=None, target_dim=None):
        """
        z: Latent tensor of shape (B, latent_dim).
        modality: 'image', 'map', 'coords', 'vector', or 'action'.
        target_shape: Optional spatial shape (H, W) to interpolate image/map outputs.
        target_dim: Output dimension for vector/action decoding.
        """
        if modality == 'image':
            x_fc = self.image_decoder_fc(z).view(-1, 128, 8, 8)
            img = self.image_decoder_net(x_fc)
            if target_shape is not None and target_shape != (64, 64):
                img = F.interpolate(img, size=target_shape, mode='bilinear', align_corners=False)
            return img
        
        elif modality == 'map':
            x_fc = self.map_decoder_fc(z).view(-1, 128, 8, 8)
            seg_map = self.map_decoder_net(x_fc)
            if target_shape is not None and target_shape != (64, 64):
                seg_map = F.interpolate(seg_map, size=target_shape, mode='bilinear', align_corners=False)
            return seg_map
        
        elif modality in ['coords', 'vector', 'action']:
            # Determine output dimension
            out_dim = target_dim if target_dim is not None else self.coords_dim
            dec = self._get_vector_decoder(out_dim)
            return dec(z)
        
        else:
            raise ValueError(f"Unknown decoder modality: {modality}")

# Global universal encoder and decoder instances
# This guarantees that all modules literally share the same parameters and weights.
encoder = UniversalEncoder(latent_dim=128)
decoder = UniversalDecoder(latent_dim=128)
