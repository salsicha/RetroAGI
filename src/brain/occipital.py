import torch
import torch.nn as nn
import torch.nn.functional as F

class OccipitalLobe(nn.Module):
    """
    Architecture: Spatial Autoencoder.
    Input: Game Frames (Batch, C, H, W)
    Output: Latent Keypoints (feature coordinates)
    Decoder: Reconstructs input frames.
    """
    def __init__(self, num_keypoints=32, channels=3, input_size=(64, 64)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.input_size = input_size

        # Encoder: Extracts feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, num_keypoints, 3, stride=1, padding=1)
        )

        # Spatial Softmax: Converts feature maps to (x,y) coordinates
        self.softmax = nn.Softmax(dim=2)

        # Decoder: Reconstructs image from keypoints
        self.decoder_fc = nn.Linear(num_keypoints * 2, 64 * 4 * 4)
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
        x = self.decoder_fc(z).view(-1, 64, 4, 4)
        return self.decoder_net(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
