"""Occipital model for processing visual input."""
import torch
import torch.nn as nn


class OccipitalLobe(nn.Module):
    """
    The Occipital Lobe model, which is a convolutional autoencoder.
    It takes in an image and reconstructs it, learning a latent representation.
    """

    def __init__(self, latent_dim=128):
        super(OccipitalLobe, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, latent_dim),  # Assuming 256x256 input image
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),  # Use Sigmoid for image reconstruction
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_latent(self, x):
        """
        Get the latent representation of the input image.
        Returns two vectors: 'what' and 'where'.
        """
        latent = self.encoder(x)
        # Split the latent vector into two halves
        what, where = torch.chunk(latent, 2, dim=1)
        return what, where