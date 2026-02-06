"""Tests for the Occipital Lobe model."""
import torch
import unittest

from src.models.occipital import OccipitalLobe


class TestOccipitalLobe(unittest.TestCase):
    """Test suite for the OccipitalLobe model."""

    def test_forward_pass(self):
        """
        Test the forward pass of the OccipitalLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OccipitalLobe(latent_dim=128).to(device)
        # Create a dummy input tensor
        # The shape is (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        
        # Pass the input through the model
        reconstructed = model(dummy_input)
        
        # Check if the output shape is the same as the input shape
        self.assertEqual(dummy_input.shape, reconstructed.shape)

    def test_get_latent(self):
        """
        Test the get_latent method of the OccipitalLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OccipitalLobe(latent_dim=128).to(device)
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 3, 256, 256).to(device)

        # Get the latent representation
        latent = model.get_latent(dummy_input)

        # Check the shape of the latent vector
        self.assertEqual(latent.shape, (1, 128))

if __name__ == '__main__':
    unittest.main()
