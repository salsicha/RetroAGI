"""Tests for the Hippocampus Lobe model."""
import torch
import unittest

from src.models import Hippocampus


class TestHippocampusLobe(unittest.TestCase):
    """Test suite for the HippocampusLobe model."""

    def test_forward_pass(self):
        """
        Test the forward pass of the HippocampusLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Hippocampus(memory_size=100, latent_dim=128).to(device)
        
        latent_v = torch.randn(1, 128).to(device)
        latent_t = torch.randn(1, 128).to(device)
        
        retrieved, spatial_map = model(latent_v, latent_t)
        
        self.assertEqual(retrieved.shape, (1, 128))
        self.assertEqual(spatial_map.shape, (1, 6, 64, 64))

    def test_process(self):
        """
        Test the process method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Hippocampus(memory_size=100, latent_dim=128).to(device)
        
        latent_v = torch.randn(1, 128).to(device)
        latent_t = torch.randn(1, 128).to(device)
        
        retrieved, spatial_map = model.process(latent_v, latent_t)
        
        self.assertEqual(retrieved.shape, (1, 128))
        self.assertEqual(spatial_map.shape, (1, 6, 64, 64))

    def test_learn(self):
        """
        Test the learn method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Hippocampus(memory_size=100, latent_dim=128).to(device)
        
        # Should run without errors
        model.learn({})
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
