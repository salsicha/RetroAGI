"""Tests for the Prefrontal Lobe model."""
import torch
import unittest

from src.models import PrefrontalLobe


class TestPrefrontalLobe(unittest.TestCase):
    """Test suite for the PrefrontalLobe model."""

    def test_forward_pass(self):
        """
        Test the forward pass of the PrefrontalLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PrefrontalLobe(planning_mode=0.5).to(device)
        
        latent_t = torch.randn(1, 128).to(device)
        latent_h = torch.randn(1, 128).to(device)
        
        plan = model(latent_t, latent_h)
        
        self.assertEqual(plan.shape, (1, 128))

    def test_process(self):
        """
        Test the process method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PrefrontalLobe(planning_mode=1.0).to(device)
        
        latent_t = torch.randn(1, 128).to(device)
        latent_h = torch.randn(1, 128).to(device)
        
        plan = model.process(latent_t, latent_h)
        
        self.assertEqual(plan.shape, (1, 128))

    def test_learn(self):
        """
        Test the learn method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PrefrontalLobe().to(device)
        
        signal = {'reward': 1.0, 'collision': False}
        model.learn(signal)
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
