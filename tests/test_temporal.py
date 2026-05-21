"""Tests for the Temporal Lobe model."""
import torch
import unittest
import numpy as np

from src.models import TemporalLobe


class TestTemporalLobe(unittest.TestCase):
    """Test suite for the TemporalLobe model."""

    def test_process_pass(self):
        """
        Test the process method of the TemporalLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalLobe(num_keypoints=16).to(device)
        
        # Create a dummy visual latent vector from Occipital Lobe
        dummy_latent = torch.randn(1, 128).to(device)
        
        # Process the latent vector
        latent_t, sprite_pos_pred = model.process(dummy_latent)
        
        # Check output shapes
        self.assertEqual(latent_t.shape, (1, 128))
        self.assertEqual(sprite_pos_pred.shape, (1, 32))

    def test_state_dict_serialization(self):
        """
        Test HGF state dictionary serialization and loading.
        """
        model = TemporalLobe(num_keypoints=16)
        
        # Modify some states
        model.nodes[0].mean = 5.0
        model.nodes[1].precision = 2.5
        
        state = model.state_dict()
        
        # Create a new model and load states
        new_model = TemporalLobe(num_keypoints=16)
        new_model.load_state_dict(state)
        
        self.assertEqual(new_model.nodes[0].mean, 5.0)
        self.assertEqual(new_model.nodes[1].precision, 2.5)


if __name__ == '__main__':
    unittest.main()
