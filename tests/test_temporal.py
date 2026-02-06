"""Tests for the Temporal Lobe model."""
import torch
import unittest

from src.models.temporal import TemporalLobe


class TestTemporalLobe(unittest.TestCase):
    """Test suite for the TemporalLobe model."""

    def test_forward_pass(self):
        """
        Test the forward pass of the TemporalLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalLobe(latent_dim=128, hidden_dim=128, vocab_size=10).to(device)
        
        # Create a dummy latent vector
        dummy_latent = torch.randn(1, 128).to(device)
        
        # Pass the latent vector through the model
        generated_sequence = model(dummy_latent)
        
        # Check the output shape
        self.assertEqual(generated_sequence.shape, (20,))

    def test_sequence_to_text(self):
        """
        Test the sequence_to_text method of the TemporalLobe model.
        """
        model = TemporalLobe(latent_dim=128, hidden_dim=128, vocab_size=10)
        
        # Create a dummy sequence of indices
        dummy_sequence = torch.tensor([2, 3, 4, 5, 1])
        
        # Convert the sequence to text
        generated_text = model.sequence_to_text(dummy_sequence)
        
        # Check if the output is a string
        self.assertIsInstance(generated_text, str)


if __name__ == '__main__':
    unittest.main()
