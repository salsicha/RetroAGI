"""Tests for the Motor Lobe model."""
import torch
import unittest

from src.models import MotorLobe


class TestMotorLobe(unittest.TestCase):
    """Test suite for the MotorLobe model."""

    def test_forward_pass(self):
        """
        Test the forward pass of the MotorLobe model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MotorLobe(num_actions=7).to(device)
        
        plan = torch.randn(1, 128).to(device)
        logits = model(plan)
        
        self.assertEqual(logits.shape, (1, 7))

    def test_decide(self):
        """
        Test the decide method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MotorLobe(num_actions=7).to(device)
        
        plan = torch.randn(1, 128).to(device)
        action = model.decide(plan)
        
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 7)

    def test_learn(self):
        """
        Test the learn method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MotorLobe(num_actions=7).to(device)
        
        signal = {'reward': 0.0, 'collision': True}
        model.learn(signal)
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
